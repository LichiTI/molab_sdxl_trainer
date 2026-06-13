"""JSON-only review for the Newbie tail8 seed2027 forward anomaly."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_newbie_tail8_forward_anomaly_review_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
FAMILY = "newbie"

CRITICAL_CONFIG_KEYS = (
    "seed",
    "newbie_target_scope",
    "max_train_steps",
    "train_batch_size",
    "gradient_accumulation_steps",
    "checkpoint_policy",
    "mixed_precision",
    "optimizer_type",
    "learning_rate",
    "resolution",
    "enable_newbie_backward_op_profile",
    "enable_newbie_module_timing_profile",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _manifest_extra(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping(_mapping(manifest).get("extra"))


def _runtime_phase_seconds(manifest: Mapping[str, Any] | None, label: str) -> float:
    timings = _mapping(_manifest_extra(manifest).get("runtime_phase_timings"))
    for item in _list(timings.get("phases")):
        row = _mapping(item)
        if str(row.get("label") or "") == label:
            return _round(row.get("dt_seconds"), 4)
    return 0.0


def _phase_profile(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    loop = _mapping(_manifest_extra(manifest).get("training_loop_runtime"))
    step_phase = _mapping(loop.get("step_phase_profile"))
    return _mapping(step_phase.get("gpu_bubble_profile"))


def _phase_mean_ms(manifest: Mapping[str, Any] | None, key: str) -> float:
    return _round(_mapping(_phase_profile(manifest).get("phase_mean_ms")).get(key), 4)


def _phase_share(manifest: Mapping[str, Any] | None, key: str) -> float:
    return _round(_mapping(_phase_profile(manifest).get("phase_share")).get(key), 6)


def _step_window(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    loop = _mapping(_manifest_extra(manifest).get("training_loop_runtime"))
    return _mapping(loop.get("step_timing_window"))


def _adapter_runtime(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping(_manifest_extra(manifest).get("adapter_runtime"))


def _critical_config(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    config = _mapping(_mapping(manifest).get("config"))
    return {key: config.get(key) for key in CRITICAL_CONFIG_KEYS}


def _semantic_config_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_config = _critical_config(left)
    right_config = _critical_config(right)
    differences = {
        key: {"candidate": left_config.get(key), "reference": right_config.get(key)}
        for key in CRITICAL_CONFIG_KEYS
        if left_config.get(key) != right_config.get(key)
    }
    expected = {"seed"}
    unexpected = sorted(key for key in differences if key not in expected)
    return {
        "critical_key_count": len(CRITICAL_CONFIG_KEYS),
        "difference_count": len(differences),
        "unexpected_difference_count": len(unexpected),
        "differences": differences,
        "unexpected_differences": unexpected,
        "only_expected_seed_difference": set(differences) <= expected,
    }


def _run_row(label: str, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(manifest)
    step_window = _step_window(payload)
    adapter = _adapter_runtime(payload)
    return {
        "label": label,
        "present": bool(payload),
        "status": str(payload.get("status") or "missing"),
        "global_step": _safe_int(payload.get("global_step")),
        "total_steps": _safe_int(payload.get("total_steps")),
        "epoch": _safe_int(payload.get("epoch")),
        "newbie_target_scope": str(_mapping(payload.get("config")).get("newbie_target_scope") or ""),
        "seed": _safe_int(_mapping(payload.get("config")).get("seed")),
        "injected_layer_count": _safe_int(adapter.get("injected_layer_count")),
        "newbie_target_module_count": _safe_int(adapter.get("newbie_target_module_count")),
        "newbie_transformer_smoke_seconds": _runtime_phase_seconds(
            payload,
            "newbie_transformer_smoke",
        ),
        "epoch_1_train_seconds": _runtime_phase_seconds(payload, "epoch_1_train"),
        "runtime_total_seconds": _round(
            _mapping(_manifest_extra(payload).get("runtime_phase_timings")).get("total_seconds"),
            4,
        ),
        "observed_steps": _safe_int(step_window.get("observed_steps")),
        "steady_mean_step_ms": _round(step_window.get("steady_mean_step_ms"), 4),
        "steady_samples_per_second": _round(step_window.get("samples_per_second"), 6),
        "forward_model_execution_mean_ms": _phase_mean_ms(
            payload,
            "train_step_compute_substage.newbie.forward_model_execution",
        ),
        "forward_model_execution_share": _phase_share(
            payload,
            "train_step_compute_substage.newbie.forward_model_execution",
        ),
        "backward_autograd_call_mean_ms": _phase_mean_ms(
            payload,
            "train_step_compute_substage.newbie.backward_autograd_call",
        ),
        "backward_autograd_call_share": _phase_share(
            payload,
            "train_step_compute_substage.newbie.backward_autograd_call",
        ),
        "data_wait_mean_ms": _phase_mean_ms(payload, "data_wait"),
        "data_wait_share": _phase_share(payload, "data_wait"),
        "dominant_bottleneck": str(_phase_profile(payload).get("dominant_bottleneck") or ""),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return _round(numerator / denominator, 6) if denominator > 0.0 else 0.0


def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _recommended_next_actions(
    *,
    candidate_present: bool,
    reference_present: bool,
    baseline_present: bool,
    forward_anomaly: bool,
    low_data_wait: bool,
) -> list[str]:
    actions: list[str] = []
    if not candidate_present:
        actions.append("restore_or_rebuild_seed2027_tail8_candidate_manifest_before_rerun")
    if not reference_present:
        actions.append("restore_seed1337_tail8_reference_manifest_before_comparison")
    if not baseline_present:
        actions.append("restore_seed2027_layer0_baseline_manifest_before_comparison")
    if candidate_present and reference_present and baseline_present and forward_anomaly:
        actions.append("capture_gpu_telemetry_and_environment_snapshot_before_seed2027_tail8_rerun")
        actions.append(
            "rerun_seed2027_tail8_long_window_once_with_same_request_config_after_runtime_state_is_known_idle"
        )
    elif candidate_present and not forward_anomaly:
        actions.append("keep_seed2027_tail8_candidate_as_missing_forward_anomaly_review_only")
    if candidate_present and low_data_wait:
        actions.append("treat_partial_manifest_as_diagnostic_only_not_repeat_evidence")
    actions.append("compare_tail4_tail8_tail12_only_after_seed2027_tail8_repeat_is_complete")
    return actions


def build_newbie_tail8_forward_anomaly_review(
    *,
    candidate_seed2027_manifest: Mapping[str, Any] | None,
    reference_tail8_seed1337_manifest: Mapping[str, Any] | None,
    baseline_seed2027_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate = _run_row("seed2027_tail8_partial", candidate_seed2027_manifest)
    tail8_ref = _run_row("seed1337_tail8_completed", reference_tail8_seed1337_manifest)
    baseline = _run_row("seed2027_layer0_completed", baseline_seed2027_manifest)
    config_diff = _semantic_config_diff(
        _mapping(candidate_seed2027_manifest),
        _mapping(reference_tail8_seed1337_manifest),
    )
    smoke_ratio_vs_tail8 = _ratio(
        _safe_float(candidate.get("newbie_transformer_smoke_seconds")),
        _safe_float(tail8_ref.get("newbie_transformer_smoke_seconds")),
    )
    step_ratio_vs_tail8 = _ratio(
        _safe_float(candidate.get("steady_mean_step_ms")),
        _safe_float(tail8_ref.get("steady_mean_step_ms")),
    )
    forward_ratio_vs_tail8 = _ratio(
        _safe_float(candidate.get("forward_model_execution_mean_ms")),
        _safe_float(tail8_ref.get("forward_model_execution_mean_ms")),
    )
    smoke_ratio_vs_baseline = _ratio(
        _safe_float(candidate.get("newbie_transformer_smoke_seconds")),
        _safe_float(baseline.get("newbie_transformer_smoke_seconds")),
    )
    low_data_wait = _safe_float(candidate.get("data_wait_share")) <= 0.01
    forward_anomaly = _safe_float(candidate.get("forward_model_execution_mean_ms")) >= 10_000.0
    incomplete = str(candidate.get("status") or "") != "completed"
    classification = (
        "seed2027_tail8_incomplete_forward_runtime_anomaly"
        if incomplete and forward_anomaly and low_data_wait
        else "seed2027_tail8_incomplete_needs_review"
        if incomplete
        else "seed2027_tail8_no_anomaly_detected"
    )
    blockers: list[str] = []
    if incomplete:
        blockers.append("candidate_run_incomplete")
    if forward_anomaly:
        blockers.append("candidate_forward_model_execution_abnormally_slow")
    if _safe_float(candidate.get("newbie_transformer_smoke_seconds")) >= 30.0:
        blockers.append("candidate_transformer_smoke_abnormally_slow")
    if low_data_wait:
        blockers.append("candidate_not_natural_load_or_dataloader_evidence")
    if config_diff["unexpected_difference_count"]:
        blockers.append("candidate_has_unexpected_critical_config_differences")
    blockers = sorted(set(blockers))
    recommended_next_actions = _recommended_next_actions(
        candidate_present=bool(candidate.get("present")),
        reference_present=bool(tail8_ref.get("present")),
        baseline_present=bool(baseline.get("present")),
        forward_anomaly=forward_anomaly,
        low_data_wait=low_data_wait,
    )
    digest_payload = {
        "candidate": candidate,
        "tail8_ref": tail8_ref,
        "baseline": baseline,
        "config_diff": config_diff,
        "classification": classification,
        "blockers": blockers,
        "recommended_next_actions": recommended_next_actions,
    }
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_newbie_tail8_forward_anomaly_review",
        "family": FAMILY,
        "candidate": "newbie_target_scope:tail8_attention seed:2027",
        "status": classification,
        "review_ready": True,
        "candidate_run": candidate,
        "reference_runs": {
            "tail8_seed1337": tail8_ref,
            "layer0_seed2027_baseline": baseline,
        },
        "critical_config_diff_vs_tail8_seed1337": config_diff,
        "comparison": {
            "transformer_smoke_ratio_vs_tail8_seed1337": smoke_ratio_vs_tail8,
            "steady_step_ms_ratio_vs_tail8_seed1337": step_ratio_vs_tail8,
            "forward_model_execution_ratio_vs_tail8_seed1337": forward_ratio_vs_tail8,
            "transformer_smoke_ratio_vs_layer0_seed2027": smoke_ratio_vs_baseline,
        },
        "diagnosis": {
            "candidate_incomplete": incomplete,
            "forward_runtime_anomaly": forward_anomaly,
            "low_data_wait": low_data_wait,
            "natural_load_or_dataloader_regression_evidence": False,
            "counts_as_tail8_repeat_pair": False,
            "root_cause_proven": False,
            "root_cause_confidence": "low_without_independent_gpu_telemetry_or_environment_snapshot",
        },
        "blockers": blockers,
        "recommended_next_actions": recommended_next_actions,
        "artifact_digest": _digest(digest_payload),
        "not_release_evidence": True,
        "fail_closed": True,
        "publishable": False,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


__all__ = [
    "FAMILY",
    "REPORT",
    "ROADMAP",
    "build_newbie_tail8_forward_anomaly_review",
]
