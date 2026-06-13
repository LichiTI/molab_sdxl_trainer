"""JSON-only review for the Newbie tail8 attention target-depth candidate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_newbie_tail8_attention_compute_review_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
FAMILY = "newbie"
DEFAULT_MIN_SEED_PAIRS = 2
DEFAULT_MIN_THROUGHPUT_GAIN_PCT = 3.0
DEFAULT_MAX_ABS_TAIL_MEAN_LOSS_DELTA = 0.25
TARGET_DEPTH_ALTERNATE_SCOPE_IDS = ("tail4_attention", "tail12_attention", "balanced", "full")


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


def _seed_from_case_id(case_id: str, default: int = 1337) -> int:
    lowered = str(case_id or "").lower()
    if "seed" not in lowered:
        return default
    suffix = lowered.split("seed", 1)[1]
    digits: list[str] = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    return int("".join(digits)) if digits else default


def _metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("metrics"))


def _loss_stability(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("loss_stability"))


def _evidence_row(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "present": False,
            "case_id": "",
            "seed": 0,
            "status": "missing",
            "dominant_bottleneck": "",
            "data_wait_share": 0.0,
            "steady_mean_step_ms": 0.0,
            "steady_samples_per_second": 0.0,
            "final_loss": 0.0,
            "non_finite_loss_count": 0,
            "steps_completed": 0,
            "release_claim_eligible": False,
        }
    metrics = _metrics(payload)
    loss = _loss_stability(payload)
    case_id = str(payload.get("case_id") or "")
    return {
        "present": True,
        "case_id": case_id,
        "seed": _seed_from_case_id(case_id),
        "status": str(payload.get("status") or ""),
        "dominant_bottleneck": str(metrics.get("dominant_bottleneck") or ""),
        "data_wait_share": _round(metrics.get("data_wait_share")),
        "steady_mean_step_ms": _round(metrics.get("steady_mean_step_ms"), 4),
        "steady_samples_per_second": _round(metrics.get("steady_samples_per_second"), 6),
        "final_loss": _round(loss.get("final_loss"), 6),
        "non_finite_loss_count": _safe_int(loss.get("non_finite_loss_count")),
        "steps_completed": _safe_int(payload.get("steps_completed")),
        "release_claim_eligible": bool(_mapping(payload.get("release_claim")).get("eligible")),
    }


def _manifest_config(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping(_mapping(manifest).get("config"))


def _manifest_extra(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping(_mapping(manifest).get("extra"))


def _adapter_runtime(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping(_manifest_extra(manifest).get("adapter_runtime"))


def _phase_profile(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    loop = _mapping(_manifest_extra(manifest).get("training_loop_runtime"))
    step_phase = _mapping(loop.get("step_phase_profile"))
    return _mapping(step_phase.get("gpu_bubble_profile"))


def _phase_mean_ms(manifest: Mapping[str, Any] | None, key: str) -> float:
    return _round(_mapping(_phase_profile(manifest).get("phase_mean_ms")).get(key), 4)


def _phase_share(manifest: Mapping[str, Any] | None, key: str) -> float:
    return _round(_mapping(_phase_profile(manifest).get("phase_share")).get(key), 6)


def _runtime_phase_seconds(manifest: Mapping[str, Any] | None, label: str) -> float:
    timings = _mapping(_manifest_extra(manifest).get("runtime_phase_timings"))
    for item in _list(timings.get("phases")):
        row = _mapping(item)
        if str(row.get("label") or "") == label:
            return _round(row.get("dt_seconds"), 4)
    return 0.0


def _manifest_progress_row(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(manifest)
    if not payload:
        return {
            "present": False,
            "status": "missing",
            "global_step": 0,
            "total_steps": 0,
            "epoch": 0,
            "complete": False,
            "observed_steps": 0,
            "steady_mean_step_ms": 0.0,
            "steady_samples_per_second": 0.0,
            "first_step_ms": 0.0,
            "last_step_ms": 0.0,
            "runtime_total_seconds": 0.0,
            "newbie_transformer_smoke_seconds": 0.0,
            "epoch_1_train_seconds": 0.0,
            "forward_model_execution_mean_ms": 0.0,
            "forward_model_execution_share": 0.0,
            "backward_autograd_call_mean_ms": 0.0,
            "backward_autograd_call_share": 0.0,
            "data_wait_mean_ms": 0.0,
            "data_wait_share": 0.0,
            "dominant_bottleneck": "",
        }
    loop = _mapping(_manifest_extra(payload).get("training_loop_runtime"))
    timing = _mapping(loop.get("step_timing_window"))
    runtime = _mapping(_manifest_extra(payload).get("runtime_phase_timings"))
    phase = _phase_profile(payload)
    status = str(payload.get("status") or "unknown")
    return {
        "present": True,
        "status": status,
        "global_step": _safe_int(payload.get("global_step")),
        "total_steps": _safe_int(payload.get("total_steps")),
        "epoch": _safe_int(payload.get("epoch")),
        "complete": status in {"completed", "unknown"},
        "observed_steps": _safe_int(timing.get("observed_steps")),
        "steady_mean_step_ms": _round(timing.get("steady_mean_step_ms"), 4),
        "steady_samples_per_second": _round(timing.get("samples_per_second"), 6),
        "first_step_ms": _round(timing.get("first_step_ms"), 4),
        "last_step_ms": _round(timing.get("last_step_ms"), 4),
        "runtime_total_seconds": _round(runtime.get("total_seconds"), 4),
        "newbie_transformer_smoke_seconds": _runtime_phase_seconds(
            payload,
            "newbie_transformer_smoke",
        ),
        "epoch_1_train_seconds": _runtime_phase_seconds(payload, "epoch_1_train"),
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
        "dominant_bottleneck": str(phase.get("dominant_bottleneck") or ""),
    }


def _partial_diagnostic_row(progress: Mapping[str, Any]) -> dict[str, Any]:
    present = bool(progress.get("present"))
    incomplete = present and not bool(progress.get("complete"))
    forward_ms = _safe_float(progress.get("forward_model_execution_mean_ms"))
    backward_ms = _safe_float(progress.get("backward_autograd_call_mean_ms"))
    data_wait_share = _safe_float(progress.get("data_wait_share"))
    transformer_smoke_s = _safe_float(progress.get("newbie_transformer_smoke_seconds"))
    is_forward_anomaly = incomplete and forward_ms >= 10_000.0
    is_low_data_wait = data_wait_share <= 0.01
    classification = "not_applicable"
    if is_forward_anomaly and is_low_data_wait:
        classification = "incomplete_compute_bound_forward_anomaly_not_natural_load_evidence"
    elif incomplete:
        classification = "incomplete_candidate_manifest_needs_review"
    return {
        "classification": classification,
        "present": present,
        "incomplete": incomplete,
        "forward_anomaly": is_forward_anomaly,
        "low_data_wait": is_low_data_wait,
        "dominant_slow_substage": (
            "forward_model_execution" if forward_ms >= backward_ms else "backward_autograd_call"
        ),
        "natural_load_or_dataloader_regression_evidence": False
        if classification == "incomplete_compute_bound_forward_anomaly_not_natural_load_evidence"
        else None,
        "should_count_as_repeat_pair": False if incomplete else None,
        "requires_complete_rerun_or_environment_snapshot": bool(incomplete),
        "forward_model_execution_mean_ms": _round(forward_ms, 4),
        "backward_autograd_call_mean_ms": _round(backward_ms, 4),
        "data_wait_share": _round(data_wait_share, 6),
        "newbie_transformer_smoke_seconds": _round(transformer_smoke_s, 4),
    }


def _losses(summary: Mapping[str, Any] | None) -> list[float]:
    runs = _mapping(_mapping(summary).get("runs"))
    run = _mapping(runs.get("standard"))
    if not run:
        for item in runs.values():
            run = _mapping(item)
            if run:
                break
    raw = run.get("losses")
    if not isinstance(raw, list):
        return []
    return [_safe_float(item, float("nan")) for item in raw]


def _finite(values: Sequence[float]) -> list[float]:
    return [float(item) for item in values if math.isfinite(float(item))]


def _mean(values: Sequence[float]) -> float:
    finite = _finite(values)
    return sum(finite) / len(finite) if finite else 0.0


def _tail(values: Sequence[float], fraction: float = 0.25) -> list[float]:
    if not values:
        return []
    count = max(1, int(math.ceil(len(values) * max(min(fraction, 1.0), 0.0))))
    return list(values[-count:])


def _non_finite_count(values: Sequence[float]) -> int:
    return sum(1 for item in values if not math.isfinite(float(item)))


def _loss_curve_row(
    *,
    pair_id: str,
    baseline_summary: Mapping[str, Any] | None,
    candidate_summary: Mapping[str, Any] | None,
    max_tail_mean_delta: float,
) -> dict[str, Any]:
    baseline = _losses(baseline_summary)
    candidate = _losses(candidate_summary)
    aligned = min(len(baseline), len(candidate))
    baseline_aligned = baseline[:aligned]
    candidate_aligned = candidate[:aligned]
    baseline_tail = _tail(baseline_aligned)
    candidate_tail = _tail(candidate_aligned)
    tail_delta = _mean(candidate_tail) - _mean(baseline_tail)
    final_delta = (
        _safe_float(candidate_aligned[-1]) - _safe_float(baseline_aligned[-1])
        if aligned
        else 0.0
    )
    blockers: list[str] = []
    if not baseline_summary or not candidate_summary:
        blockers.append("summary_missing")
    if aligned <= 0:
        blockers.append("loss_curve_missing")
    if len(baseline) != len(candidate):
        blockers.append("loss_curve_length_mismatch")
    if _non_finite_count(baseline) or _non_finite_count(candidate):
        blockers.append("non_finite_loss_observed")
    if abs(tail_delta) > float(max_tail_mean_delta):
        blockers.append("tail_mean_loss_delta_above_threshold")
    return {
        "pair_id": pair_id,
        "loss_count_baseline": len(baseline),
        "loss_count_candidate": len(candidate),
        "aligned_loss_count": aligned,
        "baseline_final_loss": _round(baseline_aligned[-1] if aligned else 0.0),
        "candidate_final_loss": _round(candidate_aligned[-1] if aligned else 0.0),
        "final_loss_delta": _round(final_delta),
        "baseline_tail_mean_loss": _round(_mean(baseline_tail)),
        "candidate_tail_mean_loss": _round(_mean(candidate_tail)),
        "tail_mean_loss_delta": _round(tail_delta),
        "ok": not blockers,
        "blocked_reasons": blockers,
    }


def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _target_depth_progression_summary(
    *,
    completed_pair_count: int,
    required_pair_count: int,
    throughput_ready: bool,
    loss_curve_ready: bool,
    blockers: Sequence[str],
) -> dict[str, Any]:
    repeat_gate_ready = completed_pair_count >= required_pair_count and throughput_ready
    quality_gate_ready = completed_pair_count >= required_pair_count and loss_curve_ready
    stable_for_depth_comparison = repeat_gate_ready and quality_gate_ready and not blockers
    blocker_ids: list[str] = []
    if completed_pair_count < required_pair_count:
        blocker_ids.append("tail8_seed2027_complete_pair_missing")
    if not throughput_ready:
        blocker_ids.append("tail8_throughput_repeat_gate_not_ready")
    if not loss_curve_ready:
        blocker_ids.append("tail8_loss_curve_quality_gate_not_ready")
    if blockers:
        blocker_ids.extend(str(blocker) for blocker in blockers)
    blocker_ids.extend(
        [
            "target_scope_changes_training_semantics",
            "release_guard_not_opened",
        ]
    )
    blocker_ids = sorted({blocker for blocker in blocker_ids if blocker})
    if stable_for_depth_comparison:
        status = "tail8_stable_ready_for_protected_depth_comparison_review"
        next_required_evidence = [
            "tail4_tail8_tail12_protected_comparison_plan",
            "release_guard_rebuild_after_depth_comparison_review",
        ]
    elif completed_pair_count < required_pair_count:
        status = "blocked_waiting_tail8_seed2027_complete_pair"
        next_required_evidence = [
            "seed2027_tail8_complete_pair",
            "tail8_attention_compute_review_refresh",
            "release_guard_rebuild_after_tail8_refresh",
        ]
    elif not quality_gate_ready:
        status = "blocked_waiting_tail8_quality_gate"
        next_required_evidence = [
            "tail8_loss_curve_quality_review",
            "tail8_attention_compute_review_refresh",
            "release_guard_rebuild_after_tail8_refresh",
        ]
    else:
        status = "blocked_waiting_tail8_repeat_gate"
        next_required_evidence = [
            "tail8_repeat_throughput_review",
            "tail8_attention_compute_review_refresh",
            "release_guard_rebuild_after_tail8_refresh",
        ]
    return {
        "schema_version": 1,
        "progression": "newbie_target_depth_progression_v1",
        "status": status,
        "family": FAMILY,
        "current_candidate_scope": "tail8_attention",
        "candidate_stage": "tail8_attention_repeat_and_quality_gate",
        "completed_seed_pair_count": completed_pair_count,
        "required_seed_pair_count": required_pair_count,
        "tail8_repeat_gate_ready": repeat_gate_ready,
        "tail8_quality_gate_ready": quality_gate_ready,
        "tail8_stable_for_depth_comparison": stable_for_depth_comparison,
        "alternate_target_depth_comparison_allowed": stable_for_depth_comparison,
        "allowed_alternate_scope_ids": list(TARGET_DEPTH_ALTERNATE_SCOPE_IDS)
        if stable_for_depth_comparison
        else [],
        "blocked_alternate_scope_ids": []
        if stable_for_depth_comparison
        else list(TARGET_DEPTH_ALTERNATE_SCOPE_IDS),
        "blocker_count": len(blocker_ids),
        "blockers": blocker_ids,
        "next_required_evidence": next_required_evidence,
        "blocked_actions": [
            "do_not_compare_tail4_tail8_tail12_before_tail8_repeat_gate_is_stable",
            "do_not_change_newbie_default_target_scope_from_tail8_review",
            "do_not_count_target_depth_progression_as_release_evidence",
            "do_not_auto_start_target_depth_followup_from_review",
        ],
        "not_release_evidence": True,
        "publishable": False,
        "fail_closed": True,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


def _pair_row(
    pair: Mapping[str, Any],
    *,
    min_throughput_gain_pct: float,
    max_abs_tail_mean_loss_delta: float,
) -> dict[str, Any]:
    pair_id = str(pair.get("pair_id") or "")
    baseline = _evidence_row(_mapping(pair.get("baseline_evidence")))
    candidate = _evidence_row(_mapping(pair.get("candidate_evidence")))
    baseline_manifest = _mapping(pair.get("baseline_manifest"))
    candidate_manifest = _mapping(pair.get("candidate_manifest"))
    baseline_step = _safe_float(baseline.get("steady_mean_step_ms"))
    candidate_step = _safe_float(candidate.get("steady_mean_step_ms"))
    baseline_sps = _safe_float(baseline.get("steady_samples_per_second"))
    candidate_sps = _safe_float(candidate.get("steady_samples_per_second"))
    step_reduction_pct = (
        ((baseline_step - candidate_step) / baseline_step) * 100.0 if baseline_step > 0.0 else 0.0
    )
    throughput_gain_pct = (
        ((candidate_sps - baseline_sps) / baseline_sps) * 100.0 if baseline_sps > 0.0 else 0.0
    )
    candidate_adapter = _adapter_runtime(candidate_manifest)
    candidate_config = _manifest_config(candidate_manifest)
    baseline_progress = _manifest_progress_row(baseline_manifest)
    candidate_progress = _manifest_progress_row(candidate_manifest)
    candidate_partial_diagnostic = _partial_diagnostic_row(candidate_progress)
    loss_curve = _loss_curve_row(
        pair_id=pair_id,
        baseline_summary=_mapping(pair.get("baseline_summary")),
        candidate_summary=_mapping(pair.get("candidate_summary")),
        max_tail_mean_delta=float(max_abs_tail_mean_loss_delta),
    )

    blockers: list[str] = []
    if not baseline.get("present") or not candidate.get("present"):
        blockers.append("natural_data_wait_evidence_missing")
    if baseline.get("status") != "no_natural_data_wait" or candidate.get("status") != "no_natural_data_wait":
        blockers.append("unexpected_natural_data_wait_status")
    if (
        baseline.get("dominant_bottleneck") != "compute_bound"
        or candidate.get("dominant_bottleneck") != "compute_bound"
    ):
        blockers.append("pair_not_compute_bound")
    if throughput_gain_pct < float(min_throughput_gain_pct):
        blockers.append("throughput_gain_below_threshold")
    if baseline.get("non_finite_loss_count") or candidate.get("non_finite_loss_count"):
        blockers.append("non_finite_loss_observed")
    if baseline.get("release_claim_eligible") or candidate.get("release_claim_eligible"):
        blockers.append("unexpected_release_claim_eligible_pair")
    if str(candidate_config.get("newbie_target_scope") or "") != "tail8_attention":
        blockers.append("candidate_target_scope_not_tail8_attention")
    if _safe_int(candidate_adapter.get("injected_layer_count")) <= 0:
        blockers.append("candidate_adapter_runtime_missing")
    if candidate_progress["present"] and not candidate_progress["complete"]:
        blockers.append("candidate_run_manifest_incomplete")
    baseline_progress_step_ms = _safe_float(baseline_progress.get("steady_mean_step_ms"))
    candidate_progress_step_ms = _safe_float(candidate_progress.get("steady_mean_step_ms"))
    if (
        candidate_progress_step_ms > 10_000.0
        and (
            baseline_progress_step_ms <= 0.0
            or candidate_progress_step_ms >= baseline_progress_step_ms * 10.0
        )
    ):
        blockers.append("candidate_partial_step_window_abnormally_slow")
    if _safe_float(candidate_progress.get("newbie_transformer_smoke_seconds")) >= 30.0:
        blockers.append("candidate_transformer_smoke_abnormally_slow")
    if not loss_curve.get("ok"):
        blockers.append("loss_curve_not_ready")

    return {
        "pair_id": pair_id,
        "seed": _safe_int(pair.get("seed")) or _safe_int(candidate.get("seed")) or _safe_int(baseline.get("seed")),
        "baseline": baseline,
        "candidate": candidate,
        "baseline_manifest_progress": baseline_progress,
        "candidate_manifest_progress": candidate_progress,
        "candidate_partial_diagnostic": candidate_partial_diagnostic,
        "candidate_target_scope": str(candidate_config.get("newbie_target_scope") or ""),
        "candidate_injected_layer_count": _safe_int(candidate_adapter.get("injected_layer_count")),
        "candidate_target_module_count": _safe_int(candidate_adapter.get("newbie_target_module_count")),
        "candidate_trainable_adapter_parameter_count": _safe_int(
            candidate_adapter.get("trainable_adapter_parameter_count")
        ),
        "candidate_backward_autograd_call_mean_ms": _phase_mean_ms(
            candidate_manifest,
            "train_step_compute_substage.newbie.backward_autograd_call",
        ),
        "candidate_backward_autograd_call_share": _phase_share(
            candidate_manifest,
            "train_step_compute_substage.newbie.backward_autograd_call",
        ),
        "candidate_forward_model_execution_mean_ms": _phase_mean_ms(
            candidate_manifest,
            "train_step_compute_substage.newbie.forward_model_execution",
        ),
        "candidate_forward_model_execution_share": _phase_share(
            candidate_manifest,
            "train_step_compute_substage.newbie.forward_model_execution",
        ),
        "baseline_backward_autograd_call_mean_ms": _phase_mean_ms(
            baseline_manifest,
            "train_step_compute_substage.newbie.backward_autograd_call",
        ),
        "baseline_backward_autograd_call_share": _phase_share(
            baseline_manifest,
            "train_step_compute_substage.newbie.backward_autograd_call",
        ),
        "step_reduction_pct": _round(step_reduction_pct, 6),
        "throughput_gain_pct": _round(throughput_gain_pct, 6),
        "loss_delta": _round(
            _safe_float(candidate.get("final_loss")) - _safe_float(baseline.get("final_loss"))
        ),
        "loss_curve": loss_curve,
        "pair_blockers": blockers,
        "throughput_candidate": throughput_gain_pct >= float(min_throughput_gain_pct),
        "loss_curve_ready": bool(loss_curve.get("ok")),
    }


def build_newbie_tail8_attention_compute_review(
    pairs: Sequence[Mapping[str, Any]],
    *,
    min_seed_pairs: int = DEFAULT_MIN_SEED_PAIRS,
    min_throughput_gain_pct: float = DEFAULT_MIN_THROUGHPUT_GAIN_PCT,
    max_abs_tail_mean_loss_delta: float = DEFAULT_MAX_ABS_TAIL_MEAN_LOSS_DELTA,
) -> dict[str, Any]:
    rows = [
        _pair_row(
            pair,
            min_throughput_gain_pct=float(min_throughput_gain_pct),
            max_abs_tail_mean_loss_delta=float(max_abs_tail_mean_loss_delta),
        )
        for pair in pairs
    ]
    completed_pair_count = sum(
        1 for row in rows if row["baseline"]["present"] and row["candidate"]["present"]
    )
    throughput_ready = completed_pair_count >= int(min_seed_pairs) and all(
        bool(row.get("throughput_candidate")) for row in rows
    )
    loss_curve_ready = completed_pair_count >= int(min_seed_pairs) and all(
        bool(row.get("loss_curve_ready")) for row in rows
    )
    blockers: list[str] = []
    if completed_pair_count < int(min_seed_pairs):
        blockers.append("repeat_seed_pair_count_below_threshold")
    if not throughput_ready:
        blockers.append("throughput_repeat_gate_not_ready")
    if not loss_curve_ready:
        blockers.append("loss_curve_quality_gate_not_ready")
    pair_blockers = sorted({str(blocker) for row in rows for blocker in row.get("pair_blockers", [])})
    blockers.extend(blocker for blocker in pair_blockers if blocker not in blockers)

    max_tail_delta = max(
        (abs(_safe_float(_mapping(row.get("loss_curve")).get("tail_mean_loss_delta"))) for row in rows),
        default=0.0,
    )
    status = (
        "tail8_attention_repeat_candidate_quality_blocked"
        if throughput_ready and blockers
        else "tail8_attention_repeat_quality_review_ready"
        if not blockers
        else "tail8_attention_repeat_evidence_incomplete"
    )
    artifact_digest = _digest(
        {
            "rows": rows,
            "thresholds": {
                "min_seed_pairs": int(min_seed_pairs),
                "min_throughput_gain_pct": float(min_throughput_gain_pct),
                "max_abs_tail_mean_loss_delta": float(max_abs_tail_mean_loss_delta),
            },
        }
    )
    target_depth_progression_summary = _target_depth_progression_summary(
        completed_pair_count=completed_pair_count,
        required_pair_count=int(min_seed_pairs),
        throughput_ready=throughput_ready,
        loss_curve_ready=loss_curve_ready,
        blockers=blockers,
    )
    return {
        "report": REPORT,
        "schema_version": 1,
        "artifact_role": "gpu_bubble_newbie_tail8_attention_compute_review",
        "roadmap": ROADMAP,
        "family": FAMILY,
        "candidate": "newbie_target_scope:tail8_attention",
        "status": status,
        "completed_seed_pair_count": completed_pair_count,
        "required_seed_pair_count": int(min_seed_pairs),
        "summary": {
            "throughput_repeat_ready": throughput_ready,
            "loss_curve_quality_ready": loss_curve_ready,
            "min_throughput_gain_pct_observed": _round(
                min([row["throughput_gain_pct"] for row in rows], default=0.0),
                6,
            ),
            "max_abs_tail_mean_loss_delta": _round(max_tail_delta),
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "thresholds": {
            "min_seed_pairs": int(min_seed_pairs),
            "min_throughput_gain_pct": float(min_throughput_gain_pct),
            "max_abs_tail_mean_loss_delta": float(max_abs_tail_mean_loss_delta),
        },
        "rows": rows,
        "target_depth_progression_summary": target_depth_progression_summary,
        "artifact_digest": artifact_digest,
        "blocked_release_reasons": [
            "target_scope_changes_training_semantics",
            "repeat_and_quality_gates_not_release_ready",
            "natural_load_canary_still_not_satisfied",
            "release_guard_not_opened",
        ],
        "recommended_next_actions": [
            "investigate_seed2027_tail8_transformer_smoke_and_forward_slowdown_before_rerun",
            "run_or_collect_tail8_attention_long_window_seed2027_complete_pair",
            "review_tail8_attention_loss_curve_quality_after_two_complete_pairs",
            "compare_tail4_tail8_tail12_only_after_tail8_repeat_gate_is_stable",
            "keep_default_target_scope_unchanged_until_release_guard_policy_explicitly_opens",
        ],
        "release_claim": {
            "eligible": False,
            "scope": "not_eligible",
            "reason": "tail8 target-depth evidence is non-release until repeat throughput, quality, and gate semantics are resolved",
        },
        "not_release_evidence": True,
        "publishable": False,
        "fail_closed": True,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


__all__ = [
    "DEFAULT_MAX_ABS_TAIL_MEAN_LOSS_DELTA",
    "DEFAULT_MIN_SEED_PAIRS",
    "DEFAULT_MIN_THROUGHPUT_GAIN_PCT",
    "FAMILY",
    "REPORT",
    "ROADMAP",
    "build_newbie_tail8_attention_compute_review",
]
