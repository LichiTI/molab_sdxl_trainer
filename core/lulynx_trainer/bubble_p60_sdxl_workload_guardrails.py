"""Guardrails for P60 SDXL workload-shape follow-up actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _base_guardrail(action_id: str, *, priority: int = 6) -> dict[str, Any]:
    return {
        "id": action_id,
        "family": "sdxl",
        "priority": int(priority),
        "action_type": "guardrail",
        "status": "active",
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "manual_start_required": False,
        "requires_gpu_if_executed": False,
        "requires_external_input": False,
        "diagnostic_only": False,
        "release_relevant": False,
        "reasons": [],
        "warnings": [],
    }


def workload_telemetry_no_gpu_98_claim_guardrail(telemetry_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = str(telemetry_evidence.get("status") or "")
    comparison = _mapping(telemetry_evidence.get("comparison"))
    before_metrics = _mapping(_mapping(telemetry_evidence.get("before")).get("metrics"))
    after = _mapping(telemetry_evidence.get("after"))
    after_metrics = _mapping(after.get("metrics"))
    after_config = _mapping(after.get("config"))
    after_active = _safe_float(after_metrics.get("active_gpu_util_pct_mean"))
    gain_pct = _safe_float(comparison.get("steady_samples_per_second_gain_pct"))
    if status not in {"keep_recommended", "needs_review"} or gain_pct <= 0.0 or not (0.0 < after_active < 90.0):
        return []

    action = _base_guardrail("guardrail_sdxl_workload_telemetry_no_gpu_98_claim_current_axis", priority=7)
    action.update(
        {
            "blocked_actions": [
                "write_sdxl_workload_shape_as_gpu_98_99_claim",
                "promote_sdxl_batch2_as_general_gpu_saturation_fix",
                "treat_sdxl_batch2_telemetry_as_release_ready_without_repeat_stability",
            ],
            "reasons": [
                "active_gpu_still_under_target_after_positive_workload_shape_signal",
                "throughput_positive_but_gpu_not_saturated",
                "release_claim_requires_repeat_stability_and_case_specific_wording",
            ],
            "summary": {
                "before_active_gpu_util_pct_mean": before_metrics.get("active_gpu_util_pct_mean"),
                "after_active_gpu_util_pct_mean": after_metrics.get("active_gpu_util_pct_mean"),
                "active_gpu_util_pct_delta": comparison.get("active_gpu_util_pct_delta"),
                "steady_samples_per_second_gain_pct": comparison.get("steady_samples_per_second_gain_pct"),
                "data_wait_share_before": before_metrics.get("data_wait_share"),
                "data_wait_share_after": after_metrics.get("data_wait_share"),
            },
            "current_axis_scope": {
                "family": "sdxl",
                "resolution": _safe_int(after_config.get("resolution")),
                "baseline_train_batch_size": 1,
                "candidate_train_batch_size": 2,
                "dataloader_workers": 0,
                "dataloader_prefetch_factor": 2,
                "attention_backend": str(after_config.get("attention_backend") or ""),
                "sdpa_backend_policy": str(after_config.get("sdpa_backend_policy") or ""),
            },
            "allowed_next_axes": [
                "repeat_stability_or_failure_review",
                "heavier_workload_with_valid_eager_anchor",
                "host_gap_step_internal_profile",
                "different_attention_or_compile_hypothesis_with_new_evidence",
            ],
            "rationale": (
                "Batch2 telemetry improves throughput on this SDXL source axis, but active-window GPU util is "
                "still far below a saturation claim; keep the evidence as case-specific review and block GPU "
                "98/99 wording."
            ),
        }
    )
    return [action]


def batch2_repeat_cuda_failure_guardrail(repeat_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = str(repeat_evidence.get("status") or "")
    summary = _mapping(repeat_evidence.get("summary"))
    failed = (
        status in {"execution_failed", "execution_failed_needs_review"}
        or _safe_int(summary.get("execution_failure_count")) > 0
        or _safe_int(summary.get("candidate_execution_failure_count")) > 0
        or _safe_int(summary.get("missing_summary_count")) > 0
    )
    if not failed:
        return []

    decision = _mapping(repeat_evidence.get("decision"))
    action = _base_guardrail("guardrail_sdxl_do_not_promote_batch2_after_repeat_cuda_failures")
    action.update(
        {
            "blocked_actions": [
                "promote_sdxl_batch2_on_sucai_6_lulu",
                "use_batch2_repeat_active_gpu_delta_as_release_claim",
                "use_failed_batch2_repeat_as_compile_eager_anchor",
                "write_sdxl_batch2_release_gain_without_successful_repeat_summary",
            ],
            "reasons": [
                "batch2_repeat_candidate_execution_failed",
                "successful_candidate_summary_required_for_release_claim",
                "active_gpu_delta_before_failure_is_diagnostic_only",
                *_string_list(decision.get("reasons")),
            ],
            "summary": dict(summary),
            "thresholds": dict(_mapping(repeat_evidence.get("thresholds"))),
            "execution_failures": [dict(_mapping(item)) for item in repeat_evidence.get("execution_failures", [])],
            "current_axis_scope": {
                "family": "sdxl",
                "source_data": str(repeat_evidence.get("source_data") or ""),
                "resolution": 1024,
                "baseline_train_batch_size": 1,
                "candidate_train_batch_size": 2,
                "dataloader_workers": 0,
                "dataloader_prefetch_factor": 2,
            },
            "allowed_next_axes": [
                "diagnose_batch2_backward_cuda_failure",
                "smaller_microbatch_or_gradient_accumulation_without_batch2_memory_path",
                "different_resolution_or_bucket_shape_with_vram_gate",
                "compile_only_after_valid_eager_anchor",
            ],
            "rationale": (
                "The longer-window batch2 repeat raised active GPU before failing, but both candidate repeats lack "
                "successful summaries; this is diagnostic signal, not promotion evidence."
            ),
        }
    )
    return [action]


__all__ = [
    "batch2_repeat_cuda_failure_guardrail",
    "workload_telemetry_no_gpu_98_claim_guardrail",
]
