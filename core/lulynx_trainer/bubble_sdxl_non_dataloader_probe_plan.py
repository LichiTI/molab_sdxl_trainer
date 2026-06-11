"""Plan SDXL probes after the current DataLoader source axis is exhausted."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SDXL_NON_DATALOADER_PROBE_PLAN_REPORT = "bubble_sdxl_non_dataloader_probe_plan_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any) -> float:
    return round(_safe_float(value), 6)


def _case_metric(case: Mapping[str, Any], side: str, name: str) -> float:
    return _safe_float(_mapping(case.get(side)).get(name))


def _case_comparison(case: Mapping[str, Any], name: str) -> float:
    return _safe_float(_mapping(case.get("comparison")).get(name))


def _max_metric(cases: Sequence[Mapping[str, Any]], name: str) -> float:
    values: list[float] = []
    for case in cases:
        values.append(_case_metric(case, "before", name))
        values.append(_case_metric(case, "after", name))
    return round(max(values), 6) if values else 0.0


def _representative_cases(cases: Sequence[Mapping[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    ranked = sorted(
        cases,
        key=lambda item: (
            _case_comparison(item, "steady_samples_per_second_gain_pct"),
            _safe_int(item.get("sample_offset"), 999999),
            str(item.get("case_id") or ""),
        ),
        reverse=True,
    )
    return [
        {
            "case_id": str(case.get("case_id") or ""),
            "sample_offset": _safe_int(case.get("sample_offset"), -1),
            "resolution": _safe_int(case.get("resolution"), 0),
            "samples": _safe_int(case.get("samples"), 0),
            "before_data_wait_share": _round(_mapping(case.get("before")).get("data_wait_share")),
            "after_data_wait_share": _round(_mapping(case.get("after")).get("data_wait_share")),
            "throughput_gain_pct": _round(_mapping(case.get("comparison")).get("steady_samples_per_second_gain_pct")),
            "loss_stability_status": str(case.get("loss_stability_status") or "unknown"),
        }
        for case in ranked[:limit]
    ]


def _base_item(
    *,
    track: str,
    category: str,
    priority: int,
    status: str,
) -> dict[str, Any]:
    return {
        "id": f"sdxl_{track}",
        "family": "sdxl",
        "track": track,
        "category": category,
        "priority": int(priority),
        "status": status,
        "safe_to_auto_start": False,
        "manual_start_required": status == "manual_review_ready",
        "requires_gpu_if_executed": False,
        "release_claim_allowed": False,
        "release_relevant": False,
        "diagnostic_only": False,
        "reason_codes": [],
        "warnings": [],
    }


def _repeat_guardrail(summary: Mapping[str, Any]) -> dict[str, Any]:
    item = _base_item(
        track="do_not_repeat_workers_prefetch_on_current_axis",
        category="guardrail",
        priority=5,
        status="active",
    )
    item.update(
        {
            "manual_start_required": False,
            "reason_codes": [
                "current_sdxl_source_axis_exhausted",
                "baseline_data_wait_gate_failed",
                "release_claim_gate_failed",
            ],
            "blocked_actions": [
                "repeat_sdxl_workers_prefetch_on_sucai_6_lulu",
                "promote_workers_prefetch_as_p60_release_gain",
            ],
            "evidence": {
                "case_count": _safe_int(summary.get("case_count")),
                "release_eligible_count": _safe_int(summary.get("release_eligible_count")),
                "baseline_data_wait_below_threshold_count": _safe_int(
                    summary.get("baseline_data_wait_below_threshold_count")
                ),
                "after_data_wait_worse_count": _safe_int(summary.get("after_data_wait_worse_count")),
            },
            "rationale": (
                "Existing SDXL real-material workers/prefetch evidence did not pass the natural "
                "data-wait release gate on this source axis."
            ),
        }
    )
    return item


def _phase_gate_guardrail(summary: Mapping[str, Any]) -> dict[str, Any]:
    item = _base_item(
        track="require_phase_gate_before_throughput_claim",
        category="guardrail",
        priority=8,
        status="active",
    )
    item.update(
        {
            "manual_start_required": False,
            "reason_codes": ["large_throughput_delta_without_data_wait_gain"],
            "blocked_actions": ["use_samples_per_second_delta_as_release_claim_without_phase_gate"],
            "evidence": {
                "large_throughput_without_data_wait_gain_count": _safe_int(
                    summary.get("large_throughput_without_data_wait_gain_count")
                ),
                "mean_throughput_gain_pct": _round(summary.get("mean_throughput_gain_pct")),
                "mean_before_data_wait_share": _round(summary.get("mean_before_data_wait_share")),
                "mean_after_data_wait_share": _round(summary.get("mean_after_data_wait_share")),
            },
            "rationale": (
                "Some SDXL throughput deltas are large while the data-wait gate still fails, so "
                "phase attribution must precede any claim."
            ),
        }
    )
    return item


def _workload_shape_probe(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case_count = max(_safe_int(summary.get("case_count")), 1)
    compute_count = _safe_int(summary.get("baseline_compute_bound_count"))
    ready = compute_count >= max(1, case_count // 2)
    item = _base_item(
        track="batch_or_microbatch_shape_probe",
        category="manual_probe",
        priority=20 if ready else 65,
        status="manual_review_ready" if ready else "low_signal_manual_review",
    )
    item.update(
        {
            "requires_gpu_if_executed": True,
            "probe_profile": "workload_shape_next_request_ab",
            "reason_codes": ["baseline_compute_bound_majority"] if ready else ["baseline_compute_bound_not_dominant"],
            "candidate_axes": [
                "train_batch_size_step_up",
                "gradient_accumulation_control",
                "resolution_or_bucket_shape_control",
                "token_length_or_caption_bucket_stability_control",
            ],
            "request_boundary": "advisor_patch_next_request_only",
            "required_inputs": [
                "before_and_after_run_manifest",
                "bubble_controller_report",
                "steady_window_samples_per_second",
                "active_window_gpu_util",
                "peak_vram_ratio",
                "loss_stability",
            ],
            "required_gates": [
                "throughput_gain_over_baseline",
                "loss_stability_required",
                "vram_ratio_within_limit",
                "phase_profile_boundary",
                "case_specific_release_wording_only",
            ],
            "evidence": {
                "baseline_compute_bound_count": compute_count,
                "case_count": case_count,
                "mean_before_data_wait_share": _round(summary.get("mean_before_data_wait_share")),
                "representative_cases": _representative_cases(cases),
            },
            "rationale": (
                "SDXL baselines are mostly compute-bound instead of naturally data-wait-bound; "
                "the next useful GPU-heavy probe is workload shape, not DataLoader workers."
            ),
        }
    )
    return item


def _optimizer_attention_probe(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    max_optimizer = _max_metric(cases, "optimizer_share")
    ready = max_optimizer >= 0.07
    item = _base_item(
        track="optimizer_or_attention_backend_probe",
        category="manual_probe",
        priority=30 if ready else 70,
        status="manual_review_ready" if ready else "low_signal_manual_review",
    )
    item.update(
        {
            "requires_gpu_if_executed": True,
            "probe_profile": "optimizer_attention_next_request_ab",
            "reason_codes": ["optimizer_share_spike_observed"] if ready else ["optimizer_share_low_signal"],
            "candidate_axes": [
                "torch_fused_or_foreach_adamw_guarded_ab",
                "attention_backend_or_compile_static_shape_review",
                "optimizer_update_phase_attribution",
            ],
            "request_boundary": "advisor_patch_next_request_or_manual_benchmark_only",
            "required_inputs": [
                "phase_profile_optimizer_share",
                "optimizer_backend_and_args",
                "attention_or_compile_backend_flags",
                "before_and_after_loss_stability",
            ],
            "required_gates": [
                "throughput_gain_over_baseline",
                "loss_stability_required",
                "optimizer_state_compatibility",
                "case_specific_release_wording_only",
            ],
            "evidence": {
                "max_optimizer_share": max_optimizer,
                "mean_throughput_gain_pct": _round(summary.get("mean_throughput_gain_pct")),
            },
            "rationale": (
                "One or more SDXL windows show enough optimizer/update share to justify a guarded "
                "optimizer or attention-backend probe after workload shape is reviewed."
            ),
        }
    )
    return item


def _transfer_probe(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    max_h2d = _max_metric(cases, "h2d_transfer_share")
    ready = max_h2d >= 0.03
    item = _base_item(
        track="transfer_or_h2d_profile_review",
        category="manual_probe",
        priority=40 if ready else 80,
        status="manual_review_ready" if ready else "low_signal_manual_review",
    )
    item.update(
        {
            "requires_gpu_if_executed": True,
            "probe_profile": "transfer_profile_review",
            "reason_codes": ["h2d_transfer_share_high"] if ready else ["h2d_transfer_share_low_signal"],
            "candidate_axes": [
                "pin_memory_and_non_blocking_transfer_audit",
                "h2d_transfer_phase_profile_review",
                "runtime_transfer_adapter_boundary_check",
            ],
            "request_boundary": "existing_transfer_runtime_adapter_or_next_request_patch",
            "required_inputs": [
                "h2d_transfer_share",
                "pin_memory_state",
                "data_transfer_non_blocking_state",
                "before_after_steady_window",
            ],
            "required_gates": [
                "h2d_share_reduction_or_throughput_gain",
                "loss_stability_required",
                "no_vram_regression",
            ],
            "evidence": {"max_h2d_transfer_share": max_h2d},
            "rationale": (
                "Current SDXL evidence does not make transfer the leading hypothesis unless H2D "
                "share is high in a new run, so keep it as a lower-priority profile review."
            ),
        }
    )
    return item


def _host_probe(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    max_host_gap = _max_metric(cases, "host_gap_share")
    ready = max_host_gap >= 0.12
    item = _base_item(
        track="host_scheduling_closed_loop_probe",
        category="manual_probe",
        priority=45 if ready else 85,
        status="manual_review_ready" if ready else "low_signal_manual_review",
    )
    item.update(
        {
            "requires_gpu_if_executed": True,
            "probe_profile": "host_scheduling_closed_loop_or_report_only",
            "reason_codes": ["host_gap_share_high"] if ready else ["host_gap_share_low_signal"],
            "candidate_axes": [
                "disable_sync_profiler_mode_if_enabled",
                "increase_logging_interval",
                "move_validation_or_checkpoint_out_of_hot_window",
            ],
            "request_boundary": "low_risk_runtime_allowlist_or_next_request_patch",
            "required_inputs": [
                "host_gap_share",
                "profiler_mode",
                "logging_interval",
                "checkpoint_or_validation_cadence",
                "closed_loop_keep_or_rollback_evidence",
            ],
            "required_gates": [
                "throughput_gain_over_baseline",
                "rollback_if_regressed",
                "loss_stability_required",
            ],
            "evidence": {"max_host_gap_share": max_host_gap},
            "rationale": (
                "Host scheduling is already supported by the low-risk closed-loop scaffold, but "
                "current SDXL evidence makes it a lower-priority path unless host gap rises."
            ),
        }
    )
    return item


def _blocked_missing_evidence_item() -> dict[str, Any]:
    item = _base_item(
        track="collect_sdxl_real_material_ab_evidence",
        category="missing_evidence",
        priority=10,
        status="blocked_missing_evidence",
    )
    item.update(
        {
            "manual_start_required": False,
            "reason_codes": ["missing_sdxl_real_material_ab_evidence"],
            "rationale": "No SDXL real-material A/B evidence is available to plan non-DataLoader probes.",
        }
    )
    return item


def _summary(items: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = sorted({str(item.get("status") or "") for item in items if item.get("status")})
    categories = sorted({str(item.get("category") or "") for item in items if item.get("category")})
    return {
        "item_count": len(items),
        "case_count": len(cases),
        "guardrail_count": sum(1 for item in items if item.get("category") == "guardrail"),
        "manual_probe_ready_count": sum(1 for item in items if item.get("status") == "manual_review_ready"),
        "low_signal_probe_count": sum(1 for item in items if item.get("status") == "low_signal_manual_review"),
        "requires_gpu_if_executed_count": sum(1 for item in items if item.get("requires_gpu_if_executed")),
        "status_counts": {status: sum(1 for item in items if item.get("status") == status) for status in statuses},
        "category_counts": {
            category: sum(1 for item in items if item.get("category") == category) for category in categories
        },
    }


def _plan_status(source_status: str, summary: Mapping[str, Any]) -> str:
    if source_status == "no_sdxl_ab_evidence":
        return "needs_sdxl_real_material_ab_evidence"
    if source_status == "sdxl_release_candidate_present":
        return "release_candidate_present_review_claim_gates"
    if _safe_int(summary.get("manual_probe_ready_count")):
        return "manual_probe_plan_ready"
    if _safe_int(summary.get("low_signal_probe_count")):
        return "guardrail_with_low_signal_probe_reviews"
    return "guardrail_only"


def build_sdxl_non_dataloader_probe_plan(sdxl_investigation: Mapping[str, Any]) -> dict[str, Any]:
    """Build a conservative SDXL probe plan from the aggregate investigation report."""

    source = _mapping(sdxl_investigation)
    source_status = str(source.get("status") or "")
    source_summary = _mapping(source.get("summary"))
    cases = [_mapping(item) for item in source.get("cases", []) if _mapping(item)]
    items: list[dict[str, Any]] = []

    if not cases:
        items.append(_blocked_missing_evidence_item())
    else:
        items.append(_repeat_guardrail(source_summary))
        if _safe_int(source_summary.get("large_throughput_without_data_wait_gain_count")):
            items.append(_phase_gate_guardrail(source_summary))
        items.append(_workload_shape_probe(source_summary, cases))
        items.append(_optimizer_attention_probe(source_summary, cases))
        items.append(_transfer_probe(cases))
        items.append(_host_probe(cases))

    items.sort(key=lambda item: (int(item.get("priority") or 999), str(item.get("id") or "")))
    summary = _summary(items, cases)
    return {
        "schema_version": 1,
        "report": SDXL_NON_DATALOADER_PROBE_PLAN_REPORT,
        "status": _plan_status(source_status, summary),
        "family": "sdxl",
        "source_investigation_report": str(source.get("report") or ""),
        "source_investigation_status": source_status,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "recommended_item_ids": [
            str(item.get("id")) for item in items if item.get("status") == "manual_review_ready"
        ][:4],
        "source_next_actions": _string_list(source_summary.get("next_actions")),
        "summary": summary,
        "items": items,
        "notes": [
            "This plan does not start GPU work.",
            "It turns negative SDXL workers/prefetch evidence into safer non-DataLoader hypotheses.",
            "Any resulting benchmark must still pass throughput, loss, VRAM and case-specific claim gates.",
        ],
    }


__all__ = ["SDXL_NON_DATALOADER_PROBE_PLAN_REPORT", "build_sdxl_non_dataloader_probe_plan"]
