"""JSON-only review gate for SDXL CUDA debug repeat promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_promotion_review_v0"
REPEAT_REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_followup_v0"
P60_REVIEW_ACTION_ID = "review_sdxl_batch2_cuda_debug_repeat_followup_results"
ROADMAP = "gpu_bubble_elimination_roadmap.md"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _gate(gate_id: str, passed: bool, reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "passed" if passed else "blocked",
        "reasons": reasons or [],
    }


def _p60_review_action(p60_plan: Mapping[str, Any]) -> Mapping[str, Any]:
    for raw in _list(p60_plan.get("actions")):
        action = _mapping(raw)
        if str(action.get("id") or "") == P60_REVIEW_ACTION_ID:
            return action
    return {}


def _repeat_comparisons(repeat_evidence: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    thresholds = _mapping(repeat_evidence.get("thresholds"))
    min_gain = _safe_float(thresholds.get("min_throughput_gain_pct"), 3.0)
    max_loss = _safe_float(thresholds.get("max_loss_regression_ratio"), 0.05)
    max_vram = _safe_float(thresholds.get("max_vram_ratio"), 0.92)
    comparisons: list[dict[str, Any]] = []
    gate_reasons: list[str] = []
    for raw in _list(repeat_evidence.get("comparisons")):
        item = _mapping(raw)
        comparison = _mapping(item.get("comparison"))
        decision = _mapping(item.get("decision"))
        after = _mapping(item.get("after_metrics"))
        gain = _safe_float(comparison.get("steady_samples_per_second_gain_pct"))
        loss_ratio = _safe_float(decision.get("loss_regression_ratio"))
        memory_ratio = _safe_float(after.get("memory_ratio"))
        status = str(item.get("status") or decision.get("status") or "")
        reasons: list[str] = []
        if status != "keep_recommended":
            reasons.append("comparison_not_keep_recommended")
        if gain < min_gain:
            reasons.append("throughput_gain_below_threshold")
        if loss_ratio > max_loss:
            reasons.append("loss_regression_ratio_above_threshold")
        if memory_ratio > max_vram:
            reasons.append("vram_ratio_above_threshold")
        comparison_id = str(item.get("id") or "")
        if reasons:
            gate_reasons.append(f"comparison_failed:{comparison_id or 'unknown'}")
        comparisons.append(
            {
                "id": comparison_id,
                "status": status,
                "steady_samples_per_second_gain_pct": gain,
                "loss_regression_ratio": loss_ratio,
                "memory_ratio": memory_ratio,
                "gate_passed": not reasons,
                "gate_reasons": reasons,
            }
        )
    if not comparisons:
        gate_reasons.append("comparison_rows_missing")
    return comparisons, gate_reasons


def _repeat_evidence_ready(repeat_evidence: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(repeat_evidence.get("report") or "") != REPEAT_REPORT:
        reasons.append("repeat_report_mismatch")
    if str(repeat_evidence.get("status") or "") != "repeat_candidate_review":
        reasons.append("repeat_not_candidate_review")
    selected = _safe_int(summary.get("selected_command_count"))
    completed_reports = _safe_int(summary.get("completed_report_count"))
    completed_summaries = _safe_int(summary.get("completed_summary_count"))
    comparison_count = _safe_int(summary.get("comparison_count"))
    pass_count = _safe_int(summary.get("repeat_candidate_pass_count"))
    fully_repeated = _safe_int(summary.get("fully_repeated_candidate_count"))
    failures_or_missing = (
        _safe_int(summary.get("execution_failure_count"))
        + _safe_int(summary.get("missing_report_count"))
        + _safe_int(summary.get("missing_summary_count"))
    )
    if not selected or completed_reports < selected or completed_summaries < selected:
        reasons.append("repeat_outputs_incomplete")
    if not comparison_count or pass_count < comparison_count or not fully_repeated:
        reasons.append("repeat_candidates_not_fully_passed")
    if failures_or_missing:
        reasons.append("repeat_failures_or_missing_outputs_present")
    return reasons


def _manual_surface_ready(action: Mapping[str, Any], cuda_review_items: Sequence[Mapping[str, Any]], cuda_blockers: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if not (
        bool(action)
        and str(action.get("action_type") or "") == "manual_promotion_review"
        and str(action.get("status") or "") == "manual_review_ready"
    ):
        reasons.append("manual_promotion_review_action_missing_or_not_ready")
    if not cuda_review_items or cuda_blockers:
        reasons.append("evidence_pack_manual_review_surface_not_ready")
    return reasons


def _release_boundaries_closed(
    repeat_evidence: Mapping[str, Any],
    release_claims: Mapping[str, Any],
    cuda_review_items: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if not (
        bool(repeat_evidence.get("diagnostic_only"))
        and bool(repeat_evidence.get("case_specific_only"))
        and not bool(repeat_evidence.get("release_claim_allowed"))
        and not bool(repeat_evidence.get("safe_to_auto_start"))
    ):
        reasons.append("repeat_release_boundary_open_or_missing")
    if any(bool(item.get("release_claim_allowed")) or bool(item.get("publishable")) for item in cuda_review_items):
        reasons.append("evidence_pack_review_item_publishable_or_release_allowed")
    if str(release_claims.get("release_readiness") or "") == "review_ready":
        reasons.append("global_release_claims_unexpectedly_review_ready")
    return reasons


def build_sdxl_batch2_cuda_debug_repeat_promotion_review(
    *,
    repeat_evidence: Mapping[str, Any],
    p60_plan: Mapping[str, Any],
    evidence_pack: Mapping[str, Any] | None = None,
    release_claims: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed promotion-review artifact without enabling claims."""

    repeat_evidence = _mapping(repeat_evidence)
    p60_plan = _mapping(p60_plan)
    evidence_pack = _mapping(evidence_pack)
    release_claims = _mapping(release_claims)
    summary = _mapping(repeat_evidence.get("summary"))
    action = _p60_review_action(p60_plan)
    comparisons, comparison_reasons = _repeat_comparisons(repeat_evidence)
    review_items = [_mapping(item) for item in _list(evidence_pack.get("review_queue"))]
    cuda_review_items = [
        item for item in review_items
        if str(item.get("type") or "") == "cuda_debug_repeat_case_specific_review"
    ]
    cuda_blockers = [
        item for item in review_items
        if str(item.get("type") or "") == "cuda_debug_repeat_blocker"
    ]

    gates = [
        _gate("repeat_candidate_evidence_ready", False, _repeat_evidence_ready(repeat_evidence, summary)),
        _gate("manual_review_surface_ready", False, _manual_surface_ready(action, cuda_review_items, cuda_blockers)),
        _gate(
            "release_boundaries_closed",
            False,
            _release_boundaries_closed(repeat_evidence, release_claims, cuda_review_items),
        ),
        _gate("case_specific_metrics_pass", False, comparison_reasons),
    ]
    for gate in gates:
        gate["status"] = "passed" if not gate["reasons"] else "blocked"
    blocked_gates = [gate for gate in gates if gate["status"] != "passed"]
    ready = not blocked_gates
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": "manual_promotion_review_ready" if ready else "blocked_review_required",
        "promotion_review_ready": ready,
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "publishable": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "diagnostic_only": True,
        "case_specific_only": True,
        "source_repeat_report": str(repeat_evidence.get("report") or ""),
        "source_repeat_status": str(repeat_evidence.get("status") or ""),
        "source_p60_action_id": str(action.get("id") or ""),
        "summary": {
            "selected_command_count": _safe_int(summary.get("selected_command_count")),
            "completed_report_count": _safe_int(summary.get("completed_report_count")),
            "completed_summary_count": _safe_int(summary.get("completed_summary_count")),
            "comparison_count": _safe_int(summary.get("comparison_count")),
            "source_candidate_count": _safe_int(summary.get("source_candidate_count")),
            "repeat_candidate_pass_count": _safe_int(summary.get("repeat_candidate_pass_count")),
            "fully_repeated_candidate_count": _safe_int(summary.get("fully_repeated_candidate_count")),
            "execution_failure_count": _safe_int(summary.get("execution_failure_count")),
            "missing_report_count": _safe_int(summary.get("missing_report_count")),
            "missing_summary_count": _safe_int(summary.get("missing_summary_count")),
        },
        "comparisons": comparisons,
        "gates": gates,
        "gate_summary": {
            "required_gate_count": len(gates),
            "passed_gate_count": len(gates) - len(blocked_gates),
            "blocked_gate_count": len(blocked_gates),
            "simplified_promotion_gate": True,
        },
        "blocked_gate_ids": [str(gate["id"]) for gate in blocked_gates],
        "allowed_followup_actions": [
            "manual_promotion_review",
            "protected_followup_axis_review",
        ],
        "blocked_actions": [
            "write_sdxl_batch2_release_gain_from_cuda_debug_repeat",
            "use_cuda_debug_repeat_as_compile_anchor",
            "enable_sdxl_batch2_by_default",
            "promote_microbatch_or_universal_gpu_utilization_claim",
        ],
        "recommended_next_action": (
            "manual_case_specific_promotion_review_without_release_claim"
            if ready
            else "fix_blocked_gates_or_rerun_protected_repeat_axis"
        ),
        "notes": [
            "This artifact is JSON-only and does not start GPU work.",
            "Ready means the evidence can be manually reviewed; it is not a release claim.",
        ],
    }


__all__ = [
    "REPORT",
    "REPEAT_REPORT",
    "build_sdxl_batch2_cuda_debug_repeat_promotion_review",
]
