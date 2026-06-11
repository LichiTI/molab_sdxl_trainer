"""Quality-review decision gate for profiled adapter target runs."""

from __future__ import annotations

from typing import Any, Mapping


def build_adapter_target_quality_review_decision(
    *,
    execution_audit: Mapping[str, Any],
    comparison_report: Mapping[str, Any],
) -> dict[str, Any]:
    audit = dict(execution_audit)
    comparison = dict(comparison_report)
    quality_delta = float(comparison.get("quality_delta", 0.0) or 0.0)
    speedup_ratio = float(comparison.get("speedup_ratio", 0.0) or 0.0)
    target_reduction_ratio = float(comparison.get("target_reduction_ratio", 0.0) or 0.0)
    min_speedup = float(comparison.get("min_speedup_ratio", 1.0) or 1.0)
    max_quality_drop = float(comparison.get("max_quality_drop", 0.0) or 0.0)
    blockers: list[str] = []

    if audit.get("scorecard") != "adapter_target_execution_audit_v0":
        blockers.append("unexpected_execution_audit")
    if not bool(audit.get("execution_audit_ready", audit.get("ok", False))):
        blockers.append("execution_audit_not_ready")
    if _unsafe_flags(audit, comparison):
        blockers.append("unsafe_child_flag")
    if not bool(comparison.get("comparison_report_present", comparison.get("ok", False))):
        blockers.append("comparison_report_missing")
    if quality_delta < -max_quality_drop:
        blockers.append("quality_drop_exceeds_limit")
    if speedup_ratio < min_speedup:
        blockers.append("speedup_below_minimum")
    if target_reduction_ratio <= 0.0:
        blockers.append("target_reduction_missing")
    if not bool(comparison.get("loss_parity_passed", False)):
        blockers.append("loss_parity_missing")
    if not bool(comparison.get("metadata_roundtrip_passed", False)):
        blockers.append("metadata_roundtrip_missing")
    if not bool(comparison.get("acknowledge_default_off", False)):
        blockers.append("default_off_acknowledgement_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_quality_review_decision_v0",
        "ok": ready,
        "quality_review_ready": ready,
        "promotion_review_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "selected_count": int(audit.get("expected_target_count") or 0),
        "quality_delta": quality_delta,
        "speedup_ratio": speedup_ratio,
        "target_reduction_ratio": target_reduction_ratio,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off adapter target rollout proposal"
            if ready
            else "complete selected-target quality comparison before rollout proposal"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    return any(
        bool(payload.get("training_path_enabled", False))
        or bool(payload.get("default_behavior_changed", False))
        or bool(payload.get("promotion_ready", False))
        for payload in payloads
    )


__all__ = ["build_adapter_target_quality_review_decision"]
