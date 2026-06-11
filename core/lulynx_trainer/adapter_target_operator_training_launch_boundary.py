"""Operator launch-review boundary for profiled adapter-target launch templates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_REVIEW_FIELDS = (
    "operator_id",
    "review_id",
    "review_timestamp",
    "launch_template_digest",
    "rollback_plan",
)


def build_adapter_target_operator_training_launch_boundary(
    *,
    training_launch_boundary: Mapping[str, Any],
    operator_review: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(training_launch_boundary)
    review = dict(operator_review)
    checklist = _items(review.get("safety_checklist") or review.get("operator_safety_checklist"))
    blockers: list[str] = []

    if boundary.get("scorecard") != "adapter_target_training_launch_boundary_v0":
        blockers.append("unexpected_training_launch_boundary")
    if not bool(boundary.get("training_launch_boundary_ready", boundary.get("ok", False))):
        blockers.append("training_launch_boundary_not_ready")
    if _unsafe_flags(boundary, review):
        blockers.append("unsafe_child_flag")
    for name in REQUIRED_REVIEW_FIELDS:
        if not str(review.get(name) or "").strip():
            blockers.append(f"operator_review_field_missing:{name}")
    if not str(review.get("launch_scope") or "").strip():
        blockers.append("launch_scope_missing")
    if not checklist:
        blockers.append("operator_safety_checklist_missing")
    if not bool(review.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(review.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(review.get("acknowledge_no_operator_launch", False)):
        blockers.append("operator_launch_ack_missing")
    if not bool(review.get("acknowledge_no_process_start", False)):
        blockers.append("process_start_ack_missing")
    if review.get("operator_training_launch_allowed") is not False:
        blockers.append("operator_training_launch_allowed_must_be_false")
    if review.get("operator_training_launch_executed") is not False:
        blockers.append("operator_training_launch_executed_must_be_false")
    if review.get("training_process_started") is not False:
        blockers.append("training_process_started_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_operator_training_launch_boundary_v0",
        "ok": ready,
        "operator_training_launch_boundary_ready": ready,
        "operator_review_recorded": ready,
        "operator_training_launch_allowed": False,
        "operator_training_launch_executed": False,
        "training_process_started": False,
        "training_runtime_started": False,
        "training_launch_allowed": False,
        "training_launch_requested": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "scheduler_dispatch_executed": False,
        "execution_job_created": False,
        "training_job_created": False,
        "job_record_written": False,
        "job_store_written": False,
        "queue_enqueued": False,
        "request_payload_submitted": False,
        "request_payload_materialized": False,
        "request_adapter_registered": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "launch_scope": str(review.get("launch_scope") or ""),
        "operator_id": str(review.get("operator_id") or ""),
        "safety_checklist_count": len(checklist),
        "parsed_module_count": int(boundary.get("parsed_module_count") or 0),
        "parsed_rank_count": int(boundary.get("parsed_rank_count") or 0),
        "training_launch_boundary_ready": bool(
            boundary.get("training_launch_boundary_ready", boundary.get("ok", False))
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold for explicit route decision before any profiled-target training execution contract"
            if ready
            else "complete default-off adapter-target operator training-launch boundary"
        ),
    }


def _items(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "runtime_activation_enabled",
        "request_adapter_registered",
        "request_payload_materialized",
        "request_payload_submitted",
        "execution_job_created",
        "training_job_created",
        "job_record_written",
        "job_store_written",
        "queue_enqueued",
        "run_dispatch_executed",
        "runs_dispatched",
        "scheduler_dispatch_executed",
        "training_launch_allowed",
        "training_launch_requested",
        "training_launch_executed",
        "training_runtime_started",
        "training_process_started",
        "operator_training_launch_allowed",
        "operator_training_launch_executed",
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_adapter_target_operator_training_launch_boundary"]
