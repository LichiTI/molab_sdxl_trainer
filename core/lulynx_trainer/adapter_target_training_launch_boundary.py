"""Training-launch boundary for profiled adapter-target run-dispatch templates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_LAUNCH_FIELDS = ("launch_id", "run_ref", "config_ref", "dispatch_ref", "rollback_plan")


def build_adapter_target_training_launch_boundary(
    *,
    run_dispatch_boundary: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(run_dispatch_boundary)
    plan = dict(launch_plan)
    launch_template = dict(plan.get("launch_template") or plan.get("training_launch_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("launch_preconditions"))
    blockers: list[str] = []

    if boundary.get("scorecard") != "adapter_target_run_dispatch_boundary_v0":
        blockers.append("unexpected_run_dispatch_boundary")
    if not bool(boundary.get("run_dispatch_boundary_ready", boundary.get("ok", False))):
        blockers.append("run_dispatch_boundary_not_ready")
    if _unsafe_flags(boundary, plan):
        blockers.append("unsafe_child_flag")
    if not str(plan.get("owner") or "").strip():
        blockers.append("owner_missing")
    if not str(plan.get("manual_approval_id") or "").strip():
        blockers.append("manual_approval_id_missing")
    if not str(plan.get("launch_scope") or "").strip():
        blockers.append("launch_scope_missing")
    if not str(plan.get("rollback_plan") or "").strip():
        blockers.append("rollback_plan_missing")
    if not launch_template:
        blockers.append("training_launch_template_missing")
    for name in REQUIRED_LAUNCH_FIELDS:
        if name not in launch_template:
            blockers.append(f"training_launch_field_missing:{name}")
    if not preconditions:
        blockers.append("training_launch_preconditions_missing")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_later_operator_launch_contract", False)):
        blockers.append("later_operator_launch_contract_missing")
    if not bool(plan.get("acknowledge_no_training_launch", False)):
        blockers.append("training_launch_ack_missing")
    if plan.get("training_launch_allowed") is not False:
        blockers.append("training_launch_allowed_must_be_false")
    if plan.get("training_launch_requested") is not False:
        blockers.append("training_launch_requested_must_be_false")
    if plan.get("training_launch_executed") is not False:
        blockers.append("training_launch_executed_must_be_false")
    if plan.get("training_runtime_started") is not False:
        blockers.append("training_runtime_started_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_training_launch_boundary_v0",
        "ok": ready,
        "training_launch_boundary_ready": ready,
        "training_launch_template_recorded": ready,
        "training_launch_allowed": False,
        "training_launch_requested": False,
        "training_launch_executed": False,
        "training_runtime_started": False,
        "training_process_started": False,
        "operator_training_launch_allowed": False,
        "operator_training_launch_executed": False,
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
        "launch_scope": str(plan.get("launch_scope") or ""),
        "training_launch_template_keys": sorted(launch_template.keys()),
        "precondition_count": len(preconditions),
        "parsed_module_count": int(boundary.get("parsed_module_count") or 0),
        "parsed_rank_count": int(boundary.get("parsed_rank_count") or 0),
        "run_dispatch_boundary_ready": bool(boundary.get("run_dispatch_boundary_ready", boundary.get("ok", False))),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare operator-launch review only after a separate route decision"
            if ready
            else "complete default-off adapter-target training-launch boundary"
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


__all__ = ["build_adapter_target_training_launch_boundary"]
