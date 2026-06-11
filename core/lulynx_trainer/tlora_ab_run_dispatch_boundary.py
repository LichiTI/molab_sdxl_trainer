"""Run-dispatch boundary for T-LoRA A/B execution-job templates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_DISPATCH_FIELDS = ("run_id", "job_ref", "dispatch_mode", "rollback_plan")


def build_tlora_ab_run_dispatch_boundary(
    *,
    execution_job_boundary: Mapping[str, Any],
    dispatch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(execution_job_boundary)
    plan = dict(dispatch_plan)
    dispatch_template = dict(plan.get("dispatch_template") or plan.get("run_dispatch_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("dispatch_preconditions"))
    blockers: list[str] = []

    if boundary.get("scorecard") != "tlora_ab_execution_job_creation_boundary_v0":
        blockers.append("unexpected_execution_job_creation_boundary")
    if not bool(boundary.get("execution_job_creation_boundary_ready", boundary.get("ok", False))):
        blockers.append("execution_job_creation_boundary_not_ready")
    if _unsafe_flags(boundary, plan):
        blockers.append("unsafe_child_flag")
    if not str(plan.get("owner") or "").strip():
        blockers.append("owner_missing")
    if not str(plan.get("manual_approval_id") or "").strip():
        blockers.append("manual_approval_id_missing")
    if not str(plan.get("dispatch_scope") or "").strip():
        blockers.append("dispatch_scope_missing")
    if not str(plan.get("rollback_plan") or "").strip():
        blockers.append("rollback_plan_missing")
    if not dispatch_template:
        blockers.append("dispatch_template_missing")
    for name in REQUIRED_DISPATCH_FIELDS:
        if name not in dispatch_template:
            blockers.append(f"dispatch_field_missing:{name}")
    if not preconditions:
        blockers.append("dispatch_preconditions_missing")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_later_training_launch_contract", False)):
        blockers.append("later_training_launch_contract_missing")
    if not bool(plan.get("acknowledge_no_run_dispatch", False)):
        blockers.append("run_dispatch_ack_missing")
    if plan.get("run_dispatch_executed") is not False:
        blockers.append("run_dispatch_executed_must_be_false")
    if plan.get("runs_dispatched") is not False:
        blockers.append("runs_dispatched_must_be_false")
    if plan.get("scheduler_dispatch_executed") is not False:
        blockers.append("scheduler_dispatch_executed_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_run_dispatch_boundary_v0",
        "ok": ready,
        "run_dispatch_boundary_ready": ready,
        "run_dispatch_template_recorded": ready,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "scheduler_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_requested": False,
        "training_launch_executed": False,
        "training_runtime_started": False,
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
        "dispatch_scope": str(plan.get("dispatch_scope") or ""),
        "dispatch_template_keys": sorted(dispatch_template.keys()),
        "precondition_count": len(preconditions),
        "execution_job_boundary_ready": bool(
            boundary.get("execution_job_creation_boundary_ready", boundary.get("ok", False))
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare training-launch boundary while keeping T-LoRA runs undispatched"
            if ready
            else "complete default-off T-LoRA run-dispatch boundary"
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
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_tlora_ab_run_dispatch_boundary"]
