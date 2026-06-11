"""Execution-job creation boundary for profiled adapter target request submissions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_JOB_FIELDS = ("job_type", "request_id", "payload_ref", "rollback_plan")


def build_adapter_target_execution_job_creation_boundary(
    *,
    request_submission: Mapping[str, Any],
    job_plan: Mapping[str, Any],
) -> dict[str, Any]:
    submission = dict(request_submission)
    plan = dict(job_plan)
    job_template = dict(plan.get("job_template") or plan.get("execution_job_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("job_preconditions"))
    blockers: list[str] = []

    if submission.get("scorecard") != "adapter_target_request_submission_boundary_v0":
        blockers.append("unexpected_request_submission_boundary")
    if not bool(submission.get("request_submission_boundary_ready", submission.get("ok", False))):
        blockers.append("request_submission_boundary_not_ready")
    if _unsafe_flags(submission, plan):
        blockers.append("unsafe_child_flag")
    if not str(plan.get("owner") or "").strip():
        blockers.append("owner_missing")
    if not str(plan.get("manual_approval_id") or "").strip():
        blockers.append("manual_approval_id_missing")
    if not str(plan.get("job_creation_scope") or "").strip():
        blockers.append("job_creation_scope_missing")
    if not job_template:
        blockers.append("job_template_missing")
    for name in REQUIRED_JOB_FIELDS:
        if name not in job_template:
            blockers.append(f"job_field_missing:{name}")
    if not preconditions:
        blockers.append("job_preconditions_missing")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_later_run_dispatch_contract", False)):
        blockers.append("later_run_dispatch_contract_missing")
    if not bool(plan.get("acknowledge_no_job_creation", False)):
        blockers.append("job_creation_ack_missing")
    if plan.get("execution_job_created") is not False:
        blockers.append("execution_job_created_must_be_false")
    if plan.get("training_job_created") is not False:
        blockers.append("training_job_created_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_execution_job_creation_boundary_v0",
        "ok": ready,
        "execution_job_creation_boundary_ready": ready,
        "job_template_recorded": ready,
        "execution_job_created": False,
        "training_job_created": False,
        "job_record_written": False,
        "job_store_written": False,
        "queue_enqueued": False,
        "runs_dispatched": False,
        "training_launch_allowed": False,
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
        "job_creation_scope": str(plan.get("job_creation_scope") or ""),
        "job_template_keys": sorted(job_template.keys()),
        "precondition_count": len(preconditions),
        "parsed_module_count": int(submission.get("parsed_module_count") or 0),
        "parsed_rank_count": int(submission.get("parsed_rank_count") or 0),
        "request_submission_ready": bool(submission.get("request_submission_boundary_ready", submission.get("ok", False))),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare run-dispatch boundary while keeping profiled-target execution job uncreated"
            if ready
            else "complete default-off adapter-target execution-job creation boundary"
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
        "training_launch_allowed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_adapter_target_execution_job_creation_boundary"]
