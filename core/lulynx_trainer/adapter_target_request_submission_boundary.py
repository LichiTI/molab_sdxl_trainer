"""Request submission boundary for profiled adapter target request-adapter plans."""

from __future__ import annotations

from typing import Any, Mapping


REQUIRED_PAYLOAD_FIELDS = (
    "adapter_target_policy",
    "adapter_target_modules",
    "adapter_target_rank_map",
    "adapter_target_profile_contract",
)


def build_adapter_target_request_submission_boundary(
    *,
    registration_boundary: Mapping[str, Any],
    submission_plan: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(registration_boundary)
    plan = dict(submission_plan)
    template = dict(plan.get("payload_template") or plan.get("request_payload_template") or {})
    blockers: list[str] = []

    if boundary.get("scorecard") != "adapter_target_request_adapter_registration_boundary_v0":
        blockers.append("unexpected_registration_boundary")
    if not bool(boundary.get("request_adapter_registration_boundary_ready", boundary.get("ok", False))):
        blockers.append("registration_boundary_not_ready")
    if _unsafe_flags(boundary, plan):
        blockers.append("unsafe_child_flag")
    if not str(plan.get("submission_scope") or "").strip():
        blockers.append("submission_scope_missing")
    if not str(plan.get("owner") or "").strip():
        blockers.append("owner_missing")
    if not str(plan.get("rollback_plan") or "").strip():
        blockers.append("rollback_plan_missing")
    if not str(plan.get("manual_approval_id") or "").strip():
        blockers.append("manual_approval_id_missing")
    if not template:
        blockers.append("payload_template_missing")
    for name in REQUIRED_PAYLOAD_FIELDS:
        if name not in template:
            blockers.append(f"payload_field_missing:{name}")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_later_execution_job_contract", False)):
        blockers.append("later_execution_job_contract_missing")
    if not bool(plan.get("acknowledge_no_request_submission", False)):
        blockers.append("request_submission_ack_missing")
    if plan.get("request_payload_materialized") is not False:
        blockers.append("request_payload_materialized_must_be_false")
    if plan.get("request_payload_submitted") is not False:
        blockers.append("request_payload_submitted_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_request_submission_boundary_v0",
        "ok": ready,
        "request_submission_boundary_ready": ready,
        "request_payload_template_recorded": ready,
        "request_payload_materialized": False,
        "request_payload_submitted": False,
        "request_adapter_registered": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "submission_scope": str(plan.get("submission_scope") or ""),
        "payload_template_keys": sorted(template.keys()),
        "parsed_module_count": int(boundary.get("parsed_module_count") or 0),
        "parsed_rank_count": int(boundary.get("parsed_rank_count") or 0),
        "registration_boundary_ready": bool(
            boundary.get("request_adapter_registration_boundary_ready", boundary.get("ok", False))
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare execution-job creation boundary while keeping profiled-target request submission disabled"
            if ready
            else "complete default-off adapter-target request submission boundary"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "request_fields_emitted",
        "request_adapter_registered",
        "registration_applied",
        "runtime_activation_enabled",
        "runtime_activation_allowed",
        "request_payload_materialized",
        "request_payload_submitted",
        "training_launch_allowed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_adapter_target_request_submission_boundary"]
