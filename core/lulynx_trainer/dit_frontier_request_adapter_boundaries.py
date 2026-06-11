"""Default-off request-channel boundaries for DiT frontier features."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .dit_frontier_request_field_contracts import FRONTIER_REQUEST_FIELD_SPECS


REQUIRED_JOB_FIELDS = ("job_type", "request_id", "payload_ref", "rollback_plan")
REQUIRED_DISPATCH_FIELDS = ("run_id", "job_ref", "dispatch_mode", "rollback_plan")
REQUIRED_LAUNCH_FIELDS = ("launch_id", "run_ref", "config_ref", "dispatch_ref", "rollback_plan")
REQUIRED_OPERATOR_REVIEW_FIELDS = (
    "operator_id",
    "review_id",
    "review_timestamp",
    "launch_template_digest",
    "rollback_plan",
)


def build_dit_frontier_request_adapter_registration_boundary(
    *,
    feature_id: str,
    config_adapter_replay: Mapping[str, Any],
    registration_plan: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    replay = dict(config_adapter_replay)
    plan = dict(registration_plan)
    mapping_fields = _items(plan.get("mapping_fields") or plan.get("request_field_names"))
    blockers = _base_blockers(
        replay,
        plan,
        expected_scorecard=spec.replay_scorecard,
        expected_ready_field="config_adapter_replay_ready",
        unexpected_reason="unexpected_config_adapter_replay",
        not_ready_reason="config_adapter_replay_not_ready",
    )
    blockers.extend(_missing_text(plan, "adapter_id", "owner", "registration_scope", "rollback_plan", "activation_policy"))
    if not mapping_fields:
        blockers.append("mapping_fields_missing")
    blockers.extend(f"mapping_field_missing:{name}" for name in spec.required_fields if name not in mapping_fields)
    blockers.extend(_common_plan_blockers(plan, "requires_later_request_submission_contract"))
    if not bool(plan.get("acknowledge_no_request_adapter_registration", False)):
        blockers.append("request_adapter_no_registration_ack_missing")
    if plan.get("request_adapter_registered") is not False:
        blockers.append("request_adapter_registered_must_be_false")
    if plan.get("registration_applied") is not False:
        blockers.append("registration_applied_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            feature_id,
            f"{feature_id}_request_adapter_registration_boundary_v0",
            ready,
            "request_adapter_registration_boundary_ready",
            "registration_inventory_recorded",
        ),
        "request_adapter_registered": False,
        "registration_applied": False,
        "adapter_id": str(plan.get("adapter_id") or ""),
        "mapping_fields": list(mapping_fields),
        "config_adapter_replay_ready": bool(replay.get("config_adapter_replay_ready", replay.get("ok", False))),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"prepare {spec.label} request submission boundary while keeping request adapter unregistered"
            if ready
            else f"complete default-off {spec.label} request-adapter registration boundary"
        ),
    }


def build_dit_frontier_request_submission_boundary(
    *,
    feature_id: str,
    registration_boundary: Mapping[str, Any],
    submission_plan: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    boundary = dict(registration_boundary)
    plan = dict(submission_plan)
    template = dict(plan.get("payload_template") or plan.get("request_payload_template") or {})
    blockers = _base_blockers(
        boundary,
        plan,
        expected_scorecard=f"{feature_id}_request_adapter_registration_boundary_v0",
        expected_ready_field="request_adapter_registration_boundary_ready",
        unexpected_reason="unexpected_registration_boundary",
        not_ready_reason="registration_boundary_not_ready",
    )
    blockers.extend(_missing_text(plan, "submission_scope", "owner", "rollback_plan", "manual_approval_id"))
    if not template:
        blockers.append("payload_template_missing")
    blockers.extend(f"payload_field_missing:{name}" for name in spec.required_fields if name not in template)
    blockers.extend(_common_plan_blockers(plan, "requires_later_execution_job_contract"))
    if not bool(plan.get("acknowledge_no_request_submission", False)):
        blockers.append("request_submission_ack_missing")
    if plan.get("request_payload_materialized") is not False:
        blockers.append("request_payload_materialized_must_be_false")
    if plan.get("request_payload_submitted") is not False:
        blockers.append("request_payload_submitted_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            feature_id,
            f"{feature_id}_request_submission_boundary_v0",
            ready,
            "request_submission_boundary_ready",
            "request_payload_template_recorded",
        ),
        "request_payload_materialized": False,
        "request_payload_submitted": False,
        "submission_scope": str(plan.get("submission_scope") or ""),
        "payload_template_keys": sorted(template.keys()),
        "registration_boundary_ready": bool(
            boundary.get("request_adapter_registration_boundary_ready", boundary.get("ok", False))
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"prepare {spec.label} execution-job boundary while keeping request submission disabled"
            if ready
            else f"complete default-off {spec.label} request submission boundary"
        ),
    }


def build_dit_frontier_execution_job_creation_boundary(
    *,
    feature_id: str,
    request_submission: Mapping[str, Any],
    job_plan: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    submission = dict(request_submission)
    plan = dict(job_plan)
    job_template = dict(plan.get("job_template") or plan.get("execution_job_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("job_preconditions"))
    blockers = _base_blockers(
        submission,
        plan,
        expected_scorecard=f"{feature_id}_request_submission_boundary_v0",
        expected_ready_field="request_submission_boundary_ready",
        unexpected_reason="unexpected_request_submission_boundary",
        not_ready_reason="request_submission_boundary_not_ready",
    )
    blockers.extend(_missing_text(plan, "owner", "manual_approval_id", "job_creation_scope"))
    if not job_template:
        blockers.append("job_template_missing")
    blockers.extend(f"job_field_missing:{name}" for name in REQUIRED_JOB_FIELDS if name not in job_template)
    if not preconditions:
        blockers.append("job_preconditions_missing")
    blockers.extend(_common_plan_blockers(plan, "requires_later_run_dispatch_contract"))
    if not bool(plan.get("acknowledge_no_job_creation", False)):
        blockers.append("job_creation_ack_missing")
    if plan.get("execution_job_created") is not False:
        blockers.append("execution_job_created_must_be_false")
    if plan.get("training_job_created") is not False:
        blockers.append("training_job_created_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            feature_id,
            f"{feature_id}_execution_job_creation_boundary_v0",
            ready,
            "execution_job_creation_boundary_ready",
            "job_template_recorded",
        ),
        "execution_job_created": False,
        "training_job_created": False,
        "job_record_written": False,
        "job_store_written": False,
        "queue_enqueued": False,
        "job_creation_scope": str(plan.get("job_creation_scope") or ""),
        "job_template_keys": sorted(job_template.keys()),
        "precondition_count": len(preconditions),
        "request_submission_ready": bool(submission.get("request_submission_boundary_ready", submission.get("ok", False))),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"prepare {spec.label} run-dispatch boundary while keeping execution job uncreated"
            if ready
            else f"complete default-off {spec.label} execution-job creation boundary"
        ),
    }


def build_dit_frontier_run_dispatch_boundary(
    *,
    feature_id: str,
    execution_job_boundary: Mapping[str, Any],
    dispatch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    boundary = dict(execution_job_boundary)
    plan = dict(dispatch_plan)
    dispatch_template = dict(plan.get("dispatch_template") or plan.get("run_dispatch_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("dispatch_preconditions"))
    blockers = _base_blockers(
        boundary,
        plan,
        expected_scorecard=f"{feature_id}_execution_job_creation_boundary_v0",
        expected_ready_field="execution_job_creation_boundary_ready",
        unexpected_reason="unexpected_execution_job_creation_boundary",
        not_ready_reason="execution_job_creation_boundary_not_ready",
    )
    blockers.extend(_missing_text(plan, "owner", "manual_approval_id", "dispatch_scope", "rollback_plan"))
    if not dispatch_template:
        blockers.append("dispatch_template_missing")
    blockers.extend(f"dispatch_field_missing:{name}" for name in REQUIRED_DISPATCH_FIELDS if name not in dispatch_template)
    if not preconditions:
        blockers.append("dispatch_preconditions_missing")
    blockers.extend(_common_plan_blockers(plan, "requires_later_training_launch_contract"))
    if not bool(plan.get("acknowledge_no_run_dispatch", False)):
        blockers.append("run_dispatch_ack_missing")
    for key in ("run_dispatch_executed", "runs_dispatched", "scheduler_dispatch_executed"):
        if plan.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            feature_id,
            f"{feature_id}_run_dispatch_boundary_v0",
            ready,
            "run_dispatch_boundary_ready",
            "run_dispatch_template_recorded",
        ),
        "run_dispatch_executed": False,
        "scheduler_dispatch_executed": False,
        "dispatch_scope": str(plan.get("dispatch_scope") or ""),
        "dispatch_template_keys": sorted(dispatch_template.keys()),
        "precondition_count": len(preconditions),
        "execution_job_boundary_ready": bool(
            boundary.get("execution_job_creation_boundary_ready", boundary.get("ok", False))
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"prepare {spec.label} training-launch boundary while keeping runs undispatched"
            if ready
            else f"complete default-off {spec.label} run-dispatch boundary"
        ),
    }


def build_dit_frontier_training_launch_boundary(
    *,
    feature_id: str,
    run_dispatch_boundary: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    boundary = dict(run_dispatch_boundary)
    plan = dict(launch_plan)
    launch_template = dict(plan.get("launch_template") or plan.get("training_launch_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("launch_preconditions"))
    blockers = _base_blockers(
        boundary,
        plan,
        expected_scorecard=f"{feature_id}_run_dispatch_boundary_v0",
        expected_ready_field="run_dispatch_boundary_ready",
        unexpected_reason="unexpected_run_dispatch_boundary",
        not_ready_reason="run_dispatch_boundary_not_ready",
    )
    blockers.extend(_missing_text(plan, "owner", "manual_approval_id", "launch_scope", "rollback_plan"))
    if not launch_template:
        blockers.append("training_launch_template_missing")
    blockers.extend(f"training_launch_field_missing:{name}" for name in REQUIRED_LAUNCH_FIELDS if name not in launch_template)
    if not preconditions:
        blockers.append("training_launch_preconditions_missing")
    blockers.extend(_common_plan_blockers(plan, "requires_later_operator_launch_contract"))
    if not bool(plan.get("acknowledge_no_training_launch", False)):
        blockers.append("training_launch_ack_missing")
    for key in ("training_launch_allowed", "training_launch_requested", "training_launch_executed", "training_runtime_started"):
        if plan.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            feature_id,
            f"{feature_id}_training_launch_boundary_v0",
            ready,
            "training_launch_boundary_ready",
            "training_launch_template_recorded",
        ),
        "training_launch_requested": False,
        "training_launch_executed": False,
        "training_runtime_started": False,
        "training_process_started": False,
        "operator_training_launch_allowed": False,
        "operator_training_launch_executed": False,
        "launch_scope": str(plan.get("launch_scope") or ""),
        "training_launch_template_keys": sorted(launch_template.keys()),
        "precondition_count": len(preconditions),
        "run_dispatch_boundary_ready": bool(boundary.get("run_dispatch_boundary_ready", boundary.get("ok", False))),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"prepare {spec.label} operator launch review only after a separate route decision"
            if ready
            else f"complete default-off {spec.label} training-launch boundary"
        ),
    }


def build_dit_frontier_operator_training_launch_boundary(
    *,
    feature_id: str,
    training_launch_boundary: Mapping[str, Any],
    operator_review: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    boundary = dict(training_launch_boundary)
    review = dict(operator_review)
    checklist = _items(review.get("safety_checklist") or review.get("operator_safety_checklist"))
    blockers = _base_blockers(
        boundary,
        review,
        expected_scorecard=f"{feature_id}_training_launch_boundary_v0",
        expected_ready_field="training_launch_boundary_ready",
        unexpected_reason="unexpected_training_launch_boundary",
        not_ready_reason="training_launch_boundary_not_ready",
    )
    for name in REQUIRED_OPERATOR_REVIEW_FIELDS:
        if not str(review.get(name) or "").strip():
            blockers.append(f"operator_review_field_missing:{name}")
    if not str(review.get("launch_scope") or "").strip():
        blockers.append("launch_scope_missing")
    if not checklist:
        blockers.append("operator_safety_checklist_missing")
    blockers.extend(_common_plan_blockers(review, None))
    if not bool(review.get("acknowledge_no_operator_launch", False)):
        blockers.append("operator_launch_ack_missing")
    if not bool(review.get("acknowledge_no_process_start", False)):
        blockers.append("process_start_ack_missing")
    for key in ("operator_training_launch_allowed", "operator_training_launch_executed", "training_process_started"):
        if review.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            feature_id,
            f"{feature_id}_operator_training_launch_boundary_v0",
            ready,
            "operator_training_launch_boundary_ready",
            "operator_review_recorded",
        ),
        "operator_training_launch_allowed": False,
        "operator_training_launch_executed": False,
        "training_process_started": False,
        "training_runtime_started": False,
        "training_launch_requested": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "scheduler_dispatch_executed": False,
        "execution_job_created": False,
        "training_job_created": False,
        "job_record_written": False,
        "job_store_written": False,
        "queue_enqueued": False,
        "request_payload_submitted": False,
        "request_payload_materialized": False,
        "launch_scope": str(review.get("launch_scope") or ""),
        "operator_id": str(review.get("operator_id") or ""),
        "safety_checklist_count": len(checklist),
        "training_launch_boundary_ready": bool(
            boundary.get("training_launch_boundary_ready", boundary.get("ok", False))
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"hold {spec.label} for explicit route decision before any training execution contract"
            if ready
            else f"complete default-off {spec.label} operator training-launch boundary"
        ),
    }


def _spec(feature_id: str):
    try:
        return FRONTIER_REQUEST_FIELD_SPECS[feature_id]
    except KeyError as exc:
        raise ValueError(f"unsupported DiT frontier feature: {feature_id}") from exc


def _safe_report(feature_id: str, scorecard: str, ready: bool, ready_field: str, recorded_field: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": scorecard,
        "feature_id": feature_id,
        "ok": ready,
        ready_field: ready,
        recorded_field: ready,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_allowed": False,
        "runtime_activation_enabled": False,
        "ab_dispatch_allowed": False,
        "ab_execution_allowed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
    }


def _base_blockers(
    upstream: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    expected_scorecard: str,
    expected_ready_field: str,
    unexpected_reason: str,
    not_ready_reason: str,
) -> list[str]:
    blockers: list[str] = []
    if upstream.get("scorecard") != expected_scorecard:
        blockers.append(unexpected_reason)
    if not bool(upstream.get(expected_ready_field, upstream.get("ok", False))):
        blockers.append(not_ready_reason)
    if _unsafe_flags(upstream, plan):
        blockers.append("unsafe_child_flag")
    return blockers


def _common_plan_blockers(plan: Mapping[str, Any], later_contract_key: str | None) -> list[str]:
    blockers: list[str] = []
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if later_contract_key and not bool(plan.get(later_contract_key, False)):
        blockers.append(f"{later_contract_key}_missing")
    return blockers


def _missing_text(payload: Mapping[str, Any], *keys: str) -> list[str]:
    return [f"{key}_missing" for key in keys if not str(payload.get(key) or "").strip()]


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
        "request_fields_emitted",
        "request_adapter_registered",
        "registration_applied",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "runtime_activation_allowed",
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
        "ab_dispatch_allowed",
        "ab_execution_allowed",
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


__all__ = [
    "build_dit_frontier_execution_job_creation_boundary",
    "build_dit_frontier_operator_training_launch_boundary",
    "build_dit_frontier_request_adapter_registration_boundary",
    "build_dit_frontier_request_submission_boundary",
    "build_dit_frontier_run_dispatch_boundary",
    "build_dit_frontier_training_launch_boundary",
]
