"""Default-off execution boundaries for compute-reducer trainer A/B runs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_JOB_FIELDS = ("job_type", "dispatch_manifest_ref", "payload_refs", "rollback_plan")
REQUIRED_DISPATCH_FIELDS = ("run_id", "job_ref", "dispatch_mode", "rollback_plan")
REQUIRED_LAUNCH_FIELDS = ("launch_id", "run_ref", "config_ref", "dispatch_ref", "rollback_plan")
REQUIRED_OPERATOR_FIELDS = (
    "operator_id",
    "review_id",
    "review_timestamp",
    "launch_template_digest",
    "rollback_plan",
)


def build_dit_compute_reducer_trainer_ab_request_submission_boundary(
    *,
    dispatch_manifest: Mapping[str, Any],
    submission_plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = dict(dispatch_manifest)
    plan = dict(submission_plan)
    payload_template = dict(plan.get("payload_template") or {})
    blockers = _upstream_blockers(
        manifest,
        expected_scorecard="dit_compute_reducer_trainer_ab_dispatch_manifest_v0",
        ready_field="dispatch_manifest_ready",
        unexpected_reason="unexpected_dispatch_manifest",
        not_ready_reason="dispatch_manifest_not_ready",
    )
    blockers.extend(_missing_text(plan, "owner", "manual_approval_id", "submission_scope", "rollback_plan"))
    blockers.extend(_common_plan_blockers(plan, "requires_later_execution_job_contract"))
    if not payload_template:
        blockers.append("payload_template_missing")
    if int(payload_template.get("payload_count") or 0) != int(manifest.get("payload_count") or 0):
        blockers.append("payload_count_mismatch")
    if not bool(plan.get("acknowledge_no_request_submission", False)):
        blockers.append("request_submission_ack_missing")
    for key in ("request_payload_materialized", "request_payload_submitted"):
        if plan.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            "dit_compute_reducer_trainer_ab_request_submission_boundary_v0",
            ready,
            "request_submission_boundary_ready",
            "request_payload_template_recorded",
        ),
        "request_payload_materialized": False,
        "request_payload_submitted": False,
        "submission_scope": str(plan.get("submission_scope") or ""),
        "payload_template_keys": sorted(payload_template.keys()),
        "dispatch_manifest_ready": bool(manifest.get("dispatch_manifest_ready", manifest.get("ok", False))),
        "blocked_reasons": blockers,
    }


def build_dit_compute_reducer_trainer_ab_execution_job_creation_boundary(
    *,
    request_submission: Mapping[str, Any],
    job_plan: Mapping[str, Any],
) -> dict[str, Any]:
    submission = dict(request_submission)
    plan = dict(job_plan)
    job_template = dict(plan.get("job_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("job_preconditions"))
    blockers = _upstream_blockers(
        submission,
        expected_scorecard="dit_compute_reducer_trainer_ab_request_submission_boundary_v0",
        ready_field="request_submission_boundary_ready",
        unexpected_reason="unexpected_request_submission_boundary",
        not_ready_reason="request_submission_boundary_not_ready",
    )
    blockers.extend(_missing_text(plan, "owner", "manual_approval_id", "job_creation_scope"))
    blockers.extend(_common_plan_blockers(plan, "requires_later_run_dispatch_contract"))
    if not job_template:
        blockers.append("job_template_missing")
    blockers.extend(f"job_field_missing:{field}" for field in REQUIRED_JOB_FIELDS if field not in job_template)
    if not preconditions:
        blockers.append("job_preconditions_missing")
    if not bool(plan.get("acknowledge_no_job_creation", False)):
        blockers.append("job_creation_ack_missing")
    for key in ("execution_job_created", "training_job_created"):
        if plan.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            "dit_compute_reducer_trainer_ab_execution_job_creation_boundary_v0",
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
    }


def build_dit_compute_reducer_trainer_ab_run_dispatch_boundary(
    *,
    execution_job_boundary: Mapping[str, Any],
    dispatch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(execution_job_boundary)
    plan = dict(dispatch_plan)
    dispatch_template = dict(plan.get("dispatch_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("dispatch_preconditions"))
    blockers = _upstream_blockers(
        boundary,
        expected_scorecard="dit_compute_reducer_trainer_ab_execution_job_creation_boundary_v0",
        ready_field="execution_job_creation_boundary_ready",
        unexpected_reason="unexpected_execution_job_creation_boundary",
        not_ready_reason="execution_job_creation_boundary_not_ready",
    )
    blockers.extend(_missing_text(plan, "owner", "manual_approval_id", "dispatch_scope", "rollback_plan"))
    blockers.extend(_common_plan_blockers(plan, "requires_later_training_launch_contract"))
    if not dispatch_template:
        blockers.append("dispatch_template_missing")
    blockers.extend(
        f"dispatch_field_missing:{field}" for field in REQUIRED_DISPATCH_FIELDS if field not in dispatch_template
    )
    if not preconditions:
        blockers.append("dispatch_preconditions_missing")
    if not bool(plan.get("acknowledge_no_run_dispatch", False)):
        blockers.append("run_dispatch_ack_missing")
    for key in ("run_dispatch_executed", "runs_dispatched", "scheduler_dispatch_executed"):
        if plan.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            "dit_compute_reducer_trainer_ab_run_dispatch_boundary_v0",
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
    }


def build_dit_compute_reducer_trainer_ab_training_launch_boundary(
    *,
    run_dispatch_boundary: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(run_dispatch_boundary)
    plan = dict(launch_plan)
    launch_template = dict(plan.get("launch_template") or {})
    preconditions = _items(plan.get("preconditions") or plan.get("launch_preconditions"))
    blockers = _upstream_blockers(
        boundary,
        expected_scorecard="dit_compute_reducer_trainer_ab_run_dispatch_boundary_v0",
        ready_field="run_dispatch_boundary_ready",
        unexpected_reason="unexpected_run_dispatch_boundary",
        not_ready_reason="run_dispatch_boundary_not_ready",
    )
    blockers.extend(_missing_text(plan, "owner", "manual_approval_id", "launch_scope", "rollback_plan"))
    blockers.extend(_common_plan_blockers(plan, "requires_later_operator_launch_contract"))
    if not launch_template:
        blockers.append("training_launch_template_missing")
    blockers.extend(
        f"training_launch_field_missing:{field}" for field in REQUIRED_LAUNCH_FIELDS if field not in launch_template
    )
    if not preconditions:
        blockers.append("training_launch_preconditions_missing")
    if not bool(plan.get("acknowledge_no_training_launch", False)):
        blockers.append("training_launch_ack_missing")
    for key in ("training_launch_allowed", "training_launch_requested", "training_launch_executed"):
        if plan.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        **_safe_report(
            "dit_compute_reducer_trainer_ab_training_launch_boundary_v0",
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
    }


def build_dit_compute_reducer_trainer_ab_operator_training_launch_boundary(
    *,
    training_launch_boundary: Mapping[str, Any],
    operator_review: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = dict(training_launch_boundary)
    review = dict(operator_review)
    checklist = _items(review.get("safety_checklist") or review.get("operator_safety_checklist"))
    blockers = _upstream_blockers(
        boundary,
        expected_scorecard="dit_compute_reducer_trainer_ab_training_launch_boundary_v0",
        ready_field="training_launch_boundary_ready",
        unexpected_reason="unexpected_training_launch_boundary",
        not_ready_reason="training_launch_boundary_not_ready",
    )
    blockers.extend(f"operator_review_field_missing:{field}" for field in REQUIRED_OPERATOR_FIELDS if not str(review.get(field) or "").strip())
    blockers.extend(_common_plan_blockers(review, None))
    if not checklist:
        blockers.append("operator_safety_checklist_missing")
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
            "dit_compute_reducer_trainer_ab_operator_training_launch_boundary_v0",
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
        "operator_id": str(review.get("operator_id") or ""),
        "safety_checklist_count": len(checklist),
        "training_launch_boundary_ready": bool(
            boundary.get("training_launch_boundary_ready", boundary.get("ok", False))
        ),
        "blocked_reasons": blockers,
    }


def _safe_report(scorecard: str, ready: bool, ready_field: str, recorded_field: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": scorecard,
        "ok": ready,
        ready_field: ready,
        recorded_field: ready,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "ab_dispatch_allowed": False,
        "ab_execution_allowed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
    }


def _upstream_blockers(
    upstream: Mapping[str, Any],
    *,
    expected_scorecard: str,
    ready_field: str,
    unexpected_reason: str,
    not_ready_reason: str,
) -> list[str]:
    blockers: list[str] = []
    if upstream.get("scorecard") != expected_scorecard:
        blockers.append(unexpected_reason)
    if not bool(upstream.get(ready_field, upstream.get("ok", False))):
        blockers.append(not_ready_reason)
    if _unsafe_flags(upstream):
        blockers.append("unsafe_upstream_flag")
    return blockers


def _common_plan_blockers(plan: Mapping[str, Any], later_contract_key: str | None) -> list[str]:
    blockers: list[str] = []
    if _unsafe_flags(plan):
        blockers.append("unsafe_plan_flag")
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
        "trainer_wiring_allowed",
        "request_fields_emitted",
        "request_adapter_registered",
        "runtime_activation_enabled",
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
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_dit_compute_reducer_trainer_ab_execution_job_creation_boundary",
    "build_dit_compute_reducer_trainer_ab_operator_training_launch_boundary",
    "build_dit_compute_reducer_trainer_ab_request_submission_boundary",
    "build_dit_compute_reducer_trainer_ab_run_dispatch_boundary",
    "build_dit_compute_reducer_trainer_ab_training_launch_boundary",
]
