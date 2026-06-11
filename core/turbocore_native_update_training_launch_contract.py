"""Default-off training-launch contract for TurboCore native updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (  # noqa: E402
    as_dict as _as_dict,
    dedupe as _dedupe,
    event_list as _event_list,
    history_summary as _history_summary,
    string_list as _string_list,
)


SCOPE = "native_update_training_launch_contract"
READY_DECISION = "native_update_training_launch_contract_recorded_default_off"
HOLD_DECISION = "native_update_training_launch_contract_hold_for_review_default_off"
BLOCKED_DECISION = "native_update_training_launch_contract_blocked_default_off"
REJECTED_DECISION = "native_update_training_launch_contract_rejected_default_off"
REQUIRED_SECTIONS = (
    "training_step_execution_contract_reference",
    "training_launch_plan_inventory",
    "training_launch_authorization_inventory",
    "training_launch_precondition_inventory",
    "request_submission_boundary",
    "job_creation_boundary",
    "queue_enqueue_boundary",
    "run_record_boundary",
    "request_ui_schema_boundary",
    "no_training_launch_boundary",
    "no_request_submission_boundary",
    "no_job_creation_boundary",
    "no_queue_enqueue_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_training_step_execution_contract_ready",
    "acknowledge_training_launch_plan_default_off",
    "acknowledge_training_launch_authorization_default_off",
    "acknowledge_training_launch_preconditions_default_off",
    "acknowledge_request_job_queue_boundaries_default_off",
    "acknowledge_no_training_launch",
    "acknowledge_no_request_job_queue_or_run_record",
    "acknowledge_no_request_ui_schema_exposure",
    "acknowledge_owner_review_required_before_product_exposure",
)
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_approved",
    "training_launch_executed",
    "training_runtime_started",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "training_step_executed",
    "request_submission_allowed",
    "request_submitted",
    "request_payload_materialized",
    "job_created",
    "job_record_written",
    "queue_enqueued",
    "run_record_written",
    "artifact_loaded",
    "execution_replay_executed",
    "ready_for_ui",
    "ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "backend_router_registered",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "rollout_authorization_allowed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_training_launch_request_fields",
    "post_training_step_execution_request_fields",
    "training_launch_request",
    "training_launch_payload",
    "training_request",
    "request_payload",
    "job_record_payload",
    "queue_payload",
    "run_record_payload",
    "request_adapter_fields",
    "request_schema_fields",
    "ui_route_registration",
    "backend_router_registration",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_native_update_training_launch_contract(
    *,
    training_step_execution_contract: Mapping[str, Any] | None = None,
    training_launch_evidence: Mapping[str, Any] | None = None,
    training_launch_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record training-launch evidence without launching training."""

    step = _training_step_contract_summary(_as_dict(training_step_execution_contract))
    evidence = _training_launch_summary(_as_dict(training_launch_evidence))
    review = _review_summary(_as_dict(training_launch_review))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(
        step=step,
        launch=evidence,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review)
    if evidence_blockers:
        decision = BLOCKED_DECISION
    elif not review.get("present"):
        decision = HOLD_DECISION
    elif review_blockers:
        decision = BLOCKED_DECISION
    elif review.get("approve_native_update_training_launch_contract") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_training_launch_review_missing")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_training_launch_contract_v0",
        "gate": "native_update_training_launch_contract",
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_ready": evidence_ready,
        "ready_for_training_launch_review": evidence_ready,
        "training_launch_contract_recorded": decision == READY_DECISION,
        "native_update_training_launch_contract_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "training_launch_review_action_required": decision == HOLD_DECISION,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_training_launch_request_fields": {},
        "training_step_execution_contract_summary": step,
        "training_launch_evidence_summary": evidence,
        "training_launch_review": review,
        "training_launch_review_template": _review_template(step, evidence),
        "progress_gates": {
            "training_step_execution_contract_ready": bool(step.get("ready")),
            "training_launch_evidence_ready": bool(evidence.get("ready")),
            "training_launch_plan_default_off_count": evidence.get("plan_default_off_count", 0),
            "training_launch_precondition_default_off_count": evidence.get("precondition_default_off_count", 0),
            "request_boundary_default_off_count": evidence.get("request_boundary_default_off_count", 0),
            "job_boundary_default_off_count": evidence.get("job_boundary_default_off_count", 0),
            "queue_boundary_default_off_count": evidence.get("queue_boundary_default_off_count", 0),
            "request_ui_schema_exposure_blocked": bool(
                step.get("request_ui_schema_exposure_blocked")
                and evidence.get("request_ui_schema_exposure_blocked")
            ),
            "signed_training_launch_review_present": bool(review.get("present")),
        },
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This contract records training-launch evidence only.",
            "It does not submit requests, create jobs, enqueue work, write run records, expose UI/schema, or launch training.",
            "Product exposure still requires explicit owner review and a separate activation decision.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.setdefault("_source_path", str(source))
        payload.setdefault("_source_digest", _digest_payload(payload))
        return payload
    return {}


def _training_step_contract_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    unsafe = _unsafe_claims(contract, "training_step_execution_contract")
    return {
        "present": bool(contract),
        "source": str(contract.get("_source_path") or ""),
        "digest": str(contract.get("_source_digest") or _digest_payload(contract) if contract else ""),
        "ok": contract.get("ok") is True,
        "evidence_ready": contract.get("evidence_ready") is True,
        "ready_for_training_step_execution_review": (
            contract.get("ready_for_training_step_execution_review") is True
        ),
        "training_step_execution_contract_recorded": (
            contract.get("training_step_execution_contract_recorded") is True
        ),
        "ready": bool(
            contract
            and contract.get("ok") is True
            and contract.get("evidence_ready") is True
            and contract.get("ready_for_training_step_execution_review") is True
            and contract.get("post_training_step_execution_request_fields", {}) == {}
            and not unsafe
        ),
        "request_ui_schema_exposure_blocked": bool(
            contract.get("request_adapter_mapping_allowed") is False
            and contract.get("request_fields_emitted") is False
            and contract.get("schema_exposure_allowed") is False
            and contract.get("ready_for_ui") is False
        ),
        "decision": str(contract.get("decision") or ""),
        "blocked_reasons": unsafe,
    }


def _training_launch_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = _section_set(evidence)
    missing = [section for section in REQUIRED_SECTIONS if section not in sections]
    plans = _rows(evidence, "training_launch_plan_inventory")
    authorizations = _rows(evidence, "training_launch_authorization_inventory")
    preconditions = _rows(evidence, "training_launch_precondition_inventory")
    requests = _rows(evidence, "request_submission_boundary")
    jobs = _rows(evidence, "job_creation_boundary")
    queues = _rows(evidence, "queue_enqueue_boundary")
    runs = _rows(evidence, "run_record_boundary")
    ui_schema = _rows(evidence, "request_ui_schema_boundary")
    rollback = _rows(evidence, "rollback_policy")
    observability = _rows(evidence, "observability_policy")
    unsafe = _unsafe_claims(evidence, "training_launch_evidence")
    blockers = _dedupe(
        _flag_blockers(evidence)
        + [f"native_update_training_launch_section_missing:{section}" for section in missing]
        + _row_blockers(plans, "training_launch_plan", require_default_off=True)
        + _row_blockers(authorizations, "training_launch_authorization", require_default_off=True)
        + _row_blockers(preconditions, "training_launch_precondition", require_default_off=True)
        + _row_blockers(requests, "request_submission_boundary", require_default_off=True)
        + _row_blockers(jobs, "job_creation_boundary", require_default_off=True)
        + _row_blockers(queues, "queue_enqueue_boundary", require_default_off=True)
        + _row_blockers(runs, "run_record_boundary", require_default_off=True)
        + _row_blockers(ui_schema, "request_ui_schema_boundary", require_default_off=True)
        + _row_blockers(rollback, "rollback_policy", require_default_off=False)
        + _row_blockers(observability, "observability_policy", require_default_off=False)
        + unsafe
        + _string_list(evidence.get("blocked_reasons"))
        + _string_list(evidence.get("promotion_blockers"))
    )
    return {
        "present": bool(evidence),
        "source": str(evidence.get("source") or evidence.get("_source_path") or ""),
        "digest": str(evidence.get("sha256") or evidence.get("artifact_digest") or _digest_payload(evidence) if evidence else ""),
        "ok": evidence.get("ok") is True,
        "ready": bool(evidence and evidence.get("ok") is True and evidence.get("training_launch_contract_ready") is True and not blockers),
        "training_launch_contract_ready": evidence.get("training_launch_contract_ready") is True,
        "plan_count": len(plans),
        "plan_default_off_count": sum(1 for row in plans if _row_default_off(row)),
        "authorization_count": len(authorizations),
        "authorization_default_off_count": sum(1 for row in authorizations if _row_default_off(row)),
        "precondition_count": len(preconditions),
        "precondition_default_off_count": sum(1 for row in preconditions if _row_default_off(row)),
        "request_boundary_default_off_count": sum(1 for row in requests if _row_default_off(row)),
        "job_boundary_default_off_count": sum(1 for row in jobs if _row_default_off(row)),
        "queue_boundary_default_off_count": sum(1 for row in queues if _row_default_off(row)),
        "run_boundary_default_off_count": sum(1 for row in runs if _row_default_off(row)),
        "ui_schema_boundary_default_off_count": sum(1 for row in ui_schema if _row_default_off(row)),
        "request_ui_schema_exposure_blocked": not bool(
            evidence.get("request_adapter_mapping_allowed") is True
            or evidence.get("request_fields_emitted") is True
            or evidence.get("schema_exposure_allowed") is True
            or evidence.get("ready_for_ui") is True
        ),
        "blocked_reasons": blockers,
    }


def _flag_blockers(evidence: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    for field in (
        "report_only",
        "contract_only",
        "training_launch_contract_only",
        "records_evidence_only",
        "manual_only",
        "internal_only",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"native_update_training_launch_{field}_missing")
    if evidence.get("default_off") is not True:
        blocked.append("native_update_training_launch_default_off_violation")
    if evidence.get("training_launch_contract_ready") is not True:
        blocked.append("native_update_training_launch_evidence_not_ready")
    if not evidence.get("source") and not evidence.get("_source_path"):
        blocked.append("native_update_training_launch_source_missing")
    return blocked


def _row_blockers(rows: list[Mapping[str, Any]], kind: str, *, require_default_off: bool) -> list[str]:
    if not rows:
        return [f"native_update_training_launch_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("plan_id") or row.get("check_id") or row.get("id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"native_update_training_launch_{kind}_not_ready:{row_id}")
        if not row.get("source"):
            blocked.append(f"native_update_training_launch_{kind}_source_missing:{row_id}")
        if require_default_off and not _row_default_off(row):
            blocked.append(f"native_update_training_launch_{kind}_not_default_off:{row_id}")
        blocked.extend(_unsafe_claims(row, row_id))
    return _dedupe(blocked)


def _row_default_off(row: Mapping[str, Any]) -> bool:
    default_off_fields = (
        "training_launch_allowed",
        "training_launch_enabled",
        "training_launch_approved",
        "training_launch_executed",
        "request_submitted",
        "request_payload_materialized",
        "job_created",
        "job_record_written",
        "queue_enqueued",
        "run_record_written",
        "request_fields_emitted",
        "ready_for_ui",
        "schema_exposure_allowed",
    )
    return all(row.get(field) is False for field in default_off_fields if field in row)


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_update_training_launch_contract": review.get(
            "approve_native_update_training_launch_contract"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _evidence_blockers(
    *,
    step: Mapping[str, Any],
    launch: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not step.get("present"):
        blocked.append("native_update_training_launch_training_step_contract_missing")
    elif not step.get("ready"):
        blocked.append("native_update_training_launch_training_step_contract_not_ready")
        blocked.extend(_string_list(step.get("blocked_reasons")))
    if not launch.get("present"):
        blocked.append("native_update_training_launch_evidence_missing")
    elif not launch.get("ready"):
        blocked.append("native_update_training_launch_evidence_not_ready")
        blocked.extend(_string_list(launch.get("blocked_reasons")))
    for event in failure_events:
        blocked.append(f"native_update_training_launch_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_training_launch_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_training_launch_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_training_launch_reviewed_at_missing")
    if review.get("requested_scope") != SCOPE:
        blocked.append("native_update_training_launch_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_training_launch_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_training_launch_review_ack_missing:{field}")
    return _dedupe(blocked)


def _review_template(step: Mapping[str, Any], launch: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_training_launch_contract": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence"] = {
        "training_step_execution_contract_digest": step.get("digest"),
        "training_launch_digest": launch.get("digest"),
        "training_launch_plan_default_off_count": launch.get("plan_default_off_count"),
    }
    return template


def _rows(payload: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    value = payload.get(field)
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("sections")))
    sections.update(_string_list(value.get("available_sections")))
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {item for item in sections if item}


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"native_update_training_launch_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_training_launch_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return ["archive_native_update_training_launch_contract", "request_owner_product_exposure_decision"]
    if evidence_ready:
        return ["collect_signed_native_update_training_launch_review"]
    return ["repair_training_step_or_training_launch_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive training-launch contract and request a separate owner decision before product exposure"
    if evidence_ready:
        return "record training-launch review while keeping request/UI/schema exposure and product launch disabled"
    return "repair training-step or training-launch evidence before review"


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {k: v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update default-off training-launch contract.")
    parser.add_argument("--training-step-execution-contract", required=True)
    parser.add_argument("--training-launch-evidence", required=True)
    parser.add_argument("--training-launch-review")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    payload = build_native_update_training_launch_contract(
        training_step_execution_contract=load_json(args.training_step_execution_contract),
        training_launch_evidence=load_json(args.training_launch_evidence),
        training_launch_review=load_json(args.training_launch_review) if args.training_launch_review else None,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
