"""Default-off runtime dispatch contract for TurboCore native updates.

This stage records runtime dispatch control evidence after the runtime
execution contract.  It is still evidence-only: no runtime dispatch, native
dispatch, kernel launch, parity, request/schema/UI exposure, or training step
is approved or executed here.
"""

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


SCOPE = "native_update_runtime_dispatch_contract"
READY_DECISION = "native_update_runtime_dispatch_contract_recorded_default_off"
HOLD_DECISION = "native_update_runtime_dispatch_contract_hold_for_review_default_off"
BLOCKED_DECISION = "native_update_runtime_dispatch_contract_blocked_default_off"
REJECTED_DECISION = "native_update_runtime_dispatch_contract_rejected_default_off"
REQUIRED_SECTIONS = (
    "runtime_execution_contract_reference",
    "runtime_dispatch_control_plan_inventory",
    "runtime_dispatch_authorization_inventory",
    "runtime_dispatch_precondition_inventory",
    "runtime_adapter_lock_inventory",
    "runtime_state_lock_inventory",
    "native_dispatch_boundary",
    "kernel_launch_boundary",
    "no_runtime_dispatch_boundary",
    "no_native_dispatch_boundary",
    "no_kernel_launch_boundary",
    "no_parity_boundary",
    "no_training_step_boundary",
    "no_request_ui_schema_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_runtime_execution_contract_ready",
    "acknowledge_runtime_dispatch_control_default_off",
    "acknowledge_runtime_dispatch_authorization_default_off",
    "acknowledge_runtime_dispatch_preconditions_default_off",
    "acknowledge_runtime_adapter_and_state_locks_default_off",
    "acknowledge_native_dispatch_and_kernel_boundaries_default_off",
    "acknowledge_no_runtime_dispatch_approved_or_executed",
    "acknowledge_no_native_dispatch_or_kernel_launch",
    "acknowledge_no_parity_or_training_step",
    "acknowledge_no_request_ui_schema_exposure",
    "acknowledge_later_native_dispatch_execution_contract_required",
)
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "runtime_execution_allowed",
    "runtime_execution_enabled",
    "runtime_execution_started",
    "runtime_execution_executed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_approved",
    "runtime_dispatch_started",
    "runtime_dispatch_executed",
    "runtime_dispatch_authorized",
    "runtime_dispatch_control_enabled",
    "runtime_dispatch_control_executed",
    "runtime_adapter_enabled",
    "runtime_adapter_handoff_enabled",
    "runtime_state_lock_enabled",
    "runtime_state_refreshed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_started",
    "native_dispatch_executed",
    "native_kernel_dispatch_enabled",
    "kernel_launch_allowed",
    "kernel_launch_started",
    "kernel_launch_executed",
    "parity_allowed",
    "parity_executed",
    "parity_check_executed",
    "training_step_allowed",
    "training_step_executed",
    "artifact_loaded",
    "execution_replay_executed",
    "ready_for_ui",
    "ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_runtime_dispatch_request_fields",
    "post_runtime_execution_request_fields",
    "runtime_dispatch_request",
    "runtime_dispatch_payload",
    "runtime_dispatch_authorization_request",
    "runtime_dispatch_authorization_payload",
    "runtime_adapter_handoff_payload",
    "runtime_state_lock_payload",
    "native_dispatch_request",
    "native_dispatch_payload",
    "kernel_launch_request",
    "kernel_launch_payload",
    "parity_request",
    "training_step_request",
    "request_adapter_fields",
    "request_schema_fields",
    "ui_route_registration",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_native_update_runtime_dispatch_contract(
    *,
    runtime_execution_contract: Mapping[str, Any] | None = None,
    runtime_dispatch_evidence: Mapping[str, Any] | None = None,
    runtime_dispatch_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record runtime dispatch control evidence without enabling dispatch."""

    execution = _runtime_execution_contract_summary(_as_dict(runtime_execution_contract))
    evidence = _runtime_dispatch_summary(_as_dict(runtime_dispatch_evidence))
    review = _review_summary(_as_dict(runtime_dispatch_review))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(
        execution=execution,
        dispatch=evidence,
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
    elif review.get("approve_native_update_runtime_dispatch_contract") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_runtime_dispatch_review_missing")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_runtime_dispatch_contract_v0",
        "gate": "native_update_runtime_dispatch_contract",
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_ready": evidence_ready,
        "ready_for_runtime_dispatch_review": evidence_ready,
        "runtime_dispatch_contract_recorded": decision == READY_DECISION,
        "native_update_runtime_dispatch_contract_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "runtime_dispatch_review_action_required": decision == HOLD_DECISION,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_runtime_dispatch_request_fields": {},
        "runtime_execution_contract_summary": execution,
        "runtime_dispatch_evidence_summary": evidence,
        "runtime_dispatch_review": review,
        "runtime_dispatch_review_template": _review_template(execution, evidence),
        "progress_gates": {
            "runtime_execution_contract_ready": bool(execution.get("ready")),
            "runtime_dispatch_evidence_ready": bool(evidence.get("ready")),
            "runtime_dispatch_control_default_off_count": evidence.get("control_default_off_count", 0),
            "runtime_dispatch_precondition_default_off_count": evidence.get("precondition_default_off_count", 0),
            "native_dispatch_boundary_default_off_count": evidence.get("native_boundary_default_off_count", 0),
            "kernel_launch_boundary_default_off_count": evidence.get("kernel_boundary_default_off_count", 0),
            "request_ui_schema_exposure_blocked": bool(
                execution.get("request_ui_schema_exposure_blocked")
                and evidence.get("request_ui_schema_exposure_blocked")
            ),
            "signed_runtime_dispatch_review_present": bool(review.get("present")),
        },
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This contract records runtime dispatch control evidence only.",
            "It does not approve or execute runtime dispatch, native dispatch, kernel launch, parity, request/UI/schema exposure, or training steps.",
            "A later native dispatch execution contract is required before runtime dispatch can become active.",
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


def _runtime_execution_contract_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    unsafe = _unsafe_claims(contract, "runtime_execution_contract")
    return {
        "present": bool(contract),
        "source": str(contract.get("_source_path") or ""),
        "digest": str(contract.get("_source_digest") or _digest_payload(contract) if contract else ""),
        "ok": contract.get("ok") is True,
        "evidence_ready": contract.get("evidence_ready") is True,
        "ready_for_runtime_execution_review": contract.get("ready_for_runtime_execution_review") is True,
        "runtime_execution_contract_recorded": contract.get("runtime_execution_contract_recorded") is True,
        "ready": bool(
            contract
            and contract.get("ok") is True
            and contract.get("evidence_ready") is True
            and contract.get("ready_for_runtime_execution_review") is True
            and contract.get("post_runtime_execution_request_fields", {}) == {}
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


def _runtime_dispatch_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = _section_set(evidence)
    missing = [section for section in REQUIRED_SECTIONS if section not in sections]
    controls = _rows(evidence, "runtime_dispatch_control_plan_inventory")
    authorizations = _rows(evidence, "runtime_dispatch_authorization_inventory")
    preconditions = _rows(evidence, "runtime_dispatch_precondition_inventory")
    adapter_locks = _rows(evidence, "runtime_adapter_lock_inventory")
    state_locks = _rows(evidence, "runtime_state_lock_inventory")
    native_boundaries = _rows(evidence, "native_dispatch_boundary")
    kernel_boundaries = _rows(evidence, "kernel_launch_boundary")
    rollback = _rows(evidence, "rollback_policy")
    observability = _rows(evidence, "observability_policy")
    unsafe = _unsafe_claims(evidence, "runtime_dispatch_evidence")
    blockers = _dedupe(
        _flag_blockers(evidence)
        + [f"native_update_runtime_dispatch_section_missing:{section}" for section in missing]
        + _row_blockers(controls, "runtime_dispatch_control", require_default_off=True)
        + _row_blockers(authorizations, "runtime_dispatch_authorization", require_default_off=True)
        + _row_blockers(preconditions, "runtime_dispatch_precondition", require_default_off=True)
        + _row_blockers(adapter_locks, "runtime_adapter_lock", require_default_off=True)
        + _row_blockers(state_locks, "runtime_state_lock", require_default_off=True)
        + _row_blockers(native_boundaries, "native_dispatch_boundary", require_default_off=True)
        + _row_blockers(kernel_boundaries, "kernel_launch_boundary", require_default_off=True)
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
        "ready": bool(evidence and evidence.get("ok") is True and evidence.get("runtime_dispatch_contract_ready") is True and not blockers),
        "runtime_dispatch_contract_ready": evidence.get("runtime_dispatch_contract_ready") is True,
        "control_count": len(controls),
        "control_default_off_count": sum(1 for row in controls if _row_default_off(row)),
        "authorization_count": len(authorizations),
        "authorization_default_off_count": sum(1 for row in authorizations if _row_default_off(row)),
        "precondition_count": len(preconditions),
        "precondition_default_off_count": sum(1 for row in preconditions if _row_default_off(row)),
        "adapter_lock_default_off_count": sum(1 for row in adapter_locks if _row_default_off(row)),
        "state_lock_default_off_count": sum(1 for row in state_locks if _row_default_off(row)),
        "native_boundary_default_off_count": sum(1 for row in native_boundaries if _row_default_off(row)),
        "kernel_boundary_default_off_count": sum(1 for row in kernel_boundaries if _row_default_off(row)),
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
        "runtime_dispatch_contract_only",
        "records_evidence_only",
        "manual_only",
        "internal_only",
        "requires_later_native_dispatch_execution_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"native_update_runtime_dispatch_{field}_missing")
    if evidence.get("default_off") is not True:
        blocked.append("native_update_runtime_dispatch_default_off_violation")
    if evidence.get("runtime_dispatch_contract_ready") is not True:
        blocked.append("native_update_runtime_dispatch_evidence_not_ready")
    if not evidence.get("source") and not evidence.get("_source_path"):
        blocked.append("native_update_runtime_dispatch_source_missing")
    return blocked


def _row_blockers(rows: list[Mapping[str, Any]], kind: str, *, require_default_off: bool) -> list[str]:
    if not rows:
        return [f"native_update_runtime_dispatch_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("plan_id") or row.get("check_id") or row.get("id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"native_update_runtime_dispatch_{kind}_not_ready:{row_id}")
        if not row.get("source"):
            blocked.append(f"native_update_runtime_dispatch_{kind}_source_missing:{row_id}")
        if require_default_off and not _row_default_off(row):
            blocked.append(f"native_update_runtime_dispatch_{kind}_not_default_off:{row_id}")
        blocked.extend(_unsafe_claims(row, row_id))
    return _dedupe(blocked)


def _row_default_off(row: Mapping[str, Any]) -> bool:
    default_off_fields = (
        "runtime_dispatch_allowed",
        "runtime_dispatch_enabled",
        "runtime_dispatch_approved",
        "runtime_dispatch_executed",
        "runtime_adapter_enabled",
        "runtime_state_refreshed",
        "native_dispatch_enabled",
        "native_dispatch_executed",
        "kernel_launch_executed",
        "parity_executed",
        "training_step_executed",
        "request_fields_emitted",
    )
    return all(row.get(field) is False for field in default_off_fields if field in row)


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_update_runtime_dispatch_contract": review.get(
            "approve_native_update_runtime_dispatch_contract"
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
    execution: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not execution.get("present"):
        blocked.append("native_update_runtime_dispatch_runtime_execution_contract_missing")
    elif not execution.get("ready"):
        blocked.append("native_update_runtime_dispatch_runtime_execution_contract_not_ready")
        blocked.extend(_string_list(execution.get("blocked_reasons")))
    if not dispatch.get("present"):
        blocked.append("native_update_runtime_dispatch_evidence_missing")
    elif not dispatch.get("ready"):
        blocked.append("native_update_runtime_dispatch_evidence_not_ready")
        blocked.extend(_string_list(dispatch.get("blocked_reasons")))
    for event in failure_events:
        blocked.append(f"native_update_runtime_dispatch_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_runtime_dispatch_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_runtime_dispatch_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_runtime_dispatch_reviewed_at_missing")
    if review.get("requested_scope") != SCOPE:
        blocked.append("native_update_runtime_dispatch_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_runtime_dispatch_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_runtime_dispatch_review_ack_missing:{field}")
    return _dedupe(blocked)


def _review_template(execution: Mapping[str, Any], dispatch: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_runtime_dispatch_contract": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence"] = {
        "runtime_execution_contract_digest": execution.get("digest"),
        "runtime_dispatch_digest": dispatch.get("digest"),
        "runtime_dispatch_control_default_off_count": dispatch.get("control_default_off_count"),
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
            blocked.append(f"native_update_runtime_dispatch_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_runtime_dispatch_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return ["archive_native_update_runtime_dispatch_contract", "prepare_later_native_dispatch_execution_contract"]
    if evidence_ready:
        return ["collect_signed_native_update_runtime_dispatch_review"]
    return ["repair_runtime_execution_or_runtime_dispatch_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive runtime dispatch contract and prepare the later native dispatch execution contract"
    if evidence_ready:
        return "record runtime dispatch review while keeping native dispatch, kernel launch, request/UI/schema exposure, and training disabled"
    return "repair runtime execution or runtime dispatch evidence before runtime dispatch review"


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {k: v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update default-off runtime dispatch contract.")
    parser.add_argument("--runtime-execution-contract", required=True)
    parser.add_argument("--runtime-dispatch-evidence", required=True)
    parser.add_argument("--runtime-dispatch-review")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    payload = build_native_update_runtime_dispatch_contract(
        runtime_execution_contract=load_json(args.runtime_execution_contract),
        runtime_dispatch_evidence=load_json(args.runtime_dispatch_evidence),
        runtime_dispatch_review=load_json(args.runtime_dispatch_review) if args.runtime_dispatch_review else None,
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
