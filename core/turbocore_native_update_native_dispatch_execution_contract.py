"""Default-off native dispatch execution contract for TurboCore native updates.

This stage records native dispatch execution evidence after the runtime
dispatch contract.  It remains evidence-only and does not approve or execute
native dispatch, kernel launch, parity, tensor transfer, request/schema/UI
exposure, or training steps.
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


SCOPE = "native_update_native_dispatch_execution_contract"
READY_DECISION = "native_update_native_dispatch_execution_contract_recorded_default_off"
HOLD_DECISION = "native_update_native_dispatch_execution_contract_hold_for_review_default_off"
BLOCKED_DECISION = "native_update_native_dispatch_execution_contract_blocked_default_off"
REJECTED_DECISION = "native_update_native_dispatch_execution_contract_rejected_default_off"
REQUIRED_SECTIONS = (
    "runtime_dispatch_contract_reference",
    "native_dispatch_execution_plan_inventory",
    "native_dispatch_authorization_inventory",
    "native_dispatch_precondition_inventory",
    "native_runtime_handoff_boundary",
    "native_dispatch_adapter_boundary",
    "kernel_launch_boundary",
    "parity_boundary",
    "tensor_transfer_boundary",
    "no_native_dispatch_execution_boundary",
    "no_kernel_launch_boundary",
    "no_parity_boundary",
    "no_tensor_transfer_boundary",
    "no_training_step_boundary",
    "no_request_ui_schema_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_runtime_dispatch_contract_ready",
    "acknowledge_native_dispatch_plan_default_off",
    "acknowledge_native_dispatch_authorization_default_off",
    "acknowledge_native_dispatch_preconditions_default_off",
    "acknowledge_native_runtime_handoff_default_off",
    "acknowledge_native_adapter_kernel_parity_boundaries_default_off",
    "acknowledge_no_native_dispatch_approved_or_executed",
    "acknowledge_no_kernel_launch",
    "acknowledge_no_parity_tensor_or_training_step",
    "acknowledge_no_request_ui_schema_exposure",
    "acknowledge_later_kernel_launch_execution_contract_required",
)
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "runtime_execution_executed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_approved",
    "runtime_dispatch_executed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_approved",
    "native_dispatch_started",
    "native_dispatch_executed",
    "native_dispatch_execution_allowed",
    "native_dispatch_execution_enabled",
    "native_dispatch_execution_executed",
    "native_runtime_handoff_enabled",
    "native_runtime_handoff_executed",
    "native_adapter_handoff_enabled",
    "native_adapter_handoff_executed",
    "native_kernel_handoff_enabled",
    "kernel_launch_allowed",
    "kernel_launch_started",
    "kernel_launch_executed",
    "parity_allowed",
    "parity_executed",
    "parity_check_executed",
    "tensor_transfer_allowed",
    "tensor_transfer_executed",
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
    "post_native_dispatch_execution_request_fields",
    "post_runtime_dispatch_request_fields",
    "native_dispatch_execution_request",
    "native_dispatch_execution_payload",
    "native_dispatch_authorization_request",
    "native_dispatch_authorization_payload",
    "native_runtime_handoff_payload",
    "native_adapter_handoff_payload",
    "native_kernel_handoff_payload",
    "kernel_launch_request",
    "kernel_launch_payload",
    "parity_request",
    "parity_payload",
    "tensor_transfer_request",
    "tensor_transfer_payload",
    "training_step_request",
    "request_adapter_fields",
    "request_schema_fields",
    "ui_route_registration",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_native_update_native_dispatch_execution_contract(
    *,
    runtime_dispatch_contract: Mapping[str, Any] | None = None,
    native_dispatch_execution_evidence: Mapping[str, Any] | None = None,
    native_dispatch_execution_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record native dispatch execution evidence without executing native work."""

    dispatch = _runtime_dispatch_contract_summary(_as_dict(runtime_dispatch_contract))
    evidence = _native_dispatch_execution_summary(_as_dict(native_dispatch_execution_evidence))
    review = _review_summary(_as_dict(native_dispatch_execution_review))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(
        dispatch=dispatch,
        execution=evidence,
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
    elif review.get("approve_native_update_native_dispatch_execution_contract") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_native_dispatch_execution_review_missing")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_native_dispatch_execution_contract_v0",
        "gate": "native_update_native_dispatch_execution_contract",
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_ready": evidence_ready,
        "ready_for_native_dispatch_execution_review": evidence_ready,
        "native_dispatch_execution_contract_recorded": decision == READY_DECISION,
        "native_update_native_dispatch_execution_contract_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "native_dispatch_execution_review_action_required": decision == HOLD_DECISION,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_native_dispatch_execution_request_fields": {},
        "runtime_dispatch_contract_summary": dispatch,
        "native_dispatch_execution_evidence_summary": evidence,
        "native_dispatch_execution_review": review,
        "native_dispatch_execution_review_template": _review_template(dispatch, evidence),
        "progress_gates": {
            "runtime_dispatch_contract_ready": bool(dispatch.get("ready")),
            "native_dispatch_execution_evidence_ready": bool(evidence.get("ready")),
            "native_dispatch_plan_default_off_count": evidence.get("plan_default_off_count", 0),
            "native_dispatch_precondition_default_off_count": evidence.get("precondition_default_off_count", 0),
            "kernel_boundary_default_off_count": evidence.get("kernel_boundary_default_off_count", 0),
            "parity_boundary_default_off_count": evidence.get("parity_boundary_default_off_count", 0),
            "request_ui_schema_exposure_blocked": bool(
                dispatch.get("request_ui_schema_exposure_blocked")
                and evidence.get("request_ui_schema_exposure_blocked")
            ),
            "signed_native_dispatch_execution_review_present": bool(review.get("present")),
        },
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This contract records native dispatch execution evidence only.",
            "It does not approve or execute native dispatch, kernel launch, parity, tensor transfer, request/UI/schema exposure, or training steps.",
            "A later kernel launch execution contract is required before native execution can become active.",
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


def _runtime_dispatch_contract_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    unsafe = _unsafe_claims(contract, "runtime_dispatch_contract")
    return {
        "present": bool(contract),
        "source": str(contract.get("_source_path") or ""),
        "digest": str(contract.get("_source_digest") or _digest_payload(contract) if contract else ""),
        "ok": contract.get("ok") is True,
        "evidence_ready": contract.get("evidence_ready") is True,
        "ready_for_runtime_dispatch_review": contract.get("ready_for_runtime_dispatch_review") is True,
        "runtime_dispatch_contract_recorded": contract.get("runtime_dispatch_contract_recorded") is True,
        "ready": bool(
            contract
            and contract.get("ok") is True
            and contract.get("evidence_ready") is True
            and contract.get("ready_for_runtime_dispatch_review") is True
            and contract.get("post_runtime_dispatch_request_fields", {}) == {}
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


def _native_dispatch_execution_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = _section_set(evidence)
    missing = [section for section in REQUIRED_SECTIONS if section not in sections]
    plans = _rows(evidence, "native_dispatch_execution_plan_inventory")
    authorizations = _rows(evidence, "native_dispatch_authorization_inventory")
    preconditions = _rows(evidence, "native_dispatch_precondition_inventory")
    runtime_handoffs = _rows(evidence, "native_runtime_handoff_boundary")
    adapters = _rows(evidence, "native_dispatch_adapter_boundary")
    kernels = _rows(evidence, "kernel_launch_boundary")
    parity = _rows(evidence, "parity_boundary")
    transfers = _rows(evidence, "tensor_transfer_boundary")
    rollback = _rows(evidence, "rollback_policy")
    observability = _rows(evidence, "observability_policy")
    unsafe = _unsafe_claims(evidence, "native_dispatch_execution_evidence")
    blockers = _dedupe(
        _flag_blockers(evidence)
        + [f"native_update_native_dispatch_execution_section_missing:{section}" for section in missing]
        + _row_blockers(plans, "native_dispatch_plan", require_default_off=True)
        + _row_blockers(authorizations, "native_dispatch_authorization", require_default_off=True)
        + _row_blockers(preconditions, "native_dispatch_precondition", require_default_off=True)
        + _row_blockers(runtime_handoffs, "native_runtime_handoff", require_default_off=True)
        + _row_blockers(adapters, "native_dispatch_adapter", require_default_off=True)
        + _row_blockers(kernels, "kernel_launch_boundary", require_default_off=True)
        + _row_blockers(parity, "parity_boundary", require_default_off=True)
        + _row_blockers(transfers, "tensor_transfer_boundary", require_default_off=True)
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
        "ready": bool(evidence and evidence.get("ok") is True and evidence.get("native_dispatch_execution_contract_ready") is True and not blockers),
        "native_dispatch_execution_contract_ready": evidence.get("native_dispatch_execution_contract_ready") is True,
        "plan_count": len(plans),
        "plan_default_off_count": sum(1 for row in plans if _row_default_off(row)),
        "authorization_count": len(authorizations),
        "authorization_default_off_count": sum(1 for row in authorizations if _row_default_off(row)),
        "precondition_count": len(preconditions),
        "precondition_default_off_count": sum(1 for row in preconditions if _row_default_off(row)),
        "runtime_handoff_default_off_count": sum(1 for row in runtime_handoffs if _row_default_off(row)),
        "adapter_boundary_default_off_count": sum(1 for row in adapters if _row_default_off(row)),
        "kernel_boundary_default_off_count": sum(1 for row in kernels if _row_default_off(row)),
        "parity_boundary_default_off_count": sum(1 for row in parity if _row_default_off(row)),
        "tensor_transfer_boundary_default_off_count": sum(1 for row in transfers if _row_default_off(row)),
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
        "native_dispatch_execution_contract_only",
        "records_evidence_only",
        "manual_only",
        "internal_only",
        "requires_later_kernel_launch_execution_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"native_update_native_dispatch_execution_{field}_missing")
    if evidence.get("default_off") is not True:
        blocked.append("native_update_native_dispatch_execution_default_off_violation")
    if evidence.get("native_dispatch_execution_contract_ready") is not True:
        blocked.append("native_update_native_dispatch_execution_evidence_not_ready")
    if not evidence.get("source") and not evidence.get("_source_path"):
        blocked.append("native_update_native_dispatch_execution_source_missing")
    return blocked


def _row_blockers(rows: list[Mapping[str, Any]], kind: str, *, require_default_off: bool) -> list[str]:
    if not rows:
        return [f"native_update_native_dispatch_execution_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("plan_id") or row.get("check_id") or row.get("id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"native_update_native_dispatch_execution_{kind}_not_ready:{row_id}")
        if not row.get("source"):
            blocked.append(f"native_update_native_dispatch_execution_{kind}_source_missing:{row_id}")
        if require_default_off and not _row_default_off(row):
            blocked.append(f"native_update_native_dispatch_execution_{kind}_not_default_off:{row_id}")
        blocked.extend(_unsafe_claims(row, row_id))
    return _dedupe(blocked)


def _row_default_off(row: Mapping[str, Any]) -> bool:
    default_off_fields = (
        "native_dispatch_allowed",
        "native_dispatch_enabled",
        "native_dispatch_approved",
        "native_dispatch_executed",
        "native_runtime_handoff_enabled",
        "native_adapter_handoff_enabled",
        "kernel_launch_executed",
        "parity_executed",
        "tensor_transfer_executed",
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
        "approve_native_update_native_dispatch_execution_contract": review.get(
            "approve_native_update_native_dispatch_execution_contract"
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
    dispatch: Mapping[str, Any],
    execution: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not dispatch.get("present"):
        blocked.append("native_update_native_dispatch_execution_runtime_dispatch_contract_missing")
    elif not dispatch.get("ready"):
        blocked.append("native_update_native_dispatch_execution_runtime_dispatch_contract_not_ready")
        blocked.extend(_string_list(dispatch.get("blocked_reasons")))
    if not execution.get("present"):
        blocked.append("native_update_native_dispatch_execution_evidence_missing")
    elif not execution.get("ready"):
        blocked.append("native_update_native_dispatch_execution_evidence_not_ready")
        blocked.extend(_string_list(execution.get("blocked_reasons")))
    for event in failure_events:
        blocked.append(f"native_update_native_dispatch_execution_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_native_dispatch_execution_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_native_dispatch_execution_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_native_dispatch_execution_reviewed_at_missing")
    if review.get("requested_scope") != SCOPE:
        blocked.append("native_update_native_dispatch_execution_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_native_dispatch_execution_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_native_dispatch_execution_review_ack_missing:{field}")
    return _dedupe(blocked)


def _review_template(dispatch: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_native_dispatch_execution_contract": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence"] = {
        "runtime_dispatch_contract_digest": dispatch.get("digest"),
        "native_dispatch_execution_digest": execution.get("digest"),
        "native_dispatch_plan_default_off_count": execution.get("plan_default_off_count"),
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
            blocked.append(f"native_update_native_dispatch_execution_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_native_dispatch_execution_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return ["archive_native_update_native_dispatch_execution_contract", "prepare_later_kernel_launch_execution_contract"]
    if evidence_ready:
        return ["collect_signed_native_update_native_dispatch_execution_review"]
    return ["repair_runtime_dispatch_or_native_dispatch_execution_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive native dispatch execution contract and prepare the later kernel launch execution contract"
    if evidence_ready:
        return "record native dispatch execution review while keeping kernel launch, request/UI/schema exposure, and training disabled"
    return "repair runtime dispatch or native dispatch execution evidence before review"


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {k: v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update default-off native dispatch execution contract.")
    parser.add_argument("--runtime-dispatch-contract", required=True)
    parser.add_argument("--native-dispatch-execution-evidence", required=True)
    parser.add_argument("--native-dispatch-execution-review")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    payload = build_native_update_native_dispatch_execution_contract(
        runtime_dispatch_contract=load_json(args.runtime_dispatch_contract),
        native_dispatch_execution_evidence=load_json(args.native_dispatch_execution_evidence),
        native_dispatch_execution_review=load_json(args.native_dispatch_execution_review)
        if args.native_dispatch_execution_review else None,
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
