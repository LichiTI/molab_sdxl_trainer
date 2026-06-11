"""Native dispatch execution contract for TurboCore V5-P61.

P61 records future native dispatch execution contract evidence after P60. It
checks native dispatch plan, authorization, preconditions, runtime handoff,
adapter, kernel, parity, tensor-transfer, rollback, and observability evidence
only; it does not approve or execute native dispatch, kernel launch, parity,
training steps, request fields, UI exposure, or training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT, REPO_ROOT = Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (
    as_dict as _as_dict,
    dedupe as _dedupe,
    default_off_confirmed as _default_off_confirmed,
    digest as _digest,
    event_list as _event_list,
    history_clear as _history_clear,
    history_summary as _history_summary,
    request_adapter_off as _request_adapter_off,
    source as _source,
    string_list as _string_list,
)
from core.turbocore_v5_owner_review_evidence_package import load_json
from core.turbocore_v5_runtime_dispatch_execution_control_contract import (
    P60_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P60_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P60_UNSAFE_TRUE_FIELDS,
)


P61_READY_DECISION = "native_dispatch_execution_contract_recorded_default_off"
P61_BLOCKED_DECISION = "native_dispatch_execution_contract_blocked_default_off"
P61_HOLD_DECISION = "native_dispatch_execution_contract_hold_for_signed_review_default_off"
P61_REJECTED_DECISION = "native_dispatch_execution_contract_rejected_default_off"
P61_SCOPE = "native_dispatch_execution_contract"
DEFAULT_REQUIRED_SECTIONS = (
    "p60_runtime_dispatch_execution_control_reference", "native_dispatch_execution_plan_inventory",
    "native_dispatch_authorization_boundary", "native_dispatch_precondition_inventory",
    "native_runtime_handoff_boundary", "native_dispatch_adapter_boundary", "kernel_launch_boundary",
    "parity_boundary", "tensor_transfer_boundary", "request_adapter_boundary",
    "rollback_preflight_inventory", "observability_preflight_inventory",
    "no_native_dispatch_execution_boundary", "no_kernel_launch_boundary",
    "no_parity_execution_boundary", "no_training_step_boundary", "no_runtime_dispatch_execution_boundary",
    "no_runtime_execution_boundary", "no_runtime_state_refresh_boundary", "no_runtime_adapter_enabled_boundary",
    "no_execution_replay_boundary", "no_artifact_load_boundary", "no_request_fields_boundary",
    "no_training_launch_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p60_runtime_dispatch_execution_control_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure",
    "acknowledge_no_native_dispatch_approved", "acknowledge_no_native_dispatch_executed",
    "acknowledge_no_kernel_launch_executed", "acknowledge_no_parity_executed",
    "acknowledge_no_tensor_transfer_executed", "acknowledge_no_training_step_executed",
    "acknowledge_no_runtime_dispatch_executed", "acknowledge_no_runtime_execution_executed",
    "acknowledge_no_runtime_state_refreshed", "acknowledge_no_runtime_adapter_enabled",
    "acknowledge_no_execution_replay_executed", "acknowledge_no_artifact_loaded",
    "acknowledge_no_request_adapter_enabled", "acknowledge_no_request_fields_emitted",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_native_dispatch_execution_contract_evidence_replayable",
    "acknowledge_later_kernel_launch_execution_contract_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P60_UNSAFE_TRUE_FIELDS,
            "native_dispatch_approved", "native_dispatch_execution_allowed",
            "native_dispatch_execution_contract_allowed", "native_dispatch_execution_contract_enabled",
            "native_dispatch_execution_contract_executed", "native_dispatch_execution_contract_applied",
            "native_dispatch_started", "native_dispatch_plan_activated", "native_dispatch_plan_executed",
            "native_dispatch_request_submitted", "native_dispatch_request_executed",
            "native_runtime_handoff_enabled", "native_runtime_handoff_executed",
            "native_adapter_handoff_enabled", "native_adapter_handoff_executed",
            "native_kernel_handoff_enabled", "kernel_launch_requested", "kernel_launch_allowed",
            "kernel_launch_started", "parity_check_allowed", "tensor_transfer_executed",
            "native_execution_result_recorded", "dispatch_control_default_enabled",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P60_UNSAFE_NON_EMPTY_FIELDS,
            "post_p61_request_fields", "post_native_dispatch_execution_contract_fields",
            "native_dispatch_execution_contract_request", "native_dispatch_execution_contract_payload",
            "native_dispatch_execution_plan_request", "native_dispatch_execution_plan_payload",
            "native_dispatch_authorization_request", "native_dispatch_authorization_payload",
            "native_runtime_handoff_payload", "native_adapter_handoff_payload",
            "native_kernel_handoff_payload", "native_execution_result_payload",
            "kernel_launch_payload", "parity_check_payload", "tensor_transfer_payload",
            "training_step_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(
    f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed"
)
P60_REQUIRED_FALSE_FIELDS = _P60_UNSAFE_TRUE_FIELDS


def build_v5_native_dispatch_execution_contract(
    *,
    p60_runtime_dispatch_execution_control: Mapping[str, Any] | None = None,
    native_dispatch_execution_evidence: Mapping[str, Any] | None = None,
    native_dispatch_execution_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record native dispatch execution evidence without executing it."""

    p60 = _as_dict(p60_runtime_dispatch_execution_control)
    evidence = _as_dict(native_dispatch_execution_evidence)
    review = _as_dict(native_dispatch_execution_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p60_summary = _p60_summary(p60)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p60_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P61_HOLD_DECISION:
        blockers.append("v5_p61_signed_native_dispatch_execution_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P61_READY_DECISION
    rejected = decision == P61_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_native_dispatch_execution_contract_v0",
        "gate": "v5_native_dispatch_execution_contract",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "native_dispatch_execution_review_recorded": decision_record_ready,
        "native_dispatch_execution_review_signed": decision_record_ready,
        "native_dispatch_execution_contract_ready": ready,
        "native_dispatch_execution_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p61_request_fields": {},
        "p60_runtime_dispatch_execution_control_summary": p60_summary,
        "native_dispatch_execution_evidence_summary": evidence_summary,
        "native_dispatch_execution_review": review_summary,
        "native_dispatch_execution_review_template": _review_template(),
        "progress_gates": _progress_gates(p60_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P61 records native dispatch execution contract evidence only.",
            "P61 does not approve or execute native dispatch, kernel launch, parity, tensor transfer, training steps, request fields, UI exposure, or training.",
            "A later kernel launch execution contract is still required before native dispatch can become active.",
        ],
    }


def _p60_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "runtime_dispatch_execution_control_ready": report.get("runtime_dispatch_execution_control_ready") is True,
        "runtime_dispatch_execution_control_evidence_recorded": report.get(
            "runtime_dispatch_execution_control_evidence_recorded"
        ) is True,
        "runtime_dispatch_execution_control_review_signed": report.get(
            "runtime_dispatch_execution_control_review_signed"
        ) is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P60_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p60_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p60"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    plan_rows = _rows(evidence, "native_dispatch_execution_plan_inventory", "native_dispatch_plan_inventory")
    auth_rows = _rows(evidence, "native_dispatch_authorization_boundary", "native_authorization_inventory")
    precondition_rows = _rows(evidence, "native_dispatch_precondition_inventory", "native_precondition_inventory")
    runtime_rows = _rows(evidence, "native_runtime_handoff_boundary", "native_runtime_handoff_inventory")
    adapter_rows = _rows(evidence, "native_dispatch_adapter_boundary", "native_adapter_inventory")
    kernel_rows = _rows(evidence, "kernel_launch_boundary", "kernel_launch_inventory")
    parity_rows = _rows(evidence, "parity_boundary", "parity_inventory")
    tensor_rows = _rows(evidence, "tensor_transfer_boundary", "tensor_transfer_inventory")
    rollback_rows = _rows(evidence, "rollback_preflight_inventory", "rollback_preflight")
    observability_rows = _rows(evidence, "observability_preflight_inventory", "observability_preflight")
    blockers = _evidence_blocker_list(
        evidence, missing_sections, plan_rows, auth_rows, precondition_rows, runtime_rows, adapter_rows, kernel_rows,
        parity_rows, tensor_rows, rollback_rows, observability_rows,
    )
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "native_dispatch_execution_contract_ready": evidence.get("native_dispatch_execution_contract_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "native_dispatch_execution_contract_only": evidence.get("native_dispatch_execution_contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_kernel_launch_execution_contract": evidence.get(
            "requires_later_kernel_launch_execution_contract"
        ) is True,
        "requires_explicit_owner_approval": evidence.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": evidence.get("requires_explicit_operator_opt_in") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "plan_count": len(plan_rows),
        "authorization_count": len(auth_rows),
        "precondition_count": len(precondition_rows),
        "runtime_handoff_count": len(runtime_rows),
        "adapter_boundary_count": len(adapter_rows),
        "kernel_boundary_count": len(kernel_rows),
        "parity_boundary_count": len(parity_rows),
        "tensor_transfer_boundary_count": len(tensor_rows),
        "rollback_preflight_count": len(rollback_rows),
        "observability_preflight_count": len(observability_rows),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_dispatch_execution_contract": review.get("approve_native_dispatch_execution_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p60: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p60_runtime_dispatch_execution_control_ready": _p60_ready(p60),
        "native_dispatch_execution_evidence_ready": bool(evidence.get("ready", False)),
        "signed_native_dispatch_execution_review_present": bool(review.get("present", False)),
        "requested_scope_valid": review.get("requested_scope") == P61_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P61_BLOCKED_DECISION
    if not review:
        return P61_HOLD_DECISION
    if review_blockers:
        return P61_BLOCKED_DECISION
    if review.get("approve_native_dispatch_execution_contract") is True:
        return P61_READY_DECISION
    return P61_REJECTED_DECISION


def _evidence_blockers(
    p60_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p60_summary.get("present", False)):
        blocked.append("v5_p61_p60_runtime_dispatch_execution_control_missing")
    elif not _p60_ready(p60_summary):
        blocked.append("v5_p61_p60_runtime_dispatch_execution_control_not_ready")
        blocked.extend(_string_list(p60_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p60_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p60_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p61_native_dispatch_execution_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p61_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p61_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p61_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p61_reviewed_at_missing")
    if review.get("requested_scope") != P61_SCOPE:
        blocked.append("v5_p61_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p61_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p61_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p60_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("runtime_dispatch_execution_control_ready")
        and summary.get("runtime_dispatch_execution_control_evidence_recorded")
        and summary.get("runtime_dispatch_execution_control_review_signed")
        and summary.get("decision") == P60_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P60_REQUIRED_FALSE_FIELDS)
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
        and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers"))
        and not _string_list(summary.get("unsafe_claims"))
    )


def _evidence_blocker_list(
    evidence: Mapping[str, Any],
    missing: list[str],
    plan_rows: list[dict[str, Any]],
    auth_rows: list[dict[str, Any]],
    precondition_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    parity_rows: list[dict[str, Any]],
    tensor_rows: list[dict[str, Any]],
    rollback_rows: list[dict[str, Any]],
    observability_rows: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p61_native_dispatch_execution_evidence_not_ok")
    if evidence.get("native_dispatch_execution_contract_ready") is not True:
        blocked.append("v5_p61_native_dispatch_execution_evidence_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "native_dispatch_execution_contract_only",
        "records_evidence_only", "manual_only", "internal_only",
        "requires_later_kernel_launch_execution_contract", "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p61_native_dispatch_execution_evidence_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p61_native_dispatch_execution_evidence_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p61_native_dispatch_execution_evidence_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p61_native_dispatch_execution_evidence_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p61_native_dispatch_execution_evidence_source_missing")
    for section in missing:
        blocked.append(f"v5_p61_native_dispatch_execution_evidence_section_missing:{section}")
    for rows, kind in (
        (plan_rows, "native_dispatch_execution_plan"), (auth_rows, "native_dispatch_authorization"),
        (precondition_rows, "native_dispatch_precondition"), (runtime_rows, "native_runtime_handoff"),
        (adapter_rows, "native_dispatch_adapter"), (kernel_rows, "kernel_launch_boundary"),
        (parity_rows, "parity_boundary"), (tensor_rows, "tensor_transfer_boundary"),
        (rollback_rows, "rollback_preflight"), (observability_rows, "observability_preflight"),
    ):
        blocked.extend(_row_blockers(rows, kind))
    blocked.extend(_unsafe_claims(evidence, "native_dispatch_execution_evidence"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _row_blockers(rows: list[Mapping[str, Any]], kind: str) -> list[str]:
    if not rows:
        return [f"v5_p61_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("plan_id") or row.get("check_id") or row.get("id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"v5_p61_{kind}_not_ready:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p61_{kind}_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p61_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p61_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _rows(payload: Mapping[str, Any], field: str, fallback: str) -> list[dict[str, Any]]:
    value = payload.get(field, payload.get(fallback))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    return {str(item).strip() for item in sections if str(item).strip()}


def _review_template() -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": P61_SCOPE,
        "approve_native_dispatch_execution_contract": False,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P61_READY_DECISION:
        return ["archive_p61_native_dispatch_execution_contract"]
    if decision == P61_REJECTED_DECISION:
        return ["record_p61_default_off_rejection_or_repair_native_dispatch_execution"]
    if decision == P61_HOLD_DECISION:
        return ["collect_signed_native_dispatch_execution_review"]
    if any("p60" in item for item in blockers):
        return ["repair_p60_runtime_dispatch_execution_control_contract"]
    if any("native" in item or "kernel" in item or "preflight" in item for item in blockers):
        return ["repair_native_dispatch_execution_evidence"]
    return ["clear_failure_or_rollback_history_before_p61_contract"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P61_READY_DECISION:
        return "archive P61 contract; kernel launch execution still requires a later explicit contract"
    if decision == P61_REJECTED_DECISION:
        return "record the signed rejection and keep native dispatch execution default-off for repair"
    if decision == P61_HOLD_DECISION:
        return "collect a signed native dispatch execution review over P60 evidence"
    if any("p60" in item for item in blockers):
        return "repair the P60 runtime dispatch execution control contract before P61"
    return "hold P61 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P61 native dispatch execution contract.")
    parser.add_argument("--p60-runtime-dispatch-execution-control", default="", help="P60 control JSON.")
    parser.add_argument("--native-dispatch-execution-evidence", default="", help="P61 evidence JSON.")
    parser.add_argument("--native-dispatch-execution-review", default="", help="Signed P61 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_native_dispatch_execution_contract(
        p60_runtime_dispatch_execution_control=load_json(args.p60_runtime_dispatch_execution_control)
        if args.p60_runtime_dispatch_execution_control else None,
        native_dispatch_execution_evidence=load_json(args.native_dispatch_execution_evidence)
        if args.native_dispatch_execution_evidence else None,
        native_dispatch_execution_review=load_json(args.native_dispatch_execution_review)
        if args.native_dispatch_execution_review else None,
        failure_history=load_json(args.failure_history) if args.failure_history else None,
        rollback_history=load_json(args.rollback_history) if args.rollback_history else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_native_dispatch_execution_contract"]
