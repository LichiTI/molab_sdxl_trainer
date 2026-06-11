"""Runtime dispatch execution control contract for TurboCore V5-P60.

P60 records future runtime dispatch execution control evidence after P59. It
checks dispatch control plan, authorization, preconditions, runtime adapter and
state locks, native/kernel boundaries, rollback, and observability evidence
only; it does not approve or execute runtime dispatch, native dispatch, kernel
launch, parity, training steps, request fields, UI exposure, or training.
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
from core.turbocore_v5_runtime_execution_contract import (
    P59_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P59_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P59_UNSAFE_TRUE_FIELDS,
)


P60_READY_DECISION = "runtime_dispatch_execution_control_recorded_default_off"
P60_BLOCKED_DECISION = "runtime_dispatch_execution_control_blocked_default_off"
P60_HOLD_DECISION = "runtime_dispatch_execution_control_hold_for_signed_review_default_off"
P60_REJECTED_DECISION = "runtime_dispatch_execution_control_rejected_default_off"
P60_SCOPE = "runtime_dispatch_execution_control_contract"
DEFAULT_REQUIRED_SECTIONS = (
    "p59_runtime_execution_contract_reference", "runtime_dispatch_execution_control_plan_inventory",
    "runtime_dispatch_authorization_boundary", "runtime_dispatch_precondition_inventory",
    "runtime_adapter_lock_boundary", "runtime_state_lock_boundary", "native_dispatch_boundary",
    "kernel_launch_boundary", "parity_training_boundary", "request_adapter_boundary",
    "rollback_preflight_inventory", "observability_preflight_inventory",
    "no_runtime_dispatch_execution_boundary", "no_native_dispatch_execution_boundary",
    "no_kernel_launch_boundary", "no_parity_execution_boundary", "no_training_step_boundary",
    "no_runtime_execution_boundary", "no_runtime_state_refresh_boundary", "no_runtime_adapter_enabled_boundary",
    "no_execution_replay_boundary", "no_artifact_load_boundary", "no_request_fields_boundary",
    "no_training_launch_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p59_runtime_execution_contract_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure",
    "acknowledge_no_runtime_dispatch_approved", "acknowledge_no_runtime_dispatch_executed",
    "acknowledge_no_native_dispatch_enabled", "acknowledge_no_native_dispatch_executed",
    "acknowledge_no_kernel_launch_executed", "acknowledge_no_parity_executed",
    "acknowledge_no_training_step_executed", "acknowledge_no_runtime_execution_executed",
    "acknowledge_no_runtime_state_refreshed", "acknowledge_no_runtime_adapter_enabled",
    "acknowledge_no_execution_replay_executed", "acknowledge_no_artifact_loaded",
    "acknowledge_no_request_adapter_enabled", "acknowledge_no_request_fields_emitted",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_runtime_dispatch_execution_control_evidence_replayable",
    "acknowledge_later_native_dispatch_execution_contract_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P59_UNSAFE_TRUE_FIELDS,
            "runtime_dispatch_approved", "runtime_dispatch_execution_allowed",
            "runtime_dispatch_execution_control_allowed", "runtime_dispatch_execution_control_enabled",
            "runtime_dispatch_execution_control_executed", "runtime_dispatch_execution_control_applied",
            "runtime_dispatch_execution_started", "runtime_dispatch_plan_activated",
            "runtime_dispatch_plan_executed", "runtime_dispatch_request_submitted",
            "runtime_dispatch_request_executed", "runtime_native_dispatch_handoff_enabled",
            "native_dispatch_allowed", "native_dispatch_started", "native_kernel_dispatch_enabled",
            "kernel_launch_allowed", "kernel_launch_started", "parity_check_allowed",
            "training_step_allowed", "dispatch_control_default_enabled",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P59_UNSAFE_NON_EMPTY_FIELDS,
            "post_p60_request_fields", "post_runtime_dispatch_execution_control_fields",
            "runtime_dispatch_execution_control_request", "runtime_dispatch_execution_control_payload",
            "runtime_dispatch_execution_plan_request", "runtime_dispatch_execution_plan_payload",
            "runtime_dispatch_authorization_request", "runtime_dispatch_authorization_payload",
            "runtime_adapter_handoff_payload", "runtime_state_lock_payload",
            "native_dispatch_handoff_payload", "kernel_launch_payload", "parity_check_payload",
            "training_step_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(
    f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed"
)
P59_REQUIRED_FALSE_FIELDS = _P59_UNSAFE_TRUE_FIELDS


def build_v5_runtime_dispatch_execution_control_contract(
    *,
    p59_runtime_execution_contract: Mapping[str, Any] | None = None,
    runtime_dispatch_execution_control_evidence: Mapping[str, Any] | None = None,
    runtime_dispatch_execution_control_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record runtime dispatch execution control evidence without executing it."""

    p59 = _as_dict(p59_runtime_execution_contract)
    evidence = _as_dict(runtime_dispatch_execution_control_evidence)
    review = _as_dict(runtime_dispatch_execution_control_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p59_summary = _p59_summary(p59)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p59_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P60_HOLD_DECISION:
        blockers.append("v5_p60_signed_runtime_dispatch_execution_control_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P60_READY_DECISION
    rejected = decision == P60_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_runtime_dispatch_execution_control_contract_v0",
        "gate": "v5_runtime_dispatch_execution_control_contract",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "runtime_dispatch_execution_control_review_recorded": decision_record_ready,
        "runtime_dispatch_execution_control_review_signed": decision_record_ready,
        "runtime_dispatch_execution_control_ready": ready,
        "runtime_dispatch_execution_control_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p60_request_fields": {},
        "p59_runtime_execution_contract_summary": p59_summary,
        "runtime_dispatch_execution_control_evidence_summary": evidence_summary,
        "runtime_dispatch_execution_control_review": review_summary,
        "runtime_dispatch_execution_control_review_template": _review_template(),
        "progress_gates": _progress_gates(p59_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P60 records runtime dispatch execution control evidence only.",
            "P60 does not approve or execute runtime dispatch, native dispatch, kernel launch, parity, training steps, request fields, UI exposure, or training.",
            "A later native dispatch execution contract is still required before runtime dispatch can become active.",
        ],
    }


def _p59_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "runtime_execution_contract_ready": report.get("runtime_execution_contract_ready") is True,
        "runtime_execution_contract_evidence_recorded": report.get("runtime_execution_contract_evidence_recorded") is True,
        "runtime_execution_contract_review_signed": report.get("runtime_execution_contract_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P59_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p59_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p59"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    plan_rows = _rows(evidence, "runtime_dispatch_execution_control_plan_inventory", "dispatch_control_plan_inventory")
    auth_rows = _rows(evidence, "runtime_dispatch_authorization_boundary", "dispatch_authorization_inventory")
    precondition_rows = _rows(evidence, "runtime_dispatch_precondition_inventory", "dispatch_precondition_inventory")
    adapter_lock_rows = _rows(evidence, "runtime_adapter_lock_boundary", "runtime_adapter_lock_inventory")
    state_lock_rows = _rows(evidence, "runtime_state_lock_boundary", "runtime_state_lock_inventory")
    native_rows = _rows(evidence, "native_dispatch_boundary", "native_dispatch_inventory")
    kernel_rows = _rows(evidence, "kernel_launch_boundary", "kernel_launch_inventory")
    rollback_rows = _rows(evidence, "rollback_preflight_inventory", "rollback_preflight")
    observability_rows = _rows(evidence, "observability_preflight_inventory", "observability_preflight")
    blockers = _evidence_blocker_list(
        evidence, missing_sections, plan_rows, auth_rows, precondition_rows, adapter_lock_rows, state_lock_rows,
        native_rows, kernel_rows, rollback_rows, observability_rows,
    )
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "runtime_dispatch_execution_control_ready": evidence.get("runtime_dispatch_execution_control_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "runtime_dispatch_execution_control_only": evidence.get("runtime_dispatch_execution_control_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_native_dispatch_execution_contract": evidence.get(
            "requires_later_native_dispatch_execution_contract"
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
        "adapter_lock_count": len(adapter_lock_rows),
        "state_lock_count": len(state_lock_rows),
        "native_boundary_count": len(native_rows),
        "kernel_boundary_count": len(kernel_rows),
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
        "approve_runtime_dispatch_execution_control_contract": review.get(
            "approve_runtime_dispatch_execution_control_contract"
        ) is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p59: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p59_runtime_execution_contract_ready": _p59_ready(p59),
        "runtime_dispatch_execution_control_evidence_ready": bool(evidence.get("ready", False)),
        "signed_runtime_dispatch_execution_control_review_present": bool(review.get("present", False)),
        "requested_scope_valid": review.get("requested_scope") == P60_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P60_BLOCKED_DECISION
    if not review:
        return P60_HOLD_DECISION
    if review_blockers:
        return P60_BLOCKED_DECISION
    if review.get("approve_runtime_dispatch_execution_control_contract") is True:
        return P60_READY_DECISION
    return P60_REJECTED_DECISION


def _evidence_blockers(
    p59_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p59_summary.get("present", False)):
        blocked.append("v5_p60_p59_runtime_execution_contract_missing")
    elif not _p59_ready(p59_summary):
        blocked.append("v5_p60_p59_runtime_execution_contract_not_ready")
        blocked.extend(_string_list(p59_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p59_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p59_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p60_runtime_dispatch_execution_control_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p60_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p60_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p60_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p60_reviewed_at_missing")
    if review.get("requested_scope") != P60_SCOPE:
        blocked.append("v5_p60_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p60_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p60_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p59_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("runtime_execution_contract_ready")
        and summary.get("runtime_execution_contract_evidence_recorded")
        and summary.get("runtime_execution_contract_review_signed")
        and summary.get("decision") == P59_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P59_REQUIRED_FALSE_FIELDS)
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
    adapter_lock_rows: list[dict[str, Any]],
    state_lock_rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    rollback_rows: list[dict[str, Any]],
    observability_rows: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p60_runtime_dispatch_execution_control_evidence_not_ok")
    if evidence.get("runtime_dispatch_execution_control_ready") is not True:
        blocked.append("v5_p60_runtime_dispatch_execution_control_evidence_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "runtime_dispatch_execution_control_only",
        "records_evidence_only", "manual_only", "internal_only",
        "requires_later_native_dispatch_execution_contract", "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p60_runtime_dispatch_execution_control_evidence_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p60_runtime_dispatch_execution_control_evidence_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p60_runtime_dispatch_execution_control_evidence_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p60_runtime_dispatch_execution_control_evidence_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p60_runtime_dispatch_execution_control_evidence_source_missing")
    for section in missing:
        blocked.append(f"v5_p60_runtime_dispatch_execution_control_evidence_section_missing:{section}")
    for rows, kind in (
        (plan_rows, "runtime_dispatch_execution_control_plan"),
        (auth_rows, "runtime_dispatch_authorization"),
        (precondition_rows, "runtime_dispatch_precondition"),
        (adapter_lock_rows, "runtime_adapter_lock"),
        (state_lock_rows, "runtime_state_lock"),
        (native_rows, "native_dispatch_boundary"),
        (kernel_rows, "kernel_launch_boundary"),
        (rollback_rows, "rollback_preflight"),
        (observability_rows, "observability_preflight"),
    ):
        blocked.extend(_row_blockers(rows, kind))
    blocked.extend(_unsafe_claims(evidence, "runtime_dispatch_execution_control_evidence"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _row_blockers(rows: list[Mapping[str, Any]], kind: str) -> list[str]:
    if not rows:
        return [f"v5_p60_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("plan_id") or row.get("check_id") or row.get("id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"v5_p60_{kind}_not_ready:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p60_{kind}_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p60_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p60_unsafe_claim:{owner}:{field}")
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
        "requested_scope": P60_SCOPE,
        "approve_runtime_dispatch_execution_control_contract": False,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P60_READY_DECISION:
        return ["archive_p60_runtime_dispatch_execution_control_contract"]
    if decision == P60_REJECTED_DECISION:
        return ["record_p60_default_off_rejection_or_repair_runtime_dispatch_execution_control"]
    if decision == P60_HOLD_DECISION:
        return ["collect_signed_runtime_dispatch_execution_control_review"]
    if any("p59" in item for item in blockers):
        return ["repair_p59_runtime_execution_contract"]
    if any("dispatch" in item or "preflight" in item or "precondition" in item for item in blockers):
        return ["repair_runtime_dispatch_execution_control_evidence"]
    return ["clear_failure_or_rollback_history_before_p60_contract"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P60_READY_DECISION:
        return "archive P60 contract; native dispatch execution still requires a later explicit contract"
    if decision == P60_REJECTED_DECISION:
        return "record the signed rejection and keep runtime dispatch execution default-off for repair"
    if decision == P60_HOLD_DECISION:
        return "collect a signed runtime dispatch execution control review over P59 evidence"
    if any("p59" in item for item in blockers):
        return "repair the P59 runtime execution contract before P60"
    return "hold P60 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P60 runtime dispatch execution control contract.")
    parser.add_argument("--p59-runtime-execution-contract", default="", help="P59 runtime execution contract JSON.")
    parser.add_argument("--runtime-dispatch-execution-control-evidence", default="", help="P60 evidence JSON.")
    parser.add_argument("--runtime-dispatch-execution-control-review", default="", help="Signed P60 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_runtime_dispatch_execution_control_contract(
        p59_runtime_execution_contract=load_json(args.p59_runtime_execution_contract)
        if args.p59_runtime_execution_contract else None,
        runtime_dispatch_execution_control_evidence=load_json(args.runtime_dispatch_execution_control_evidence)
        if args.runtime_dispatch_execution_control_evidence else None,
        runtime_dispatch_execution_control_review=load_json(args.runtime_dispatch_execution_control_review)
        if args.runtime_dispatch_execution_control_review else None,
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


__all__ = ["build_v5_runtime_dispatch_execution_control_contract"]
