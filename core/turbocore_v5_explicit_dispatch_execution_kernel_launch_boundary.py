"""Explicit dispatch execution / kernel launch boundary for TurboCore V5-P50.

P50 records future dispatch execution and kernel launch boundary evidence after
P49. It does not execute runtime/native dispatch, launch kernels, execute
training steps, emit request fields, expose UI, or launch training.
"""

from __future__ import annotations

import argparse, json, sys
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


P49_READY_DECISION = "runtime_dispatch_contract_boundary_recorded_default_off"
P50_READY_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_recorded_default_off"
P50_BLOCKED_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_blocked_default_off"
P50_HOLD_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_hold_for_signed_review_default_off"
P50_REJECTED_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_rejected_default_off"
P50_SCOPE = "explicit_dispatch_execution_kernel_launch_boundary"

DEFAULT_REQUIRED_SECTIONS = (
    "p49_runtime_dispatch_boundary_reference", "explicit_dispatch_execution_plan_inventory",
    "dispatch_execution_precondition_inventory", "runtime_dispatch_execution_boundary",
    "native_dispatch_execution_boundary", "kernel_launch_boundary", "training_step_boundary",
    "request_adapter_boundary", "no_runtime_dispatch_execution_boundary",
    "no_native_dispatch_execution_boundary", "no_kernel_launch_execution_boundary",
    "no_training_step_boundary", "no_request_fields_boundary", "no_training_launch_boundary",
    "rollback_policy", "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p49_runtime_dispatch_boundary_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure",
    "acknowledge_no_runtime_dispatch_executed", "acknowledge_no_native_dispatch_executed",
    "acknowledge_no_kernel_launch_executed", "acknowledge_no_training_step_executed",
    "acknowledge_no_request_adapter_enabled", "acknowledge_no_request_fields_emitted",
    "acknowledge_no_default_or_auto_rollout", "acknowledge_dispatch_execution_kernel_launch_evidence_replayable",
    "acknowledge_later_native_execution_readiness_contract_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = (
    "training_launch_allowed", "auto_launch_allowed", "runs_dispatched", "default_training_path_enabled",
    "training_path_enabled", "default_rollout_allowed", "auto_rollout_allowed", "ui_exposure_allowed",
    "product_ui_exposure_allowed", "launcher_exposure_allowed", "webui_exposure_allowed",
    "request_adapter_mapping_allowed", "request_fields_emitted", "request_adapter_registered",
    "request_adapter_enabled", "runtime_adapter_registered", "runtime_adapter_enabled",
    "runtime_enablement_allowed", "runtime_enablement_enabled", "runtime_enablement_executed",
    "runtime_enablement_applied", "runtime_enablement_execution_allowed",
    "runtime_enablement_execution_enabled", "runtime_enablement_execution_executed",
    "runtime_enablement_execution_applied", "runtime_execution_allowed", "runtime_execution_executed",
    "runtime_dispatch_allowed", "runtime_dispatch_enabled", "runtime_dispatch_executed",
    "runtime_dispatch_applied", "explicit_dispatch_execution_allowed", "explicit_dispatch_execution_enabled",
    "explicit_dispatch_execution_executed", "runtime_activation_allowed", "runtime_activation_enabled",
    "native_runtime_enabled", "native_dispatch_allowed", "native_dispatch_enabled", "native_dispatch_started",
    "native_dispatch_executed", "kernel_launch_allowed", "kernel_launch_enabled", "kernel_launch_executed",
    "training_step_allowed", "training_step_enabled", "training_step_executed",
    "generation_request_patch_allowed", "config_adapter_patch_allowed", "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed", "training_manager_patch_allowed", "rollout_authorization_allowed",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p49_request_fields", "post_p50_request_fields", "post_dispatch_execution_kernel_launch_fields",
    "runtime_dispatch_request", "runtime_dispatch_payload", "dispatch_execution_request",
    "dispatch_execution_payload", "explicit_dispatch_execution_request", "explicit_dispatch_execution_payload",
    "native_dispatch_request", "native_dispatch_payload", "kernel_launch_request", "kernel_launch_payload",
    "training_step_request", "training_step_payload", "request_adapter_fields", "request_fields",
    "api_route_registration", "backend_route_registration", "ui_route_registration", "launcher_menu_entry",
    "webui_tab_entry", "launch_request", "training_request", "generation_request_patch",
    "config_adapter_patch", "runtime_resolver_patch", "execution_resolver_patch", "entry_train_patch",
    "training_manager_patch", "auto_launch_plan",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed")
P49_REQUIRED_FALSE_FIELDS = (
    "default_behavior_changed", "training_launch_allowed", "auto_launch_allowed", "runs_dispatched",
    "ui_exposure_allowed", "request_fields_emitted", "request_adapter_enabled", "runtime_dispatch_enabled",
    "runtime_dispatch_executed", "native_dispatch_enabled", "native_dispatch_started", "native_dispatch_executed",
    "kernel_launch_executed", "training_step_executed",
)


def build_v5_explicit_dispatch_execution_kernel_launch_boundary(
    *,
    p49_runtime_dispatch_contract_boundary: Mapping[str, Any] | None = None,
    dispatch_execution_kernel_launch_evidence: Mapping[str, Any] | None = None,
    dispatch_execution_kernel_launch_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record explicit dispatch execution / kernel launch evidence without executing it."""

    p49 = _as_dict(p49_runtime_dispatch_contract_boundary)
    evidence = _as_dict(dispatch_execution_kernel_launch_evidence)
    review = _as_dict(dispatch_execution_kernel_launch_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p49_summary = _p49_summary(p49)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p49_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P50_HOLD_DECISION:
        blockers.append("v5_p50_signed_dispatch_execution_kernel_launch_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P50_READY_DECISION
    rejected = decision == P50_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_explicit_dispatch_execution_kernel_launch_boundary_v0",
        "gate": "v5_explicit_dispatch_execution_kernel_launch_boundary",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "dispatch_execution_kernel_launch_review_recorded": decision_record_ready,
        "dispatch_execution_kernel_launch_review_signed": decision_record_ready,
        "explicit_dispatch_execution_kernel_launch_boundary_ready": ready,
        "dispatch_execution_kernel_launch_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p50_request_fields": {},
        "p49_runtime_dispatch_boundary_summary": p49_summary,
        "dispatch_execution_kernel_launch_evidence_summary": evidence_summary,
        "dispatch_execution_kernel_launch_review": review_summary,
        "dispatch_execution_kernel_launch_review_template": _review_template(),
        "progress_gates": _progress_gates(p49_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P50 records explicit dispatch execution and kernel launch boundary evidence only.",
            "P50 does not execute runtime/native dispatch, launch kernels, execute training steps, emit request fields, or launch training.",
            "A later native execution readiness or kernel implementation contract is still required before runtime behavior can become active.",
        ],
    }


def _p49_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "runtime_dispatch_contract_boundary_ready": report.get("runtime_dispatch_contract_boundary_ready") is True,
        "runtime_dispatch_evidence_recorded": report.get("runtime_dispatch_evidence_recorded") is True,
        "runtime_dispatch_review_signed": report.get("runtime_dispatch_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P49_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p49_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p49"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    plan_rows = _rows(evidence, "explicit_dispatch_execution_plan_inventory", "dispatch_execution_plan_inventory")
    precondition_rows = _rows(evidence, "dispatch_execution_precondition_inventory", "kernel_launch_precondition_inventory")
    blockers = _evidence_blocker_list(evidence, missing_sections, plan_rows, precondition_rows)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "explicit_dispatch_execution_kernel_launch_boundary_ready": evidence.get(
            "explicit_dispatch_execution_kernel_launch_boundary_ready"
        )
        is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "execution_only": evidence.get("execution_only") is True,
        "kernel_launch_boundary_only": evidence.get("kernel_launch_boundary_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_native_execution_readiness_contract": _any_true(
            evidence,
            "requires_later_native_execution_readiness_contract",
            "requires_later_kernel_launch_implementation_readiness_contract",
            "requires_later_runtime_execution_contract",
        ),
        "requires_explicit_owner_approval": evidence.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": evidence.get("requires_explicit_operator_opt_in") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "blocked_reasons": _string_list(evidence.get("blocked_reasons")),
        "promotion_blockers": _string_list(evidence.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(evidence, "dispatch_execution_kernel_launch"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    approval = review.get("approve_explicit_dispatch_execution_kernel_launch_boundary")
    if approval is None:
        approval = review.get("approve_dispatch_execution_kernel_launch_boundary")
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_explicit_dispatch_execution_kernel_launch_boundary": approval is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p49: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p49_runtime_dispatch_boundary_present": bool(p49.get("present", False)),
        "p49_runtime_dispatch_boundary_ready": _p49_ready(p49),
        "dispatch_execution_kernel_launch_evidence_present": bool(evidence.get("present", False)),
        "dispatch_execution_kernel_launch_evidence_ready": bool(evidence.get("ready", False)),
        "signed_dispatch_execution_kernel_launch_review_present": bool(review.get("present", False)),
        "reviewer_present": bool(review.get("reviewer")),
        "reviewed_at_present": bool(review.get("reviewed_at")),
        "requested_scope_valid": review.get("requested_scope") == P50_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P50_BLOCKED_DECISION
    if not review:
        return P50_HOLD_DECISION
    if review_blockers:
        return P50_BLOCKED_DECISION
    if review.get("approve_explicit_dispatch_execution_kernel_launch_boundary") is True or review.get(
        "approve_dispatch_execution_kernel_launch_boundary"
    ) is True:
        return P50_READY_DECISION
    return P50_REJECTED_DECISION


def _evidence_blockers(
    p49_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p49_summary.get("present", False)):
        blocked.append("v5_p50_p49_runtime_dispatch_boundary_missing")
    elif not _p49_ready(p49_summary):
        blocked.append("v5_p50_p49_runtime_dispatch_boundary_not_ready")
        blocked.extend(_string_list(p49_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p49_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p49_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p50_dispatch_execution_kernel_launch_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p50_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p50_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p50_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p50_reviewed_at_missing")
    if review.get("requested_scope") != P50_SCOPE:
        blocked.append("v5_p50_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p50_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p50_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p49_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("runtime_dispatch_contract_boundary_ready")
        and summary.get("runtime_dispatch_evidence_recorded")
        and summary.get("runtime_dispatch_review_signed")
        and summary.get("decision") == P49_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P49_REQUIRED_FALSE_FIELDS)
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
    missing_sections: list[str],
    plan_rows: list[dict[str, Any]],
    precondition_rows: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p50_dispatch_execution_kernel_launch_not_ok")
    if evidence.get("explicit_dispatch_execution_kernel_launch_boundary_ready") is not True:
        blocked.append("v5_p50_dispatch_execution_kernel_launch_evidence_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "execution_only", "kernel_launch_boundary_only",
        "records_evidence_only", "manual_only", "internal_only",
        "requires_explicit_owner_approval", "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p50_dispatch_execution_kernel_launch_{field}_missing")
    if not _any_true(evidence, "requires_later_native_execution_readiness_contract", "requires_later_kernel_launch_implementation_readiness_contract", "requires_later_runtime_execution_contract"):
        blocked.append("v5_p50_dispatch_execution_kernel_launch_later_contract_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p50_dispatch_execution_kernel_launch_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p50_dispatch_execution_kernel_launch_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p50_dispatch_execution_kernel_launch_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p50_dispatch_execution_kernel_launch_source_missing")
    for section in missing_sections:
        blocked.append(f"v5_p50_dispatch_execution_kernel_launch_section_missing:{section}")
    blocked.extend(_plan_blockers(plan_rows))
    blocked.extend(_precondition_blockers(precondition_rows))
    blocked.extend(_unsafe_claims(evidence, "dispatch_execution_kernel_launch"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _plan_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p50_dispatch_execution_kernel_launch_plan_inventory_missing"]
    for index, row in enumerate(rows):
        plan_id = str(row.get("plan_id") or row.get("id") or f"plan_{index}")
        for field in (
            "runtime_dispatch_executed", "native_dispatch_started", "native_dispatch_enabled",
            "native_dispatch_executed", "kernel_launch_allowed", "kernel_launch_enabled",
            "kernel_launch_executed", "training_step_allowed", "training_step_executed",
            "request_adapter_enabled", "request_fields_emitted", "training_launch_allowed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p50_dispatch_execution_kernel_launch_claim:{plan_id}:{field}")
        blocked.extend(_unsafe_claims(row, plan_id))
    return _dedupe(blocked)


def _precondition_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p50_dispatch_execution_kernel_launch_precondition_inventory_missing"]
    for index, row in enumerate(rows):
        check_id = str(row.get("check_id") or row.get("id") or f"check_{index}")
        for field in (
            "dispatch_execution_precondition_active", "dispatch_execution_check_registered",
            "dispatch_execution_check_enabled", "kernel_launch_precondition_active",
            "kernel_launch_check_registered", "kernel_launch_check_enabled",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p50_dispatch_execution_kernel_launch_precondition_claim:{check_id}:{field}")
        blocked.extend(_unsafe_claims(row, check_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p50_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p50_unsafe_claim:{owner}:{field}")
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
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {str(item).strip() for item in sections if str(item).strip()}


def _any_true(value: Mapping[str, Any], *fields: str) -> bool:
    return any(value.get(field) is True for field in fields)


def _review_template() -> dict[str, Any]:
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P50_SCOPE, "approve_explicit_dispatch_execution_kernel_launch_boundary": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P50_READY_DECISION:
        return ["archive_p50_explicit_dispatch_execution_kernel_launch_boundary"]
    if decision == P50_REJECTED_DECISION:
        return ["record_p50_default_off_rejection_or_repair_dispatch_execution_kernel_launch_evidence"]
    if decision == P50_HOLD_DECISION:
        return ["collect_signed_dispatch_execution_kernel_launch_review"]
    if any("p49" in item for item in blockers):
        return ["repair_p49_runtime_dispatch_boundary"]
    if any("dispatch_execution" in item or "kernel_launch" in item or "precondition" in item for item in blockers):
        return ["repair_dispatch_execution_kernel_launch_boundary_evidence"]
    return ["clear_failure_or_rollback_history_before_p50_boundary"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P50_READY_DECISION:
        return "archive P50 boundary; native execution readiness still requires a later explicit contract"
    if decision == P50_REJECTED_DECISION:
        return "record the signed rejection and keep dispatch execution / kernel launch default-off for repair"
    if decision == P50_HOLD_DECISION:
        return "collect a signed dispatch execution / kernel launch review over P49 evidence"
    if any("p49" in item for item in blockers):
        return "repair the P49 runtime dispatch boundary before P50"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable dispatch execution / kernel launch source and digest evidence"
    return "hold P50 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P50 dispatch execution / kernel launch boundary.")
    parser.add_argument("--p49-runtime-dispatch-contract-boundary", default="", help="P49 boundary JSON.")
    parser.add_argument("--dispatch-execution-kernel-launch-evidence", default="", help="P50 evidence JSON.")
    parser.add_argument("--dispatch-execution-kernel-launch-review", default="", help="Signed P50 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    p49 = load_json(args.p49_runtime_dispatch_contract_boundary) if args.p49_runtime_dispatch_contract_boundary else None
    report = build_v5_explicit_dispatch_execution_kernel_launch_boundary(
        p49_runtime_dispatch_contract_boundary=p49,
        dispatch_execution_kernel_launch_evidence=load_json(args.dispatch_execution_kernel_launch_evidence) if args.dispatch_execution_kernel_launch_evidence else None,
        dispatch_execution_kernel_launch_review=load_json(args.dispatch_execution_kernel_launch_review) if args.dispatch_execution_kernel_launch_review else None,
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
