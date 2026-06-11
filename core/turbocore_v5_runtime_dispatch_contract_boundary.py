"""Runtime dispatch contract boundary for TurboCore V5-P49.

P49 records future runtime dispatch boundary evidence after P48; it does not
enable runtime/native dispatch, launch kernels, execute training steps, emit
request fields, expose UI, or launch training.
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


P48_READY_DECISION = "runtime_enablement_execution_contract_boundary_recorded_default_off"
P49_READY_DECISION = "runtime_dispatch_contract_boundary_recorded_default_off"
P49_BLOCKED_DECISION = "runtime_dispatch_contract_boundary_blocked_default_off"
P49_HOLD_DECISION = "runtime_dispatch_contract_boundary_hold_for_signed_review_default_off"
P49_REJECTED_DECISION = "runtime_dispatch_contract_boundary_rejected_default_off"
P49_SCOPE = "runtime_dispatch_contract_boundary"

DEFAULT_REQUIRED_SECTIONS = (
    "p48_runtime_enablement_execution_boundary_reference", "runtime_dispatch_plan_inventory",
    "runtime_dispatch_precondition_inventory", "runtime_dispatch_boundary", "runtime_adapter_boundary",
    "native_dispatch_boundary", "kernel_launch_boundary", "training_step_boundary", "request_adapter_boundary",
    "no_runtime_dispatch_boundary", "no_native_dispatch_boundary", "no_kernel_launch_boundary", "no_training_step_boundary",
    "no_request_fields_boundary", "no_training_launch_boundary", "rollback_policy", "observability_policy",
)

REQUIRED_REVIEW_ACKS = (
    "acknowledge_p48_runtime_enablement_execution_boundary_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure", "acknowledge_no_runtime_dispatch_enabled",
    "acknowledge_no_native_dispatch_enabled", "acknowledge_no_kernel_launch_executed",
    "acknowledge_no_training_step_executed", "acknowledge_no_request_adapter_enabled",
    "acknowledge_no_request_fields_emitted", "acknowledge_no_default_or_auto_rollout",
    "acknowledge_runtime_dispatch_evidence_replayable", "acknowledge_later_runtime_dispatch_execution_contract_required",
    "acknowledge_manual_review_only",
)

UNSAFE_TRUE_FIELDS = (
    "training_launch_allowed",
    "auto_launch_allowed",
    "runs_dispatched",
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "ui_exposure_allowed",
    "product_ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "request_adapter_registered",
    "request_adapter_enabled",
    "runtime_adapter_registered",
    "runtime_adapter_enabled",
    "runtime_enablement_allowed",
    "runtime_enablement_enabled",
    "runtime_enablement_executed",
    "runtime_enablement_applied",
    "runtime_enablement_execution_allowed",
    "runtime_enablement_execution_enabled",
    "runtime_enablement_execution_executed",
    "runtime_enablement_execution_applied",
    "runtime_execution_allowed",
    "runtime_execution_executed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_executed",
    "runtime_dispatch_applied",
    "runtime_activation_allowed",
    "runtime_activation_enabled",
    "native_runtime_enabled",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_started",
    "native_dispatch_executed",
    "kernel_launch_allowed",
    "kernel_launch_enabled",
    "kernel_launch_executed",
    "training_step_allowed",
    "training_step_executed",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed",
    "training_manager_patch_allowed",
    "rollout_authorization_allowed",
    "default_behavior_changed",
)

UNSAFE_NON_EMPTY_FIELDS = (
    "post_p48_request_fields",
    "post_p49_request_fields",
    "post_runtime_dispatch_fields",
    "runtime_dispatch_request",
    "runtime_dispatch_payload",
    "runtime_execution_payload",
    "runtime_enablement_execution_request",
    "runtime_enablement_execution_payload",
    "runtime_enablement_request",
    "runtime_enablement_payload",
    "runtime_activation_request",
    "runtime_activation_env",
    "runtime_activation_flags",
    "active_runtime_adapter",
    "native_dispatch_request",
    "native_dispatch_payload",
    "kernel_launch_request",
    "kernel_launch_payload",
    "training_step_request",
    "training_step_payload",
    "request_adapter_fields",
    "request_fields",
    "api_route_registration",
    "backend_route_registration",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
    "launch_request",
    "training_request",
    "generation_request_patch",
    "config_adapter_patch",
    "runtime_resolver_patch",
    "execution_resolver_patch",
    "entry_train_patch",
    "training_manager_patch",
    "auto_launch_plan",
)

UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed")
P48_REQUIRED_FALSE_FIELDS = (
    "default_behavior_changed", "training_launch_allowed", "auto_launch_allowed", "runs_dispatched", "ui_exposure_allowed",
    "request_fields_emitted", "request_adapter_enabled", "runtime_enablement_execution_executed",
    "runtime_dispatch_enabled", "native_dispatch_enabled", "kernel_launch_executed", "training_step_executed",
)


def build_v5_runtime_dispatch_contract_boundary(
    *,
    p48_runtime_enablement_execution_contract_boundary: Mapping[str, Any] | None = None,
    runtime_dispatch_evidence: Mapping[str, Any] | None = None,
    runtime_dispatch_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record runtime dispatch boundary evidence without enabling dispatch."""

    p48 = _as_dict(p48_runtime_enablement_execution_contract_boundary)
    evidence = _as_dict(runtime_dispatch_evidence)
    review = _as_dict(runtime_dispatch_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)

    p48_summary = _p48_summary(p48)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p48_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P49_HOLD_DECISION:
        blockers.append("v5_p49_signed_runtime_dispatch_review_missing")
    blockers = _dedupe(blockers)

    ready = decision == P49_READY_DECISION
    rejected = decision == P49_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_runtime_dispatch_contract_boundary_v0",
        "gate": "v5_runtime_dispatch_contract_boundary",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "runtime_dispatch_review_recorded": decision_record_ready,
        "runtime_dispatch_review_signed": decision_record_ready,
        "runtime_dispatch_contract_boundary_ready": ready,
        "runtime_dispatch_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p49_request_fields": {},
        "p48_runtime_enablement_execution_boundary_summary": p48_summary,
        "runtime_dispatch_evidence_summary": evidence_summary,
        "runtime_dispatch_review": review_summary,
        "runtime_dispatch_review_template": _review_template(),
        "progress_gates": _progress_gates(p48_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P49 records runtime dispatch contract boundary evidence only.",
            "P49 does not enable runtime/native dispatch, launch kernels, execute training steps, emit request fields, or launch training.",
            "A later explicit runtime dispatch execution or kernel launch boundary contract is still required before runtime behavior can become active.",
        ],
    }


def _p48_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "runtime_enablement_execution_contract_boundary_ready": report.get(
            "runtime_enablement_execution_contract_boundary_ready"
        )
        is True,
        "runtime_enablement_execution_evidence_recorded": report.get(
            "runtime_enablement_execution_evidence_recorded"
        )
        is True,
        "runtime_enablement_execution_review_signed": report.get("runtime_enablement_execution_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_behavior_changed": report.get("default_behavior_changed"),
        "training_launch_allowed": report.get("training_launch_allowed"),
        "auto_launch_allowed": report.get("auto_launch_allowed"),
        "runs_dispatched": report.get("runs_dispatched"),
        "ui_exposure_allowed": report.get("ui_exposure_allowed"),
        "request_fields_emitted": report.get("request_fields_emitted"),
        "request_adapter_enabled": report.get("request_adapter_enabled"),
        "runtime_enablement_execution_executed": report.get("runtime_enablement_execution_executed"),
        "runtime_dispatch_enabled": report.get("runtime_dispatch_enabled"),
        "native_dispatch_enabled": report.get("native_dispatch_enabled"),
        "kernel_launch_executed": report.get("kernel_launch_executed"),
        "training_step_executed": report.get("training_step_executed"),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p48_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p48"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    plan_rows = _rows(evidence, "runtime_dispatch_plan_inventory", "dispatch_plan_inventory")
    precondition_rows = _rows(evidence, "runtime_dispatch_precondition_inventory", "dispatch_precondition_inventory")
    blockers = _evidence_blocker_list(evidence, missing_sections, plan_rows, precondition_rows)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "runtime_dispatch_contract_boundary_ready": evidence.get("runtime_dispatch_contract_boundary_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_runtime_dispatch_execution_contract": evidence.get(
            "requires_later_runtime_dispatch_execution_contract"
        )
        is True,
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
        "unsafe_claims": _unsafe_claims(evidence, "runtime_dispatch"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    approval = review.get("approve_runtime_dispatch_contract_boundary")
    if approval is None:
        approval = review.get("approve_runtime_dispatch_boundary")
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_runtime_dispatch_contract_boundary": approval is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p48: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p48_runtime_enablement_execution_boundary_present": bool(p48.get("present", False)),
        "p48_runtime_enablement_execution_boundary_ready": _p48_ready(p48),
        "runtime_dispatch_evidence_present": bool(evidence.get("present", False)),
        "runtime_dispatch_evidence_ready": bool(evidence.get("ready", False)),
        "signed_runtime_dispatch_review_present": bool(review.get("present", False)),
        "reviewer_present": bool(review.get("reviewer")),
        "reviewed_at_present": bool(review.get("reviewed_at")),
        "requested_scope_valid": review.get("requested_scope") == P49_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P49_BLOCKED_DECISION
    if not review:
        return P49_HOLD_DECISION
    if review_blockers:
        return P49_BLOCKED_DECISION
    if (
        review.get("approve_runtime_dispatch_contract_boundary") is True
        or review.get("approve_runtime_dispatch_boundary") is True
    ):
        return P49_READY_DECISION
    return P49_REJECTED_DECISION


def _evidence_blockers(
    p48_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p48_summary.get("present", False)):
        blocked.append("v5_p49_p48_runtime_enablement_execution_boundary_missing")
    elif not _p48_ready(p48_summary):
        blocked.append("v5_p49_p48_runtime_enablement_execution_boundary_not_ready")
        blocked.extend(_string_list(p48_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p48_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p48_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p49_runtime_dispatch_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p49_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p49_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p49_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p49_reviewed_at_missing")
    if review.get("requested_scope") != P49_SCOPE:
        blocked.append("v5_p49_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p49_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p49_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p48_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("runtime_enablement_execution_contract_boundary_ready")
        and summary.get("runtime_enablement_execution_evidence_recorded")
        and summary.get("runtime_enablement_execution_review_signed")
        and summary.get("decision") == P48_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P48_REQUIRED_FALSE_FIELDS)
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
        blocked.append("v5_p49_runtime_dispatch_not_ok")
    if evidence.get("runtime_dispatch_contract_boundary_ready") is not True:
        blocked.append("v5_p49_runtime_dispatch_evidence_not_ready")
    for field in (
        "report_only",
        "boundary_only",
        "contract_only",
        "records_evidence_only",
        "manual_only",
        "internal_only",
        "requires_later_runtime_dispatch_execution_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p49_runtime_dispatch_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p49_runtime_dispatch_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p49_runtime_dispatch_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p49_runtime_dispatch_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p49_runtime_dispatch_source_missing")
    for section in missing_sections:
        blocked.append(f"v5_p49_runtime_dispatch_section_missing:{section}")
    blocked.extend(_runtime_dispatch_plan_blockers(plan_rows))
    blocked.extend(_precondition_blockers(precondition_rows))
    blocked.extend(_unsafe_claims(evidence, "runtime_dispatch"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _runtime_dispatch_plan_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p49_runtime_dispatch_plan_inventory_missing"]
    for index, row in enumerate(rows):
        plan_id = str(row.get("plan_id") or row.get("id") or f"plan_{index}")
        for field in (
            "runtime_dispatch_allowed",
            "runtime_dispatch_enabled",
            "runtime_dispatch_executed",
            "native_dispatch_allowed",
            "native_dispatch_enabled",
            "native_dispatch_started",
            "kernel_launch_allowed",
            "kernel_launch_executed",
            "training_step_executed",
            "request_adapter_enabled",
            "request_fields_emitted",
            "training_launch_allowed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p49_runtime_dispatch_claim:{plan_id}:{field}")
        blocked.extend(_unsafe_claims(row, plan_id))
    return _dedupe(blocked)


def _precondition_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p49_runtime_dispatch_precondition_inventory_missing"]
    for index, row in enumerate(rows):
        check_id = str(row.get("check_id") or row.get("id") or f"check_{index}")
        for field in (
            "runtime_dispatch_precondition_active",
            "runtime_dispatch_check_registered",
            "runtime_dispatch_check_enabled",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p49_runtime_dispatch_precondition_claim:{check_id}:{field}")
        blocked.extend(_unsafe_claims(row, check_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p49_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p49_unsafe_claim:{owner}:{field}")
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


def _review_template() -> dict[str, Any]:
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P49_SCOPE, "approve_runtime_dispatch_contract_boundary": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P49_READY_DECISION:
        return ["archive_p49_runtime_dispatch_contract_boundary"]
    if decision == P49_REJECTED_DECISION:
        return ["record_p49_default_off_rejection_or_repair_runtime_dispatch_evidence"]
    if decision == P49_HOLD_DECISION:
        return ["collect_signed_runtime_dispatch_review"]
    if any("p48" in item for item in blockers):
        return ["repair_p48_runtime_enablement_execution_boundary"]
    if any("runtime_dispatch" in item or "precondition" in item for item in blockers):
        return ["repair_runtime_dispatch_boundary_evidence"]
    return ["clear_failure_or_rollback_history_before_runtime_dispatch_boundary"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P49_READY_DECISION:
        return "archive P49 boundary; runtime dispatch execution still requires a later explicit contract"
    if decision == P49_REJECTED_DECISION:
        return "record the signed rejection and keep runtime dispatch default-off for repair"
    if decision == P49_HOLD_DECISION:
        return "collect a signed runtime dispatch review over P48 evidence"
    if any("p48" in item for item in blockers):
        return "repair the P48 runtime enablement execution boundary before P49"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable runtime dispatch source and digest evidence"
    return "hold P49 until runtime dispatch evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P49 runtime dispatch boundary.")
    parser.add_argument("--p48-runtime-enablement-execution-contract-boundary", default="", help="P48 boundary JSON.")
    parser.add_argument("--runtime-dispatch-evidence", default="", help="P49 runtime dispatch evidence JSON.")
    parser.add_argument("--runtime-dispatch-review", default="", help="Signed P49 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    p48 = load_json(args.p48_runtime_enablement_execution_contract_boundary) if args.p48_runtime_enablement_execution_contract_boundary else None
    report = build_v5_runtime_dispatch_contract_boundary(
        p48_runtime_enablement_execution_contract_boundary=p48,
        runtime_dispatch_evidence=load_json(args.runtime_dispatch_evidence) if args.runtime_dispatch_evidence else None,
        runtime_dispatch_review=load_json(args.runtime_dispatch_review) if args.runtime_dispatch_review else None,
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
