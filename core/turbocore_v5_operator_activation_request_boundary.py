"""Operator activation request boundary for TurboCore V5-P45.

P45 records a signed boundary review for a future operator activation request
after P44. It does not submit activation requests, enable runtime/native
dispatch, emit request fields, expose UI, or launch training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
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


P44_READY_DECISION = "runtime_activation_contract_boundary_recorded_default_off"
P45_READY_DECISION = "operator_activation_request_boundary_recorded_default_off"
P45_BLOCKED_DECISION = "operator_activation_request_boundary_blocked_default_off"
P45_HOLD_DECISION = "operator_activation_request_boundary_hold_for_signed_review_default_off"
P45_REJECTED_DECISION = "operator_activation_request_boundary_rejected_default_off"
P45_SCOPE = "operator_activation_request_boundary"
DEFAULT_REQUIRED_SECTIONS = (
    "p44_runtime_activation_boundary_reference",
    "operator_activation_request_inventory",
    "operator_identity_boundary",
    "activation_scope_boundary",
    "no_operator_request_submission_boundary",
    "no_runtime_activation_boundary",
    "no_runtime_adapter_enabled_boundary",
    "no_request_fields_boundary",
    "no_training_launch_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p44_runtime_activation_boundary_recorded",
    "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch",
    "acknowledge_no_ui_exposure",
    "acknowledge_no_operator_activation_request_submitted",
    "acknowledge_no_runtime_activation",
    "acknowledge_no_runtime_adapter_enabled",
    "acknowledge_no_request_fields_emitted",
    "acknowledge_no_request_config_runtime_patch",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_operator_request_evidence_replayable",
    "acknowledge_later_operator_activation_execution_contract_required",
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
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "request_adapter_registered",
    "runtime_adapter_registered",
    "operator_activation_request_allowed",
    "operator_activation_request_submitted",
    "activation_request_submitted",
    "operator_activation_requested",
    "runtime_activation_allowed",
    "runtime_activation_enabled",
    "runtime_adapter_enabled",
    "native_runtime_enabled",
    "native_dispatch_enabled",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed",
    "training_manager_patch_allowed",
    "rollout_authorization_allowed",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p44_request_fields",
    "post_p45_request_fields",
    "post_operator_activation_request_fields",
    "operator_activation_request",
    "activation_request_payload",
    "operator_request_payload",
    "request_adapter",
    "request_adapter_fields",
    "request_fields",
    "launch_request",
    "training_request",
    "generation_request_patch",
    "config_adapter_patch",
    "runtime_resolver_patch",
    "execution_resolver_patch",
    "runtime_activation_request",
    "runtime_activation_env",
    "runtime_activation_flags",
    "active_runtime_adapter",
    "backend_route_registration",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
    "entry_train_patch",
    "training_manager_patch",
    "auto_launch_plan",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed")


def build_v5_operator_activation_request_boundary(
    *,
    p44_runtime_activation_contract_boundary: Mapping[str, Any] | None = None,
    operator_activation_request_evidence: Mapping[str, Any] | None = None,
    operator_activation_request_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record operator activation request evidence without submitting a request."""

    p44 = _as_dict(p44_runtime_activation_contract_boundary)
    evidence = _as_dict(operator_activation_request_evidence)
    review = _as_dict(operator_activation_request_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p44_summary = _p44_summary(p44)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p44_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P45_HOLD_DECISION:
        blockers.append("v5_p45_signed_operator_activation_request_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P45_READY_DECISION
    rejected = decision == P45_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_operator_activation_request_boundary_v0",
        "gate": "v5_operator_activation_request_boundary",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "operator_activation_request_review_recorded": decision_record_ready,
        "operator_activation_request_review_signed": decision_record_ready,
        "operator_activation_request_boundary_ready": ready,
        "operator_activation_request_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p45_request_fields": {},
        "p44_runtime_activation_contract_boundary_summary": p44_summary,
        "operator_activation_request_evidence_summary": evidence_summary,
        "operator_activation_request_review": review_summary,
        "operator_activation_request_review_template": _review_template(evidence_summary),
        "progress_gates": _progress_gates(p44_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P45 records operator activation request boundary evidence only.",
            "P45 does not submit activation requests, enable runtime/native dispatch, emit request fields, or launch training.",
            "A later explicit operator activation execution contract is still required before any request can become active.",
        ],
    }


def _p44_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "runtime_activation_contract_boundary_ready": report.get("runtime_activation_contract_boundary_ready") is True,
        "runtime_activation_contract_evidence_recorded": report.get("runtime_activation_contract_evidence_recorded")
        is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_behavior_changed": report.get("default_behavior_changed"),
        "training_launch_allowed": report.get("training_launch_allowed"),
        "auto_launch_allowed": report.get("auto_launch_allowed"),
        "runs_dispatched": report.get("runs_dispatched"),
        "ui_exposure_allowed": report.get("ui_exposure_allowed"),
        "request_adapter_mapping_allowed": report.get("request_adapter_mapping_allowed"),
        "request_fields_emitted": report.get("request_fields_emitted"),
        "runtime_activation_enabled": report.get("runtime_activation_enabled"),
        "runtime_adapter_enabled": report.get("runtime_adapter_enabled"),
        "native_runtime_enabled": report.get("native_runtime_enabled"),
        "native_dispatch_enabled": report.get("native_dispatch_enabled"),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p44_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p44"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    available_sections = _section_set(evidence)
    missing_sections = [item for item in required_sections if item not in available_sections]
    request_rows = _rows(evidence, "operator_activation_request_inventory", "activation_request_inventory")
    scope_rows = _rows(evidence, "activation_scope_inventory", "operator_scope_inventory")
    blockers = _evidence_blocker_list(evidence, missing_sections, request_rows, scope_rows)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "operator_activation_request_boundary_ready": evidence.get("operator_activation_request_boundary_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "request_only": evidence.get("request_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_operator_activation_execution_contract": evidence.get(
            "requires_later_operator_activation_execution_contract"
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
        "unsafe_claims": _unsafe_claims(evidence, "operator_activation_request"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_operator_activation_request_boundary": review.get("approve_operator_activation_request_boundary")
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p44_summary: Mapping[str, Any], evidence_summary: Mapping[str, Any], review_summary: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p44_runtime_activation_boundary_present": bool(p44_summary.get("present", False)),
        "p44_runtime_activation_boundary_ready": _p44_ready(p44_summary),
        "operator_activation_request_evidence_present": bool(evidence_summary.get("present", False)),
        "operator_activation_request_evidence_ready": bool(evidence_summary.get("ready", False)),
        "signed_operator_activation_request_review_present": bool(review_summary.get("present", False)),
        "reviewer_present": bool(review_summary.get("reviewer")),
        "reviewed_at_present": bool(review_summary.get("reviewed_at")),
        "requested_scope_valid": review_summary.get("requested_scope") == P45_SCOPE,
        "required_acknowledgements_present": all(bool(review_summary.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P45_BLOCKED_DECISION
    if not review:
        return P45_HOLD_DECISION
    if review_blockers:
        return P45_BLOCKED_DECISION
    if review.get("approve_operator_activation_request_boundary") is True:
        return P45_READY_DECISION
    return P45_REJECTED_DECISION


def _evidence_blockers(
    p44_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p44_summary.get("present", False)):
        blocked.append("v5_p45_p44_runtime_activation_boundary_missing")
    elif not _p44_ready(p44_summary):
        blocked.append("v5_p45_p44_runtime_activation_boundary_not_ready")
        blocked.extend(_string_list(p44_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p44_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p44_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p45_operator_activation_request_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p45_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p45_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p45_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p45_reviewed_at_missing")
    if review.get("requested_scope") != P45_SCOPE:
        blocked.append("v5_p45_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p45_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p45_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p44_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("runtime_activation_contract_boundary_ready")
        and summary.get("runtime_activation_contract_evidence_recorded")
        and summary.get("decision") == P44_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_behavior_changed") is False
        and summary.get("training_launch_allowed") is False
        and summary.get("auto_launch_allowed") is False
        and summary.get("runs_dispatched") is False
        and summary.get("ui_exposure_allowed") is False
        and summary.get("request_adapter_mapping_allowed") is False
        and summary.get("request_fields_emitted") is False
        and summary.get("runtime_activation_enabled") is False
        and summary.get("runtime_adapter_enabled") is False
        and summary.get("native_runtime_enabled") is False
        and summary.get("native_dispatch_enabled") is False
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
    request_rows: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p45_operator_request_not_ok")
    if evidence.get("operator_activation_request_boundary_ready") is not True:
        blocked.append("v5_p45_operator_request_evidence_not_ready")
    for field in (
        "report_only",
        "boundary_only",
        "request_only",
        "manual_only",
        "internal_only",
        "requires_later_operator_activation_execution_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p45_operator_request_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p45_operator_request_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p45_operator_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p45_operator_request_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p45_operator_request_source_missing")
    for section in missing_sections:
        blocked.append(f"v5_p45_operator_request_section_missing:{section}")
    blocked.extend(_request_inventory_blockers(request_rows))
    blocked.extend(_scope_inventory_blockers(scope_rows))
    blocked.extend(_unsafe_claims(evidence, "operator_activation_request"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _rows(payload: Mapping[str, Any], field: str, fallback: str) -> list[dict[str, Any]]:
    value = payload.get(field, payload.get(fallback))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _request_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p45_operator_activation_request_inventory_missing"]
    for index, row in enumerate(rows):
        request_id = str(row.get("request_id") or row.get("id") or f"request_{index}")
        for field in (
            "operator_activation_request_submitted",
            "activation_request_submitted",
            "operator_activation_requested",
            "operator_activation_request_allowed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p45_operator_request_claim:{request_id}:{field}")
        blocked.extend(_unsafe_claims(row, request_id))
    return _dedupe(blocked)


def _scope_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p45_activation_scope_inventory_missing"]
    for index, row in enumerate(rows):
        scope_id = str(row.get("scope_id") or row.get("id") or f"scope_{index}")
        for field in ("activation_scope_enabled", "runtime_activation_enabled", "runtime_adapter_enabled", "native_dispatch_enabled"):
            if row.get(field) is True:
                blocked.append(f"v5_p45_activation_scope_claim:{scope_id}:{field}")
        blocked.extend(_unsafe_claims(row, scope_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p45_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p45_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {str(item).strip() for item in sections if str(item).strip()}


def _review_template(evidence_summary: Mapping[str, Any]) -> dict[str, Any]:
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P45_SCOPE, "approve_operator_activation_request_boundary": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P45_READY_DECISION:
        return ["archive_p45_operator_activation_request_boundary"]
    if decision == P45_REJECTED_DECISION:
        return ["record_p45_default_off_rejection_or_repair_operator_request_evidence"]
    if decision == P45_HOLD_DECISION:
        return ["collect_signed_operator_activation_request_review"]
    if any("p44" in item for item in blockers):
        return ["repair_p44_runtime_activation_contract_boundary"]
    if any("operator_request" in item or "activation_scope" in item for item in blockers):
        return ["repair_operator_activation_request_evidence"]
    return ["clear_failure_or_rollback_history_before_operator_activation_request_boundary"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P45_READY_DECISION:
        return "archive P45 boundary; operator activation execution still requires a later explicit contract"
    if decision == P45_REJECTED_DECISION:
        return "record the signed rejection and keep operator activation request default-off for repair"
    if decision == P45_HOLD_DECISION:
        return "collect a signed operator activation request review over P44 evidence"
    if any("p44" in item for item in blockers):
        return "repair the P44 runtime activation boundary before P45"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable operator activation request source and digest evidence"
    return "hold P45 until operator request evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P45 operator activation request boundary.")
    parser.add_argument("--p44-runtime-activation-contract-boundary", default="", help="P44 boundary JSON.")
    parser.add_argument("--operator-activation-request-evidence", default="", help="P45 request evidence JSON.")
    parser.add_argument("--operator-activation-request-review", default="", help="Signed P45 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_operator_activation_request_boundary(
        p44_runtime_activation_contract_boundary=load_json(args.p44_runtime_activation_contract_boundary)
        if args.p44_runtime_activation_contract_boundary
        else None,
        operator_activation_request_evidence=load_json(args.operator_activation_request_evidence)
        if args.operator_activation_request_evidence
        else None,
        operator_activation_request_review=load_json(args.operator_activation_request_review)
        if args.operator_activation_request_review
        else None,
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
