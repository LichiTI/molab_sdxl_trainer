"""Runtime wiring contract boundary for TurboCore V5-P43.

P43 records a signed boundary review for a future runtime wiring contract after
P42. It does not register runtime adapters, emit request fields, patch
GenerationRequest/config/runtime resolvers, expose UI, or launch training.
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


P42_READY_DECISION = "explicit_adapter_integration_contract_scaffold_recorded_default_off"
P43_READY_DECISION = "runtime_wiring_contract_boundary_recorded_default_off"
P43_BLOCKED_DECISION = "runtime_wiring_contract_boundary_blocked_default_off"
P43_HOLD_DECISION = "runtime_wiring_contract_boundary_hold_for_signed_review_default_off"
P43_REJECTED_DECISION = "runtime_wiring_contract_boundary_rejected_default_off"
P43_SCOPE = "runtime_wiring_contract_boundary"
DEFAULT_REQUIRED_SECTIONS = (
    "p42_scaffold_reference",
    "runtime_adapter_inventory",
    "runtime_wiring_plan_inventory",
    "generation_request_boundary",
    "config_adapter_boundary",
    "runtime_resolver_boundary",
    "execution_resolver_boundary",
    "no_runtime_adapter_registration_boundary",
    "no_request_fields_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p42_scaffold_recorded",
    "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch",
    "acknowledge_no_ui_exposure",
    "acknowledge_no_runtime_adapter_registration",
    "acknowledge_no_request_fields_emitted",
    "acknowledge_no_generation_request_patch",
    "acknowledge_no_config_adapter_patch",
    "acknowledge_no_runtime_resolver_patch",
    "acknowledge_no_execution_resolver_patch",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_runtime_wiring_evidence_replayable",
    "acknowledge_later_runtime_activation_contract_required",
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
    "adapter_integration_allowed",
    "adapter_wiring_allowed",
    "runtime_wiring_allowed",
    "runtime_adapter_wiring_allowed",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed",
    "training_manager_patch_allowed",
    "rollout_authorization_allowed",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p42_request_fields",
    "post_p43_request_fields",
    "post_runtime_wiring_request_fields",
    "request_adapter",
    "request_adapter_fields",
    "request_fields",
    "launch_request",
    "training_request",
    "generation_request_patch",
    "config_adapter_patch",
    "runtime_resolver_patch",
    "execution_resolver_patch",
    "request_schema_patch",
    "runtime_request_mapping",
    "adapter_registration",
    "runtime_adapter_registration",
    "backend_route_registration",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
    "entry_train_patch",
    "training_manager_patch",
    "auto_launch_plan",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed")


def build_v5_runtime_wiring_contract_boundary(
    *,
    p42_adapter_integration_contract_scaffold: Mapping[str, Any] | None = None,
    runtime_wiring_contract_evidence: Mapping[str, Any] | None = None,
    runtime_wiring_contract_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record runtime wiring contract evidence without wiring runtime adapters."""

    p42 = _as_dict(p42_adapter_integration_contract_scaffold)
    evidence = _as_dict(runtime_wiring_contract_evidence)
    review = _as_dict(runtime_wiring_contract_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p42_summary = _p42_summary(p42)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p42_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P43_HOLD_DECISION:
        blockers.append("v5_p43_signed_runtime_wiring_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P43_READY_DECISION
    rejected = decision == P43_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_runtime_wiring_contract_boundary_v0",
        "gate": "v5_runtime_wiring_contract_boundary",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "runtime_wiring_contract_review_recorded": decision_record_ready,
        "runtime_wiring_contract_review_signed": decision_record_ready,
        "runtime_wiring_contract_boundary_ready": ready,
        "runtime_wiring_contract_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p43_request_fields": {},
        "p42_adapter_integration_contract_scaffold_summary": p42_summary,
        "runtime_wiring_contract_evidence_summary": evidence_summary,
        "runtime_wiring_contract_review": review_summary,
        "runtime_wiring_contract_review_template": _review_template(evidence_summary),
        "progress_gates": _progress_gates(p42_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P43 records runtime wiring contract boundary evidence only.",
            "P43 does not register runtime adapters, emit request fields, patch request/config/runtime, or launch training.",
            "A later explicit runtime activation contract is still required before any wiring can become active.",
        ],
    }


def _p42_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "adapter_integration_contract_scaffold_ready": report.get("adapter_integration_contract_scaffold_ready") is True,
        "adapter_integration_contract_scaffold_recorded": report.get("adapter_integration_contract_scaffold_recorded")
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
        "request_adapter_registered": report.get("request_adapter_registered"),
        "runtime_adapter_registered": report.get("runtime_adapter_registered"),
        "generation_request_patch_allowed": report.get("generation_request_patch_allowed"),
        "config_adapter_patch_allowed": report.get("config_adapter_patch_allowed"),
        "runtime_resolver_patch_allowed": report.get("runtime_resolver_patch_allowed"),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p42_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p42"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    available_sections = _section_set(evidence)
    missing_sections = [item for item in required_sections if item not in available_sections]
    adapter_rows = _rows(evidence, "runtime_adapter_inventory", "adapter_inventory")
    wiring_rows = _rows(evidence, "runtime_wiring_plan_inventory", "wiring_plan_inventory")
    blockers = _evidence_blocker_list(evidence, required_sections, missing_sections, adapter_rows, wiring_rows)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "runtime_wiring_contract_boundary_ready": evidence.get("runtime_wiring_contract_boundary_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_runtime_activation_contract": evidence.get("requires_later_runtime_activation_contract") is True,
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
        "unsafe_claims": _unsafe_claims(evidence, "runtime_wiring"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_runtime_wiring_contract_boundary": review.get("approve_runtime_wiring_contract_boundary") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p42_summary: Mapping[str, Any], evidence_summary: Mapping[str, Any], review_summary: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p42_scaffold_present": bool(p42_summary.get("present", False)),
        "p42_scaffold_ready": _p42_ready(p42_summary),
        "runtime_wiring_evidence_present": bool(evidence_summary.get("present", False)),
        "runtime_wiring_evidence_ready": bool(evidence_summary.get("ready", False)),
        "signed_runtime_wiring_review_present": bool(review_summary.get("present", False)),
        "reviewer_present": bool(review_summary.get("reviewer")),
        "reviewed_at_present": bool(review_summary.get("reviewed_at")),
        "requested_scope_valid": review_summary.get("requested_scope") == P43_SCOPE,
        "required_acknowledgements_present": all(bool(review_summary.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P43_BLOCKED_DECISION
    if not review:
        return P43_HOLD_DECISION
    if review_blockers:
        return P43_BLOCKED_DECISION
    if review.get("approve_runtime_wiring_contract_boundary") is True:
        return P43_READY_DECISION
    return P43_REJECTED_DECISION


def _evidence_blockers(
    p42_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p42_summary.get("present", False)):
        blocked.append("v5_p43_p42_adapter_contract_scaffold_missing")
    elif not _p42_ready(p42_summary):
        blocked.append("v5_p43_p42_adapter_contract_scaffold_not_ready")
        blocked.extend(_string_list(p42_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p42_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p42_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p43_runtime_wiring_contract_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p43_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p43_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p43_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p43_reviewed_at_missing")
    if review.get("requested_scope") != P43_SCOPE:
        blocked.append("v5_p43_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p43_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p43_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p42_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("adapter_integration_contract_scaffold_ready")
        and summary.get("adapter_integration_contract_scaffold_recorded")
        and summary.get("decision") == P42_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_behavior_changed") is False
        and summary.get("training_launch_allowed") is False
        and summary.get("auto_launch_allowed") is False
        and summary.get("runs_dispatched") is False
        and summary.get("ui_exposure_allowed") is False
        and summary.get("request_adapter_mapping_allowed") is False
        and summary.get("request_fields_emitted") is False
        and summary.get("request_adapter_registered") is False
        and summary.get("runtime_adapter_registered") is False
        and summary.get("generation_request_patch_allowed") is False
        and summary.get("config_adapter_patch_allowed") is False
        and summary.get("runtime_resolver_patch_allowed") is False
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
    required_sections: list[str],
    missing_sections: list[str],
    adapter_rows: list[dict[str, Any]],
    wiring_rows: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p43_runtime_wiring_not_ok")
    if evidence.get("runtime_wiring_contract_boundary_ready") is not True:
        blocked.append("v5_p43_runtime_wiring_evidence_not_ready")
    for field in (
        "report_only",
        "boundary_only",
        "contract_only",
        "manual_only",
        "internal_only",
        "requires_later_runtime_activation_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p43_runtime_wiring_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p43_runtime_wiring_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p43_runtime_wiring_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p43_runtime_wiring_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p43_runtime_wiring_source_missing")
    if not required_sections:
        blocked.append("v5_p43_runtime_wiring_required_sections_empty")
    for section in missing_sections:
        blocked.append(f"v5_p43_runtime_wiring_section_missing:{section}")
    blocked.extend(_adapter_inventory_blockers(adapter_rows))
    blocked.extend(_wiring_plan_blockers(wiring_rows))
    blocked.extend(_unsafe_claims(evidence, "runtime_wiring"))
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


def _adapter_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p43_runtime_adapter_inventory_missing"]
    for index, row in enumerate(rows):
        adapter_id = str(row.get("adapter_id") or row.get("id") or f"adapter_{index}")
        for field in ("adapter_registered", "runtime_adapter_registered", "runtime_adapter_wiring_allowed"):
            if row.get(field) is True:
                blocked.append(f"v5_p43_runtime_adapter_claim:{adapter_id}:{field}")
        blocked.extend(_unsafe_claims(row, adapter_id))
    return _dedupe(blocked)


def _wiring_plan_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p43_runtime_wiring_plan_inventory_missing"]
    for index, row in enumerate(rows):
        plan_id = str(row.get("plan_id") or row.get("id") or f"plan_{index}")
        for field in (
            "wiring_enabled",
            "runtime_wiring_allowed",
            "request_fields_emitted",
            "generation_request_patch_applied",
            "config_adapter_patch_applied",
            "runtime_resolver_patch_applied",
            "execution_resolver_patch_applied",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p43_runtime_wiring_plan_claim:{plan_id}:{field}")
        blocked.extend(_unsafe_claims(row, plan_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p43_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p43_unsafe_claim:{owner}:{field}")
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
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P43_SCOPE, "approve_runtime_wiring_contract_boundary": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P43_READY_DECISION:
        return ["archive_p43_runtime_wiring_contract_boundary"]
    if decision == P43_REJECTED_DECISION:
        return ["record_p43_default_off_rejection_or_repair_runtime_wiring_evidence"]
    if decision == P43_HOLD_DECISION:
        return ["collect_signed_runtime_wiring_contract_review"]
    if any("p42" in item for item in blockers):
        return ["repair_p42_adapter_integration_contract_scaffold"]
    if any("runtime_wiring" in item or "runtime_adapter" in item for item in blockers):
        return ["repair_runtime_wiring_contract_evidence"]
    return ["clear_failure_or_rollback_history_before_runtime_wiring_boundary"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P43_READY_DECISION:
        return "archive P43 boundary; runtime activation still requires a later explicit contract"
    if decision == P43_REJECTED_DECISION:
        return "record the signed rejection and keep runtime wiring default-off for repair"
    if decision == P43_HOLD_DECISION:
        return "collect a signed runtime wiring contract review over P42 evidence"
    if any("p42" in item for item in blockers):
        return "repair the P42 adapter-integration scaffold before P43"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable runtime wiring source and digest evidence"
    return "hold P43 until runtime wiring evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P43 runtime wiring contract boundary.")
    parser.add_argument("--p42-adapter-integration-contract-scaffold", default="", help="P42 scaffold JSON.")
    parser.add_argument("--runtime-wiring-contract-evidence", default="", help="P43 runtime wiring evidence JSON.")
    parser.add_argument("--runtime-wiring-contract-review", default="", help="Signed P43 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_runtime_wiring_contract_boundary(
        p42_adapter_integration_contract_scaffold=load_json(args.p42_adapter_integration_contract_scaffold)
        if args.p42_adapter_integration_contract_scaffold
        else None,
        runtime_wiring_contract_evidence=load_json(args.runtime_wiring_contract_evidence)
        if args.runtime_wiring_contract_evidence
        else None,
        runtime_wiring_contract_review=load_json(args.runtime_wiring_contract_review)
        if args.runtime_wiring_contract_review
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
