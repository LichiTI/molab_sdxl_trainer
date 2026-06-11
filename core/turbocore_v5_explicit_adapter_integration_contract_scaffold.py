"""Explicit adapter-integration contract scaffold for TurboCore V5-P42.

P42 records a signed scaffold review for a future adapter-integration contract
after P41. It does not register adapters, emit request fields, patch
GenerationRequest/config adapters/runtime resolvers, expose UI, or launch
training.
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


P41_READY_DECISION = "request_adapter_mapping_boundary_evidence_recorded_default_off"
P42_READY_DECISION = "explicit_adapter_integration_contract_scaffold_recorded_default_off"
P42_BLOCKED_DECISION = "explicit_adapter_integration_contract_scaffold_blocked_default_off"
P42_HOLD_DECISION = "explicit_adapter_integration_contract_scaffold_hold_for_signed_review_default_off"
P42_REJECTED_DECISION = "explicit_adapter_integration_contract_scaffold_rejected_default_off"
P42_SCOPE = "explicit_adapter_integration_contract_scaffold"
DEFAULT_REQUIRED_SECTIONS = (
    "p41_mapping_boundary_reference",
    "adapter_contract_inventory",
    "future_request_field_inventory",
    "generation_request_contract_boundary",
    "config_adapter_contract_boundary",
    "runtime_resolver_contract_boundary",
    "no_adapter_registration_boundary",
    "no_request_fields_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p41_mapping_boundary_recorded",
    "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch",
    "acknowledge_no_ui_exposure",
    "acknowledge_no_request_adapter_registration",
    "acknowledge_no_request_fields_emitted",
    "acknowledge_no_generation_request_patch",
    "acknowledge_no_config_adapter_patch",
    "acknowledge_no_runtime_resolver_patch",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_scaffold_evidence_replayable",
    "acknowledge_later_runtime_wiring_contract_required",
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
    "ui_entry_enabled",
    "ready_for_ui",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "request_adapter_registered",
    "runtime_adapter_registered",
    "adapter_integration_allowed",
    "adapter_wiring_allowed",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "rollout_authorization_allowed",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p41_request_fields",
    "post_p42_request_fields",
    "post_scaffold_request_fields",
    "request_adapter",
    "request_adapter_fields",
    "request_fields",
    "launch_request",
    "training_request",
    "generation_request_patch",
    "config_adapter_patch",
    "runtime_resolver_patch",
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
UNSAFE_REVIEW_APPROVAL_FIELDS = (
    "approve_training_launch_allowed",
    "approve_auto_launch_allowed",
    "approve_runs_dispatched",
    "approve_default_training_path_enabled",
    "approve_training_path_enabled",
    "approve_default_rollout_allowed",
    "approve_auto_rollout_allowed",
    "approve_ui_exposure_allowed",
    "approve_product_ui_exposure_allowed",
    "approve_request_adapter_mapping_allowed",
    "approve_request_fields_emitted",
    "approve_request_adapter_registered",
    "approve_runtime_adapter_registered",
    "approve_adapter_integration_allowed",
    "approve_adapter_wiring_allowed",
    "approve_generation_request_patch_allowed",
    "approve_config_adapter_patch_allowed",
    "approve_runtime_resolver_patch_allowed",
    "approve_rollout_authorization_allowed",
)


def build_v5_explicit_adapter_integration_contract_scaffold(
    *,
    p41_request_adapter_mapping_boundary: Mapping[str, Any] | None = None,
    adapter_integration_contract_scaffold: Mapping[str, Any] | None = None,
    adapter_integration_contract_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record an adapter-integration contract scaffold without wiring it."""

    p41 = _as_dict(p41_request_adapter_mapping_boundary)
    scaffold = _as_dict(adapter_integration_contract_scaffold)
    review = _as_dict(adapter_integration_contract_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p41_summary = _p41_summary(p41)
    scaffold_summary = _scaffold_summary(scaffold)
    review_summary = _review_summary(review)
    progress = _progress_gates(p41_summary, scaffold_summary, review_summary)
    evidence_blockers = _evidence_blockers(
        p41_summary=p41_summary,
        scaffold_summary=scaffold_summary,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P42_HOLD_DECISION:
        blockers.append("v5_p42_signed_adapter_integration_contract_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P42_READY_DECISION
    rejected = decision == P42_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_explicit_adapter_integration_contract_scaffold_v0",
        "gate": "v5_explicit_adapter_integration_contract_scaffold",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "adapter_integration_contract_review_recorded": decision_record_ready,
        "adapter_integration_contract_review_signed": decision_record_ready,
        "adapter_integration_contract_scaffold_ready": ready,
        "adapter_integration_contract_scaffold_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p42_request_fields": {},
        "p41_request_adapter_mapping_boundary_summary": p41_summary,
        "adapter_integration_contract_scaffold_summary": scaffold_summary,
        "adapter_integration_contract_review": review_summary,
        "adapter_integration_contract_review_template": _review_template(scaffold_summary),
        "progress_gates": progress,
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P42 records explicit adapter-integration contract scaffold evidence only.",
            "P42 does not register adapters, emit request fields, patch request/config/runtime, or launch training.",
            "A later explicit runtime wiring contract is still required before any adapter integration can be wired.",
        ],
    }


def _p41_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "request_adapter_mapping_boundary_ready": report.get("request_adapter_mapping_boundary_ready") is True,
        "request_adapter_mapping_evidence_recorded": report.get("request_adapter_mapping_evidence_recorded") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_behavior_changed": report.get("default_behavior_changed"),
        "training_launch_allowed": report.get("training_launch_allowed"),
        "auto_launch_allowed": report.get("auto_launch_allowed"),
        "runs_dispatched": report.get("runs_dispatched"),
        "ui_exposure_allowed": report.get("ui_exposure_allowed"),
        "product_ui_exposure_allowed": report.get("product_ui_exposure_allowed"),
        "request_adapter_mapping_allowed": report.get("request_adapter_mapping_allowed"),
        "request_fields_emitted": report.get("request_fields_emitted"),
        "request_adapter_registered": report.get("request_adapter_registered"),
        "generation_request_patch_allowed": report.get("generation_request_patch_allowed"),
        "config_adapter_patch_allowed": report.get("config_adapter_patch_allowed"),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p41_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p41"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _scaffold_summary(scaffold: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(scaffold.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    available_sections = _section_set(scaffold)
    missing_sections = [item for item in required_sections if item not in available_sections]
    contract_rows = _contract_rows(scaffold)
    field_rows = _future_field_rows(scaffold)
    blockers = _scaffold_blockers(scaffold, required_sections, missing_sections)
    return {
        "present": bool(scaffold),
        "evidence_id": str(scaffold.get("evidence_id") or scaffold.get("id") or ""),
        "evidence_version": str(scaffold.get("evidence_version") or scaffold.get("version") or ""),
        "ok": scaffold.get("ok") is True,
        "ready": not blockers,
        "adapter_integration_contract_scaffold_ready": scaffold.get(
            "adapter_integration_contract_scaffold_ready"
        )
        is True,
        "report_only": scaffold.get("report_only") is True,
        "boundary_only": scaffold.get("boundary_only") is True,
        "scaffold_only": scaffold.get("scaffold_only") is True,
        "manual_only": scaffold.get("manual_only") is True,
        "internal_only": scaffold.get("internal_only") is True,
        "requires_later_runtime_wiring_contract": scaffold.get("requires_later_runtime_wiring_contract") is True,
        "requires_explicit_owner_approval": scaffold.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": scaffold.get("requires_explicit_operator_opt_in") is True,
        "default_off": scaffold.get("default_off") is True and _default_off_confirmed(scaffold),
        "request_adapter_off": scaffold.get("request_adapter_off") is True and _request_adapter_off(scaffold),
        "digest": _digest(scaffold),
        "source": _source(scaffold),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "blocked_reasons": _string_list(scaffold.get("blocked_reasons")),
        "promotion_blockers": _string_list(scaffold.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(scaffold, "scaffold"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_adapter_integration_contract_scaffold": review.get(
            "approve_adapter_integration_contract_scaffold"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(
    p41_summary: Mapping[str, Any],
    scaffold_summary: Mapping[str, Any],
    review_summary: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "p41_mapping_boundary_present": bool(p41_summary.get("present", False)),
        "p41_mapping_boundary_ready": _p41_ready(p41_summary),
        "adapter_integration_scaffold_present": bool(scaffold_summary.get("present", False)),
        "adapter_integration_scaffold_ready": bool(scaffold_summary.get("ready", False)),
        "signed_adapter_integration_review_present": bool(review_summary.get("present", False)),
        "reviewer_present": bool(review_summary.get("reviewer")),
        "reviewed_at_present": bool(review_summary.get("reviewed_at")),
        "requested_scope_valid": review_summary.get("requested_scope") == P42_SCOPE,
        "required_acknowledgements_present": all(bool(review_summary.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P42_BLOCKED_DECISION
    if not review:
        return P42_HOLD_DECISION
    if review_blockers:
        return P42_BLOCKED_DECISION
    if review.get("approve_adapter_integration_contract_scaffold") is True:
        return P42_READY_DECISION
    return P42_REJECTED_DECISION


def _evidence_blockers(
    *,
    p41_summary: Mapping[str, Any],
    scaffold_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p41_summary.get("present", False)):
        blocked.append("v5_p42_p41_request_adapter_mapping_boundary_missing")
    elif not _p41_ready(p41_summary):
        blocked.append("v5_p42_p41_request_adapter_mapping_boundary_not_ready")
        blocked.extend(_string_list(p41_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p41_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p41_summary.get("unsafe_claims")))
    if not bool(scaffold_summary.get("present", False)):
        blocked.append("v5_p42_adapter_integration_contract_scaffold_missing")
    elif not bool(scaffold_summary.get("ready", False)):
        blocked.extend(_string_list(scaffold_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p42_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p42_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p42_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p42_reviewed_at_missing")
    if review.get("requested_scope") != P42_SCOPE:
        blocked.append("v5_p42_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p42_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p42_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p41_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("request_adapter_mapping_boundary_ready")
        and summary.get("request_adapter_mapping_evidence_recorded")
        and summary.get("decision") == P41_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_behavior_changed") is False
        and summary.get("training_launch_allowed") is False
        and summary.get("auto_launch_allowed") is False
        and summary.get("runs_dispatched") is False
        and summary.get("ui_exposure_allowed") is False
        and summary.get("product_ui_exposure_allowed") is False
        and summary.get("request_adapter_mapping_allowed") is False
        and summary.get("request_fields_emitted") is False
        and summary.get("request_adapter_registered") is False
        and summary.get("generation_request_patch_allowed") is False
        and summary.get("config_adapter_patch_allowed") is False
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
        and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers"))
        and not _string_list(summary.get("unsafe_claims"))
    )


def _scaffold_blockers(scaffold: Mapping[str, Any], required_sections: list[str], missing_sections: list[str]) -> list[str]:
    blocked: list[str] = []
    if scaffold.get("ok") is not True:
        blocked.append("v5_p42_scaffold_not_ok")
    if scaffold.get("adapter_integration_contract_scaffold_ready") is not True:
        blocked.append("v5_p42_scaffold_evidence_not_ready")
    for field in (
        "report_only",
        "boundary_only",
        "scaffold_only",
        "manual_only",
        "internal_only",
        "requires_later_runtime_wiring_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if scaffold.get(field) is not True:
            blocked.append(f"v5_p42_scaffold_{field}_missing")
    if scaffold.get("default_off") is not True or not _default_off_confirmed(scaffold):
        blocked.append("v5_p42_scaffold_default_off_violation")
    if scaffold.get("request_adapter_off") is not True or not _request_adapter_off(scaffold):
        blocked.append("v5_p42_scaffold_request_adapter_violation")
    if not _digest(scaffold):
        blocked.append("v5_p42_scaffold_digest_missing")
    if not _source(scaffold):
        blocked.append("v5_p42_scaffold_source_missing")
    if not required_sections:
        blocked.append("v5_p42_scaffold_required_sections_empty")
    for section in missing_sections:
        blocked.append(f"v5_p42_scaffold_section_missing:{section}")
    blocked.extend(_contract_inventory_blockers(_contract_rows(scaffold)))
    blocked.extend(_future_field_inventory_blockers(_future_field_rows(scaffold)))
    blocked.extend(_unsafe_claims(scaffold, "scaffold"))
    blocked.extend(_string_list(scaffold.get("blocked_reasons")))
    blocked.extend(_string_list(scaffold.get("promotion_blockers")))
    return _dedupe(blocked)


def _contract_rows(scaffold: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = scaffold.get("adapter_contract_inventory", scaffold.get("contract_inventory"))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _future_field_rows(scaffold: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = scaffold.get("future_request_field_inventory", scaffold.get("future_field_inventory"))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _contract_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p42_adapter_contract_inventory_missing"]
    for index, row in enumerate(rows):
        contract_id = str(row.get("contract_id") or row.get("mapping_id") or row.get("id") or f"contract_{index}")
        if row.get("scaffold_enabled") is not False:
            blocked.append(f"v5_p42_contract_scaffold_not_disabled:{contract_id}")
        if row.get("contract_enabled") is True:
            blocked.append(f"v5_p42_contract_enabled:{contract_id}")
        for field in (
            "adapter_registered",
            "request_fields_emitted",
            "generation_request_patch_applied",
            "config_adapter_patch_applied",
            "runtime_resolver_patch_applied",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p42_contract_wiring_claim:{contract_id}:{field}")
        blocked.extend(_unsafe_claims(row, contract_id))
    return _dedupe(blocked)


def _future_field_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p42_future_request_field_inventory_missing"]
    for index, row in enumerate(rows):
        field_id = str(row.get("field_id") or row.get("name") or row.get("id") or f"field_{index}")
        for field in ("field_enabled", "field_emitted", "default_value_materialized"):
            if row.get(field) is True:
                blocked.append(f"v5_p42_future_field_claim:{field_id}:{field}")
        blocked.extend(_unsafe_claims(row, field_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p42_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p42_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {str(item).strip() for item in sections if str(item).strip()}


def _review_template(scaffold_summary: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": P42_SCOPE,
        "approve_adapter_integration_contract_scaffold": False,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P42_READY_DECISION:
        return ["archive_p42_adapter_integration_contract_scaffold"]
    if decision == P42_REJECTED_DECISION:
        return ["record_p42_default_off_rejection_or_repair_scaffold_evidence"]
    if decision == P42_HOLD_DECISION:
        return ["collect_signed_adapter_integration_contract_scaffold_review"]
    if any("p41" in item for item in blockers):
        return ["repair_p41_request_adapter_mapping_boundary"]
    if any("scaffold" in item or "contract" in item for item in blockers):
        return ["repair_adapter_integration_contract_scaffold_evidence"]
    return ["clear_failure_or_rollback_history_before_adapter_contract_scaffold"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P42_READY_DECISION:
        return "archive P42 scaffold; runtime adapter wiring still requires a later explicit contract"
    if decision == P42_REJECTED_DECISION:
        return "record the signed rejection and keep adapter integration default-off for repair"
    if decision == P42_HOLD_DECISION:
        return "collect a signed adapter-integration scaffold review over P41 evidence"
    if any("p41" in item for item in blockers):
        return "repair the P41 request-adapter mapping boundary before P42"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable adapter-integration scaffold source and digest evidence"
    return "hold P42 until scaffold evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P42 explicit adapter-integration scaffold.")
    parser.add_argument("--p41-request-adapter-mapping-boundary", default="", help="P41 boundary JSON.")
    parser.add_argument("--adapter-integration-contract-scaffold", default="", help="P42 scaffold evidence JSON.")
    parser.add_argument("--adapter-integration-contract-review", default="", help="Signed P42 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_explicit_adapter_integration_contract_scaffold(
        p41_request_adapter_mapping_boundary=load_json(args.p41_request_adapter_mapping_boundary)
        if args.p41_request_adapter_mapping_boundary
        else None,
        adapter_integration_contract_scaffold=load_json(args.adapter_integration_contract_scaffold)
        if args.adapter_integration_contract_scaffold
        else None,
        adapter_integration_contract_review=load_json(args.adapter_integration_contract_review)
        if args.adapter_integration_contract_review
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
