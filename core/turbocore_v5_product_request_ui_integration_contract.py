"""Product/request/UI integration contract for TurboCore V5-P67.

P67 records future product/request/UI integration evidence after P66. It checks
surface inventory, request schema, UI exposure, product wiring, rollback, and
observability evidence only; it does not register UI routes, emit request
fields, enable request adapters, or launch training.
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
from core.turbocore_v5_rollout_default_enable_contract import (
    P66_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P66_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P66_UNSAFE_TRUE_FIELDS,
)


P67_READY_DECISION = "product_request_ui_integration_contract_recorded_default_off"
P67_BLOCKED_DECISION = "product_request_ui_integration_contract_blocked_default_off"
P67_HOLD_DECISION = "product_request_ui_integration_contract_hold_for_signed_review_default_off"
P67_REJECTED_DECISION = "product_request_ui_integration_contract_rejected_default_off"
P67_SCOPE = "product_request_ui_integration_contract"
DEFAULT_REQUIRED_SECTIONS = (
    "p66_rollout_default_enable_contract_reference", "product_surface_inventory",
    "request_schema_inventory", "request_adapter_mapping_boundary", "ui_exposure_boundary",
    "product_wiring_boundary", "launcher_boundary", "webui_boundary", "backend_router_boundary",
    "training_request_boundary", "default_behavior_boundary", "operator_visibility_boundary",
    "monitoring_observability_boundary", "rollback_policy_boundary", "docs_release_boundary",
    "no_product_integration_enable_boundary", "no_ui_route_registration_boundary",
    "no_request_adapter_enable_boundary", "no_request_fields_emit_boundary",
    "no_default_enable_boundary", "no_default_rollout_boundary", "no_auto_rollout_boundary",
    "no_training_launch_boundary", "no_training_job_creation_boundary", "no_run_dispatch_boundary",
    "no_training_step_execution_boundary", "no_parity_execution_boundary",
    "no_tensor_transfer_execution_boundary", "no_kernel_launch_execution_boundary",
    "no_native_dispatch_execution_boundary", "no_runtime_dispatch_execution_boundary",
    "no_runtime_execution_boundary", "no_runtime_state_refresh_boundary", "no_runtime_adapter_enabled_boundary",
    "no_execution_replay_boundary", "no_artifact_load_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p66_rollout_default_enable_contract_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_product_integration_enabled", "acknowledge_no_ui_route_registered",
    "acknowledge_no_ui_exposure", "acknowledge_no_request_adapter_enabled",
    "acknowledge_no_request_fields_emitted", "acknowledge_no_default_enable",
    "acknowledge_no_default_rollout", "acknowledge_no_auto_rollout",
    "acknowledge_no_training_launch", "acknowledge_no_training_job_created",
    "acknowledge_no_runs_dispatched", "acknowledge_no_training_step_executed",
    "acknowledge_no_parity_executed", "acknowledge_no_tensor_transfer_executed",
    "acknowledge_no_kernel_launch_executed", "acknowledge_no_native_dispatch_executed",
    "acknowledge_no_runtime_dispatch_executed", "acknowledge_no_runtime_execution_executed",
    "acknowledge_no_runtime_state_refreshed", "acknowledge_no_runtime_adapter_enabled",
    "acknowledge_no_execution_replay_executed", "acknowledge_no_artifact_loaded",
    "acknowledge_product_request_ui_integration_evidence_replayable",
    "acknowledge_later_request_adapter_activation_contract_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P66_UNSAFE_TRUE_FIELDS,
            "product_request_ui_integration_approved", "product_request_ui_integration_allowed",
            "product_request_ui_integration_enabled", "product_request_ui_integration_executed",
            "product_integration_enabled", "product_wiring_enabled", "ui_route_registered",
            "ui_entry_registered", "ui_exposure_allowed", "launcher_integration_enabled",
            "webui_integration_enabled", "backend_router_registered", "request_schema_registered",
            "request_adapter_enabled", "request_adapter_mapping_allowed", "request_fields_emitted",
            "generation_request_fields_added", "training_request_fields_added", "default_behavior_changed",
            "default_enable_enabled", "training_path_enabled", "default_rollout_allowed",
            "auto_rollout_allowed", "training_launch_allowed", "training_job_created",
            "runs_dispatched", "training_run_dispatched",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P66_UNSAFE_NON_EMPTY_FIELDS,
            "post_p67_request_fields", "post_product_request_ui_integration_fields",
            "product_request_ui_integration_contract_request",
            "product_request_ui_integration_contract_payload", "product_integration_payload",
            "product_wiring_payload", "ui_route_registration", "ui_route_payload",
            "ui_exposure_payload", "launcher_payload", "webui_payload", "backend_router_payload",
            "request_schema_payload", "request_adapter_payload", "request_fields",
            "generation_request_payload", "training_request_payload", "default_enable_payload",
            "training_launch_payload", "training_job_payload", "training_run_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)
P66_REQUIRED_FALSE_FIELDS = _P66_UNSAFE_TRUE_FIELDS


def build_v5_product_request_ui_integration_contract(
    *,
    p66_rollout_default_enable_contract: Mapping[str, Any] | None = None,
    product_request_ui_evidence: Mapping[str, Any] | None = None,
    product_request_ui_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record product/request/UI integration evidence without integrating it."""

    p66 = _as_dict(p66_rollout_default_enable_contract)
    evidence = _as_dict(product_request_ui_evidence)
    review = _as_dict(product_request_ui_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p66_summary = _p66_summary(p66)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p66_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P67_HOLD_DECISION:
        blockers.append("v5_p67_signed_product_request_ui_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P67_READY_DECISION
    rejected = decision == P67_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_product_request_ui_integration_contract_v0",
        "gate": "v5_product_request_ui_integration_contract",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "product_request_ui_review_recorded": decision_record_ready,
        "product_request_ui_review_signed": decision_record_ready,
        "product_request_ui_integration_contract_ready": ready,
        "product_request_ui_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p67_request_fields": {},
        "p66_rollout_default_enable_summary": p66_summary,
        "product_request_ui_evidence_summary": evidence_summary,
        "product_request_ui_review": review_summary,
        "product_request_ui_review_template": _review_template(),
        "progress_gates": _progress_gates(p66_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P67 records product/request/UI integration evidence only.",
            "P67 does not register UI routes, emit request fields, enable request adapters, or launch training.",
            "A later request-adapter activation contract is still required before user-visible integration.",
        ],
    }


def _p66_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "rollout_default_enable_contract_ready": report.get("rollout_default_enable_contract_ready") is True,
        "rollout_default_enable_evidence_recorded": report.get("rollout_default_enable_evidence_recorded") is True,
        "rollout_default_enable_review_signed": report.get("rollout_default_enable_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P66_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p66_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p66"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    row_specs = _row_specs(evidence)
    blockers = _evidence_blocker_list(evidence, missing_sections, row_specs)
    summary = {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "product_request_ui_integration_contract_ready": evidence.get("product_request_ui_integration_contract_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "product_request_ui_integration_contract_only": evidence.get("product_request_ui_integration_contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_request_adapter_activation_contract": evidence.get("requires_later_request_adapter_activation_contract") is True,
        "requires_explicit_owner_approval": evidence.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": evidence.get("requires_explicit_operator_opt_in") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "blockers": blockers,
    }
    summary.update({f"{label}_count": len(rows) for label, rows, _kind in row_specs})
    return summary


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review), "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_product_request_ui_integration_contract": review.get("approve_product_request_ui_integration_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p66: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p66_rollout_default_enable_contract_ready": _p66_ready(p66),
        "product_request_ui_evidence_ready": bool(evidence.get("ready", False)),
        "signed_product_request_ui_review_present": bool(review.get("present", False)),
        "requested_scope_valid": review.get("requested_scope") == P67_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P67_BLOCKED_DECISION
    if not review:
        return P67_HOLD_DECISION
    if review_blockers:
        return P67_BLOCKED_DECISION
    if review.get("approve_product_request_ui_integration_contract") is True:
        return P67_READY_DECISION
    return P67_REJECTED_DECISION


def _evidence_blockers(
    p66_summary: Mapping[str, Any], evidence_summary: Mapping[str, Any],
    failure_events: list[str], rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p66_summary.get("present", False)):
        blocked.append("v5_p67_p66_rollout_default_enable_contract_missing")
    elif not _p66_ready(p66_summary):
        blocked.append("v5_p67_p66_rollout_default_enable_contract_not_ready")
        blocked.extend(_string_list(p66_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p66_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p66_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p67_product_request_ui_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p67_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p67_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p67_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p67_reviewed_at_missing")
    if review.get("requested_scope") != P67_SCOPE:
        blocked.append("v5_p67_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p67_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p67_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p66_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present") and summary.get("ok") and summary.get("decision_record_ready")
        and summary.get("rollout_default_enable_contract_ready")
        and summary.get("rollout_default_enable_evidence_recorded")
        and summary.get("rollout_default_enable_review_signed")
        and summary.get("decision") == P66_READY_DECISION and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P66_REQUIRED_FALSE_FIELDS)
        and summary.get("default_off") and summary.get("request_adapter_off")
        and summary.get("post_fields_empty") and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear") and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers")) and not _string_list(summary.get("unsafe_claims"))
    )


def _row_specs(evidence: Mapping[str, Any]) -> list[tuple[str, list[dict[str, Any]], str]]:
    specs = (
        ("product_surface", "product_surface_inventory", "product_surface", "product_surface"),
        ("request_schema", "request_schema_inventory", "request_schema", "request_schema"),
        ("request_adapter", "request_adapter_mapping_boundary", "request_adapter_mapping_inventory", "request_adapter_mapping"),
        ("ui_exposure", "ui_exposure_boundary", "ui_exposure_inventory", "ui_exposure_boundary"),
        ("product_wiring", "product_wiring_boundary", "product_wiring_inventory", "product_wiring"),
        ("launcher", "launcher_boundary", "launcher_inventory", "launcher_boundary"),
        ("webui", "webui_boundary", "webui_inventory", "webui_boundary"),
        ("backend_router", "backend_router_boundary", "backend_router_inventory", "backend_router_boundary"),
        ("training_request", "training_request_boundary", "training_request_inventory", "training_request_boundary"),
        ("default_behavior", "default_behavior_boundary", "default_behavior_inventory", "default_behavior_boundary"),
        ("operator_visibility", "operator_visibility_boundary", "operator_visibility_inventory", "operator_visibility"),
        ("monitoring", "monitoring_observability_boundary", "monitoring_observability_inventory", "monitoring_observability"),
        ("rollback", "rollback_policy_boundary", "rollback_policy_inventory", "rollback_policy"),
        ("docs_release", "docs_release_boundary", "docs_release_inventory", "docs_release"),
    )
    return [(label, _rows(evidence, field, fallback), kind) for label, field, fallback, kind in specs]


def _evidence_blocker_list(
    evidence: Mapping[str, Any], missing: list[str], row_specs: list[tuple[str, list[dict[str, Any]], str]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p67_product_request_ui_evidence_not_ok")
    if evidence.get("product_request_ui_integration_contract_ready") is not True:
        blocked.append("v5_p67_product_request_ui_evidence_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "product_request_ui_integration_contract_only",
        "records_evidence_only", "manual_only", "internal_only",
        "requires_later_request_adapter_activation_contract", "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p67_product_request_ui_evidence_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p67_product_request_ui_evidence_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p67_product_request_ui_evidence_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p67_product_request_ui_evidence_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p67_product_request_ui_evidence_source_missing")
    for section in missing:
        blocked.append(f"v5_p67_product_request_ui_evidence_section_missing:{section}")
    for _label, rows, kind in row_specs:
        blocked.extend(_row_blockers(rows, kind))
    blocked.extend(_unsafe_claims(evidence, "product_request_ui_evidence"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _row_blockers(rows: list[Mapping[str, Any]], kind: str) -> list[str]:
    if not rows:
        return [f"v5_p67_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("plan_id") or row.get("check_id") or row.get("id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"v5_p67_{kind}_not_ready:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p67_{kind}_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p67_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p67_unsafe_claim:{owner}:{field}")
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
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P67_SCOPE, "approve_product_request_ui_integration_contract": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P67_READY_DECISION:
        return ["archive_p67_product_request_ui_integration_contract"]
    if decision == P67_REJECTED_DECISION:
        return ["record_p67_default_off_rejection_or_repair_product_request_ui"]
    if decision == P67_HOLD_DECISION:
        return ["collect_signed_product_request_ui_review"]
    if any("p66" in item for item in blockers):
        return ["repair_p66_rollout_default_enable_contract"]
    if any(term in item for item in blockers for term in ("product", "request", "ui", "launcher", "webui", "router")):
        return ["repair_product_request_ui_evidence"]
    return ["clear_failure_or_rollback_history_before_p67_contract"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P67_READY_DECISION:
        return "archive P67 contract; request-adapter activation still requires a later explicit contract"
    if decision == P67_REJECTED_DECISION:
        return "record the signed rejection and keep product/request/UI integration default-off for repair"
    if decision == P67_HOLD_DECISION:
        return "collect a signed product/request/UI integration review over P66 evidence"
    if any("p66" in item for item in blockers):
        return "repair the P66 rollout/default-enable contract before P67"
    return "hold P67 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P67 product/request/UI integration contract.")
    parser.add_argument("--p66-rollout-default-enable-contract", default="", help="P66 rollout/default-enable JSON.")
    parser.add_argument("--product-request-ui-evidence", default="", help="P67 evidence JSON.")
    parser.add_argument("--product-request-ui-review", default="", help="Signed P67 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_product_request_ui_integration_contract(
        p66_rollout_default_enable_contract=load_json(args.p66_rollout_default_enable_contract)
        if args.p66_rollout_default_enable_contract else None,
        product_request_ui_evidence=load_json(args.product_request_ui_evidence)
        if args.product_request_ui_evidence else None,
        product_request_ui_review=load_json(args.product_request_ui_review)
        if args.product_request_ui_review else None,
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


__all__ = ["build_v5_product_request_ui_integration_contract"]
