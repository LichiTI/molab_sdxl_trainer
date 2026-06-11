"""Product/UI exposure readiness boundary for TurboCore V5-P40.

P40 records a signed readiness-boundary review after P39. It deliberately does
not expose UI, emit request-adapter fields, launch training, or permit default
rollout. A ready P40 record only means the evidence can be archived for a later
integration contract.
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
    int_value as _int,
    request_adapter_off as _request_adapter_off,
    source as _source,
    string_list as _string_list,
)
from core.turbocore_v5_owner_review_evidence_package import load_json


P39_READY_DECISION = "controlled_rollout_policy_evidence_recorded_default_off"
P40_READY_DECISION = "product_ui_exposure_readiness_evidence_recorded_default_off"
P40_BLOCKED_DECISION = "product_ui_exposure_readiness_blocked_default_off"
P40_HOLD_DECISION = "product_ui_exposure_hold_for_signed_review_default_off"
P40_REJECTED_DECISION = "product_ui_exposure_readiness_rejected_default_off"
P40_SCOPE = "product_ui_exposure_readiness_boundary"
DEFAULT_REQUIRED_SECTIONS = (
    "surface_inventory",
    "no_ui_exposure_boundary",
    "no_request_adapter_boundary",
    "operator_opt_in_policy",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p39_policy_evidence_recorded",
    "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch",
    "acknowledge_no_ui_exposure",
    "acknowledge_no_ui_entry",
    "acknowledge_no_request_adapter_mapping",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_readiness_evidence_replayable",
    "acknowledge_later_integration_contract_required",
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
    "rollout_authorization_allowed",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p39_request_fields",
    "post_p40_request_fields",
    "post_ui_request_fields",
    "request_adapter",
    "request_adapter_fields",
    "launch_request",
    "training_request",
    "generation_request_patch",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
    "ui_launch_plan",
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
    "approve_launcher_exposure_allowed",
    "approve_webui_exposure_allowed",
    "approve_ui_entry_enabled",
    "approve_ready_for_ui",
    "approve_request_adapter_mapping_allowed",
    "approve_request_fields_emitted",
    "approve_rollout_authorization_allowed",
)


def build_v5_product_ui_exposure_readiness_boundary(
    *,
    p39_controlled_rollout_policy_gate: Mapping[str, Any] | None = None,
    product_ui_exposure_readiness_evidence: Mapping[str, Any] | None = None,
    product_ui_exposure_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record UI exposure readiness evidence without exposing UI."""

    p39 = _as_dict(p39_controlled_rollout_policy_gate)
    evidence = _as_dict(product_ui_exposure_readiness_evidence)
    review = _as_dict(product_ui_exposure_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p39_summary = _p39_summary(p39)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    progress = _progress_gates(p39_summary, evidence_summary, review_summary)
    evidence_blockers = _evidence_blockers(
        p39_summary=p39_summary,
        evidence_summary=evidence_summary,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P40_HOLD_DECISION:
        blockers.append("v5_p40_signed_product_ui_exposure_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P40_READY_DECISION
    rejected = decision == P40_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_product_ui_exposure_readiness_boundary_v0",
        "gate": "v5_product_ui_exposure_readiness_boundary",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "product_ui_exposure_review_recorded": decision_record_ready,
        "product_ui_exposure_review_signed": decision_record_ready,
        "product_ui_exposure_readiness_boundary_ready": ready,
        "product_ui_exposure_readiness_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "ui_exposure_allowed": False,
        "product_ui_exposure_allowed": False,
        "launcher_exposure_allowed": False,
        "webui_exposure_allowed": False,
        "ui_entry_enabled": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "rollout_authorization_allowed": False,
        "post_p40_request_fields": {},
        "p39_controlled_rollout_policy_summary": p39_summary,
        "product_ui_exposure_readiness_evidence_summary": evidence_summary,
        "product_ui_exposure_review": review_summary,
        "product_ui_exposure_review_template": _review_template(evidence_summary),
        "progress_gates": progress,
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P40 records product/UI exposure readiness evidence only.",
            "P40 does not expose UI, create request-adapter mappings, launch training, or change defaults.",
            "A later explicit integration contract is still required before any product surface can be wired.",
        ],
    }


def _p39_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "controlled_rollout_policy_evidence_ready": report.get("controlled_rollout_policy_evidence_ready") is True,
        "controlled_rollout_policy_evidence_recorded": report.get("controlled_rollout_policy_evidence_recorded") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_behavior_changed": report.get("default_behavior_changed"),
        "training_launch_allowed": report.get("training_launch_allowed"),
        "auto_launch_allowed": report.get("auto_launch_allowed"),
        "runs_dispatched": report.get("runs_dispatched"),
        "ui_exposure_allowed": report.get("ui_exposure_allowed"),
        "product_ui_exposure_allowed": report.get("product_ui_exposure_allowed"),
        "ui_entry_enabled": report.get("ui_entry_enabled"),
        "ready_for_ui": report.get("ready_for_ui"),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p39_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p39"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    available_sections = _section_set(evidence)
    missing_sections = [item for item in required_sections if item not in available_sections]
    surface_rows = _surface_rows(evidence)
    blockers = _readiness_evidence_blockers(evidence, required_sections, missing_sections)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "product_ui_exposure_readiness_evidence_ready": evidence.get(
            "product_ui_exposure_readiness_evidence_ready"
        )
        is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_integration_contract": evidence.get("requires_later_integration_contract") is True,
        "requires_explicit_owner_approval": evidence.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": evidence.get("requires_explicit_operator_opt_in") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "required_sections": required_sections,
        "available_sections": sorted(available_sections),
        "missing_sections": missing_sections,
        "surface_count": len(surface_rows),
        "surface_inventory_blockers": _surface_inventory_blockers(surface_rows),
        "blocked_reasons": _string_list(evidence.get("blocked_reasons")),
        "promotion_blockers": _string_list(evidence.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(evidence, "ui_readiness"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_product_ui_exposure_readiness_evidence": review.get(
            "approve_product_ui_exposure_readiness_evidence"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(
    p39_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    review_summary: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "p39_gate_present": bool(p39_summary.get("present", False)),
        "p39_gate_ready": _p39_ready(p39_summary),
        "product_ui_exposure_readiness_evidence_present": bool(evidence_summary.get("present", False)),
        "product_ui_exposure_readiness_evidence_ready": bool(evidence_summary.get("ready", False)),
        "signed_product_ui_exposure_review_present": bool(review_summary.get("present", False)),
        "reviewer_present": bool(review_summary.get("reviewer")),
        "reviewed_at_present": bool(review_summary.get("reviewed_at")),
        "requested_scope_valid": review_summary.get("requested_scope") == P40_SCOPE,
        "review_no_launch_requested": not bool(review_summary.get("approve_training_launch_allowed", False))
        and not bool(review_summary.get("approve_auto_launch_allowed", False))
        and not bool(review_summary.get("approve_runs_dispatched", False)),
        "review_no_ui_requested": not bool(review_summary.get("approve_ui_exposure_allowed", False))
        and not bool(review_summary.get("approve_product_ui_exposure_allowed", False))
        and not bool(review_summary.get("approve_ui_entry_enabled", False))
        and not bool(review_summary.get("approve_ready_for_ui", False)),
        "review_no_request_adapter_requested": not bool(
            review_summary.get("approve_request_adapter_mapping_allowed", False)
        )
        and not bool(review_summary.get("approve_request_fields_emitted", False)),
        "required_acknowledgements_present": all(bool(review_summary.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P40_BLOCKED_DECISION
    if not review:
        return P40_HOLD_DECISION
    if review_blockers:
        return P40_BLOCKED_DECISION
    if review.get("approve_product_ui_exposure_readiness_evidence") is True:
        return P40_READY_DECISION
    return P40_REJECTED_DECISION


def _evidence_blockers(
    *,
    p39_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p39_summary.get("present", False)):
        blocked.append("v5_p40_p39_controlled_policy_gate_missing")
    elif not _p39_ready(p39_summary):
        blocked.append("v5_p40_p39_controlled_policy_gate_not_ready")
        blocked.extend(_string_list(p39_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p39_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p39_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p40_product_ui_exposure_readiness_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p40_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p40_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p40_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p40_reviewed_at_missing")
    if review.get("requested_scope") != P40_SCOPE:
        blocked.append("v5_p40_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p40_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p40_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p39_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("controlled_rollout_policy_evidence_ready")
        and summary.get("controlled_rollout_policy_evidence_recorded")
        and summary.get("decision") == P39_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_behavior_changed") is False
        and summary.get("training_launch_allowed") is False
        and summary.get("auto_launch_allowed") is False
        and summary.get("runs_dispatched") is False
        and summary.get("ui_exposure_allowed") is False
        and summary.get("product_ui_exposure_allowed") is False
        and summary.get("ui_entry_enabled") is False
        and summary.get("ready_for_ui") is False
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
        and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers"))
        and not _string_list(summary.get("unsafe_claims"))
    )


def _readiness_evidence_blockers(
    evidence: Mapping[str, Any],
    required_sections: list[str],
    missing_sections: list[str],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p40_ui_readiness_not_ok")
    if evidence.get("product_ui_exposure_readiness_evidence_ready") is not True:
        blocked.append("v5_p40_ui_readiness_evidence_not_ready")
    for field in (
        "report_only",
        "boundary_only",
        "manual_only",
        "internal_only",
        "requires_later_integration_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p40_ui_readiness_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p40_ui_readiness_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p40_ui_readiness_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p40_ui_readiness_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p40_ui_readiness_source_missing")
    if not required_sections:
        blocked.append("v5_p40_ui_readiness_required_sections_empty")
    for section in missing_sections:
        blocked.append(f"v5_p40_ui_readiness_section_missing:{section}")
    blocked.extend(_surface_inventory_blockers(_surface_rows(evidence)))
    blocked.extend(_unsafe_claims(evidence, "ui_readiness"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _surface_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = evidence.get("product_surface_inventory", evidence.get("surface_inventory"))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _surface_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p40_ui_readiness_surface_inventory_missing"]
    for index, row in enumerate(rows):
        surface_id = str(row.get("surface_id") or row.get("id") or row.get("name") or f"surface_{index}")
        if row.get("exposure_enabled") is not False:
            blocked.append(f"v5_p40_surface_exposure_not_disabled:{surface_id}")
        if row.get("entry_registered") is True:
            blocked.append(f"v5_p40_surface_entry_registered:{surface_id}")
        blocked.extend(_unsafe_claims(row, surface_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p40_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p40_unsafe_claim:{owner}:{field}")
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
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": P40_SCOPE,
        "approve_product_ui_exposure_readiness_evidence": False,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence_id"] = evidence_summary.get("evidence_id")
    template["acknowledged_evidence_version"] = evidence_summary.get("evidence_version")
    template["acknowledged_evidence_digest"] = evidence_summary.get("digest")
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P40_READY_DECISION:
        return ["archive_p40_product_ui_exposure_readiness_boundary"]
    if decision == P40_REJECTED_DECISION:
        return ["record_p40_default_off_rejection_or_repair_readiness_evidence"]
    if decision == P40_HOLD_DECISION:
        return ["collect_signed_product_ui_exposure_readiness_review"]
    if any("p39" in item for item in blockers):
        return ["repair_p39_controlled_rollout_policy_gate"]
    if any("ui_readiness" in item for item in blockers):
        return ["repair_product_ui_exposure_readiness_evidence"]
    return ["clear_failure_or_rollback_history_before_ui_readiness_boundary"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P40_READY_DECISION:
        return "archive P40 readiness boundary; product wiring still requires a later explicit integration contract"
    if decision == P40_REJECTED_DECISION:
        return "record the signed rejection and keep all product surfaces default-off for repair"
    if decision == P40_HOLD_DECISION:
        return "collect a signed product/UI exposure readiness review over P39 evidence"
    if any("p39" in item for item in blockers):
        return "repair the P39 controlled policy evidence gate before P40"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable UI exposure readiness source and digest evidence"
    return "hold P40 until readiness evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P40 product/UI exposure readiness boundary.")
    parser.add_argument("--p39-controlled-rollout-policy-gate", default="", help="P39 policy gate JSON.")
    parser.add_argument("--product-ui-exposure-readiness-evidence", default="", help="P40 readiness evidence JSON.")
    parser.add_argument("--product-ui-exposure-review", default="", help="Signed P40 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_product_ui_exposure_readiness_boundary(
        p39_controlled_rollout_policy_gate=load_json(args.p39_controlled_rollout_policy_gate)
        if args.p39_controlled_rollout_policy_gate
        else None,
        product_ui_exposure_readiness_evidence=load_json(args.product_ui_exposure_readiness_evidence)
        if args.product_ui_exposure_readiness_evidence
        else None,
        product_ui_exposure_review=load_json(args.product_ui_exposure_review)
        if args.product_ui_exposure_review
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


__all__ = ["build_v5_product_ui_exposure_readiness_boundary"]
