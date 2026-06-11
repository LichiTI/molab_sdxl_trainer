"""Request-adapter mapping boundary for TurboCore V5-P41.

P41 records a signed boundary review for a future request-adapter mapping
proposal after P40. It never emits request fields, patches GenerationRequest,
registers an adapter, launches training, exposes UI, or changes defaults.
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


P40_READY_DECISION = "product_ui_exposure_readiness_evidence_recorded_default_off"
P41_READY_DECISION = "request_adapter_mapping_boundary_evidence_recorded_default_off"
P41_BLOCKED_DECISION = "request_adapter_mapping_boundary_blocked_default_off"
P41_HOLD_DECISION = "request_adapter_mapping_hold_for_signed_review_default_off"
P41_REJECTED_DECISION = "request_adapter_mapping_boundary_rejected_default_off"
P41_SCOPE = "request_adapter_mapping_boundary"
DEFAULT_REQUIRED_SECTIONS = (
    "schema_inventory",
    "generation_request_boundary",
    "config_adapter_boundary",
    "no_request_fields_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p40_product_ui_boundary_recorded",
    "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch",
    "acknowledge_no_ui_exposure",
    "acknowledge_no_request_adapter_mapping",
    "acknowledge_no_request_fields_emitted",
    "acknowledge_no_generation_request_patch",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_mapping_evidence_replayable",
    "acknowledge_later_adapter_contract_required",
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
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "rollout_authorization_allowed",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p40_request_fields",
    "post_p41_request_fields",
    "post_mapping_request_fields",
    "request_adapter",
    "request_adapter_fields",
    "request_fields",
    "launch_request",
    "training_request",
    "generation_request_patch",
    "config_adapter_patch",
    "request_schema_patch",
    "runtime_request_mapping",
    "adapter_registration",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
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
    "approve_generation_request_patch_allowed",
    "approve_config_adapter_patch_allowed",
    "approve_rollout_authorization_allowed",
)


def build_v5_request_adapter_mapping_boundary(
    *,
    p40_product_ui_exposure_boundary: Mapping[str, Any] | None = None,
    request_adapter_mapping_evidence: Mapping[str, Any] | None = None,
    request_adapter_mapping_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record request-adapter mapping evidence without registering an adapter."""

    p40 = _as_dict(p40_product_ui_exposure_boundary)
    evidence = _as_dict(request_adapter_mapping_evidence)
    review = _as_dict(request_adapter_mapping_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p40_summary = _p40_summary(p40)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    progress = _progress_gates(p40_summary, evidence_summary, review_summary)
    evidence_blockers = _evidence_blockers(
        p40_summary=p40_summary,
        evidence_summary=evidence_summary,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P41_HOLD_DECISION:
        blockers.append("v5_p41_signed_request_adapter_mapping_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P41_READY_DECISION
    rejected = decision == P41_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_request_adapter_mapping_boundary_v0",
        "gate": "v5_request_adapter_mapping_boundary",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "request_adapter_mapping_review_recorded": decision_record_ready,
        "request_adapter_mapping_review_signed": decision_record_ready,
        "request_adapter_mapping_boundary_ready": ready,
        "request_adapter_mapping_evidence_recorded": ready,
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
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "generation_request_patch_allowed": False,
        "config_adapter_patch_allowed": False,
        "rollout_authorization_allowed": False,
        "post_p41_request_fields": {},
        "p40_product_ui_exposure_boundary_summary": p40_summary,
        "request_adapter_mapping_evidence_summary": evidence_summary,
        "request_adapter_mapping_review": review_summary,
        "request_adapter_mapping_review_template": _review_template(evidence_summary),
        "progress_gates": progress,
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P41 records request-adapter mapping boundary evidence only.",
            "P41 does not register adapters, emit request fields, patch GenerationRequest, or launch training.",
            "A later explicit adapter integration contract is still required before any mapping can be wired.",
        ],
    }


def _p40_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "product_ui_exposure_readiness_boundary_ready": report.get(
            "product_ui_exposure_readiness_boundary_ready"
        )
        is True,
        "product_ui_exposure_readiness_evidence_recorded": report.get(
            "product_ui_exposure_readiness_evidence_recorded"
        )
        is True,
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
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p40_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p40"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    available_sections = _section_set(evidence)
    missing_sections = [item for item in required_sections if item not in available_sections]
    mapping_rows = _mapping_rows(evidence)
    blockers = _mapping_evidence_blockers(evidence, required_sections, missing_sections)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "request_adapter_mapping_boundary_evidence_ready": evidence.get(
            "request_adapter_mapping_boundary_evidence_ready"
        )
        is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_adapter_contract": evidence.get("requires_later_adapter_contract") is True,
        "requires_explicit_owner_approval": evidence.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": evidence.get("requires_explicit_operator_opt_in") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "required_sections": required_sections,
        "available_sections": sorted(available_sections),
        "missing_sections": missing_sections,
        "mapping_count": len(mapping_rows),
        "mapping_inventory_blockers": _mapping_inventory_blockers(mapping_rows),
        "blocked_reasons": _string_list(evidence.get("blocked_reasons")),
        "promotion_blockers": _string_list(evidence.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(evidence, "mapping"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_request_adapter_mapping_boundary_evidence": review.get(
            "approve_request_adapter_mapping_boundary_evidence"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(
    p40_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    review_summary: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "p40_boundary_present": bool(p40_summary.get("present", False)),
        "p40_boundary_ready": _p40_ready(p40_summary),
        "request_adapter_mapping_evidence_present": bool(evidence_summary.get("present", False)),
        "request_adapter_mapping_evidence_ready": bool(evidence_summary.get("ready", False)),
        "signed_request_adapter_mapping_review_present": bool(review_summary.get("present", False)),
        "reviewer_present": bool(review_summary.get("reviewer")),
        "reviewed_at_present": bool(review_summary.get("reviewed_at")),
        "requested_scope_valid": review_summary.get("requested_scope") == P41_SCOPE,
        "review_no_launch_requested": not bool(review_summary.get("approve_training_launch_allowed", False))
        and not bool(review_summary.get("approve_auto_launch_allowed", False))
        and not bool(review_summary.get("approve_runs_dispatched", False)),
        "review_no_request_adapter_requested": not bool(
            review_summary.get("approve_request_adapter_mapping_allowed", False)
        )
        and not bool(review_summary.get("approve_request_fields_emitted", False))
        and not bool(review_summary.get("approve_request_adapter_registered", False)),
        "review_no_request_patch_requested": not bool(
            review_summary.get("approve_generation_request_patch_allowed", False)
        )
        and not bool(review_summary.get("approve_config_adapter_patch_allowed", False)),
        "required_acknowledgements_present": all(bool(review_summary.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P41_BLOCKED_DECISION
    if not review:
        return P41_HOLD_DECISION
    if review_blockers:
        return P41_BLOCKED_DECISION
    if review.get("approve_request_adapter_mapping_boundary_evidence") is True:
        return P41_READY_DECISION
    return P41_REJECTED_DECISION


def _evidence_blockers(
    *,
    p40_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p40_summary.get("present", False)):
        blocked.append("v5_p41_p40_product_ui_boundary_missing")
    elif not _p40_ready(p40_summary):
        blocked.append("v5_p41_p40_product_ui_boundary_not_ready")
        blocked.extend(_string_list(p40_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p40_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p40_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p41_request_adapter_mapping_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p41_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p41_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p41_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p41_reviewed_at_missing")
    if review.get("requested_scope") != P41_SCOPE:
        blocked.append("v5_p41_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p41_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p41_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p40_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("product_ui_exposure_readiness_boundary_ready")
        and summary.get("product_ui_exposure_readiness_evidence_recorded")
        and summary.get("decision") == P40_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_behavior_changed") is False
        and summary.get("training_launch_allowed") is False
        and summary.get("auto_launch_allowed") is False
        and summary.get("runs_dispatched") is False
        and summary.get("ui_exposure_allowed") is False
        and summary.get("product_ui_exposure_allowed") is False
        and summary.get("request_adapter_mapping_allowed") is False
        and summary.get("request_fields_emitted") is False
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
        and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers"))
        and not _string_list(summary.get("unsafe_claims"))
    )


def _mapping_evidence_blockers(
    evidence: Mapping[str, Any],
    required_sections: list[str],
    missing_sections: list[str],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p41_mapping_not_ok")
    if evidence.get("request_adapter_mapping_boundary_evidence_ready") is not True:
        blocked.append("v5_p41_mapping_evidence_not_ready")
    for field in (
        "report_only",
        "boundary_only",
        "manual_only",
        "internal_only",
        "requires_later_adapter_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p41_mapping_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p41_mapping_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p41_mapping_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p41_mapping_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p41_mapping_source_missing")
    if not required_sections:
        blocked.append("v5_p41_mapping_required_sections_empty")
    for section in missing_sections:
        blocked.append(f"v5_p41_mapping_section_missing:{section}")
    blocked.extend(_mapping_inventory_blockers(_mapping_rows(evidence)))
    blocked.extend(_unsafe_claims(evidence, "mapping"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _mapping_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = evidence.get("request_adapter_mapping_inventory", evidence.get("mapping_inventory"))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _mapping_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p41_mapping_inventory_missing"]
    for index, row in enumerate(rows):
        mapping_id = str(row.get("mapping_id") or row.get("id") or row.get("name") or f"mapping_{index}")
        if row.get("mapping_enabled") is not False:
            blocked.append(f"v5_p41_mapping_not_disabled:{mapping_id}")
        if row.get("adapter_registered") is True:
            blocked.append(f"v5_p41_adapter_registered:{mapping_id}")
        if row.get("request_fields_emitted") is True:
            blocked.append(f"v5_p41_mapping_request_fields_emitted:{mapping_id}")
        blocked.extend(_unsafe_claims(row, mapping_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p41_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p41_unsafe_claim:{owner}:{field}")
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
        "requested_scope": P41_SCOPE,
        "approve_request_adapter_mapping_boundary_evidence": False,
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
    if decision == P41_READY_DECISION:
        return ["archive_p41_request_adapter_mapping_boundary"]
    if decision == P41_REJECTED_DECISION:
        return ["record_p41_default_off_rejection_or_repair_mapping_evidence"]
    if decision == P41_HOLD_DECISION:
        return ["collect_signed_request_adapter_mapping_review"]
    if any("p40" in item for item in blockers):
        return ["repair_p40_product_ui_exposure_boundary"]
    if any("mapping" in item for item in blockers):
        return ["repair_request_adapter_mapping_boundary_evidence"]
    return ["clear_failure_or_rollback_history_before_mapping_boundary"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P41_READY_DECISION:
        return "archive P41 mapping boundary; adapter wiring still requires a later explicit integration contract"
    if decision == P41_REJECTED_DECISION:
        return "record the signed rejection and keep request-adapter mapping default-off for repair"
    if decision == P41_HOLD_DECISION:
        return "collect a signed request-adapter mapping review over P40 evidence"
    if any("p40" in item for item in blockers):
        return "repair the P40 product/UI exposure boundary before P41"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable request-adapter mapping source and digest evidence"
    return "hold P41 until mapping evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P41 request-adapter mapping boundary.")
    parser.add_argument("--p40-product-ui-exposure-boundary", default="", help="P40 boundary JSON.")
    parser.add_argument("--request-adapter-mapping-evidence", default="", help="P41 mapping evidence JSON.")
    parser.add_argument("--request-adapter-mapping-review", default="", help="Signed P41 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_request_adapter_mapping_boundary(
        p40_product_ui_exposure_boundary=load_json(args.p40_product_ui_exposure_boundary)
        if args.p40_product_ui_exposure_boundary
        else None,
        request_adapter_mapping_evidence=load_json(args.request_adapter_mapping_evidence)
        if args.request_adapter_mapping_evidence
        else None,
        request_adapter_mapping_review=load_json(args.request_adapter_mapping_review)
        if args.request_adapter_mapping_review
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


__all__ = ["build_v5_request_adapter_mapping_boundary"]
