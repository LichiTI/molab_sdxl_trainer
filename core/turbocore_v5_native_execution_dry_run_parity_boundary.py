"""Native execution dry-run / parity boundary for TurboCore V5-P52.

P52 records future native execution dry-run and parity boundary evidence after
P51. It does not execute native dispatch, launch kernels, run parity, execute
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


P51_READY_DECISION = "kernel_launch_native_execution_readiness_boundary_recorded_default_off"
P52_READY_DECISION = "native_execution_dry_run_parity_boundary_recorded_default_off"
P52_BLOCKED_DECISION = "native_execution_dry_run_parity_boundary_blocked_default_off"
P52_HOLD_DECISION = "native_execution_dry_run_parity_boundary_hold_for_signed_review_default_off"
P52_REJECTED_DECISION = "native_execution_dry_run_parity_boundary_rejected_default_off"
P52_SCOPE = "native_execution_dry_run_parity_boundary"
DEFAULT_REQUIRED_SECTIONS = (
    "p51_kernel_native_readiness_boundary_reference", "native_execution_dry_run_plan_inventory",
    "parity_check_plan_inventory", "dry_run_boundary", "parity_boundary", "kernel_artifact_boundary",
    "native_dispatch_boundary", "request_adapter_boundary", "no_native_execution_boundary",
    "no_native_dispatch_execution_boundary", "no_kernel_launch_execution_boundary", "no_parity_execution_boundary",
    "no_training_step_boundary", "no_request_fields_boundary", "no_training_launch_boundary",
    "rollback_policy", "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p51_kernel_native_readiness_boundary_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure", "acknowledge_no_native_execution_executed",
    "acknowledge_no_native_dispatch_executed", "acknowledge_no_kernel_launch_executed",
    "acknowledge_no_parity_executed", "acknowledge_no_training_step_executed",
    "acknowledge_no_request_adapter_enabled", "acknowledge_no_request_fields_emitted",
    "acknowledge_no_default_or_auto_rollout", "acknowledge_native_dry_run_parity_evidence_replayable",
    "acknowledge_later_kernel_artifact_or_result_ingestion_contract_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = (
    "training_launch_allowed", "auto_launch_allowed", "runs_dispatched", "default_training_path_enabled",
    "training_path_enabled", "default_rollout_allowed", "auto_rollout_allowed", "ui_exposure_allowed",
    "product_ui_exposure_allowed", "launcher_exposure_allowed", "webui_exposure_allowed",
    "request_adapter_mapping_allowed", "request_fields_emitted", "request_adapter_registered",
    "request_adapter_enabled", "runtime_adapter_registered", "runtime_adapter_enabled",
    "runtime_execution_allowed", "runtime_execution_executed", "runtime_dispatch_allowed",
    "runtime_dispatch_enabled", "runtime_dispatch_executed", "native_runtime_enabled",
    "native_execution_allowed", "native_execution_enabled", "native_execution_executed",
    "native_execution_dry_run_allowed", "native_execution_dry_run_enabled", "native_execution_dry_run_executed",
    "native_dispatch_allowed", "native_dispatch_enabled", "native_dispatch_started", "native_dispatch_executed",
    "kernel_launch_allowed", "kernel_launch_enabled", "kernel_launch_executed",
    "parity_check_allowed", "parity_check_enabled", "parity_check_executed",
    "parity_result_recorded", "kernel_artifact_registered", "kernel_artifact_enabled",
    "training_step_allowed", "training_step_enabled", "training_step_executed",
    "generation_request_patch_allowed", "config_adapter_patch_allowed", "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed", "training_manager_patch_allowed", "rollout_authorization_allowed",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p51_request_fields", "post_p52_request_fields", "post_native_dry_run_parity_fields",
    "native_execution_request", "native_execution_payload", "native_dry_run_request", "native_dry_run_payload",
    "native_dispatch_request", "native_dispatch_payload", "kernel_launch_request", "kernel_launch_payload",
    "parity_check_request", "parity_check_payload", "parity_result_payload", "training_step_request",
    "training_step_payload", "request_adapter_fields", "request_fields", "api_route_registration",
    "backend_route_registration", "ui_route_registration", "launcher_menu_entry", "webui_tab_entry",
    "launch_request", "training_request", "generation_request_patch", "config_adapter_patch",
    "runtime_resolver_patch", "execution_resolver_patch", "entry_train_patch", "training_manager_patch",
    "auto_launch_plan",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed")
P51_REQUIRED_FALSE_FIELDS = (
    "default_behavior_changed", "training_launch_allowed", "auto_launch_allowed", "runs_dispatched",
    "ui_exposure_allowed", "request_fields_emitted", "request_adapter_enabled", "native_execution_executed",
    "native_dispatch_started", "native_dispatch_executed", "kernel_launch_executed", "training_step_executed",
)


def build_v5_native_execution_dry_run_parity_boundary(
    *,
    p51_kernel_launch_native_execution_readiness_boundary: Mapping[str, Any] | None = None,
    native_dry_run_parity_evidence: Mapping[str, Any] | None = None,
    native_dry_run_parity_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record native dry-run/parity evidence without executing it."""

    p51 = _as_dict(p51_kernel_launch_native_execution_readiness_boundary)
    evidence = _as_dict(native_dry_run_parity_evidence)
    review = _as_dict(native_dry_run_parity_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p51_summary = _p51_summary(p51)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p51_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P52_HOLD_DECISION:
        blockers.append("v5_p52_signed_native_dry_run_parity_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P52_READY_DECISION
    rejected = decision == P52_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_native_execution_dry_run_parity_boundary_v0",
        "gate": "v5_native_execution_dry_run_parity_boundary",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "native_dry_run_parity_review_recorded": decision_record_ready,
        "native_dry_run_parity_review_signed": decision_record_ready,
        "native_execution_dry_run_parity_boundary_ready": ready,
        "native_dry_run_parity_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p52_request_fields": {},
        "p51_kernel_native_readiness_boundary_summary": p51_summary,
        "native_dry_run_parity_evidence_summary": evidence_summary,
        "native_dry_run_parity_review": review_summary,
        "native_dry_run_parity_review_template": _review_template(),
        "progress_gates": _progress_gates(p51_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P52 records native execution dry-run and parity boundary evidence only.",
            "P52 does not execute native dispatch, launch kernels, run parity, emit request fields, or launch training.",
            "A later kernel artifact gate or native dry-run result ingestion contract is still required before runtime behavior can become active.",
        ],
    }


def _p51_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "kernel_launch_native_execution_readiness_boundary_ready": report.get(
            "kernel_launch_native_execution_readiness_boundary_ready"
        )
        is True,
        "kernel_native_readiness_evidence_recorded": report.get("kernel_native_readiness_evidence_recorded") is True,
        "kernel_native_readiness_review_signed": report.get("kernel_native_readiness_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P51_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p51_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p51"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    dry_run_rows = _rows(evidence, "native_execution_dry_run_plan_inventory", "native_dry_run_plan_inventory")
    parity_rows = _rows(evidence, "parity_check_plan_inventory", "parity_plan_inventory")
    blockers = _evidence_blocker_list(evidence, missing_sections, dry_run_rows, parity_rows)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "native_execution_dry_run_parity_boundary_ready": evidence.get("native_execution_dry_run_parity_boundary_ready")
        is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "dry_run_boundary_only": evidence.get("dry_run_boundary_only") is True,
        "parity_boundary_only": evidence.get("parity_boundary_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_kernel_artifact_or_result_ingestion_contract": evidence.get(
            "requires_later_kernel_artifact_or_result_ingestion_contract"
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
        "unsafe_claims": _unsafe_claims(evidence, "native_dry_run_parity"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    approval = review.get("approve_native_execution_dry_run_parity_boundary")
    if approval is None:
        approval = review.get("approve_native_dry_run_parity_boundary")
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_execution_dry_run_parity_boundary": approval is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p51: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p51_kernel_native_readiness_boundary_present": bool(p51.get("present", False)),
        "p51_kernel_native_readiness_boundary_ready": _p51_ready(p51),
        "native_dry_run_parity_evidence_present": bool(evidence.get("present", False)),
        "native_dry_run_parity_evidence_ready": bool(evidence.get("ready", False)),
        "signed_native_dry_run_parity_review_present": bool(review.get("present", False)),
        "reviewer_present": bool(review.get("reviewer")),
        "reviewed_at_present": bool(review.get("reviewed_at")),
        "requested_scope_valid": review.get("requested_scope") == P52_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P52_BLOCKED_DECISION
    if not review:
        return P52_HOLD_DECISION
    if review_blockers:
        return P52_BLOCKED_DECISION
    if review.get("approve_native_execution_dry_run_parity_boundary") is True or review.get(
        "approve_native_dry_run_parity_boundary"
    ) is True:
        return P52_READY_DECISION
    return P52_REJECTED_DECISION


def _evidence_blockers(
    p51_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p51_summary.get("present", False)):
        blocked.append("v5_p52_p51_kernel_native_readiness_boundary_missing")
    elif not _p51_ready(p51_summary):
        blocked.append("v5_p52_p51_kernel_native_readiness_boundary_not_ready")
        blocked.extend(_string_list(p51_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p51_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p51_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p52_native_dry_run_parity_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p52_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p52_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p52_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p52_reviewed_at_missing")
    if review.get("requested_scope") != P52_SCOPE:
        blocked.append("v5_p52_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p52_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p52_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p51_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("kernel_launch_native_execution_readiness_boundary_ready")
        and summary.get("kernel_native_readiness_evidence_recorded")
        and summary.get("kernel_native_readiness_review_signed")
        and summary.get("decision") == P51_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P51_REQUIRED_FALSE_FIELDS)
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
    dry_run_rows: list[dict[str, Any]],
    parity_rows: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p52_native_dry_run_parity_not_ok")
    if evidence.get("native_execution_dry_run_parity_boundary_ready") is not True:
        blocked.append("v5_p52_native_dry_run_parity_evidence_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "dry_run_boundary_only", "parity_boundary_only",
        "records_evidence_only", "manual_only", "internal_only",
        "requires_later_kernel_artifact_or_result_ingestion_contract",
        "requires_explicit_owner_approval", "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p52_native_dry_run_parity_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p52_native_dry_run_parity_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p52_native_dry_run_parity_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p52_native_dry_run_parity_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p52_native_dry_run_parity_source_missing")
    for section in missing_sections:
        blocked.append(f"v5_p52_native_dry_run_parity_section_missing:{section}")
    blocked.extend(_dry_run_blockers(dry_run_rows))
    blocked.extend(_parity_blockers(parity_rows))
    blocked.extend(_unsafe_claims(evidence, "native_dry_run_parity"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _dry_run_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p52_native_execution_dry_run_plan_inventory_missing"]
    for index, row in enumerate(rows):
        item_id = str(row.get("plan_id") or row.get("id") or f"dry_run_{index}")
        for field in (
            "native_execution_dry_run_enabled", "native_execution_dry_run_executed", "native_execution_executed",
            "native_dispatch_started", "native_dispatch_executed", "kernel_launch_executed", "training_step_executed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p52_native_dry_run_claim:{item_id}:{field}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _parity_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    if not rows:
        return ["v5_p52_parity_check_plan_inventory_missing"]
    for index, row in enumerate(rows):
        check_id = str(row.get("check_id") or row.get("id") or f"parity_{index}")
        for field in (
            "parity_check_enabled", "parity_check_executed", "parity_result_recorded",
            "kernel_launch_executed", "native_dispatch_executed", "training_step_executed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p52_parity_claim:{check_id}:{field}")
        blocked.extend(_unsafe_claims(row, check_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p52_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p52_unsafe_claim:{owner}:{field}")
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
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P52_SCOPE, "approve_native_execution_dry_run_parity_boundary": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P52_READY_DECISION:
        return ["archive_p52_native_execution_dry_run_parity_boundary"]
    if decision == P52_REJECTED_DECISION:
        return ["record_p52_default_off_rejection_or_repair_native_dry_run_parity_evidence"]
    if decision == P52_HOLD_DECISION:
        return ["collect_signed_native_dry_run_parity_review"]
    if any("p51" in item for item in blockers):
        return ["repair_p51_kernel_native_readiness_boundary"]
    if any("dry_run" in item or "parity" in item for item in blockers):
        return ["repair_native_dry_run_parity_evidence"]
    return ["clear_failure_or_rollback_history_before_p52_boundary"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P52_READY_DECISION:
        return "archive P52 boundary; kernel artifact or result ingestion still requires a later explicit contract"
    if decision == P52_REJECTED_DECISION:
        return "record the signed rejection and keep native dry-run/parity default-off for repair"
    if decision == P52_HOLD_DECISION:
        return "collect a signed native dry-run/parity review over P51 evidence"
    if any("p51" in item for item in blockers):
        return "repair the P51 kernel/native readiness boundary before P52"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable native dry-run/parity source and digest evidence"
    return "hold P52 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P52 native execution dry-run/parity boundary.")
    parser.add_argument("--p51-kernel-launch-native-execution-readiness-boundary", default="", help="P51 boundary JSON.")
    parser.add_argument("--native-dry-run-parity-evidence", default="", help="P52 evidence JSON.")
    parser.add_argument("--native-dry-run-parity-review", default="", help="Signed P52 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    p51 = load_json(args.p51_kernel_launch_native_execution_readiness_boundary) if args.p51_kernel_launch_native_execution_readiness_boundary else None
    report = build_v5_native_execution_dry_run_parity_boundary(
        p51_kernel_launch_native_execution_readiness_boundary=p51,
        native_dry_run_parity_evidence=load_json(args.native_dry_run_parity_evidence) if args.native_dry_run_parity_evidence else None,
        native_dry_run_parity_review=load_json(args.native_dry_run_parity_review) if args.native_dry_run_parity_review else None,
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
