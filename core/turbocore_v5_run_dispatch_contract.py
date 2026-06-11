"""Run-dispatch contract for TurboCore V5-P72.

P72 records future run-dispatch evidence after P71. It checks dispatch plans,
run-record, scheduler, training-launch, rollback, runtime boundary, and
observability evidence only; it does not dispatch runs, write run records,
enqueue schedulers, or launch training.
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
from core.turbocore_v5_request_execution_job_creation_contract import (
    P71_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P71_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P71_UNSAFE_TRUE_FIELDS,
)


P72_READY_DECISION = "run_dispatch_contract_recorded_default_off"
P72_BLOCKED_DECISION = "run_dispatch_contract_blocked_default_off"
P72_HOLD_DECISION = "run_dispatch_contract_hold_for_signed_review_default_off"
P72_REJECTED_DECISION = "run_dispatch_contract_rejected_default_off"
P72_SCOPE = "run_dispatch_contract"
DEFAULT_REQUIRED_SECTIONS = (
    "p71_request_execution_job_creation_contract_reference", "run_dispatch_plan_inventory",
    "run_dispatch_precondition_inventory", "run_record_boundary", "scheduler_dispatch_boundary",
    "training_launch_boundary", "runtime_dispatch_boundary", "execution_resolution_boundary",
    "operator_dispatch_boundary", "observability_boundary", "rollback_policy_boundary",
    "no_request_execution_boundary", "no_job_creation_boundary", "no_job_record_write_boundary",
    "no_queue_enqueue_boundary", "no_run_dispatch_boundary", "no_run_record_write_boundary",
    "no_scheduler_dispatch_boundary", "no_training_launch_boundary",
    "no_request_submission_boundary", "no_request_payload_materialization_boundary",
    "no_request_fields_emit_boundary", "no_generation_request_patch_boundary",
    "no_request_schema_patch_boundary", "no_config_adapter_patch_boundary",
    "no_backend_router_registration_boundary", "no_router_submission_boundary",
    "no_training_request_submit_boundary", "no_request_adapter_enable_boundary",
    "no_ui_route_registration_boundary", "no_ui_exposure_boundary", "no_default_enable_boundary",
    "no_default_rollout_boundary", "no_auto_rollout_boundary", "no_training_step_execution_boundary",
    "no_parity_execution_boundary", "no_tensor_transfer_execution_boundary",
    "no_kernel_launch_execution_boundary", "no_native_dispatch_execution_boundary",
    "no_runtime_dispatch_execution_boundary", "no_runtime_execution_boundary",
    "no_runtime_state_refresh_boundary", "no_runtime_adapter_enabled_boundary",
    "no_execution_replay_boundary", "no_artifact_load_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p71_request_execution_job_creation_contract_recorded",
    "acknowledge_default_off_boundary", "acknowledge_no_request_execution_executed",
    "acknowledge_no_job_created", "acknowledge_no_job_record_written",
    "acknowledge_no_queue_enqueued", "acknowledge_no_run_dispatch_executed",
    "acknowledge_no_run_record_written", "acknowledge_no_scheduler_dispatch_executed",
    "acknowledge_no_training_launch", "acknowledge_no_training_launch_executed",
    "acknowledge_no_request_submission_executed", "acknowledge_no_request_submitted",
    "acknowledge_no_request_payload_materialized", "acknowledge_no_request_fields_emitted",
    "acknowledge_no_generation_request_patch", "acknowledge_no_request_schema_patch",
    "acknowledge_no_config_adapter_patch", "acknowledge_no_backend_router_registered",
    "acknowledge_no_router_submission", "acknowledge_no_training_request_submitted",
    "acknowledge_no_request_adapter_enabled", "acknowledge_no_ui_route_registered",
    "acknowledge_no_ui_exposure", "acknowledge_no_default_enable",
    "acknowledge_no_default_rollout", "acknowledge_no_auto_rollout",
    "acknowledge_no_training_step_executed", "acknowledge_no_parity_executed",
    "acknowledge_no_tensor_transfer_executed", "acknowledge_no_kernel_launch_executed",
    "acknowledge_no_native_dispatch_executed", "acknowledge_no_runtime_dispatch_executed",
    "acknowledge_no_runtime_execution_executed", "acknowledge_no_runtime_state_refreshed",
    "acknowledge_no_runtime_adapter_enabled", "acknowledge_no_execution_replay_executed",
    "acknowledge_no_artifact_loaded", "acknowledge_run_dispatch_evidence_replayable",
    "acknowledge_later_training_launch_execution_contract_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P71_UNSAFE_TRUE_FIELDS,
            "run_dispatch_approved", "run_dispatch_allowed", "run_dispatch_enabled",
            "run_dispatch_executed", "runs_dispatched", "run_record_written",
            "scheduler_dispatch_allowed", "scheduler_dispatch_enabled", "scheduler_dispatch_executed",
            "training_launch_allowed", "training_launch_enabled", "training_launch_executed",
            "runtime_dispatch_allowed", "runtime_dispatch_enabled", "runtime_dispatch_executed",
            "runtime_execution_executed", "native_dispatch_executed", "kernel_launch_executed",
            "operator_dispatch_executed", "execution_resolution_invoked",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P71_UNSAFE_NON_EMPTY_FIELDS,
            "post_p72_request_fields", "run_dispatch_contract_request", "run_dispatch_contract_payload",
            "run_dispatch_payload", "run_record_payload", "scheduler_dispatch_payload",
            "training_launch_payload", "runtime_dispatch_payload", "training_run_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)
P71_REQUIRED_FALSE_FIELDS = _P71_UNSAFE_TRUE_FIELDS


def build_v5_run_dispatch_contract(
    *,
    p71_request_execution_job_creation_contract: Mapping[str, Any] | None = None,
    run_dispatch_evidence: Mapping[str, Any] | None = None,
    run_dispatch_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record run-dispatch evidence without dispatching a run."""

    p71 = _as_dict(p71_request_execution_job_creation_contract)
    evidence = _as_dict(run_dispatch_evidence)
    review = _as_dict(run_dispatch_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p71_summary = _p71_summary(p71)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p71_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P72_HOLD_DECISION:
        blockers.append("v5_p72_signed_run_dispatch_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P72_READY_DECISION
    rejected = decision == P72_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_run_dispatch_contract_v0",
        "gate": "v5_run_dispatch_contract",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "run_dispatch_review_recorded": decision_record_ready,
        "run_dispatch_review_signed": decision_record_ready,
        "run_dispatch_contract_ready": ready,
        "run_dispatch_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p72_request_fields": {},
        "p71_request_execution_job_creation_summary": p71_summary,
        "run_dispatch_evidence_summary": evidence_summary,
        "run_dispatch_review": review_summary,
        "run_dispatch_review_template": _review_template(),
        "progress_gates": _progress_gates(p71_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P72 records run-dispatch contract evidence only.",
            "P72 does not dispatch runs, write run records, enqueue schedulers, or launch training.",
            "A later training-launch execution contract is still required.",
        ],
    }


def _p71_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "request_execution_job_creation_contract_ready": report.get("request_execution_job_creation_contract_ready") is True,
        "request_execution_job_creation_evidence_recorded": report.get("request_execution_job_creation_evidence_recorded") is True,
        "request_execution_job_creation_review_signed": report.get("request_execution_job_creation_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P71_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p71_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p71"),
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
        "run_dispatch_contract_ready": evidence.get("run_dispatch_contract_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "run_dispatch_contract_only": evidence.get("run_dispatch_contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_training_launch_execution_contract": (
            evidence.get("requires_later_training_launch_execution_contract") is True
        ),
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
        "approve_run_dispatch_contract": review.get("approve_run_dispatch_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p71: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p71_request_execution_job_creation_contract_ready": _p71_ready(p71),
        "run_dispatch_evidence_ready": bool(evidence.get("ready", False)),
        "signed_run_dispatch_review_present": bool(review.get("present", False)),
        "requested_scope_valid": review.get("requested_scope") == P72_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P72_BLOCKED_DECISION
    if not review:
        return P72_HOLD_DECISION
    if review_blockers:
        return P72_BLOCKED_DECISION
    if review.get("approve_run_dispatch_contract") is True:
        return P72_READY_DECISION
    return P72_REJECTED_DECISION


def _evidence_blockers(
    p71_summary: Mapping[str, Any], evidence_summary: Mapping[str, Any],
    failure_events: list[str], rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p71_summary.get("present", False)):
        blocked.append("v5_p72_p71_request_execution_job_creation_contract_missing")
    elif not _p71_ready(p71_summary):
        blocked.append("v5_p72_p71_request_execution_job_creation_contract_not_ready")
        blocked.extend(_string_list(p71_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p71_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p71_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p72_run_dispatch_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p72_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p72_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p72_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p72_reviewed_at_missing")
    if review.get("requested_scope") != P72_SCOPE:
        blocked.append("v5_p72_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p72_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p72_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p71_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present") and summary.get("ok") and summary.get("decision_record_ready")
        and summary.get("request_execution_job_creation_contract_ready")
        and summary.get("request_execution_job_creation_evidence_recorded")
        and summary.get("request_execution_job_creation_review_signed")
        and summary.get("decision") == P71_READY_DECISION and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P71_REQUIRED_FALSE_FIELDS)
        and summary.get("default_off") and summary.get("request_adapter_off")
        and summary.get("post_fields_empty") and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear") and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers")) and not _string_list(summary.get("unsafe_claims"))
    )


def _row_specs(evidence: Mapping[str, Any]) -> list[tuple[str, list[dict[str, Any]], str]]:
    specs = (
        ("dispatch_plan", "run_dispatch_plan_inventory", "run_dispatch_plan", "run_dispatch_plan"),
        ("dispatch_precondition", "run_dispatch_precondition_inventory", "run_dispatch_precondition", "run_dispatch_precondition"),
        ("run_record", "run_record_boundary", "run_record_inventory", "run_record_boundary"),
        ("scheduler", "scheduler_dispatch_boundary", "scheduler_dispatch_inventory", "scheduler_dispatch_boundary"),
        ("training_launch", "training_launch_boundary", "training_launch_inventory", "training_launch_boundary"),
        ("runtime_dispatch", "runtime_dispatch_boundary", "runtime_dispatch_inventory", "runtime_dispatch_boundary"),
        ("execution_resolution", "execution_resolution_boundary", "execution_resolution_inventory", "execution_resolution_boundary"),
        ("operator_dispatch", "operator_dispatch_boundary", "operator_dispatch_inventory", "operator_dispatch_boundary"),
        ("observability", "observability_boundary", "observability_inventory", "observability_boundary"),
        ("rollback", "rollback_policy_boundary", "rollback_policy_inventory", "rollback_policy"),
    )
    return [(label, _rows(evidence, field, fallback), kind) for label, field, fallback, kind in specs]


def _evidence_blocker_list(
    evidence: Mapping[str, Any], missing: list[str], row_specs: list[tuple[str, list[dict[str, Any]], str]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p72_run_dispatch_evidence_not_ok")
    if evidence.get("run_dispatch_contract_ready") is not True:
        blocked.append("v5_p72_run_dispatch_evidence_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "run_dispatch_contract_only",
        "records_evidence_only", "manual_only", "internal_only",
        "requires_later_training_launch_execution_contract", "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p72_run_dispatch_evidence_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p72_run_dispatch_evidence_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p72_run_dispatch_evidence_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p72_run_dispatch_evidence_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p72_run_dispatch_evidence_source_missing")
    for section in missing:
        blocked.append(f"v5_p72_run_dispatch_evidence_section_missing:{section}")
    for _label, rows, kind in row_specs:
        blocked.extend(_row_blockers(rows, kind))
    blocked.extend(_unsafe_claims(evidence, "run_dispatch_evidence"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _row_blockers(rows: list[Mapping[str, Any]], kind: str) -> list[str]:
    if not rows:
        return [f"v5_p72_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("plan_id") or row.get("check_id") or row.get("id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"v5_p72_{kind}_not_ready:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p72_{kind}_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p72_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p72_unsafe_claim:{owner}:{field}")
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
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P72_SCOPE, "approve_run_dispatch_contract": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P72_READY_DECISION:
        return ["archive_p72_run_dispatch_contract"]
    if decision == P72_REJECTED_DECISION:
        return ["record_p72_default_off_rejection_or_repair_run_dispatch"]
    if decision == P72_HOLD_DECISION:
        return ["collect_signed_run_dispatch_review"]
    if any("p71" in item for item in blockers):
        return ["repair_p71_request_execution_job_creation_contract"]
    if any(term in item for item in blockers for term in ("dispatch", "run_record", "scheduler", "launch", "runtime")):
        return ["repair_run_dispatch_evidence"]
    return ["clear_failure_or_rollback_history_before_p72_contract"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P72_READY_DECISION:
        return "archive P72 contract; training launch still requires a later explicit execution contract"
    if decision == P72_REJECTED_DECISION:
        return "record the signed rejection and keep run dispatch default-off for repair"
    if decision == P72_HOLD_DECISION:
        return "collect a signed run-dispatch review over P71 evidence"
    if any("p71" in item for item in blockers):
        return "repair the P71 request-execution/job-creation contract before P72"
    return "hold P72 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P72 run-dispatch contract.")
    parser.add_argument("--p71-request-execution-job-creation-contract", default="", help="P71 contract JSON.")
    parser.add_argument("--run-dispatch-evidence", default="", help="P72 evidence JSON.")
    parser.add_argument("--run-dispatch-review", default="", help="Signed P72 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_run_dispatch_contract(
        p71_request_execution_job_creation_contract=load_json(args.p71_request_execution_job_creation_contract)
        if args.p71_request_execution_job_creation_contract else None,
        run_dispatch_evidence=load_json(args.run_dispatch_evidence) if args.run_dispatch_evidence else None,
        run_dispatch_review=load_json(args.run_dispatch_review) if args.run_dispatch_review else None,
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


__all__ = ["build_v5_run_dispatch_contract"]
