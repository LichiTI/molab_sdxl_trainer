"""Optimizer batch keep/rollback review contract for TurboCore V5-P86.

P86 records the signed review shape for future optimizer batch-validation
results. It may record a proposed keep/rollback intent, but it never applies
that decision, promotes optimizer results, loads artifacts, executes kernels,
launches training, emits request fields, patches schemas, or exposes UI.
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
from core.turbocore_v5_optimizer_batch_result_ingestion_contract_p85 import (
    P85_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P85_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P85_UNSAFE_TRUE_FIELDS,
)
from core.turbocore_v5_optimizer_batch_validation_contract_p82 import DEFAULT_OPTIMIZER_KINDS
from core.turbocore_v5_owner_review_evidence_package import load_json


P86_READY_DECISION = "optimizer_batch_keep_rollback_review_contract_p86_recorded_default_off"
P86_BLOCKED_DECISION = "optimizer_batch_keep_rollback_review_contract_p86_blocked_default_off"
P86_HOLD_DECISION = "optimizer_batch_keep_rollback_review_contract_p86_hold_for_signed_review_default_off"
P86_REJECTED_DECISION = "optimizer_batch_keep_rollback_review_contract_p86_rejected_default_off"
P86_SCOPE = "optimizer_batch_keep_rollback_review_contract"
ALLOWED_REVIEW_INTENTS = {"keep_candidate", "rollback_candidate", "hold_for_more_evidence"}
REQUIRED_SECTIONS = (
    "p85_result_ingestion_contract_reference",
    "keep_rollback_review_package",
    "per_optimizer_keep_rollback_rows",
    "parity_regression_summary",
    "benchmark_regression_summary",
    "state_roundtrip_regression_summary",
    "debug_failure_summary",
    "rollback_policy_summary",
    "no_keep_decision_apply_boundary",
    "no_rollback_decision_apply_boundary",
    "no_optimizer_result_promotion_boundary",
    "no_artifact_load_boundary",
    "no_training_launch_boundary",
    "no_request_ui_schema_patch_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p85_result_ingestion_contract_recorded",
    "acknowledge_keep_rollback_review_only",
    "acknowledge_no_keep_decision_applied",
    "acknowledge_no_rollback_decision_applied",
    "acknowledge_no_optimizer_result_promoted",
    "acknowledge_no_artifact_loaded",
    "acknowledge_no_training_launch",
    "acknowledge_no_request_ui_schema_patch",
    "acknowledge_later_optimizer_integration_review_required",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P85_UNSAFE_TRUE_FIELDS,
            "keep_decision_executed",
            "rollback_decision_executed",
            "optimizer_result_kept",
            "optimizer_result_rolled_back",
            "optimizer_integration_approved",
            "optimizer_integration_enabled",
            "optimizer_training_path_enabled",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P85_UNSAFE_NON_EMPTY_FIELDS,
            "post_p86_request_fields",
            "keep_decision_execution_payload",
            "rollback_decision_execution_payload",
            "optimizer_integration_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_v5_optimizer_batch_keep_rollback_review_contract_p86(
    *,
    p85_result_ingestion_contract: Mapping[str, Any] | None = None,
    keep_rollback_review_evidence: Mapping[str, Any] | None = None,
    keep_rollback_signed_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record a keep/rollback review package without applying it."""

    p85 = _as_dict(p85_result_ingestion_contract)
    evidence = _as_dict(keep_rollback_review_evidence)
    review = _as_dict(keep_rollback_signed_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p85_summary = _p85_summary(p85)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p85_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P86_HOLD_DECISION:
        blockers.append("v5_p86_signed_keep_rollback_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P86_READY_DECISION
    rejected = decision == P86_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_optimizer_batch_keep_rollback_review_contract_p86_v0",
        "gate": "v5_optimizer_batch_keep_rollback_review_contract_p86",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "keep_rollback_review_recorded": decision_record_ready,
        "keep_rollback_review_signed": decision_record_ready,
        "keep_rollback_review_contract_ready": ready,
        "keep_rollback_review_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p86_request_fields": {},
        "p85_result_ingestion_summary": p85_summary,
        "keep_rollback_review_evidence_summary": evidence_summary,
        "keep_rollback_signed_review": review_summary,
        "keep_rollback_review_template": _review_template(),
        "progress_gates": _progress_gates(p85_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P86 records keep/rollback review evidence only.",
            "P86 does not apply keep/rollback decisions.",
            "A later optimizer integration review contract is still required.",
        ],
    }


def _p85_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "result_ingestion_contract_ready": report.get("optimizer_batch_result_ingestion_contract_ready") is True,
        "result_evidence_recorded": report.get("optimizer_batch_result_evidence_recorded") is True,
        "result_review_signed": report.get("optimizer_batch_result_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p85_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p85"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = set(_string_list(evidence.get("available_sections"))) | set(str(key) for key in evidence.keys())
    missing_sections = [item for item in REQUIRED_SECTIONS if item not in sections]
    rows = _review_rows(evidence)
    missing_optimizers = _missing_optimizers(rows)
    row_blockers = _row_blockers(rows)
    blockers = _evidence_blocker_list(evidence, missing_sections, missing_optimizers, row_blockers)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "keep_rollback_review_package_ready": evidence.get("keep_rollback_review_package_ready") is True,
        "keep_rollback_policy_ready": evidence.get("keep_rollback_policy_ready") is True,
        "later_optimizer_integration_review_required": (
            evidence.get("later_optimizer_integration_review_required") is True
        ),
        "review_intent": str(evidence.get("review_intent") or ""),
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "review_rows": rows,
        "missing_optimizers": missing_optimizers,
        "row_blockers": row_blockers,
        "missing_sections": missing_sections,
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_keep_rollback_review_contract": review.get("approve_keep_rollback_review_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p85: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p85_result_ingestion_contract_ready": _p85_ready(p85),
        "keep_rollback_review_evidence_ready": bool(evidence.get("ready", False)),
        "signed_keep_rollback_review_present": bool(review.get("present", False)),
        "all_default_optimizer_rows_present": not bool(evidence.get("missing_optimizers")),
    }


def _evidence_blockers(
    p85: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_events: Sequence[str],
    rollback_events: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not _p85_ready(p85):
        blockers.append("v5_p86_p85_optimizer_batch_result_ingestion_contract_not_ready")
    blockers.extend(f"v5_p86_unsafe_upstream_claim:p85:{item}" for item in p85.get("unsafe_claims") or [])
    if not evidence.get("present"):
        blockers.append("v5_p86_keep_rollback_review_evidence_missing")
    blockers.extend(_string_list(evidence.get("blockers")))
    if failure_events:
        blockers.extend(f"v5_p86_failure_history_not_clear:{item}" for item in failure_events)
    if rollback_events:
        blockers.extend(f"v5_p86_rollback_history_not_clear:{item}" for item in rollback_events)
    return _dedupe(blockers)


def _evidence_blocker_list(
    evidence: Mapping[str, Any],
    missing_sections: Sequence[str],
    missing_optimizers: Sequence[str],
    row_blockers: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    checks = (
        ("ok", True, "evidence_not_ok"),
        ("keep_rollback_review_package_ready", True, "package_not_ready"),
        ("keep_rollback_policy_ready", True, "policy_not_ready"),
        ("later_optimizer_integration_review_required", True, "later_integration_review_missing"),
        ("report_only", True, "report_only_missing"),
        ("boundary_only", True, "boundary_only_missing"),
        ("contract_only", True, "contract_only_missing"),
        ("records_evidence_only", True, "records_evidence_only_missing"),
        ("manual_only", True, "manual_only_missing"),
        ("internal_only", True, "internal_only_missing"),
    )
    for field, expected, reason in checks:
        if evidence.get(field) is not expected:
            blockers.append(f"v5_p86_keep_rollback_evidence_{reason}")
    if str(evidence.get("review_intent") or "") not in ALLOWED_REVIEW_INTENTS:
        blockers.append("v5_p86_keep_rollback_review_intent_invalid")
    if not evidence.get("default_off") or not _default_off_confirmed(evidence):
        blockers.append("v5_p86_keep_rollback_evidence_default_off_violation")
    if not evidence.get("request_adapter_off") or not _request_adapter_off(evidence):
        blockers.append("v5_p86_keep_rollback_evidence_request_adapter_boundary_violation")
    if not _source(evidence):
        blockers.append("v5_p86_keep_rollback_evidence_source_missing")
    if not _digest(evidence):
        blockers.append("v5_p86_keep_rollback_evidence_digest_missing")
    blockers.extend(f"v5_p86_required_section_missing:{item}" for item in missing_sections)
    blockers.extend(f"v5_p86_optimizer_review_row_missing:{item}" for item in missing_optimizers)
    blockers.extend(row_blockers)
    blockers.extend(_unsafe_claims(evidence, "keep_rollback_review_evidence"))
    blockers.extend(_non_empty_claims(evidence, "keep_rollback_review_evidence"))
    blockers.extend(_string_list(evidence.get("blocked_reasons")))
    blockers.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blockers)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blockers: list[str] = []
    if review.get("requested_scope") != P86_SCOPE:
        blockers.append("v5_p86_review_scope_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        blockers.append("v5_p86_review_identity_or_timestamp_missing")
    for field in REQUIRED_REVIEW_ACKS:
        if review.get(field) is not True:
            blockers.append(f"v5_p86_review_ack_missing:{field}")
    blockers.extend(
        f"v5_p86_unsafe_review_approval:{field}"
        for field in UNSAFE_REVIEW_APPROVAL_FIELDS
        if review.get(field) is True
    )
    return _dedupe(blockers)


def _decision(blockers: Sequence[str], review: Mapping[str, Any], review_blockers: Sequence[str]) -> str:
    if blockers or review_blockers:
        return P86_BLOCKED_DECISION
    if not review:
        return P86_HOLD_DECISION
    if review.get("approve_keep_rollback_review_contract") is not True:
        return P86_REJECTED_DECISION
    return P86_READY_DECISION


def _p85_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("result_ingestion_contract_ready")
        and summary.get("result_evidence_recorded")
        and summary.get("result_review_signed")
        and summary.get("decision") == P85_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and not summary.get("unsafe_claims")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
    )


def _review_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = evidence.get("optimizer_review_rows") or evidence.get("review_rows") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [_review_row(row) for row in rows if isinstance(row, Mapping)]


def _review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_kind": str(row.get("optimizer_kind") or ""),
        "source": _source(row),
        "ready": row.get("ready") is True,
        "result_summary_ready": row.get("result_summary_ready") is True,
        "rollback_policy_ready": row.get("rollback_policy_ready") is True,
        "integration_review_required": row.get("integration_review_required") is True,
        "keep_decision_applied": row.get("keep_decision_applied") is True,
        "rollback_decision_applied": row.get("rollback_decision_applied") is True,
        "optimizer_result_promoted": row.get("optimizer_result_promoted") is True,
        "blocked_reasons": _string_list(row.get("blocked_reasons")),
    }


def _missing_optimizers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row.get("optimizer_kind") or "") for row in rows}
    return [item for item in DEFAULT_OPTIMIZER_KINDS if item not in present]


def _row_blockers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        kind = str(row.get("optimizer_kind") or "unknown")
        if not row.get("source"):
            blockers.append(f"v5_p86_optimizer_review_row_source_missing:{kind}")
        for field in ("ready", "result_summary_ready", "rollback_policy_ready", "integration_review_required"):
            if row.get(field) is not True:
                blockers.append(f"v5_p86_optimizer_review_row_{field}_missing:{kind}")
        for field in ("keep_decision_applied", "rollback_decision_applied", "optimizer_result_promoted"):
            if row.get(field) is True:
                blockers.append(f"v5_p86_optimizer_review_row_unsafe_claim:{kind}:{field}")
        blockers.extend(f"v5_p86_optimizer_review_row_blocker:{kind}:{item}" for item in row.get("blocked_reasons") or [])
    return _dedupe(blockers)


def _unsafe_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p86_unsafe_claim:{label}:{field}" for field in UNSAFE_TRUE_FIELDS if value.get(field) is True]


def _non_empty_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p86_unsafe_claim:{label}:{field}" for field in UNSAFE_NON_EMPTY_FIELDS if bool(value.get(field))]


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": P86_SCOPE,
        "approve_keep_rollback_review_contract": True,
        **{field: True for field in REQUIRED_REVIEW_ACKS},
        **{field: False for field in UNSAFE_REVIEW_APPROVAL_FIELDS},
    }


def _allowed_next_actions(decision: str, blockers: Sequence[str]) -> list[str]:
    if decision == P86_READY_DECISION:
        return ["prepare_optimizer_integration_review_contract_default_off"]
    if decision == P86_REJECTED_DECISION:
        return ["keep_default_off", "refresh_keep_rollback_review_evidence"]
    if decision == P86_HOLD_DECISION:
        return ["collect_signed_keep_rollback_review"]
    return ["resolve_blockers", *list(blockers[:6])]


def _recommended_next_step(decision: str, blockers: Sequence[str]) -> str:
    if decision == P86_READY_DECISION:
        return "draft optimizer integration review contract"
    if decision == P86_HOLD_DECISION:
        return "collect signed owner review for optimizer batch keep/rollback contract"
    if decision == P86_REJECTED_DECISION:
        return "keep optimizer batch keep/rollback default-off and refresh evidence"
    return blockers[0] if blockers else "complete optimizer batch keep/rollback evidence"


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p85-result-ingestion-contract")
    parser.add_argument("--keep-rollback-review-evidence")
    parser.add_argument("--keep-rollback-signed-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_batch_keep_rollback_review_contract_p86(
        p85_result_ingestion_contract=_load_optional(args.p85_result_ingestion_contract),
        keep_rollback_review_evidence=_load_optional(args.keep_rollback_review_evidence),
        keep_rollback_signed_review=_load_optional(args.keep_rollback_signed_review),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ALLOWED_REVIEW_INTENTS",
    "P86_SCOPE",
    "REQUIRED_REVIEW_ACKS",
    "REQUIRED_SECTIONS",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_batch_keep_rollback_review_contract_p86",
]
