"""Optimizer batch-validation result-ingestion contract for TurboCore V5-P85.

P85 records the contract for accepting future optimizer batch-validation
results. It checks result-bundle shape, per-optimizer summaries, artifact
digests, parity/benchmark/debug references, and keep/rollback intent. It does
not ingest results, load artifacts, execute kernels, launch training, emit
request fields, patch schemas, or expose UI.
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
from core.turbocore_v5_optimizer_debug_failure_archive_contract_p84 import (
    P84_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P84_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P84_UNSAFE_TRUE_FIELDS,
)
from core.turbocore_v5_optimizer_batch_validation_contract_p82 import (
    DEFAULT_OPTIMIZER_KINDS,
    P82_CANARY_REPEAT_COUNT,
    P82_CANARY_STEP_COUNT,
)
from core.turbocore_v5_owner_review_evidence_package import load_json


P85_READY_DECISION = "optimizer_batch_result_ingestion_contract_p85_recorded_default_off"
P85_BLOCKED_DECISION = "optimizer_batch_result_ingestion_contract_p85_blocked_default_off"
P85_HOLD_DECISION = "optimizer_batch_result_ingestion_contract_p85_hold_for_signed_review_default_off"
P85_REJECTED_DECISION = "optimizer_batch_result_ingestion_contract_p85_rejected_default_off"
P85_SCOPE = "optimizer_batch_result_ingestion_contract"
REQUIRED_SECTIONS = (
    "p84_debug_failure_archive_contract_reference",
    "result_bundle_inventory",
    "per_optimizer_result_rows",
    "parity_result_summary",
    "benchmark_result_summary",
    "state_roundtrip_summary",
    "debug_failure_archive_summary",
    "artifact_digest_ledger",
    "keep_or_rollback_policy",
    "no_result_ingestion_execution_boundary",
    "no_artifact_load_boundary",
    "no_runner_execution_boundary",
    "no_optimizer_kernel_execution_boundary",
    "no_training_launch_boundary",
    "no_request_ui_schema_patch_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p84_debug_failure_archive_contract_recorded",
    "acknowledge_result_bundle_shape_only",
    "acknowledge_no_result_ingestion_executed",
    "acknowledge_no_artifact_loaded",
    "acknowledge_no_runner_executed",
    "acknowledge_no_optimizer_kernel_executed",
    "acknowledge_no_training_launch",
    "acknowledge_no_request_ui_schema_patch",
    "acknowledge_keep_or_rollback_requires_later_review",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P84_UNSAFE_TRUE_FIELDS,
            "result_ingestion_executed",
            "result_bundle_applied",
            "optimizer_result_accepted",
            "optimizer_result_promoted",
            "artifact_digest_ledger_applied",
            "keep_decision_applied",
            "rollback_decision_applied",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P84_UNSAFE_NON_EMPTY_FIELDS,
            "post_p85_request_fields",
            "result_ingestion_execution_payload",
            "optimizer_result_application_payload",
            "artifact_load_payload",
            "keep_or_rollback_execution_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_v5_optimizer_batch_result_ingestion_contract_p85(
    *,
    p84_debug_failure_archive_contract: Mapping[str, Any] | None = None,
    optimizer_batch_result_evidence: Mapping[str, Any] | None = None,
    optimizer_batch_result_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record optimizer batch result-ingestion evidence without ingesting."""

    p84 = _as_dict(p84_debug_failure_archive_contract)
    evidence = _as_dict(optimizer_batch_result_evidence)
    review = _as_dict(optimizer_batch_result_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p84_summary = _p84_summary(p84)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p84_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P85_HOLD_DECISION:
        blockers.append("v5_p85_signed_optimizer_batch_result_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P85_READY_DECISION
    rejected = decision == P85_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_optimizer_batch_result_ingestion_contract_p85_v0",
        "gate": "v5_optimizer_batch_result_ingestion_contract_p85",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "optimizer_batch_result_review_recorded": decision_record_ready,
        "optimizer_batch_result_review_signed": decision_record_ready,
        "optimizer_batch_result_ingestion_contract_ready": ready,
        "optimizer_batch_result_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        "canary_step_count": P82_CANARY_STEP_COUNT,
        "canary_repeat_count": P82_CANARY_REPEAT_COUNT,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p85_request_fields": {},
        "p84_debug_failure_archive_summary": p84_summary,
        "optimizer_batch_result_evidence_summary": evidence_summary,
        "optimizer_batch_result_review": review_summary,
        "optimizer_batch_result_review_template": _review_template(),
        "progress_gates": _progress_gates(p84_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P85 records optimizer batch result-ingestion evidence only.",
            "P85 does not ingest results or apply keep/rollback decisions.",
            "A later keep/rollback review contract is still required.",
        ],
    }


def _p84_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "debug_failure_archive_contract_ready": report.get("debug_failure_archive_contract_ready") is True,
        "debug_failure_archive_evidence_recorded": report.get("debug_failure_archive_evidence_recorded") is True,
        "debug_failure_archive_review_signed": report.get("debug_failure_archive_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p84_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p84"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = set(_string_list(evidence.get("available_sections"))) | set(str(key) for key in evidence.keys())
    missing_sections = [item for item in REQUIRED_SECTIONS if item not in sections]
    rows = _result_rows(evidence)
    missing_optimizers = _missing_optimizers(rows)
    row_blockers = _row_blockers(rows)
    blockers = _evidence_blocker_list(evidence, missing_sections, missing_optimizers, row_blockers)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "result_ingestion_contract_ready": evidence.get("result_ingestion_contract_ready") is True,
        "result_bundle_shape_ready": evidence.get("result_bundle_shape_ready") is True,
        "artifact_digest_ledger_ready": evidence.get("artifact_digest_ledger_ready") is True,
        "keep_or_rollback_policy_ready": evidence.get("keep_or_rollback_policy_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "later_keep_rollback_review_required": evidence.get("later_keep_rollback_review_required") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "canary_step_count": _int(evidence.get("canary_step_count")),
        "canary_repeat_count": _int(evidence.get("canary_repeat_count")),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "result_rows": rows,
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
        "approve_optimizer_batch_result_ingestion_contract": (
            review.get("approve_optimizer_batch_result_ingestion_contract") is True
        ),
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p84: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p84_debug_failure_archive_contract_ready": _p84_ready(p84),
        "optimizer_batch_result_evidence_ready": bool(evidence.get("ready", False)),
        "signed_optimizer_batch_result_review_present": bool(review.get("present", False)),
        "all_default_optimizer_rows_present": not bool(evidence.get("missing_optimizers")),
    }


def _evidence_blockers(
    p84: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_events: Sequence[str],
    rollback_events: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not _p84_ready(p84):
        blockers.append("v5_p85_p84_optimizer_debug_failure_archive_contract_not_ready")
    blockers.extend(f"v5_p85_unsafe_upstream_claim:p84:{item}" for item in p84.get("unsafe_claims") or [])
    if not evidence.get("present"):
        blockers.append("v5_p85_optimizer_batch_result_evidence_missing")
    blockers.extend(_string_list(evidence.get("blockers")))
    if failure_events:
        blockers.extend(f"v5_p85_failure_history_not_clear:{item}" for item in failure_events)
    if rollback_events:
        blockers.extend(f"v5_p85_rollback_history_not_clear:{item}" for item in rollback_events)
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
        ("result_ingestion_contract_ready", True, "not_ready"),
        ("result_bundle_shape_ready", True, "result_bundle_shape_missing"),
        ("artifact_digest_ledger_ready", True, "artifact_digest_ledger_missing"),
        ("keep_or_rollback_policy_ready", True, "keep_or_rollback_policy_missing"),
        ("report_only", True, "report_only_missing"),
        ("boundary_only", True, "boundary_only_missing"),
        ("contract_only", True, "contract_only_missing"),
        ("records_evidence_only", True, "records_evidence_only_missing"),
        ("manual_only", True, "manual_only_missing"),
        ("internal_only", True, "internal_only_missing"),
        ("later_keep_rollback_review_required", True, "later_keep_rollback_review_missing"),
    )
    for field, expected, reason in checks:
        if evidence.get(field) is not expected:
            blockers.append(f"v5_p85_result_evidence_{reason}")
    if _int(evidence.get("canary_step_count")) != P82_CANARY_STEP_COUNT:
        blockers.append("v5_p85_canary_step_count_must_be_20")
    if _int(evidence.get("canary_repeat_count")) != P82_CANARY_REPEAT_COUNT:
        blockers.append("v5_p85_canary_repeat_count_must_be_5")
    if not evidence.get("default_off") or not _default_off_confirmed(evidence):
        blockers.append("v5_p85_result_evidence_default_off_violation")
    if not evidence.get("request_adapter_off") or not _request_adapter_off(evidence):
        blockers.append("v5_p85_result_evidence_request_adapter_boundary_violation")
    if not _source(evidence):
        blockers.append("v5_p85_result_evidence_source_missing")
    if not _digest(evidence):
        blockers.append("v5_p85_result_evidence_digest_missing")
    blockers.extend(f"v5_p85_required_section_missing:{item}" for item in missing_sections)
    blockers.extend(f"v5_p85_optimizer_result_row_missing:{item}" for item in missing_optimizers)
    blockers.extend(row_blockers)
    blockers.extend(_unsafe_claims(evidence, "optimizer_batch_result_evidence"))
    blockers.extend(_non_empty_claims(evidence, "optimizer_batch_result_evidence"))
    blockers.extend(_string_list(evidence.get("blocked_reasons")))
    blockers.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blockers)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blockers: list[str] = []
    if review.get("requested_scope") != P85_SCOPE:
        blockers.append("v5_p85_review_scope_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        blockers.append("v5_p85_review_identity_or_timestamp_missing")
    for field in REQUIRED_REVIEW_ACKS:
        if review.get(field) is not True:
            blockers.append(f"v5_p85_review_ack_missing:{field}")
    blockers.extend(
        f"v5_p85_unsafe_review_approval:{field}"
        for field in UNSAFE_REVIEW_APPROVAL_FIELDS
        if review.get(field) is True
    )
    return _dedupe(blockers)


def _decision(blockers: Sequence[str], review: Mapping[str, Any], review_blockers: Sequence[str]) -> str:
    if blockers or review_blockers:
        return P85_BLOCKED_DECISION
    if not review:
        return P85_HOLD_DECISION
    if review.get("approve_optimizer_batch_result_ingestion_contract") is not True:
        return P85_REJECTED_DECISION
    return P85_READY_DECISION


def _p84_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("debug_failure_archive_contract_ready")
        and summary.get("debug_failure_archive_evidence_recorded")
        and summary.get("debug_failure_archive_review_signed")
        and summary.get("decision") == P84_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and not summary.get("unsafe_claims")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
    )


def _result_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = evidence.get("optimizer_result_rows") or evidence.get("result_rows") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [_result_row(row) for row in rows if isinstance(row, Mapping)]


def _result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_kind": str(row.get("optimizer_kind") or ""),
        "source": _source(row),
        "digest": _digest(row),
        "ready": row.get("ready") is True,
        "parity_summary_ready": row.get("parity_summary_ready") is True,
        "benchmark_summary_ready": row.get("benchmark_summary_ready") is True,
        "state_roundtrip_summary_ready": row.get("state_roundtrip_summary_ready") is True,
        "debug_archive_reference_ready": row.get("debug_archive_reference_ready") is True,
        "result_ingestion_executed": row.get("result_ingestion_executed") is True,
        "optimizer_result_accepted": row.get("optimizer_result_accepted") is True,
        "request_fields_emitted": row.get("request_fields_emitted") is True,
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
            blockers.append(f"v5_p85_optimizer_result_row_source_missing:{kind}")
        if not row.get("digest"):
            blockers.append(f"v5_p85_optimizer_result_row_digest_missing:{kind}")
        for field in ("ready", "parity_summary_ready", "benchmark_summary_ready", "state_roundtrip_summary_ready", "debug_archive_reference_ready"):
            if row.get(field) is not True:
                blockers.append(f"v5_p85_optimizer_result_row_{field}_missing:{kind}")
        for field in ("result_ingestion_executed", "optimizer_result_accepted", "request_fields_emitted"):
            if row.get(field) is True:
                blockers.append(f"v5_p85_optimizer_result_row_unsafe_claim:{kind}:{field}")
        blockers.extend(f"v5_p85_optimizer_result_row_blocker:{kind}:{item}" for item in row.get("blocked_reasons") or [])
    return _dedupe(blockers)


def _unsafe_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p85_unsafe_claim:{label}:{field}" for field in UNSAFE_TRUE_FIELDS if value.get(field) is True]


def _non_empty_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p85_unsafe_claim:{label}:{field}" for field in UNSAFE_NON_EMPTY_FIELDS if bool(value.get(field))]


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": P85_SCOPE,
        "approve_optimizer_batch_result_ingestion_contract": True,
        **{field: True for field in REQUIRED_REVIEW_ACKS},
        **{field: False for field in UNSAFE_REVIEW_APPROVAL_FIELDS},
    }


def _allowed_next_actions(decision: str, blockers: Sequence[str]) -> list[str]:
    if decision == P85_READY_DECISION:
        return ["prepare_optimizer_batch_keep_rollback_review_contract_default_off"]
    if decision == P85_REJECTED_DECISION:
        return ["keep_default_off", "refresh_optimizer_batch_result_evidence"]
    if decision == P85_HOLD_DECISION:
        return ["collect_signed_optimizer_batch_result_review"]
    return ["resolve_blockers", *list(blockers[:6])]


def _recommended_next_step(decision: str, blockers: Sequence[str]) -> str:
    if decision == P85_READY_DECISION:
        return "draft optimizer batch keep/rollback review contract"
    if decision == P85_HOLD_DECISION:
        return "collect signed owner review for optimizer batch result ingestion contract"
    if decision == P85_REJECTED_DECISION:
        return "keep optimizer batch result ingestion default-off and refresh evidence"
    return blockers[0] if blockers else "complete optimizer batch result evidence"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p84-debug-failure-archive-contract")
    parser.add_argument("--optimizer-batch-result-evidence")
    parser.add_argument("--optimizer-batch-result-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_batch_result_ingestion_contract_p85(
        p84_debug_failure_archive_contract=_load_optional(args.p84_debug_failure_archive_contract),
        optimizer_batch_result_evidence=_load_optional(args.optimizer_batch_result_evidence),
        optimizer_batch_result_review=_load_optional(args.optimizer_batch_result_review),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "P85_SCOPE",
    "REQUIRED_REVIEW_ACKS",
    "REQUIRED_SECTIONS",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_batch_result_ingestion_contract_p85",
]
