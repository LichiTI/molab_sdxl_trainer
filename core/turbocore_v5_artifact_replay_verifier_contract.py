"""Artifact replay verifier contract for TurboCore V5-P55.

P55 records future artifact replay package evidence after P54. It verifies
manifest/digest/precondition evidence only; it does not execute replay, load
artifacts, dispatch native work, launch kernels, run parity, emit request
fields, expose UI, or launch training.
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
from core.turbocore_v5_native_dry_run_result_ingestion_contract import (
    P54_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P54_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P54_UNSAFE_TRUE_FIELDS,
)
from core.turbocore_v5_owner_review_evidence_package import load_json


P55_READY_DECISION = "artifact_replay_verifier_recorded_default_off"
P55_BLOCKED_DECISION = "artifact_replay_verifier_blocked_default_off"
P55_HOLD_DECISION = "artifact_replay_verifier_hold_for_signed_review_default_off"
P55_REJECTED_DECISION = "artifact_replay_verifier_rejected_default_off"
P55_SCOPE = "artifact_replay_verifier_contract"
DEFAULT_REQUIRED_SECTIONS = (
    "p54_native_dry_run_result_ingestion_reference", "artifact_replay_manifest",
    "artifact_replay_digest_inventory", "artifact_replay_precondition_inventory",
    "artifact_replay_digest_comparison", "artifact_replay_boundary", "request_adapter_boundary",
    "no_artifact_replay_execution_boundary", "no_native_execution_boundary", "no_native_dispatch_boundary",
    "no_kernel_launch_boundary", "no_parity_execution_boundary", "no_training_step_boundary",
    "no_request_fields_boundary", "no_training_launch_boundary", "rollback_policy", "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p54_native_dry_run_result_ingestion_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure", "acknowledge_no_artifact_replay_executed",
    "acknowledge_no_artifact_loaded", "acknowledge_no_native_dispatch_executed",
    "acknowledge_no_kernel_launch_executed", "acknowledge_no_parity_executed",
    "acknowledge_no_training_step_executed", "acknowledge_no_request_adapter_enabled",
    "acknowledge_no_request_fields_emitted", "acknowledge_no_default_or_auto_rollout",
    "acknowledge_artifact_replay_package_evidence_replayable",
    "acknowledge_later_execution_or_replay_contract_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (*_P54_UNSAFE_TRUE_FIELDS, "artifact_replay_verifier_enabled", "artifact_replay_verifier_executed",
         "artifact_replay_package_verified", "artifact_replay_package_applied", "execution_replay_package_applied")
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (*_P54_UNSAFE_NON_EMPTY_FIELDS, "post_p55_request_fields", "post_artifact_replay_verifier_fields",
         "artifact_replay_verifier_request", "artifact_replay_verifier_payload",
         "execution_replay_package_request", "execution_replay_package_payload")
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(
    f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed"
)
P54_REQUIRED_FALSE_FIELDS = _P54_UNSAFE_TRUE_FIELDS


def build_v5_artifact_replay_verifier_contract(
    *,
    p54_native_dry_run_result_ingestion: Mapping[str, Any] | None = None,
    artifact_replay_package: Mapping[str, Any] | None = None,
    artifact_replay_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record artifact replay verifier evidence without executing replay."""

    p54 = _as_dict(p54_native_dry_run_result_ingestion)
    package = _as_dict(artifact_replay_package)
    review = _as_dict(artifact_replay_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p54_summary = _p54_summary(p54)
    package_summary = _package_summary(package)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p54_summary, package_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P55_HOLD_DECISION:
        blockers.append("v5_p55_signed_artifact_replay_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P55_READY_DECISION
    rejected = decision == P55_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_artifact_replay_verifier_contract_v0",
        "gate": "v5_artifact_replay_verifier_contract",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "artifact_replay_review_recorded": decision_record_ready,
        "artifact_replay_review_signed": decision_record_ready,
        "artifact_replay_verifier_contract_ready": ready,
        "artifact_replay_package_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p55_request_fields": {},
        "p54_native_dry_run_result_ingestion_summary": p54_summary,
        "artifact_replay_package_summary": package_summary,
        "artifact_replay_digest_comparisons": package_summary.get("digest_comparisons", []),
        "artifact_replay_review": review_summary,
        "artifact_replay_review_template": _review_template(),
        "progress_gates": _progress_gates(p54_summary, package_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P55 records artifact replay package verifier evidence only.",
            "P55 does not load artifacts, execute replay, dispatch native work, launch kernels, run parity, emit request fields, or launch training.",
            "A later execution or replay contract is still required before runtime behavior can become active.",
        ],
    }


def _p54_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "native_dry_run_result_ingestion_contract_ready": report.get(
            "native_dry_run_result_ingestion_contract_ready"
        ) is True,
        "native_dry_run_result_ingestion_evidence_recorded": report.get(
            "native_dry_run_result_ingestion_evidence_recorded"
        ) is True,
        "native_dry_run_result_review_signed": report.get("native_dry_run_result_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P54_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p54_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p54"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _package_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(package.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(package)]
    manifest_rows = _rows(package, "artifact_replay_manifest", "replay_manifest")
    digest_rows = _rows(package, "artifact_replay_digest_inventory", "replay_digest_inventory")
    precondition_rows = _rows(package, "artifact_replay_precondition_inventory", "replay_precondition_inventory")
    comparisons = _digest_comparisons(manifest_rows, digest_rows)
    blockers = _package_blocker_list(package, missing_sections, manifest_rows, digest_rows, precondition_rows, comparisons)
    return {
        "present": bool(package),
        "package_id": str(package.get("package_id") or package.get("evidence_id") or package.get("id") or ""),
        "package_version": str(package.get("package_version") or package.get("evidence_version") or ""),
        "ok": package.get("ok") is True,
        "ready": not blockers,
        "artifact_replay_verifier_contract_ready": package.get("artifact_replay_verifier_contract_ready") is True,
        "report_only": package.get("report_only") is True,
        "boundary_only": package.get("boundary_only") is True,
        "contract_only": package.get("contract_only") is True,
        "artifact_replay_verifier_only": package.get("artifact_replay_verifier_only") is True,
        "records_evidence_only": package.get("records_evidence_only") is True,
        "manual_only": package.get("manual_only") is True,
        "internal_only": package.get("internal_only") is True,
        "requires_later_execution_or_replay_contract": package.get(
            "requires_later_execution_or_replay_contract"
        ) is True,
        "requires_explicit_owner_approval": package.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": package.get("requires_explicit_operator_opt_in") is True,
        "default_off": package.get("default_off") is True and _default_off_confirmed(package),
        "request_adapter_off": package.get("request_adapter_off") is True and _request_adapter_off(package),
        "digest": _digest(package),
        "source": _source(package),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "manifest_count": len(manifest_rows),
        "digest_inventory_count": len(digest_rows),
        "precondition_count": len(precondition_rows),
        "digest_comparisons": comparisons,
        "blocked_reasons": _string_list(package.get("blocked_reasons")),
        "promotion_blockers": _string_list(package.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(package, "artifact_replay_package"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_artifact_replay_verifier_contract": review.get("approve_artifact_replay_verifier_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p54: Mapping[str, Any], package: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p54_native_dry_run_result_ingestion_present": bool(p54.get("present", False)),
        "p54_native_dry_run_result_ingestion_ready": _p54_ready(p54),
        "artifact_replay_package_present": bool(package.get("present", False)),
        "artifact_replay_package_ready": bool(package.get("ready", False)),
        "artifact_replay_digest_comparisons_ready": all(
            bool(item.get("match", False)) for item in package.get("digest_comparisons", [])
        ),
        "signed_artifact_replay_review_present": bool(review.get("present", False)),
        "reviewer_present": bool(review.get("reviewer")),
        "reviewed_at_present": bool(review.get("reviewed_at")),
        "requested_scope_valid": review.get("requested_scope") == P55_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P55_BLOCKED_DECISION
    if not review:
        return P55_HOLD_DECISION
    if review_blockers:
        return P55_BLOCKED_DECISION
    if review.get("approve_artifact_replay_verifier_contract") is True:
        return P55_READY_DECISION
    return P55_REJECTED_DECISION


def _evidence_blockers(
    p54_summary: Mapping[str, Any],
    package_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p54_summary.get("present", False)):
        blocked.append("v5_p55_p54_native_dry_run_result_ingestion_missing")
    elif not _p54_ready(p54_summary):
        blocked.append("v5_p55_p54_native_dry_run_result_ingestion_not_ready")
        blocked.extend(_string_list(p54_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p54_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p54_summary.get("unsafe_claims")))
    if not bool(package_summary.get("present", False)):
        blocked.append("v5_p55_artifact_replay_package_missing")
    elif not bool(package_summary.get("ready", False)):
        blocked.extend(_string_list(package_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p55_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p55_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p55_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p55_reviewed_at_missing")
    if review.get("requested_scope") != P55_SCOPE:
        blocked.append("v5_p55_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p55_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p55_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p54_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("native_dry_run_result_ingestion_contract_ready")
        and summary.get("native_dry_run_result_ingestion_evidence_recorded")
        and summary.get("native_dry_run_result_review_signed")
        and summary.get("decision") == P54_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P54_REQUIRED_FALSE_FIELDS)
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
        and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers"))
        and not _string_list(summary.get("unsafe_claims"))
    )


def _package_blocker_list(
    package: Mapping[str, Any],
    missing_sections: list[str],
    manifest_rows: list[dict[str, Any]],
    digest_rows: list[dict[str, Any]],
    precondition_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if package.get("ok") is not True:
        blocked.append("v5_p55_artifact_replay_package_not_ok")
    if package.get("artifact_replay_verifier_contract_ready") is not True:
        blocked.append("v5_p55_artifact_replay_package_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "artifact_replay_verifier_only",
        "records_evidence_only", "manual_only", "internal_only",
        "requires_later_execution_or_replay_contract", "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if package.get(field) is not True:
            blocked.append(f"v5_p55_artifact_replay_package_{field}_missing")
    if package.get("default_off") is not True or not _default_off_confirmed(package):
        blocked.append("v5_p55_artifact_replay_package_default_off_violation")
    if package.get("request_adapter_off") is not True or not _request_adapter_off(package):
        blocked.append("v5_p55_artifact_replay_package_request_adapter_violation")
    if not _digest(package):
        blocked.append("v5_p55_artifact_replay_package_digest_missing")
    if not _source(package):
        blocked.append("v5_p55_artifact_replay_package_source_missing")
    for section in missing_sections:
        blocked.append(f"v5_p55_artifact_replay_package_section_missing:{section}")
    blocked.extend(_manifest_blockers(manifest_rows))
    blocked.extend(_digest_inventory_blockers(digest_rows))
    blocked.extend(_precondition_blockers(precondition_rows))
    for item in comparisons:
        if not bool(item.get("match", False)):
            blocked.append(str(item.get("reason") or "v5_p55_artifact_replay_digest_comparison_failed"))
    blocked.extend(_unsafe_claims(package, "artifact_replay_package"))
    blocked.extend(_string_list(package.get("blocked_reasons")))
    blocked.extend(_string_list(package.get("promotion_blockers")))
    return _dedupe(blocked)


def _manifest_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p55_artifact_replay_manifest_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = _row_id(row, index, "artifact_replay")
        if not _digest(row):
            blocked.append(f"v5_p55_artifact_replay_manifest_digest_missing:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p55_artifact_replay_manifest_source_missing:{item_id}")
        for field in (
            "artifact_replay_executed", "artifact_loaded", "artifact_replay_package_applied",
            "native_dispatch_executed", "kernel_launch_executed", "parity_check_executed", "training_step_executed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p55_artifact_replay_manifest_claim:{item_id}:{field}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _digest_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p55_artifact_replay_digest_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = _row_id(row, index, "digest")
        if not _digest(row):
            blocked.append(f"v5_p55_artifact_replay_digest_missing:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p55_artifact_replay_digest_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _precondition_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p55_artifact_replay_precondition_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("precondition_id") or row.get("id") or f"precondition_{index}")
        if row.get("ready") is not True:
            blocked.append(f"v5_p55_artifact_replay_precondition_not_ready:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p55_artifact_replay_precondition_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _digest_comparisons(
    manifest_rows: list[Mapping[str, Any]],
    digest_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest = {_row_id(row, index, "artifact_replay"): row for index, row in enumerate(manifest_rows)}
    digests = {_row_id(row, index, "digest"): row for index, row in enumerate(digest_rows)}
    comparisons: list[dict[str, Any]] = []
    for item_id in sorted(set(manifest) | set(digests)):
        manifest_row = _as_dict(manifest.get(item_id))
        digest_row = _as_dict(digests.get(item_id))
        manifest_digest = _digest(manifest_row)
        inventory_digest = _digest(digest_row)
        reason = ""
        if not manifest_row:
            reason = f"v5_p55_artifact_replay_manifest_entry_missing:{item_id}"
        elif not digest_row:
            reason = f"v5_p55_artifact_replay_digest_inventory_entry_missing:{item_id}"
        elif not manifest_digest or not inventory_digest or manifest_digest != inventory_digest:
            reason = f"v5_p55_artifact_replay_digest_mismatch:{item_id}"
        comparisons.append(
            {
                "artifact_id": item_id,
                "manifest_present": bool(manifest_row),
                "digest_inventory_present": bool(digest_row),
                "manifest_sha256": manifest_digest,
                "digest_inventory_sha256": inventory_digest,
                "manifest_source": _source(manifest_row),
                "digest_inventory_source": _source(digest_row),
                "match": not bool(reason),
                "reason": reason,
            }
        )
    return comparisons


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p55_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p55_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _rows(payload: Mapping[str, Any], field: str, fallback: str) -> list[dict[str, Any]]:
    value = payload.get(field, payload.get(fallback))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _row_id(row: Mapping[str, Any], index: int, fallback: str) -> str:
    return str(row.get("artifact_id") or row.get("replay_id") or row.get("package_id") or row.get("id") or f"{fallback}_{index}")


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {str(item).strip() for item in sections if str(item).strip()}


def _review_template() -> dict[str, Any]:
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P55_SCOPE, "approve_artifact_replay_verifier_contract": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P55_READY_DECISION:
        return ["archive_p55_artifact_replay_verifier_contract"]
    if decision == P55_REJECTED_DECISION:
        return ["record_p55_default_off_rejection_or_repair_artifact_replay_package"]
    if decision == P55_HOLD_DECISION:
        return ["collect_signed_artifact_replay_verifier_review"]
    if any("p54" in item for item in blockers):
        return ["repair_p54_native_dry_run_result_ingestion_contract"]
    if any("artifact_replay" in item or "digest" in item for item in blockers):
        return ["repair_artifact_replay_package_evidence"]
    return ["clear_failure_or_rollback_history_before_p55_contract"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P55_READY_DECISION:
        return "archive P55 contract; execution/replay package refresh still requires a later explicit contract"
    if decision == P55_REJECTED_DECISION:
        return "record the signed rejection and keep artifact replay default-off for repair"
    if decision == P55_HOLD_DECISION:
        return "collect a signed artifact replay verifier review over P54 evidence"
    if any("p54" in item for item in blockers):
        return "repair the P54 native dry-run result ingestion contract before P55"
    if any("digest" in item or "manifest" in item for item in blockers):
        return "repair artifact replay manifest and digest inventory evidence"
    return "hold P55 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P55 artifact replay verifier contract.")
    parser.add_argument("--p54-native-dry-run-result-ingestion", default="", help="P54 result ingestion JSON.")
    parser.add_argument("--artifact-replay-package", default="", help="P55 artifact replay package evidence JSON.")
    parser.add_argument("--artifact-replay-review", default="", help="Signed P55 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_artifact_replay_verifier_contract(
        p54_native_dry_run_result_ingestion=load_json(args.p54_native_dry_run_result_ingestion)
        if args.p54_native_dry_run_result_ingestion
        else None,
        artifact_replay_package=load_json(args.artifact_replay_package) if args.artifact_replay_package else None,
        artifact_replay_review=load_json(args.artifact_replay_review) if args.artifact_replay_review else None,
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


__all__ = ["build_v5_artifact_replay_verifier_contract"]
