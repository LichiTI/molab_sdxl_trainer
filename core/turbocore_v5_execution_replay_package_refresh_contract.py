"""Execution replay package refresh contract for TurboCore V5-P56.

P56 records future execution/replay package refresh evidence after P55. It checks
manifest, digest, freshness, and precondition evidence only; it does not refresh
runtime state, load artifacts, execute replay, dispatch native work, launch
kernels, run parity, emit request fields, expose UI, or launch training.
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

from core.turbocore_v5_artifact_replay_verifier_contract import (
    P55_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P55_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P55_UNSAFE_TRUE_FIELDS,
)
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


P56_READY_DECISION = "execution_replay_package_refresh_recorded_default_off"
P56_BLOCKED_DECISION = "execution_replay_package_refresh_blocked_default_off"
P56_HOLD_DECISION = "execution_replay_package_refresh_hold_for_signed_review_default_off"
P56_REJECTED_DECISION = "execution_replay_package_refresh_rejected_default_off"
P56_SCOPE = "execution_replay_package_refresh_contract"
DEFAULT_REQUIRED_SECTIONS = (
    "p55_artifact_replay_verifier_reference", "execution_replay_package_manifest",
    "execution_replay_digest_inventory", "execution_replay_precondition_inventory",
    "execution_replay_freshness_inventory", "execution_replay_digest_comparison",
    "execution_replay_refresh_boundary", "request_adapter_boundary", "no_execution_replay_boundary",
    "no_artifact_load_boundary", "no_native_execution_boundary", "no_native_dispatch_boundary",
    "no_kernel_launch_boundary", "no_parity_execution_boundary", "no_training_step_boundary",
    "no_request_fields_boundary", "no_training_launch_boundary", "rollback_policy", "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p55_artifact_replay_verifier_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure",
    "acknowledge_no_execution_replay_executed", "acknowledge_no_artifact_replay_executed",
    "acknowledge_no_artifact_loaded", "acknowledge_no_native_dispatch_executed",
    "acknowledge_no_kernel_launch_executed", "acknowledge_no_parity_executed",
    "acknowledge_no_training_step_executed", "acknowledge_no_request_adapter_enabled",
    "acknowledge_no_request_fields_emitted", "acknowledge_no_default_or_auto_rollout",
    "acknowledge_execution_replay_package_refresh_evidence_replayable",
    "acknowledge_later_execution_readiness_review_required", "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P55_UNSAFE_TRUE_FIELDS,
            "execution_replay_allowed", "execution_replay_enabled", "execution_replay_executed",
            "execution_replay_package_refresh_allowed", "execution_replay_package_refresh_enabled",
            "execution_replay_package_refresh_executed", "execution_replay_package_refreshed",
            "execution_replay_package_registered", "execution_replay_package_applied",
            "execution_readiness_approved", "runtime_state_refreshed",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P55_UNSAFE_NON_EMPTY_FIELDS,
            "post_p56_request_fields", "post_execution_replay_package_refresh_fields",
            "execution_replay_refresh_request", "execution_replay_refresh_payload",
            "execution_replay_request", "execution_replay_payload", "execution_readiness_request",
            "execution_readiness_payload", "runtime_state_refresh_patch",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(
    f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed"
)
P55_REQUIRED_FALSE_FIELDS = _P55_UNSAFE_TRUE_FIELDS


def build_v5_execution_replay_package_refresh_contract(
    *,
    p55_artifact_replay_verifier: Mapping[str, Any] | None = None,
    execution_replay_package: Mapping[str, Any] | None = None,
    execution_replay_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record execution/replay package refresh evidence without executing it."""

    p55 = _as_dict(p55_artifact_replay_verifier)
    package = _as_dict(execution_replay_package)
    review = _as_dict(execution_replay_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p55_summary = _p55_summary(p55)
    package_summary = _package_summary(package)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p55_summary, package_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P56_HOLD_DECISION:
        blockers.append("v5_p56_signed_execution_replay_refresh_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P56_READY_DECISION
    rejected = decision == P56_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_execution_replay_package_refresh_contract_v0",
        "gate": "v5_execution_replay_package_refresh_contract",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "execution_replay_refresh_review_recorded": decision_record_ready,
        "execution_replay_refresh_review_signed": decision_record_ready,
        "execution_replay_package_refresh_contract_ready": ready,
        "execution_replay_package_refresh_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p56_request_fields": {},
        "p55_artifact_replay_verifier_summary": p55_summary,
        "execution_replay_package_summary": package_summary,
        "execution_replay_digest_comparisons": package_summary.get("digest_comparisons", []),
        "execution_replay_review": review_summary,
        "execution_replay_review_template": _review_template(),
        "progress_gates": _progress_gates(p55_summary, package_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P56 records execution/replay package refresh evidence only.",
            "P56 does not refresh runtime state, load artifacts, execute replay, dispatch native work, launch kernels, run parity, emit request fields, or launch training.",
            "A later execution readiness review is still required before runtime behavior can become active.",
        ],
    }


def _p55_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "artifact_replay_verifier_contract_ready": report.get("artifact_replay_verifier_contract_ready") is True,
        "artifact_replay_package_evidence_recorded": report.get("artifact_replay_package_evidence_recorded") is True,
        "artifact_replay_review_signed": report.get("artifact_replay_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P55_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p55_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p55"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _package_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(package.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(package)]
    manifest_rows = _rows(package, "execution_replay_package_manifest", "execution_replay_manifest")
    digest_rows = _rows(package, "execution_replay_digest_inventory", "replay_digest_inventory")
    precondition_rows = _rows(package, "execution_replay_precondition_inventory", "replay_precondition_inventory")
    freshness_rows = _rows(package, "execution_replay_freshness_inventory", "refresh_freshness_inventory")
    comparisons = _digest_comparisons(manifest_rows, digest_rows)
    blockers = _package_blocker_list(package, missing_sections, manifest_rows, digest_rows, precondition_rows, freshness_rows, comparisons)
    return {
        "present": bool(package),
        "package_id": str(package.get("package_id") or package.get("evidence_id") or package.get("id") or ""),
        "package_version": str(package.get("package_version") or package.get("evidence_version") or ""),
        "ok": package.get("ok") is True,
        "ready": not blockers,
        "execution_replay_package_refresh_contract_ready": package.get(
            "execution_replay_package_refresh_contract_ready"
        ) is True,
        "report_only": package.get("report_only") is True,
        "boundary_only": package.get("boundary_only") is True,
        "contract_only": package.get("contract_only") is True,
        "execution_replay_package_refresh_only": package.get("execution_replay_package_refresh_only") is True,
        "records_evidence_only": package.get("records_evidence_only") is True,
        "manual_only": package.get("manual_only") is True,
        "internal_only": package.get("internal_only") is True,
        "requires_later_execution_readiness_review": package.get("requires_later_execution_readiness_review") is True,
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
        "freshness_count": len(freshness_rows),
        "digest_comparisons": comparisons,
        "blocked_reasons": _string_list(package.get("blocked_reasons")),
        "promotion_blockers": _string_list(package.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(package, "execution_replay_package"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_execution_replay_package_refresh_contract": review.get(
            "approve_execution_replay_package_refresh_contract"
        ) is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p55: Mapping[str, Any], package: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p55_artifact_replay_verifier_present": bool(p55.get("present", False)),
        "p55_artifact_replay_verifier_ready": _p55_ready(p55),
        "execution_replay_package_present": bool(package.get("present", False)),
        "execution_replay_package_ready": bool(package.get("ready", False)),
        "execution_replay_digest_comparisons_ready": all(
            bool(item.get("match", False)) for item in package.get("digest_comparisons", [])
        ),
        "signed_execution_replay_review_present": bool(review.get("present", False)),
        "requested_scope_valid": review.get("requested_scope") == P56_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P56_BLOCKED_DECISION
    if not review:
        return P56_HOLD_DECISION
    if review_blockers:
        return P56_BLOCKED_DECISION
    if review.get("approve_execution_replay_package_refresh_contract") is True:
        return P56_READY_DECISION
    return P56_REJECTED_DECISION


def _evidence_blockers(
    p55_summary: Mapping[str, Any],
    package_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p55_summary.get("present", False)):
        blocked.append("v5_p56_p55_artifact_replay_verifier_missing")
    elif not _p55_ready(p55_summary):
        blocked.append("v5_p56_p55_artifact_replay_verifier_not_ready")
        blocked.extend(_string_list(p55_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p55_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p55_summary.get("unsafe_claims")))
    if not bool(package_summary.get("present", False)):
        blocked.append("v5_p56_execution_replay_package_missing")
    elif not bool(package_summary.get("ready", False)):
        blocked.extend(_string_list(package_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p56_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p56_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p56_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p56_reviewed_at_missing")
    if review.get("requested_scope") != P56_SCOPE:
        blocked.append("v5_p56_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p56_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p56_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p55_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("artifact_replay_verifier_contract_ready")
        and summary.get("artifact_replay_package_evidence_recorded")
        and summary.get("artifact_replay_review_signed")
        and summary.get("decision") == P55_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P55_REQUIRED_FALSE_FIELDS)
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
    freshness_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if package.get("ok") is not True:
        blocked.append("v5_p56_execution_replay_package_not_ok")
    if package.get("execution_replay_package_refresh_contract_ready") is not True:
        blocked.append("v5_p56_execution_replay_package_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "execution_replay_package_refresh_only",
        "records_evidence_only", "manual_only", "internal_only", "requires_later_execution_readiness_review",
        "requires_explicit_owner_approval", "requires_explicit_operator_opt_in",
    ):
        if package.get(field) is not True:
            blocked.append(f"v5_p56_execution_replay_package_{field}_missing")
    if package.get("default_off") is not True or not _default_off_confirmed(package):
        blocked.append("v5_p56_execution_replay_package_default_off_violation")
    if package.get("request_adapter_off") is not True or not _request_adapter_off(package):
        blocked.append("v5_p56_execution_replay_package_request_adapter_violation")
    if not _digest(package):
        blocked.append("v5_p56_execution_replay_package_digest_missing")
    if not _source(package):
        blocked.append("v5_p56_execution_replay_package_source_missing")
    for section in missing_sections:
        blocked.append(f"v5_p56_execution_replay_package_section_missing:{section}")
    blocked.extend(_manifest_blockers(manifest_rows))
    blocked.extend(_digest_inventory_blockers(digest_rows))
    blocked.extend(_precondition_blockers(precondition_rows))
    blocked.extend(_freshness_blockers(freshness_rows))
    for item in comparisons:
        if not bool(item.get("match", False)):
            blocked.append(str(item.get("reason") or "v5_p56_execution_replay_digest_comparison_failed"))
    blocked.extend(_unsafe_claims(package, "execution_replay_package"))
    blocked.extend(_string_list(package.get("blocked_reasons")))
    blocked.extend(_string_list(package.get("promotion_blockers")))
    return _dedupe(blocked)


def _manifest_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p56_execution_replay_manifest_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = _row_id(row, index, "execution_replay")
        if not _digest(row):
            blocked.append(f"v5_p56_execution_replay_manifest_digest_missing:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p56_execution_replay_manifest_source_missing:{item_id}")
        for field in (
            "execution_replay_executed", "execution_replay_package_refreshed", "runtime_state_refreshed",
            "artifact_replay_executed", "artifact_loaded", "native_dispatch_executed", "kernel_launch_executed",
            "parity_check_executed", "training_step_executed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p56_execution_replay_manifest_claim:{item_id}:{field}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _digest_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p56_execution_replay_digest_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = _row_id(row, index, "digest")
        if not _digest(row):
            blocked.append(f"v5_p56_execution_replay_digest_missing:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p56_execution_replay_digest_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _precondition_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p56_execution_replay_precondition_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("precondition_id") or row.get("id") or f"precondition_{index}")
        if row.get("ready") is not True:
            blocked.append(f"v5_p56_execution_replay_precondition_not_ready:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p56_execution_replay_precondition_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _freshness_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p56_execution_replay_freshness_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("refresh_id") or row.get("id") or f"freshness_{index}")
        if row.get("fresh") is not True:
            blocked.append(f"v5_p56_execution_replay_package_not_fresh:{item_id}")
        if row.get("expires_review_before_execution") is not True:
            blocked.append(f"v5_p56_execution_replay_expiry_review_missing:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p56_execution_replay_freshness_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _digest_comparisons(manifest_rows: list[Mapping[str, Any]], digest_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    manifest = {_row_id(row, index, "execution_replay"): row for index, row in enumerate(manifest_rows)}
    digests = {_row_id(row, index, "digest"): row for index, row in enumerate(digest_rows)}
    comparisons: list[dict[str, Any]] = []
    for item_id in sorted(set(manifest) | set(digests)):
        manifest_row = _as_dict(manifest.get(item_id))
        digest_row = _as_dict(digests.get(item_id))
        manifest_digest = _digest(manifest_row)
        inventory_digest = _digest(digest_row)
        reason = ""
        if not manifest_row:
            reason = f"v5_p56_execution_replay_manifest_entry_missing:{item_id}"
        elif not digest_row:
            reason = f"v5_p56_execution_replay_digest_inventory_entry_missing:{item_id}"
        elif not manifest_digest or not inventory_digest or manifest_digest != inventory_digest:
            reason = f"v5_p56_execution_replay_digest_mismatch:{item_id}"
        comparisons.append({"package_id": item_id, "match": not bool(reason), "reason": reason})
    return comparisons


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p56_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p56_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _rows(payload: Mapping[str, Any], field: str, fallback: str) -> list[dict[str, Any]]:
    value = payload.get(field, payload.get(fallback))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _row_id(row: Mapping[str, Any], index: int, fallback: str) -> str:
    return str(row.get("package_id") or row.get("execution_replay_id") or row.get("artifact_id") or row.get("id") or f"{fallback}_{index}")


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    return {str(item).strip() for item in sections if str(item).strip()}


def _review_template() -> dict[str, Any]:
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P56_SCOPE, "approve_execution_replay_package_refresh_contract": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P56_READY_DECISION:
        return ["archive_p56_execution_replay_package_refresh_contract"]
    if decision == P56_REJECTED_DECISION:
        return ["record_p56_default_off_rejection_or_repair_execution_replay_package"]
    if decision == P56_HOLD_DECISION:
        return ["collect_signed_execution_replay_refresh_review"]
    if any("p55" in item for item in blockers):
        return ["repair_p55_artifact_replay_verifier_contract"]
    if any("execution_replay" in item or "digest" in item for item in blockers):
        return ["repair_execution_replay_package_refresh_evidence"]
    return ["clear_failure_or_rollback_history_before_p56_contract"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P56_READY_DECISION:
        return "archive P56 contract; execution readiness review still requires a later explicit contract"
    if decision == P56_REJECTED_DECISION:
        return "record the signed rejection and keep execution replay default-off for repair"
    if decision == P56_HOLD_DECISION:
        return "collect a signed execution replay package refresh review over P55 evidence"
    if any("p55" in item for item in blockers):
        return "repair the P55 artifact replay verifier contract before P56"
    if any("digest" in item or "manifest" in item for item in blockers):
        return "repair execution replay manifest and digest inventory evidence"
    return "hold P56 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P56 execution replay package refresh contract.")
    parser.add_argument("--p55-artifact-replay-verifier", default="", help="P55 artifact replay verifier JSON.")
    parser.add_argument("--execution-replay-package", default="", help="P56 execution replay package evidence JSON.")
    parser.add_argument("--execution-replay-review", default="", help="Signed P56 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_execution_replay_package_refresh_contract(
        p55_artifact_replay_verifier=load_json(args.p55_artifact_replay_verifier)
        if args.p55_artifact_replay_verifier else None,
        execution_replay_package=load_json(args.execution_replay_package) if args.execution_replay_package else None,
        execution_replay_review=load_json(args.execution_replay_review) if args.execution_replay_review else None,
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


__all__ = ["build_v5_execution_replay_package_refresh_contract"]
