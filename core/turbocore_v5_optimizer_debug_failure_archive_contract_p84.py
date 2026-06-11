"""Optimizer debug/failure-archive contract for TurboCore V5-P84.

P84 turns public fused-optimizer debugging lessons into a local, default-off
failure archive contract: minimal repro, environment capture, CUDA debug hooks,
state round-trip evidence, and fallback notes. It records evidence only and does
not run validators, kernels, CUDA tools, training, request adapters, or UI.
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
from core.turbocore_v5_optimizer_batch_validation_runner_contract_p83 import (
    P83_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P83_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P83_UNSAFE_TRUE_FIELDS,
)
from core.turbocore_v5_owner_review_evidence_package import load_json


P84_READY_DECISION = "optimizer_debug_failure_archive_contract_p84_recorded_default_off"
P84_BLOCKED_DECISION = "optimizer_debug_failure_archive_contract_p84_blocked_default_off"
P84_HOLD_DECISION = "optimizer_debug_failure_archive_contract_p84_hold_for_signed_review_default_off"
P84_REJECTED_DECISION = "optimizer_debug_failure_archive_contract_p84_rejected_default_off"
P84_SCOPE = "optimizer_debug_failure_archive_contract"
REQUIRED_SECTIONS = (
    "p83_runner_contract_reference",
    "public_debug_lessons_inventory",
    "environment_capture_plan",
    "minimal_repro_plan",
    "cuda_debug_hooks_plan",
    "state_roundtrip_plan",
    "parity_drift_triage_plan",
    "performance_fallback_plan",
    "failure_archive_schema",
    "no_debug_tool_execution_boundary",
    "no_runner_execution_boundary",
    "no_optimizer_kernel_execution_boundary",
    "no_training_launch_boundary",
    "no_request_ui_schema_patch_boundary",
)
REQUIRED_PUBLIC_LESSONS = (
    "bitsandbytes_state_quantization_and_paged_optimizer",
    "apex_or_deepspeed_fused_adam_extension_gate",
    "pytorch_extension_parity_or_gradcheck_reference",
    "cuda_launch_blocking_or_nsight_min_repro_debug",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p83_runner_contract_recorded",
    "acknowledge_public_debug_lessons_captured",
    "acknowledge_minimal_repro_required",
    "acknowledge_environment_capture_required",
    "acknowledge_cuda_debug_hooks_plan_only",
    "acknowledge_no_debug_tool_executed",
    "acknowledge_no_runner_executed",
    "acknowledge_no_optimizer_kernel_executed",
    "acknowledge_no_training_launch",
    "acknowledge_no_request_ui_schema_patch",
    "acknowledge_later_result_ingestion_required",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P83_UNSAFE_TRUE_FIELDS,
            "debug_tool_executed",
            "cuda_launch_blocking_run_executed",
            "nsight_profile_executed",
            "cuda_gdb_executed",
            "minimal_repro_executed",
            "failure_archive_written",
            "debug_result_ingested",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P83_UNSAFE_NON_EMPTY_FIELDS,
            "post_p84_request_fields",
            "debug_tool_command",
            "cuda_debug_execution_payload",
            "failure_archive_payload",
            "minimal_repro_execution_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_v5_optimizer_debug_failure_archive_contract_p84(
    *,
    p83_runner_contract: Mapping[str, Any] | None = None,
    debug_failure_archive_evidence: Mapping[str, Any] | None = None,
    debug_failure_archive_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record optimizer debug/failure-archive evidence without execution."""

    p83 = _as_dict(p83_runner_contract)
    evidence = _as_dict(debug_failure_archive_evidence)
    review = _as_dict(debug_failure_archive_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p83_summary = _p83_summary(p83)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p83_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P84_HOLD_DECISION:
        blockers.append("v5_p84_signed_optimizer_debug_failure_archive_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P84_READY_DECISION
    rejected = decision == P84_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_optimizer_debug_failure_archive_contract_p84_v0",
        "gate": "v5_optimizer_debug_failure_archive_contract_p84",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "debug_failure_archive_review_recorded": decision_record_ready,
        "debug_failure_archive_review_signed": decision_record_ready,
        "debug_failure_archive_contract_ready": ready,
        "debug_failure_archive_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p84_request_fields": {},
        "p83_runner_contract_summary": p83_summary,
        "debug_failure_archive_evidence_summary": evidence_summary,
        "debug_failure_archive_review": review_summary,
        "debug_failure_archive_review_template": _review_template(),
        "progress_gates": _progress_gates(p83_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P84 records optimizer debugging and failure-archive evidence only.",
            "P84 captures public fused-optimizer lessons without executing tools.",
            "A later result-ingestion contract is still required.",
        ],
    }


def _p83_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "runner_contract_ready": report.get("optimizer_batch_validation_runner_contract_ready") is True,
        "runner_manifest_recorded": report.get("optimizer_batch_validation_runner_manifest_recorded") is True,
        "runner_review_signed": report.get("optimizer_batch_validation_runner_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p83_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p83"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = set(_string_list(evidence.get("available_sections"))) | set(str(key) for key in evidence.keys())
    missing_sections = [item for item in REQUIRED_SECTIONS if item not in sections]
    lessons = _string_list(evidence.get("public_debug_lessons"))
    missing_lessons = [item for item in REQUIRED_PUBLIC_LESSONS if item not in lessons]
    blockers = _evidence_blocker_list(evidence, missing_sections, missing_lessons)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "debug_failure_archive_contract_ready": evidence.get("debug_failure_archive_contract_ready") is True,
        "public_debug_lessons_captured": evidence.get("public_debug_lessons_captured") is True,
        "minimal_repro_plan_ready": evidence.get("minimal_repro_plan_ready") is True,
        "environment_capture_plan_ready": evidence.get("environment_capture_plan_ready") is True,
        "cuda_debug_hooks_plan_ready": evidence.get("cuda_debug_hooks_plan_ready") is True,
        "failure_archive_schema_ready": evidence.get("failure_archive_schema_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "result_ingestion_contract_required": evidence.get("result_ingestion_contract_required") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "public_debug_lessons": lessons,
        "missing_public_debug_lessons": missing_lessons,
        "missing_sections": missing_sections,
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_debug_failure_archive_contract": review.get("approve_debug_failure_archive_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p83: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p83_runner_contract_ready": _p83_ready(p83),
        "debug_failure_archive_evidence_ready": bool(evidence.get("ready", False)),
        "signed_debug_failure_archive_review_present": bool(review.get("present", False)),
        "public_debug_lessons_complete": not bool(evidence.get("missing_public_debug_lessons")),
    }


def _evidence_blockers(
    p83: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_events: Sequence[str],
    rollback_events: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not _p83_ready(p83):
        blockers.append("v5_p84_p83_optimizer_batch_validation_runner_contract_not_ready")
    blockers.extend(f"v5_p84_unsafe_upstream_claim:p83:{item}" for item in p83.get("unsafe_claims") or [])
    if not evidence.get("present"):
        blockers.append("v5_p84_debug_failure_archive_evidence_missing")
    blockers.extend(_string_list(evidence.get("blockers")))
    if failure_events:
        blockers.extend(f"v5_p84_failure_history_not_clear:{item}" for item in failure_events)
    if rollback_events:
        blockers.extend(f"v5_p84_rollback_history_not_clear:{item}" for item in rollback_events)
    return _dedupe(blockers)


def _evidence_blocker_list(
    evidence: Mapping[str, Any],
    missing_sections: Sequence[str],
    missing_lessons: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    checks = (
        ("ok", True, "evidence_not_ok"),
        ("debug_failure_archive_contract_ready", True, "not_ready"),
        ("public_debug_lessons_captured", True, "public_debug_lessons_missing"),
        ("minimal_repro_plan_ready", True, "minimal_repro_plan_missing"),
        ("environment_capture_plan_ready", True, "environment_capture_plan_missing"),
        ("cuda_debug_hooks_plan_ready", True, "cuda_debug_hooks_plan_missing"),
        ("failure_archive_schema_ready", True, "failure_archive_schema_missing"),
        ("report_only", True, "report_only_missing"),
        ("boundary_only", True, "boundary_only_missing"),
        ("contract_only", True, "contract_only_missing"),
        ("records_evidence_only", True, "records_evidence_only_missing"),
        ("manual_only", True, "manual_only_missing"),
        ("internal_only", True, "internal_only_missing"),
        ("result_ingestion_contract_required", True, "result_ingestion_contract_missing"),
    )
    for field, expected, reason in checks:
        if evidence.get(field) is not expected:
            blockers.append(f"v5_p84_debug_failure_archive_evidence_{reason}")
    if not evidence.get("default_off") or not _default_off_confirmed(evidence):
        blockers.append("v5_p84_debug_failure_archive_default_off_violation")
    if not evidence.get("request_adapter_off") or not _request_adapter_off(evidence):
        blockers.append("v5_p84_debug_failure_archive_request_adapter_boundary_violation")
    if not _source(evidence):
        blockers.append("v5_p84_debug_failure_archive_source_missing")
    if not _digest(evidence):
        blockers.append("v5_p84_debug_failure_archive_digest_missing")
    blockers.extend(f"v5_p84_required_section_missing:{item}" for item in missing_sections)
    blockers.extend(f"v5_p84_public_debug_lesson_missing:{item}" for item in missing_lessons)
    blockers.extend(_unsafe_claims(evidence, "debug_failure_archive_evidence"))
    blockers.extend(_non_empty_claims(evidence, "debug_failure_archive_evidence"))
    blockers.extend(_string_list(evidence.get("blocked_reasons")))
    blockers.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blockers)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blockers: list[str] = []
    if review.get("requested_scope") != P84_SCOPE:
        blockers.append("v5_p84_review_scope_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        blockers.append("v5_p84_review_identity_or_timestamp_missing")
    for field in REQUIRED_REVIEW_ACKS:
        if review.get(field) is not True:
            blockers.append(f"v5_p84_review_ack_missing:{field}")
    blockers.extend(
        f"v5_p84_unsafe_review_approval:{field}"
        for field in UNSAFE_REVIEW_APPROVAL_FIELDS
        if review.get(field) is True
    )
    return _dedupe(blockers)


def _decision(blockers: Sequence[str], review: Mapping[str, Any], review_blockers: Sequence[str]) -> str:
    if blockers or review_blockers:
        return P84_BLOCKED_DECISION
    if not review:
        return P84_HOLD_DECISION
    if review.get("approve_debug_failure_archive_contract") is not True:
        return P84_REJECTED_DECISION
    return P84_READY_DECISION


def _p83_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("runner_contract_ready")
        and summary.get("runner_manifest_recorded")
        and summary.get("runner_review_signed")
        and summary.get("decision") == P83_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and not summary.get("unsafe_claims")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
    )


def _unsafe_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p84_unsafe_claim:{label}:{field}" for field in UNSAFE_TRUE_FIELDS if value.get(field) is True]


def _non_empty_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p84_unsafe_claim:{label}:{field}" for field in UNSAFE_NON_EMPTY_FIELDS if bool(value.get(field))]


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": P84_SCOPE,
        "approve_debug_failure_archive_contract": True,
        **{field: True for field in REQUIRED_REVIEW_ACKS},
        **{field: False for field in UNSAFE_REVIEW_APPROVAL_FIELDS},
    }


def _allowed_next_actions(decision: str, blockers: Sequence[str]) -> list[str]:
    if decision == P84_READY_DECISION:
        return ["prepare_optimizer_batch_validation_result_ingestion_contract_default_off"]
    if decision == P84_REJECTED_DECISION:
        return ["keep_default_off", "refresh_debug_failure_archive_evidence"]
    if decision == P84_HOLD_DECISION:
        return ["collect_signed_debug_failure_archive_review"]
    return ["resolve_blockers", *list(blockers[:6])]


def _recommended_next_step(decision: str, blockers: Sequence[str]) -> str:
    if decision == P84_READY_DECISION:
        return "draft optimizer batch-validation result ingestion contract"
    if decision == P84_HOLD_DECISION:
        return "collect signed owner review for debug failure archive contract"
    if decision == P84_REJECTED_DECISION:
        return "keep debug failure archive default-off and refresh evidence"
    return blockers[0] if blockers else "complete debug failure archive evidence"


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p83-runner-contract")
    parser.add_argument("--debug-failure-archive-evidence")
    parser.add_argument("--debug-failure-archive-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_debug_failure_archive_contract_p84(
        p83_runner_contract=_load_optional(args.p83_runner_contract),
        debug_failure_archive_evidence=_load_optional(args.debug_failure_archive_evidence),
        debug_failure_archive_review=_load_optional(args.debug_failure_archive_review),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "P84_SCOPE",
    "REQUIRED_PUBLIC_LESSONS",
    "REQUIRED_REVIEW_ACKS",
    "REQUIRED_SECTIONS",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_debug_failure_archive_contract_p84",
]
