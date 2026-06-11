"""Optimizer kernel launch contract for TurboCore V5-P92.

P92 records the review shape for a future optimizer kernel launch stage. It
never launches kernels, transfers tensors, runs parity, executes training steps,
emits request fields, patches schemas, exposes UI, or changes rollout behavior.
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
from core.turbocore_v5_optimizer_batch_validation_contract_p82 import DEFAULT_OPTIMIZER_KINDS
from core.turbocore_v5_optimizer_native_dispatch_contract_p91 import (
    P91_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P91_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P91_UNSAFE_TRUE_FIELDS,
)
from core.turbocore_v5_owner_review_evidence_package import load_json


P92_READY_DECISION = "optimizer_kernel_launch_contract_p92_recorded_default_off"
P92_BLOCKED_DECISION = "optimizer_kernel_launch_contract_p92_blocked_default_off"
P92_HOLD_DECISION = "optimizer_kernel_launch_contract_p92_hold_for_signed_review_default_off"
P92_REJECTED_DECISION = "optimizer_kernel_launch_contract_p92_rejected_default_off"
P92_SCOPE = "optimizer_kernel_launch_contract"
ALLOWED_REVIEW_INTENTS = {"kernel_launch_candidate", "hold_for_more_evidence", "reject_kernel_launch"}
REQUIRED_SECTIONS = (
    "p91_optimizer_native_dispatch_contract_reference",
    "optimizer_kernel_launch_package",
    "per_optimizer_kernel_launch_rows",
    "kernel_launch_boundary_inventory",
    "tensor_transfer_boundary_inventory",
    "parity_boundary_inventory",
    "training_step_boundary_inventory",
    "request_schema_router_ui_boundary_inventory",
    "default_rollout_boundary_inventory",
    "rollback_policy_summary",
    "observability_summary",
    "no_kernel_launch_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_parity_execution_boundary",
    "no_training_step_or_launch_boundary",
    "no_request_field_emission_boundary",
    "no_schema_config_router_ui_patch_boundary",
    "no_default_rollout_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p91_optimizer_native_dispatch_contract_recorded",
    "acknowledge_optimizer_kernel_launch_contract_only",
    "acknowledge_no_kernel_launch_executed",
    "acknowledge_no_tensor_transfer_executed",
    "acknowledge_no_parity_executed",
    "acknowledge_no_training_step_or_launch",
    "acknowledge_no_request_fields_emitted",
    "acknowledge_no_schema_config_router_ui_patch",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_later_optimizer_tensor_transfer_contract_required",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P91_UNSAFE_TRUE_FIELDS,
            "optimizer_kernel_launch_applied",
            "optimizer_kernel_launch_enabled",
            "optimizer_kernel_launch_executed",
            "kernel_launch_requested",
            "kernel_launch_submitted",
            "kernel_launch_executed",
            "tensor_transfer_requested",
            "tensor_transfer_executed",
            "parity_execution_requested",
            "parity_execution_executed",
            "training_step_requested",
            "training_step_executed",
            "training_launch_requested",
            "training_launch_executed",
            "request_fields_emitted",
            "schema_config_router_ui_patched",
            "default_rollout_allowed",
            "auto_rollout_allowed",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P91_UNSAFE_NON_EMPTY_FIELDS,
            "post_p92_request_fields",
            "optimizer_kernel_launch_payload",
            "kernel_launch_payload",
            "tensor_transfer_payload",
            "parity_execution_payload",
            "training_step_payload",
            "training_launch_payload",
            "request_field_payload",
            "schema_config_router_ui_patch_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_v5_optimizer_kernel_launch_contract_p92(
    *,
    p91_optimizer_native_dispatch_contract: Mapping[str, Any] | None = None,
    optimizer_kernel_launch_evidence: Mapping[str, Any] | None = None,
    optimizer_kernel_launch_signed_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record optimizer kernel launch evidence without launching kernels."""

    p91 = _as_dict(p91_optimizer_native_dispatch_contract)
    evidence = _as_dict(optimizer_kernel_launch_evidence)
    review = _as_dict(optimizer_kernel_launch_signed_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p91_summary = _p91_summary(p91)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p91_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P92_HOLD_DECISION:
        blockers.append("v5_p92_signed_optimizer_kernel_launch_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P92_READY_DECISION
    rejected = decision == P92_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_optimizer_kernel_launch_contract_p92_v0",
        "gate": "v5_optimizer_kernel_launch_contract_p92",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "optimizer_kernel_launch_recorded": decision_record_ready,
        "optimizer_kernel_launch_signed": decision_record_ready,
        "optimizer_kernel_launch_contract_ready": ready,
        "optimizer_kernel_launch_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p92_request_fields": {},
        "p91_optimizer_native_dispatch_summary": p91_summary,
        "optimizer_kernel_launch_evidence_summary": evidence_summary,
        "optimizer_kernel_launch_signed_review": review_summary,
        "optimizer_kernel_launch_review_template": _review_template(),
        "progress_gates": _progress_gates(p91_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P92 records optimizer kernel launch evidence only.",
            "P92 does not execute kernel launch or tensor transfer.",
            "A later optimizer tensor transfer contract is still required.",
        ],
    }


def _p91_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "optimizer_native_dispatch_contract_ready": report.get("optimizer_native_dispatch_contract_ready") is True,
        "optimizer_native_dispatch_evidence_recorded": report.get("optimizer_native_dispatch_evidence_recorded") is True,
        "optimizer_native_dispatch_signed": report.get("optimizer_native_dispatch_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p91_request_fields"))),
        "unsafe_claims": _unsafe_claims(report, "p91"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = set(_string_list(evidence.get("available_sections"))) | set(str(key) for key in evidence.keys())
    missing_sections = [item for item in REQUIRED_SECTIONS if item not in sections]
    rows = _kernel_rows(evidence)
    missing_optimizers = _missing_optimizers(rows)
    row_blockers = _row_blockers(rows)
    blockers = _evidence_blocker_list(evidence, missing_sections, missing_optimizers, row_blockers)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "optimizer_kernel_launch_package_ready": evidence.get("optimizer_kernel_launch_package_ready") is True,
        "kernel_launch_policy_ready": evidence.get("kernel_launch_policy_ready") is True,
        "later_optimizer_tensor_transfer_contract_required": (
            evidence.get("later_optimizer_tensor_transfer_contract_required") is True
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
        "kernel_rows": rows,
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
        "approve_optimizer_kernel_launch_contract": review.get("approve_optimizer_kernel_launch_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p91: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p91_optimizer_native_dispatch_contract_ready": _p91_ready(p91),
        "optimizer_kernel_launch_evidence_ready": bool(evidence.get("ready", False)),
        "signed_optimizer_kernel_launch_review_present": bool(review.get("present", False)),
        "all_default_optimizer_rows_present": not bool(evidence.get("missing_optimizers")),
    }


def _evidence_blockers(
    p91: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_events: Sequence[str],
    rollback_events: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not _p91_ready(p91):
        blockers.append("v5_p92_p91_optimizer_native_dispatch_contract_not_ready")
    blockers.extend(f"v5_p92_unsafe_upstream_claim:p91:{item}" for item in p91.get("unsafe_claims") or [])
    if not evidence.get("present"):
        blockers.append("v5_p92_optimizer_kernel_launch_evidence_missing")
    blockers.extend(_string_list(evidence.get("blockers")))
    if failure_events:
        blockers.extend(f"v5_p92_failure_history_not_clear:{item}" for item in failure_events)
    if rollback_events:
        blockers.extend(f"v5_p92_rollback_history_not_clear:{item}" for item in rollback_events)
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
        ("optimizer_kernel_launch_package_ready", True, "package_not_ready"),
        ("kernel_launch_policy_ready", True, "policy_not_ready"),
        ("later_optimizer_tensor_transfer_contract_required", True, "later_tensor_transfer_contract_missing"),
        ("report_only", True, "report_only_missing"),
        ("boundary_only", True, "boundary_only_missing"),
        ("contract_only", True, "contract_only_missing"),
        ("records_evidence_only", True, "records_evidence_only_missing"),
        ("manual_only", True, "manual_only_missing"),
        ("internal_only", True, "internal_only_missing"),
    )
    for field, expected, reason in checks:
        if evidence.get(field) is not expected:
            blockers.append(f"v5_p92_optimizer_kernel_launch_evidence_{reason}")
    if str(evidence.get("review_intent") or "") not in ALLOWED_REVIEW_INTENTS:
        blockers.append("v5_p92_optimizer_kernel_launch_review_intent_invalid")
    if not evidence.get("default_off") or not _default_off_confirmed(evidence):
        blockers.append("v5_p92_optimizer_kernel_launch_evidence_default_off_violation")
    if not evidence.get("request_adapter_off") or not _request_adapter_off(evidence):
        blockers.append("v5_p92_optimizer_kernel_launch_evidence_request_adapter_boundary_violation")
    if not _source(evidence):
        blockers.append("v5_p92_optimizer_kernel_launch_evidence_source_missing")
    if not _digest(evidence):
        blockers.append("v5_p92_optimizer_kernel_launch_evidence_digest_missing")
    blockers.extend(f"v5_p92_required_section_missing:{item}" for item in missing_sections)
    blockers.extend(f"v5_p92_optimizer_kernel_launch_row_missing:{item}" for item in missing_optimizers)
    blockers.extend(row_blockers)
    blockers.extend(_unsafe_claims(evidence, "optimizer_kernel_launch_evidence"))
    blockers.extend(_non_empty_claims(evidence, "optimizer_kernel_launch_evidence"))
    blockers.extend(_string_list(evidence.get("blocked_reasons")))
    blockers.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blockers)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blockers: list[str] = []
    if review.get("requested_scope") != P92_SCOPE:
        blockers.append("v5_p92_review_scope_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        blockers.append("v5_p92_review_identity_or_timestamp_missing")
    for field in REQUIRED_REVIEW_ACKS:
        if review.get(field) is not True:
            blockers.append(f"v5_p92_review_ack_missing:{field}")
    blockers.extend(
        f"v5_p92_unsafe_review_approval:{field}"
        for field in UNSAFE_REVIEW_APPROVAL_FIELDS
        if review.get(field) is True
    )
    return _dedupe(blockers)


def _decision(blockers: Sequence[str], review: Mapping[str, Any], review_blockers: Sequence[str]) -> str:
    if blockers or review_blockers:
        return P92_BLOCKED_DECISION
    if not review:
        return P92_HOLD_DECISION
    if review.get("approve_optimizer_kernel_launch_contract") is not True:
        return P92_REJECTED_DECISION
    return P92_READY_DECISION


def _p91_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("optimizer_native_dispatch_contract_ready")
        and summary.get("optimizer_native_dispatch_evidence_recorded")
        and summary.get("optimizer_native_dispatch_signed")
        and summary.get("decision") == P91_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and not summary.get("unsafe_claims")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
    )


def _kernel_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = evidence.get("optimizer_kernel_rows") or evidence.get("optimizer_kernel_launch_rows") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [_kernel_row(row) for row in rows if isinstance(row, Mapping)]


def _kernel_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_kind": str(row.get("optimizer_kind") or ""),
        "source": _source(row),
        "ready": row.get("ready") is True,
        "kernel_launch_review_ready": row.get("kernel_launch_review_ready") is True,
        "kernel_launch_boundary_ready": row.get("kernel_launch_boundary_ready") is True,
        "tensor_transfer_boundary_ready": row.get("tensor_transfer_boundary_ready") is True,
        "parity_boundary_ready": row.get("parity_boundary_ready") is True,
        "training_step_boundary_ready": row.get("training_step_boundary_ready") is True,
        "rollback_policy_ready": row.get("rollback_policy_ready") is True,
        "later_tensor_transfer_contract_required": row.get("later_tensor_transfer_contract_required") is True,
        "kernel_launch_executed": row.get("kernel_launch_executed") is True,
        "tensor_transfer_executed": row.get("tensor_transfer_executed") is True,
        "parity_execution_executed": row.get("parity_execution_executed") is True,
        "training_step_executed": row.get("training_step_executed") is True,
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
            blockers.append(f"v5_p92_optimizer_kernel_launch_row_source_missing:{kind}")
        for field in (
            "ready",
            "kernel_launch_review_ready",
            "kernel_launch_boundary_ready",
            "tensor_transfer_boundary_ready",
            "parity_boundary_ready",
            "training_step_boundary_ready",
            "rollback_policy_ready",
        ):
            if row.get(field) is not True:
                blockers.append(f"v5_p92_optimizer_kernel_launch_row_{field}_missing:{kind}")
        if row.get("later_tensor_transfer_contract_required") is not True:
            blockers.append(f"v5_p92_optimizer_kernel_launch_row_later_tensor_transfer_contract_required_missing:{kind}")
        for field in (
            "kernel_launch_executed",
            "tensor_transfer_executed",
            "parity_execution_executed",
            "training_step_executed",
            "request_fields_emitted",
        ):
            if row.get(field) is True:
                blockers.append(f"v5_p92_optimizer_kernel_launch_row_unsafe_claim:{kind}:{field}")
        blockers.extend(
            f"v5_p92_optimizer_kernel_launch_row_blocker:{kind}:{item}" for item in row.get("blocked_reasons") or []
        )
    return _dedupe(blockers)


def _unsafe_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p92_unsafe_claim:{label}:{field}" for field in UNSAFE_TRUE_FIELDS if value.get(field) is True]


def _non_empty_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p92_unsafe_claim:{label}:{field}" for field in UNSAFE_NON_EMPTY_FIELDS if bool(value.get(field))]


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": P92_SCOPE,
        "approve_optimizer_kernel_launch_contract": True,
        **{field: True for field in REQUIRED_REVIEW_ACKS},
        **{field: False for field in UNSAFE_REVIEW_APPROVAL_FIELDS},
    }


def _allowed_next_actions(decision: str, blockers: Sequence[str]) -> list[str]:
    if decision == P92_READY_DECISION:
        return ["prepare_optimizer_tensor_transfer_contract_default_off"]
    if decision == P92_REJECTED_DECISION:
        return ["keep_default_off", "refresh_optimizer_kernel_launch_evidence"]
    if decision == P92_HOLD_DECISION:
        return ["collect_signed_optimizer_kernel_launch_review"]
    return ["resolve_blockers", *list(blockers[:6])]


def _recommended_next_step(decision: str, blockers: Sequence[str]) -> str:
    if decision == P92_READY_DECISION:
        return "draft optimizer tensor transfer contract"
    if decision == P92_HOLD_DECISION:
        return "collect signed owner review for optimizer kernel launch contract"
    if decision == P92_REJECTED_DECISION:
        return "keep optimizer kernel launch default-off and refresh evidence"
    return blockers[0] if blockers else "complete optimizer kernel launch evidence"


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p91-optimizer-native-dispatch-contract")
    parser.add_argument("--optimizer-kernel-launch-evidence")
    parser.add_argument("--optimizer-kernel-launch-signed-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_kernel_launch_contract_p92(
        p91_optimizer_native_dispatch_contract=_load_optional(args.p91_optimizer_native_dispatch_contract),
        optimizer_kernel_launch_evidence=_load_optional(args.optimizer_kernel_launch_evidence),
        optimizer_kernel_launch_signed_review=_load_optional(args.optimizer_kernel_launch_signed_review),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ALLOWED_REVIEW_INTENTS",
    "P92_SCOPE",
    "REQUIRED_REVIEW_ACKS",
    "REQUIRED_SECTIONS",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_kernel_launch_contract_p92",
]
