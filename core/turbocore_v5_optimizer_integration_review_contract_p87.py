"""Optimizer integration review contract for TurboCore V5-P87.

P87 records the signed review shape for a future optimizer integration stage.
It may record that integration evidence is ready for a later contract, but it
never approves optimizer integration, enables optimizer training paths, emits
request fields, patches schemas, registers routers/UI, launches training,
executes runtimes, dispatches native work, or launches kernels.
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
from core.turbocore_v5_optimizer_batch_keep_rollback_review_contract_p86 import (
    P86_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P86_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P86_UNSAFE_TRUE_FIELDS,
)
from core.turbocore_v5_optimizer_batch_validation_contract_p82 import DEFAULT_OPTIMIZER_KINDS
from core.turbocore_v5_owner_review_evidence_package import load_json


P87_READY_DECISION = "optimizer_integration_review_contract_p87_recorded_default_off"
P87_BLOCKED_DECISION = "optimizer_integration_review_contract_p87_blocked_default_off"
P87_HOLD_DECISION = "optimizer_integration_review_contract_p87_hold_for_signed_review_default_off"
P87_REJECTED_DECISION = "optimizer_integration_review_contract_p87_rejected_default_off"
P87_SCOPE = "optimizer_integration_review_contract"
ALLOWED_REVIEW_INTENTS = {"integration_candidate", "hold_for_more_evidence", "reject_integration_candidate"}
REQUIRED_SECTIONS = (
    "p86_keep_rollback_review_contract_reference",
    "optimizer_integration_review_package",
    "per_optimizer_integration_review_rows",
    "integration_boundary_inventory",
    "request_adapter_boundary_inventory",
    "runtime_dispatch_boundary_inventory",
    "native_dispatch_boundary_inventory",
    "training_path_boundary_inventory",
    "schema_router_ui_boundary_inventory",
    "default_rollout_boundary_inventory",
    "rollback_policy_summary",
    "observability_summary",
    "no_optimizer_integration_enable_boundary",
    "no_training_path_enable_boundary",
    "no_request_field_emission_boundary",
    "no_schema_config_router_ui_patch_boundary",
    "no_training_launch_boundary",
    "no_runtime_native_kernel_execution_boundary",
    "no_default_rollout_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p86_keep_rollback_review_contract_recorded",
    "acknowledge_optimizer_integration_review_only",
    "acknowledge_no_optimizer_integration_approved",
    "acknowledge_no_optimizer_integration_enabled",
    "acknowledge_no_optimizer_training_path_enabled",
    "acknowledge_no_runtime_dispatch_enabled",
    "acknowledge_no_native_dispatch_enabled",
    "acknowledge_no_kernel_launch_execution",
    "acknowledge_no_request_fields_emitted",
    "acknowledge_no_schema_config_router_ui_patch",
    "acknowledge_no_training_launch",
    "acknowledge_no_runtime_native_kernel_execution",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_later_optimizer_runtime_enablement_contract_required",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P86_UNSAFE_TRUE_FIELDS,
            "optimizer_integration_review_applied",
            "optimizer_integration_applied",
            "optimizer_integration_approved",
            "optimizer_integration_enabled",
            "optimizer_training_path_enabled",
            "optimizer_runtime_dispatch_enabled",
            "optimizer_native_dispatch_enabled",
            "optimizer_kernel_launch_enabled",
            "optimizer_kernel_launch_executed",
            "optimizer_request_adapter_enabled",
            "request_adapter_enabled",
            "request_adapter_mapping_allowed",
            "request_fields_emitted",
            "request_schema_patched",
            "generation_request_schema_patched",
            "config_adapter_patched",
            "backend_router_registered",
            "ui_route_registered",
            "ui_exposed",
            "ui_exposure_allowed",
            "default_rollout_allowed",
            "auto_rollout_allowed",
            "training_launch_allowed",
            "training_launch_executed",
            "training_step_executed",
            "runtime_adapter_enabled",
            "runtime_dispatch_enabled",
            "runtime_dispatch_executed",
            "native_dispatch_enabled",
            "native_dispatch_executed",
            "kernel_launch_executed",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P86_UNSAFE_NON_EMPTY_FIELDS,
            "post_p87_request_fields",
            "optimizer_integration_execution_payload",
            "optimizer_integration_apply_payload",
            "optimizer_runtime_dispatch_payload",
            "optimizer_native_dispatch_payload",
            "optimizer_kernel_launch_payload",
            "optimizer_request_adapter_payload",
            "request_adapter_payload",
            "schema_patch_payload",
            "schema_router_ui_patch_payload",
            "router_registration_payload",
            "ui_exposure_payload",
            "training_launch_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_v5_optimizer_integration_review_contract_p87(
    *,
    p86_keep_rollback_review_contract: Mapping[str, Any] | None = None,
    optimizer_integration_review_evidence: Mapping[str, Any] | None = None,
    optimizer_integration_signed_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record an optimizer integration review package without applying it."""

    p86 = _as_dict(p86_keep_rollback_review_contract)
    evidence = _as_dict(optimizer_integration_review_evidence)
    review = _as_dict(optimizer_integration_signed_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p86_summary = _p86_summary(p86)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p86_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P87_HOLD_DECISION:
        blockers.append("v5_p87_signed_optimizer_integration_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P87_READY_DECISION
    rejected = decision == P87_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_optimizer_integration_review_contract_p87_v0",
        "gate": "v5_optimizer_integration_review_contract_p87",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "optimizer_integration_review_recorded": decision_record_ready,
        "optimizer_integration_review_signed": decision_record_ready,
        "optimizer_integration_review_contract_ready": ready,
        "optimizer_integration_review_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p87_request_fields": {},
        "p86_keep_rollback_review_summary": p86_summary,
        "optimizer_integration_review_evidence_summary": evidence_summary,
        "optimizer_integration_signed_review": review_summary,
        "optimizer_integration_review_template": _review_template(),
        "progress_gates": _progress_gates(p86_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P87 records optimizer integration review evidence only.",
            "P87 does not approve or enable optimizer integration.",
            "A later optimizer runtime enablement contract is still required.",
        ],
    }


def _p86_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "keep_rollback_review_contract_ready": report.get("keep_rollback_review_contract_ready") is True,
        "keep_rollback_review_evidence_recorded": report.get("keep_rollback_review_evidence_recorded") is True,
        "keep_rollback_review_signed": report.get("keep_rollback_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p86_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p86"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = set(_string_list(evidence.get("available_sections"))) | set(str(key) for key in evidence.keys())
    missing_sections = [item for item in REQUIRED_SECTIONS if item not in sections]
    rows = _integration_rows(evidence)
    missing_optimizers = _missing_optimizers(rows)
    row_blockers = _row_blockers(rows)
    blockers = _evidence_blocker_list(evidence, missing_sections, missing_optimizers, row_blockers)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "optimizer_integration_review_package_ready": evidence.get("optimizer_integration_review_package_ready") is True,
        "integration_policy_ready": evidence.get("integration_policy_ready") is True,
        "later_optimizer_runtime_enablement_contract_required": (
            evidence.get("later_optimizer_runtime_enablement_contract_required") is True
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
        "integration_rows": rows,
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
        "approve_optimizer_integration_review_contract": (
            review.get("approve_optimizer_integration_review_contract") is True
        ),
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p86: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p86_keep_rollback_review_contract_ready": _p86_ready(p86),
        "optimizer_integration_review_evidence_ready": bool(evidence.get("ready", False)),
        "signed_optimizer_integration_review_present": bool(review.get("present", False)),
        "all_default_optimizer_rows_present": not bool(evidence.get("missing_optimizers")),
    }


def _evidence_blockers(
    p86: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_events: Sequence[str],
    rollback_events: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not _p86_ready(p86):
        blockers.append("v5_p87_p86_optimizer_batch_keep_rollback_review_contract_not_ready")
    blockers.extend(f"v5_p87_unsafe_upstream_claim:p86:{item}" for item in p86.get("unsafe_claims") or [])
    if not evidence.get("present"):
        blockers.append("v5_p87_optimizer_integration_review_evidence_missing")
    blockers.extend(_string_list(evidence.get("blockers")))
    if failure_events:
        blockers.extend(f"v5_p87_failure_history_not_clear:{item}" for item in failure_events)
    if rollback_events:
        blockers.extend(f"v5_p87_rollback_history_not_clear:{item}" for item in rollback_events)
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
        ("optimizer_integration_review_package_ready", True, "package_not_ready"),
        ("integration_policy_ready", True, "policy_not_ready"),
        ("later_optimizer_runtime_enablement_contract_required", True, "later_runtime_enablement_contract_missing"),
        ("report_only", True, "report_only_missing"),
        ("boundary_only", True, "boundary_only_missing"),
        ("contract_only", True, "contract_only_missing"),
        ("records_evidence_only", True, "records_evidence_only_missing"),
        ("manual_only", True, "manual_only_missing"),
        ("internal_only", True, "internal_only_missing"),
    )
    for field, expected, reason in checks:
        if evidence.get(field) is not expected:
            blockers.append(f"v5_p87_optimizer_integration_evidence_{reason}")
    if str(evidence.get("review_intent") or "") not in ALLOWED_REVIEW_INTENTS:
        blockers.append("v5_p87_optimizer_integration_review_intent_invalid")
    if not evidence.get("default_off") or not _default_off_confirmed(evidence):
        blockers.append("v5_p87_optimizer_integration_evidence_default_off_violation")
    if not evidence.get("request_adapter_off") or not _request_adapter_off(evidence):
        blockers.append("v5_p87_optimizer_integration_evidence_request_adapter_boundary_violation")
    if not _source(evidence):
        blockers.append("v5_p87_optimizer_integration_evidence_source_missing")
    if not _digest(evidence):
        blockers.append("v5_p87_optimizer_integration_evidence_digest_missing")
    blockers.extend(f"v5_p87_required_section_missing:{item}" for item in missing_sections)
    blockers.extend(f"v5_p87_optimizer_integration_row_missing:{item}" for item in missing_optimizers)
    blockers.extend(row_blockers)
    blockers.extend(_unsafe_claims(evidence, "optimizer_integration_review_evidence"))
    blockers.extend(_non_empty_claims(evidence, "optimizer_integration_review_evidence"))
    blockers.extend(_string_list(evidence.get("blocked_reasons")))
    blockers.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blockers)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blockers: list[str] = []
    if review.get("requested_scope") != P87_SCOPE:
        blockers.append("v5_p87_review_scope_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        blockers.append("v5_p87_review_identity_or_timestamp_missing")
    for field in REQUIRED_REVIEW_ACKS:
        if review.get(field) is not True:
            blockers.append(f"v5_p87_review_ack_missing:{field}")
    blockers.extend(
        f"v5_p87_unsafe_review_approval:{field}"
        for field in UNSAFE_REVIEW_APPROVAL_FIELDS
        if review.get(field) is True
    )
    return _dedupe(blockers)


def _decision(blockers: Sequence[str], review: Mapping[str, Any], review_blockers: Sequence[str]) -> str:
    if blockers or review_blockers:
        return P87_BLOCKED_DECISION
    if not review:
        return P87_HOLD_DECISION
    if review.get("approve_optimizer_integration_review_contract") is not True:
        return P87_REJECTED_DECISION
    return P87_READY_DECISION


def _p86_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("keep_rollback_review_contract_ready")
        and summary.get("keep_rollback_review_evidence_recorded")
        and summary.get("keep_rollback_review_signed")
        and summary.get("decision") == P86_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and not summary.get("unsafe_claims")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
    )


def _integration_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = (
        evidence.get("optimizer_integration_rows")
        or evidence.get("optimizer_integration_review_rows")
        or evidence.get("integration_rows")
        or []
    )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [_integration_row(row) for row in rows if isinstance(row, Mapping)]


def _integration_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_kind": str(row.get("optimizer_kind") or ""),
        "source": _source(row),
        "ready": row.get("ready") is True,
        "integration_review_ready": row.get("integration_review_ready") is True,
        "request_adapter_boundary_ready": row.get("request_adapter_boundary_ready") is True,
        "runtime_dispatch_boundary_ready": row.get("runtime_dispatch_boundary_ready") is True,
        "native_dispatch_boundary_ready": row.get("native_dispatch_boundary_ready") is True,
        "training_path_boundary_ready": row.get("training_path_boundary_ready") is True,
        "rollback_policy_ready": row.get("rollback_policy_ready") is True,
        "later_runtime_enablement_contract_required": row.get("later_runtime_enablement_contract_required") is True,
        "integration_enabled": row.get("integration_enabled") is True,
        "runtime_dispatch_enabled": row.get("runtime_dispatch_enabled") is True,
        "native_dispatch_enabled": row.get("native_dispatch_enabled") is True,
        "training_path_enabled": row.get("training_path_enabled") is True,
        "request_fields_emitted": row.get("request_fields_emitted") is True,
        "ui_exposed": row.get("ui_exposed") is True,
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
            blockers.append(f"v5_p87_optimizer_integration_row_source_missing:{kind}")
        for field in (
            "ready",
            "integration_review_ready",
            "request_adapter_boundary_ready",
            "runtime_dispatch_boundary_ready",
            "native_dispatch_boundary_ready",
            "training_path_boundary_ready",
            "rollback_policy_ready",
        ):
            if row.get(field) is not True:
                blockers.append(f"v5_p87_optimizer_integration_row_{field}_missing:{kind}")
        if row.get("later_runtime_enablement_contract_required") is not True:
            blockers.append(
                f"v5_p87_optimizer_integration_row_later_runtime_enablement_contract_required_missing:{kind}"
            )
        for field in (
            "integration_enabled",
            "runtime_dispatch_enabled",
            "native_dispatch_enabled",
            "training_path_enabled",
            "request_fields_emitted",
            "ui_exposed",
        ):
            if row.get(field) is True:
                blockers.append(f"v5_p87_optimizer_integration_row_unsafe_claim:{kind}:{field}")
        blockers.extend(
            f"v5_p87_optimizer_integration_row_blocker:{kind}:{item}" for item in row.get("blocked_reasons") or []
        )
    return _dedupe(blockers)


def _unsafe_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p87_unsafe_claim:{label}:{field}" for field in UNSAFE_TRUE_FIELDS if value.get(field) is True]


def _non_empty_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p87_unsafe_claim:{label}:{field}" for field in UNSAFE_NON_EMPTY_FIELDS if bool(value.get(field))]


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": P87_SCOPE,
        "approve_optimizer_integration_review_contract": True,
        **{field: True for field in REQUIRED_REVIEW_ACKS},
        **{field: False for field in UNSAFE_REVIEW_APPROVAL_FIELDS},
    }


def _allowed_next_actions(decision: str, blockers: Sequence[str]) -> list[str]:
    if decision == P87_READY_DECISION:
        return ["prepare_optimizer_runtime_enablement_contract_default_off"]
    if decision == P87_REJECTED_DECISION:
        return ["keep_default_off", "refresh_optimizer_integration_review_evidence"]
    if decision == P87_HOLD_DECISION:
        return ["collect_signed_optimizer_integration_review"]
    return ["resolve_blockers", *list(blockers[:6])]


def _recommended_next_step(decision: str, blockers: Sequence[str]) -> str:
    if decision == P87_READY_DECISION:
        return "draft optimizer runtime enablement contract"
    if decision == P87_HOLD_DECISION:
        return "collect signed owner review for optimizer integration review contract"
    if decision == P87_REJECTED_DECISION:
        return "keep optimizer integration default-off and refresh evidence"
    return blockers[0] if blockers else "complete optimizer integration review evidence"


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p86-keep-rollback-review-contract")
    parser.add_argument("--optimizer-integration-review-evidence")
    parser.add_argument("--optimizer-integration-signed-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_integration_review_contract_p87(
        p86_keep_rollback_review_contract=_load_optional(args.p86_keep_rollback_review_contract),
        optimizer_integration_review_evidence=_load_optional(args.optimizer_integration_review_evidence),
        optimizer_integration_signed_review=_load_optional(args.optimizer_integration_signed_review),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ALLOWED_REVIEW_INTENTS",
    "P87_SCOPE",
    "REQUIRED_REVIEW_ACKS",
    "REQUIRED_SECTIONS",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_integration_review_contract_p87",
]
