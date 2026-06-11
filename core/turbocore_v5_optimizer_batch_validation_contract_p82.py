"""Optimizer batch-validation contract for TurboCore V5-P82.

P82 records the shared validation harness for future optimizer native kernels.
It standardizes a 20-step by 5-repeat canary matrix and shared safety checks,
but does not launch training, enable native dispatch, register request/UI
surfaces, or execute optimizer kernels.
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
from core.turbocore_v5_training_launch_execution_contract_p81 import (
    P81_READY_DECISION,
    UNSAFE_NON_EMPTY_FIELDS as _P81_UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS as _P81_UNSAFE_TRUE_FIELDS,
)


P82_READY_DECISION = "optimizer_batch_validation_contract_p82_recorded_default_off"
P82_BLOCKED_DECISION = "optimizer_batch_validation_contract_p82_blocked_default_off"
P82_HOLD_DECISION = "optimizer_batch_validation_contract_p82_hold_for_signed_review_default_off"
P82_REJECTED_DECISION = "optimizer_batch_validation_contract_p82_rejected_default_off"
P82_SCOPE = "optimizer_batch_validation_contract"
P82_CANARY_STEP_COUNT = 20
P82_CANARY_REPEAT_COUNT = 5
DEFAULT_OPTIMIZER_KINDS = (
    "adamw_exact",
    "paged_adamw_8bit",
    "kahan_adamw_8bit",
    "lion",
    "sgd_nesterov",
)
DEFAULT_REQUIRED_SECTIONS = (
    "p81_training_launch_execution_contract_reference",
    "optimizer_batch_manifest_inventory",
    "shared_safety_gate_inventory",
    "optimizer_kernel_matrix_inventory",
    "optimizer_state_schema_inventory",
    "optimizer_parity_gate_inventory",
    "optimizer_benchmark_gate_inventory",
    "optimizer_observability_inventory",
    "optimizer_rollback_policy_inventory",
    "no_training_launch_execution_boundary",
    "no_training_runtime_start_boundary",
    "no_runtime_execution_boundary",
    "no_runtime_dispatch_boundary",
    "no_native_dispatch_boundary",
    "no_kernel_launch_execution_boundary",
    "no_training_step_execution_boundary",
    "no_request_execution_boundary",
    "no_request_payload_materialization_boundary",
    "no_request_fields_emit_boundary",
    "no_generation_request_patch_boundary",
    "no_request_schema_patch_boundary",
    "no_config_adapter_patch_boundary",
    "no_backend_router_registration_boundary",
    "no_ui_route_registration_boundary",
    "no_ui_exposure_boundary",
    "no_default_rollout_boundary",
    "no_auto_rollout_boundary",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p81_training_launch_execution_contract_recorded",
    "acknowledge_default_off_boundary",
    "acknowledge_batch_validation_framework_only",
    "acknowledge_20_steps_5_repeats_canary_matrix",
    "acknowledge_no_training_launch",
    "acknowledge_no_training_runtime_started",
    "acknowledge_no_runtime_execution",
    "acknowledge_no_native_dispatch",
    "acknowledge_no_kernel_launch_execution",
    "acknowledge_no_training_step_execution",
    "acknowledge_no_request_fields_emitted",
    "acknowledge_no_schema_or_config_patch",
    "acknowledge_no_backend_router_or_ui_route",
    "acknowledge_later_optimizer_training_integration_required",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P81_UNSAFE_TRUE_FIELDS,
            "batch_validation_executed",
            "optimizer_kernel_executed",
            "optimizer_training_integration_enabled",
            "optimizer_native_dispatch_enabled",
            "optimizer_request_adapter_enabled",
            "optimizer_request_fields_emitted",
            "optimizer_ui_exposed",
        )
    )
)
UNSAFE_NON_EMPTY_FIELDS = tuple(
    dict.fromkeys(
        (
            *_P81_UNSAFE_NON_EMPTY_FIELDS,
            "post_p82_request_fields",
            "optimizer_training_request_payload",
            "optimizer_kernel_launch_payload",
            "optimizer_runtime_dispatch_payload",
            "optimizer_batch_validation_execution_payload",
        )
    )
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)
P81_REQUIRED_FALSE_FIELDS = _P81_UNSAFE_TRUE_FIELDS


def build_v5_optimizer_batch_validation_contract_p82(
    *,
    p81_training_launch_execution_contract: Mapping[str, Any] | None = None,
    optimizer_batch_validation_evidence: Mapping[str, Any] | None = None,
    optimizer_batch_validation_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record a shared optimizer-kernel validation framework without execution."""

    p81 = _as_dict(p81_training_launch_execution_contract)
    evidence = _as_dict(optimizer_batch_validation_evidence)
    review = _as_dict(optimizer_batch_validation_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p81_summary = _p81_summary(p81)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p81_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P82_HOLD_DECISION:
        blockers.append("v5_p82_signed_optimizer_batch_validation_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P82_READY_DECISION
    rejected = decision == P82_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_optimizer_batch_validation_contract_p82_v0",
        "gate": "v5_optimizer_batch_validation_contract_p82",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "optimizer_batch_validation_review_recorded": decision_record_ready,
        "optimizer_batch_validation_review_signed": decision_record_ready,
        "optimizer_batch_validation_contract_ready": ready,
        "optimizer_batch_validation_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        "canary_step_count": P82_CANARY_STEP_COUNT,
        "canary_repeat_count": P82_CANARY_REPEAT_COUNT,
        "canary_sample_count": P82_CANARY_STEP_COUNT * P82_CANARY_REPEAT_COUNT,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p82_request_fields": {},
        "p81_training_launch_execution_summary": p81_summary,
        "optimizer_batch_validation_evidence_summary": evidence_summary,
        "optimizer_batch_validation_review": review_summary,
        "optimizer_batch_validation_review_template": _review_template(),
        "progress_gates": _progress_gates(p81_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P82 records an optimizer batch-validation framework only.",
            "The default per-optimizer canary matrix is 20 steps by 5 repeats.",
            "A later optimizer training-integration contract is still required.",
        ],
    }


def _p81_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "training_launch_execution_contract_ready": report.get("training_launch_execution_contract_ready") is True,
        "training_launch_execution_evidence_recorded": report.get("training_launch_execution_evidence_recorded") is True,
        "training_launch_execution_review_signed": report.get("training_launch_execution_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P81_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p81_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p81"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    optimizer_rows = _optimizer_rows(evidence)
    missing_optimizers = _missing_optimizers(optimizer_rows)
    row_blockers = _optimizer_row_blockers(optimizer_rows)
    blockers = _evidence_blocker_list(evidence, missing_sections, missing_optimizers, row_blockers)
    summary = {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "optimizer_batch_validation_contract_ready": evidence.get("optimizer_batch_validation_contract_ready") is True,
        "batch_validation_framework_ready": evidence.get("batch_validation_framework_ready") is True,
        "shared_safety_harness_ready": evidence.get("shared_safety_harness_ready") is True,
        "per_optimizer_canary_matrix": evidence.get("per_optimizer_canary_matrix") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "optimizer_batch_validation_contract_only": evidence.get("optimizer_batch_validation_contract_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_optimizer_training_integration_contract": (
            evidence.get("requires_later_optimizer_training_integration_contract") is True
        ),
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "canary_step_count": _int(evidence.get("canary_step_count")),
        "canary_repeat_count": _int(evidence.get("canary_repeat_count")),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "optimizer_rows": optimizer_rows,
        "optimizer_count": len(optimizer_rows),
        "missing_optimizers": missing_optimizers,
        "row_blockers": row_blockers,
        "blockers": blockers,
    }
    return summary


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_optimizer_batch_validation_contract": review.get("approve_optimizer_batch_validation_contract") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p81: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p81_training_launch_execution_contract_ready": _p81_ready(p81),
        "optimizer_batch_validation_evidence_ready": bool(evidence.get("ready", False)),
        "signed_optimizer_batch_validation_review_present": bool(review.get("present", False)),
        "canary_matrix_20x5": (
            evidence.get("canary_step_count") == P82_CANARY_STEP_COUNT
            and evidence.get("canary_repeat_count") == P82_CANARY_REPEAT_COUNT
        ),
        "all_default_optimizer_rows_present": not bool(evidence.get("missing_optimizers")),
    }


def _evidence_blockers(
    p81: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_events: Sequence[str],
    rollback_events: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not _p81_ready(p81):
        blockers.append("v5_p82_p81_training_launch_execution_contract_not_ready")
    blockers.extend(_unsafe_summary_claims(p81, "p81"))
    if not evidence.get("present"):
        blockers.append("v5_p82_optimizer_batch_validation_evidence_missing")
    blockers.extend(_string_list(evidence.get("blockers")))
    if failure_events:
        blockers.extend(f"v5_p82_failure_history_not_clear:{item}" for item in failure_events)
    if rollback_events:
        blockers.extend(f"v5_p82_rollback_history_not_clear:{item}" for item in rollback_events)
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
        ("optimizer_batch_validation_contract_ready", True, "not_ready"),
        ("batch_validation_framework_ready", True, "framework_not_ready"),
        ("shared_safety_harness_ready", True, "shared_safety_harness_not_ready"),
        ("per_optimizer_canary_matrix", True, "per_optimizer_canary_matrix_missing"),
        ("report_only", True, "report_only_missing"),
        ("boundary_only", True, "boundary_only_missing"),
        ("contract_only", True, "contract_only_missing"),
        ("optimizer_batch_validation_contract_only", True, "contract_only_scope_missing"),
        ("records_evidence_only", True, "records_evidence_only_missing"),
        ("manual_only", True, "manual_only_missing"),
        ("internal_only", True, "internal_only_missing"),
        ("requires_later_optimizer_training_integration_contract", True, "later_integration_contract_missing"),
    )
    for field, expected, reason in checks:
        if evidence.get(field) is not expected:
            blockers.append(f"v5_p82_optimizer_batch_validation_evidence_{reason}")
    if _int(evidence.get("canary_step_count")) != P82_CANARY_STEP_COUNT:
        blockers.append("v5_p82_canary_step_count_must_be_20")
    if _int(evidence.get("canary_repeat_count")) != P82_CANARY_REPEAT_COUNT:
        blockers.append("v5_p82_canary_repeat_count_must_be_5")
    if not evidence.get("default_off") or not _default_off_confirmed(evidence):
        blockers.append("v5_p82_optimizer_batch_validation_evidence_default_off_violation")
    if not evidence.get("request_adapter_off") or not _request_adapter_off(evidence):
        blockers.append("v5_p82_optimizer_batch_validation_evidence_request_adapter_boundary_violation")
    if not _source(evidence):
        blockers.append("v5_p82_optimizer_batch_validation_evidence_source_missing")
    if not _digest(evidence):
        blockers.append("v5_p82_optimizer_batch_validation_evidence_digest_missing")
    blockers.extend(f"v5_p82_required_section_missing:{item}" for item in missing_sections)
    blockers.extend(f"v5_p82_optimizer_row_missing:{item}" for item in missing_optimizers)
    blockers.extend(row_blockers)
    blockers.extend(_unsafe_claims(evidence, "optimizer_batch_validation_evidence"))
    blockers.extend(_non_empty_claims(evidence, "optimizer_batch_validation_evidence"))
    blockers.extend(_string_list(evidence.get("blocked_reasons")))
    blockers.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blockers)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blockers: list[str] = []
    if review.get("requested_scope") != P82_SCOPE:
        blockers.append("v5_p82_review_scope_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        blockers.append("v5_p82_review_identity_or_timestamp_missing")
    for field in REQUIRED_REVIEW_ACKS:
        if review.get(field) is not True:
            blockers.append(f"v5_p82_review_ack_missing:{field}")
    unsafe = [field for field in UNSAFE_REVIEW_APPROVAL_FIELDS if review.get(field) is True]
    blockers.extend(f"v5_p82_unsafe_review_approval:{field}" for field in unsafe)
    return _dedupe(blockers)


def _decision(blockers: Sequence[str], review: Mapping[str, Any], review_blockers: Sequence[str]) -> str:
    if blockers or review_blockers:
        return P82_BLOCKED_DECISION
    if not review:
        return P82_HOLD_DECISION
    if review.get("approve_optimizer_batch_validation_contract") is not True:
        return P82_REJECTED_DECISION
    return P82_READY_DECISION


def _p81_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("training_launch_execution_contract_ready")
        and summary.get("training_launch_execution_evidence_recorded")
        and summary.get("training_launch_execution_review_signed")
        and summary.get("decision") == P81_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and not summary.get("unsafe_claims")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
    )


def _optimizer_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = evidence.get("optimizer_kernel_matrix") or evidence.get("optimizer_rows") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [_optimizer_row(row) for row in rows if isinstance(row, Mapping)]


def _optimizer_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_kind": str(row.get("optimizer_kind") or ""),
        "source": _source(row),
        "ready": row.get("ready") is True,
        "state_schema_boundary_ready": row.get("state_schema_boundary_ready") is True,
        "parity_gate_ready": row.get("parity_gate_ready") is True,
        "benchmark_gate_ready": row.get("benchmark_gate_ready") is True,
        "safety_gate_ready": row.get("safety_gate_ready") is True,
        "canary_step_count": _int(row.get("canary_step_count")),
        "canary_repeat_count": _int(row.get("canary_repeat_count")),
        "training_path_enabled": row.get("training_path_enabled") is True,
        "native_dispatch_enabled": row.get("native_dispatch_enabled") is True,
        "request_fields_emitted": row.get("request_fields_emitted") is True,
        "blocked_reasons": _string_list(row.get("blocked_reasons")),
    }


def _missing_optimizers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row.get("optimizer_kind") or "") for row in rows}
    return [item for item in DEFAULT_OPTIMIZER_KINDS if item not in present]


def _optimizer_row_blockers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        kind = str(row.get("optimizer_kind") or "unknown")
        if not row.get("source"):
            blockers.append(f"v5_p82_optimizer_row_source_missing:{kind}")
        for field in ("ready", "state_schema_boundary_ready", "parity_gate_ready", "benchmark_gate_ready", "safety_gate_ready"):
            if row.get(field) is not True:
                blockers.append(f"v5_p82_optimizer_row_{field}_missing:{kind}")
        if row.get("canary_step_count") != P82_CANARY_STEP_COUNT:
            blockers.append(f"v5_p82_optimizer_row_step_count_mismatch:{kind}")
        if row.get("canary_repeat_count") != P82_CANARY_REPEAT_COUNT:
            blockers.append(f"v5_p82_optimizer_row_repeat_count_mismatch:{kind}")
        for field in ("training_path_enabled", "native_dispatch_enabled", "request_fields_emitted"):
            if row.get(field) is True:
                blockers.append(f"v5_p82_optimizer_row_unsafe_claim:{kind}:{field}")
        blockers.extend(f"v5_p82_optimizer_row_blocker:{kind}:{item}" for item in row.get("blocked_reasons") or [])
    return _dedupe(blockers)


def _section_set(evidence: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(evidence.get("available_sections")))
    sections.update(str(key) for key in evidence.keys())
    return sections


def _unsafe_summary_claims(summary: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p82_unsafe_upstream_claim:{label}:{item}" for item in summary.get("unsafe_claims") or []]


def _unsafe_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p82_unsafe_claim:{label}:{field}" for field in UNSAFE_TRUE_FIELDS if value.get(field) is True]


def _non_empty_claims(value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p82_unsafe_claim:{label}:{field}" for field in UNSAFE_NON_EMPTY_FIELDS if bool(value.get(field))]


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": P82_SCOPE,
        "approve_optimizer_batch_validation_contract": True,
        **{field: True for field in REQUIRED_REVIEW_ACKS},
        **{field: False for field in UNSAFE_REVIEW_APPROVAL_FIELDS},
    }


def _allowed_next_actions(decision: str, blockers: Sequence[str]) -> list[str]:
    if decision == P82_READY_DECISION:
        return ["prepare_optimizer_batch_validation_runner_contract_default_off"]
    if decision == P82_REJECTED_DECISION:
        return ["keep_default_off", "refresh_optimizer_batch_validation_evidence"]
    if decision == P82_HOLD_DECISION:
        return ["collect_signed_optimizer_batch_validation_review"]
    return ["resolve_blockers", *list(blockers[:6])]


def _recommended_next_step(decision: str, blockers: Sequence[str]) -> str:
    if decision == P82_READY_DECISION:
        return "draft default-off optimizer batch validation runner contract"
    if decision == P82_HOLD_DECISION:
        return "collect signed owner review for optimizer batch validation contract"
    if decision == P82_REJECTED_DECISION:
        return "keep optimizer batch validation default-off and refresh evidence"
    return blockers[0] if blockers else "complete optimizer batch validation evidence"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p81-training-launch-execution-contract")
    parser.add_argument("--optimizer-batch-validation-evidence")
    parser.add_argument("--optimizer-batch-validation-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_batch_validation_contract_p82(
        p81_training_launch_execution_contract=_load_optional(args.p81_training_launch_execution_contract),
        optimizer_batch_validation_evidence=_load_optional(args.optimizer_batch_validation_evidence),
        optimizer_batch_validation_review=_load_optional(args.optimizer_batch_validation_review),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "DEFAULT_OPTIMIZER_KINDS",
    "DEFAULT_REQUIRED_SECTIONS",
    "P82_CANARY_REPEAT_COUNT",
    "P82_CANARY_STEP_COUNT",
    "P82_SCOPE",
    "REQUIRED_REVIEW_ACKS",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_batch_validation_contract_p82",
]
