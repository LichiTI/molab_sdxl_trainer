"""Shared default-off optimizer late-stage contract helpers.

This module is used by P95+ optimizer contracts to avoid duplicating the same
review-only/default-off gate logic for each later boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True)
class OptimizerLateStageSpec:
    stage_id: int
    token: str
    scope: str
    title: str
    previous_token: str
    previous_label: str
    previous_ready_decision: str
    previous_ready_field: str
    previous_evidence_field: str
    previous_signed_field: str
    previous_post_fields: str
    previous_ack: str
    package_ready_field: str
    policy_ready_field: str
    row_keys: tuple[str, ...]
    row_ready_field: str
    later_field: str
    later_ack: str
    next_contract: str
    allowed_intents: frozenset[str]
    required_sections: tuple[str, ...]
    unsafe_true_fields: tuple[str, ...]
    unsafe_non_empty_fields: tuple[str, ...]
    inherited_unsafe_true_fields: tuple[str, ...]
    inherited_unsafe_non_empty_fields: tuple[str, ...]

    @property
    def ready_decision(self) -> str:
        return f"{self.token}_contract_p{self.stage_id}_recorded_default_off"

    @property
    def blocked_decision(self) -> str:
        return f"{self.token}_contract_p{self.stage_id}_blocked_default_off"

    @property
    def hold_decision(self) -> str:
        return f"{self.token}_contract_p{self.stage_id}_hold_for_signed_review_default_off"

    @property
    def rejected_decision(self) -> str:
        return f"{self.token}_contract_p{self.stage_id}_rejected_default_off"

    @property
    def review_approval_field(self) -> str:
        return f"approve_{self.token}_contract"

    @property
    def post_fields(self) -> str:
        return f"post_p{self.stage_id}_request_fields"

    @property
    def all_unsafe_true_fields(self) -> tuple[str, ...]:
        common = (
            "request_fields_emitted",
            "schema_config_router_ui_patched",
            "default_rollout_allowed",
            "auto_rollout_allowed",
        )
        return tuple(dict.fromkeys((*self.inherited_unsafe_true_fields, *self.unsafe_true_fields, *common)))

    @property
    def all_unsafe_non_empty_fields(self) -> tuple[str, ...]:
        common = (
            self.post_fields,
            "request_field_payload",
            "schema_config_router_ui_patch_payload",
        )
        return tuple(dict.fromkeys((*self.inherited_unsafe_non_empty_fields, *self.unsafe_non_empty_fields, *common)))

    @property
    def review_acks(self) -> tuple[str, ...]:
        return (
            self.previous_ack,
            f"acknowledge_{self.token}_contract_only",
            "acknowledge_no_kernel_launch_executed",
            "acknowledge_no_tensor_transfer_executed",
            "acknowledge_no_parity_executed",
            "acknowledge_no_training_step_or_launch",
            "acknowledge_no_request_fields_emitted",
            "acknowledge_no_schema_config_router_ui_patch",
            "acknowledge_no_default_or_auto_rollout",
            self.later_ack,
            "acknowledge_manual_review_only",
        )


def build_optimizer_late_stage_contract(
    spec: OptimizerLateStageSpec,
    *,
    previous_contract: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    signed_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    previous = _as_dict(previous_contract)
    evidence_map = _as_dict(evidence)
    review = _as_dict(signed_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    previous_summary = _previous_summary(spec, previous)
    evidence_summary = _evidence_summary(spec, evidence_map)
    review_summary = _review_summary(spec, review)
    evidence_blockers = _evidence_blockers(spec, previous_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(spec, review_summary)
    decision = _decision(spec, evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == spec.hold_decision:
        blockers.append(f"v5_p{spec.stage_id}_signed_{spec.token}_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == spec.ready_decision
    rejected = decision == spec.rejected_decision
    decision_record_ready = ready or rejected
    unsafe_false = {field: False for field in spec.all_unsafe_true_fields}
    return {
        "schema_version": 1,
        "package": f"turbocore_v5_{spec.token}_contract_p{spec.stage_id}_v0",
        "gate": f"v5_{spec.token}_contract_p{spec.stage_id}",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        f"{spec.token}_recorded": decision_record_ready,
        f"{spec.token}_signed": decision_record_ready,
        f"{spec.token}_contract_ready": ready,
        f"{spec.token}_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        "default_off": True,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "request_adapter_off": True,
        "request_adapter_mapping_allowed": False,
        **unsafe_false,
        spec.post_fields: {},
        f"{spec.previous_token}_summary": previous_summary,
        f"{spec.token}_evidence_summary": evidence_summary,
        f"{spec.token}_signed_review": review_summary,
        f"{spec.token}_review_template": _review_template(spec),
        "progress_gates": _progress_gates(spec, previous_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(spec, decision, blockers),
        "recommended_next_step": _recommended_next_step(spec, decision, blockers),
        "notes": [
            f"P{spec.stage_id} records {spec.title} evidence only.",
            f"P{spec.stage_id} remains default-off and does not execute training work.",
            f"A later {spec.next_contract} contract is still required.",
        ],
    }


def _previous_summary(spec: OptimizerLateStageSpec, report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        spec.previous_ready_field: report.get(spec.previous_ready_field) is True,
        spec.previous_evidence_field: report.get(spec.previous_evidence_field) is True,
        spec.previous_signed_field: report.get(spec.previous_signed_field) is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get(spec.previous_post_fields))),
        "unsafe_claims": _unsafe_claims(spec, report, spec.previous_token),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(spec: OptimizerLateStageSpec, evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = set(_string_list(evidence.get("available_sections"))) | set(str(key) for key in evidence.keys())
    missing_sections = [item for item in spec.required_sections if item not in sections]
    rows = _rows(spec, evidence)
    missing_optimizers = _missing_optimizers(rows)
    row_blockers = _row_blockers(spec, rows)
    blockers = _evidence_blocker_list(spec, evidence, missing_sections, missing_optimizers, row_blockers)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        spec.package_ready_field: evidence.get(spec.package_ready_field) is True,
        spec.policy_ready_field: evidence.get(spec.policy_ready_field) is True,
        spec.later_field: evidence.get(spec.later_field) is True,
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
        "rows": rows,
        "missing_optimizers": missing_optimizers,
        "row_blockers": row_blockers,
        "missing_sections": missing_sections,
        "blockers": blockers,
    }


def _review_summary(spec: OptimizerLateStageSpec, review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        spec.review_approval_field: review.get(spec.review_approval_field) is True,
    }
    for field in _unsafe_review_approval_fields(spec):
        summary[field] = bool(review.get(field, False))
    for field in spec.review_acks:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(
    spec: OptimizerLateStageSpec,
    previous: Mapping[str, Any],
    evidence: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        f"{spec.previous_token}_ready": _previous_ready(spec, previous),
        f"{spec.token}_evidence_ready": bool(evidence.get("ready", False)),
        f"signed_{spec.token}_review_present": bool(review.get("present", False)),
        "all_default_optimizer_rows_present": not bool(evidence.get("missing_optimizers")),
    }


def _evidence_blockers(
    spec: OptimizerLateStageSpec,
    previous: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_events: Sequence[str],
    rollback_events: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if not _previous_ready(spec, previous):
        blockers.append(f"v5_p{spec.stage_id}_{spec.previous_token}_not_ready")
    blockers.extend(f"v5_p{spec.stage_id}_unsafe_upstream_claim:{spec.previous_token}:{item}" for item in previous.get("unsafe_claims") or [])
    if not evidence.get("present"):
        blockers.append(f"v5_p{spec.stage_id}_{spec.token}_evidence_missing")
    blockers.extend(_string_list(evidence.get("blockers")))
    blockers.extend(f"v5_p{spec.stage_id}_failure_history_not_clear:{item}" for item in failure_events)
    blockers.extend(f"v5_p{spec.stage_id}_rollback_history_not_clear:{item}" for item in rollback_events)
    return _dedupe(blockers)


def _evidence_blocker_list(
    spec: OptimizerLateStageSpec,
    evidence: Mapping[str, Any],
    missing_sections: Sequence[str],
    missing_optimizers: Sequence[str],
    row_blockers: Sequence[str],
) -> list[str]:
    checks = (
        ("ok", True, "evidence_not_ok"),
        (spec.package_ready_field, True, "package_not_ready"),
        (spec.policy_ready_field, True, "policy_not_ready"),
        (spec.later_field, True, "later_contract_missing"),
        ("report_only", True, "report_only_missing"),
        ("boundary_only", True, "boundary_only_missing"),
        ("contract_only", True, "contract_only_missing"),
        ("records_evidence_only", True, "records_evidence_only_missing"),
        ("manual_only", True, "manual_only_missing"),
        ("internal_only", True, "internal_only_missing"),
    )
    blockers: list[str] = []
    for field, expected, reason in checks:
        if evidence.get(field) is not expected:
            blockers.append(f"v5_p{spec.stage_id}_{spec.token}_evidence_{reason}")
    if str(evidence.get("review_intent") or "") not in spec.allowed_intents:
        blockers.append(f"v5_p{spec.stage_id}_{spec.token}_review_intent_invalid")
    if not evidence.get("default_off") or not _default_off_confirmed(evidence):
        blockers.append(f"v5_p{spec.stage_id}_{spec.token}_evidence_default_off_violation")
    if not evidence.get("request_adapter_off") or not _request_adapter_off(evidence):
        blockers.append(f"v5_p{spec.stage_id}_{spec.token}_evidence_request_adapter_boundary_violation")
    if not _source(evidence):
        blockers.append(f"v5_p{spec.stage_id}_{spec.token}_evidence_source_missing")
    if not _digest(evidence):
        blockers.append(f"v5_p{spec.stage_id}_{spec.token}_evidence_digest_missing")
    blockers.extend(f"v5_p{spec.stage_id}_required_section_missing:{item}" for item in missing_sections)
    blockers.extend(f"v5_p{spec.stage_id}_{spec.token}_row_missing:{item}" for item in missing_optimizers)
    blockers.extend(row_blockers)
    blockers.extend(_unsafe_claims(spec, evidence, f"{spec.token}_evidence"))
    blockers.extend(_non_empty_claims(spec, evidence, f"{spec.token}_evidence"))
    blockers.extend(_string_list(evidence.get("blocked_reasons")))
    blockers.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blockers)


def _review_blockers(spec: OptimizerLateStageSpec, review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blockers: list[str] = []
    if review.get("requested_scope") != spec.scope:
        blockers.append(f"v5_p{spec.stage_id}_review_scope_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at"):
        blockers.append(f"v5_p{spec.stage_id}_review_identity_or_timestamp_missing")
    for field in spec.review_acks:
        if review.get(field) is not True:
            blockers.append(f"v5_p{spec.stage_id}_review_ack_missing:{field}")
    blockers.extend(
        f"v5_p{spec.stage_id}_unsafe_review_approval:{field}"
        for field in _unsafe_review_approval_fields(spec)
        if review.get(field) is True
    )
    return _dedupe(blockers)


def _decision(
    spec: OptimizerLateStageSpec,
    blockers: Sequence[str],
    review: Mapping[str, Any],
    review_blockers: Sequence[str],
) -> str:
    if blockers or review_blockers:
        return spec.blocked_decision
    if not review:
        return spec.hold_decision
    if review.get(spec.review_approval_field) is not True:
        return spec.rejected_decision
    return spec.ready_decision


def _previous_ready(spec: OptimizerLateStageSpec, summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get(spec.previous_ready_field)
        and summary.get(spec.previous_evidence_field)
        and summary.get(spec.previous_signed_field)
        and summary.get("decision") == spec.previous_ready_decision
        and summary.get("manual_review_required")
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and not summary.get("unsafe_claims")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
    )


def _rows(spec: OptimizerLateStageSpec, evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: Any = []
    for key in spec.row_keys:
        if evidence.get(key):
            rows = evidence.get(key)
            break
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [_row(spec, row) for row in rows if isinstance(row, Mapping)]


def _row(spec: OptimizerLateStageSpec, row: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "optimizer_kind": str(row.get("optimizer_kind") or ""),
        "source": _source(row),
        "ready": row.get("ready") is True,
        spec.row_ready_field: row.get(spec.row_ready_field) is True,
        spec.later_field: row.get(spec.later_field) is True,
        "blocked_reasons": _string_list(row.get("blocked_reasons")),
    }
    for field in _row_required_fields(spec):
        out[field] = row.get(field) is True
    for field in _row_unsafe_fields():
        out[field] = row.get(field) is True
    return out


def _missing_optimizers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row.get("optimizer_kind") or "") for row in rows}
    return [item for item in DEFAULT_OPTIMIZER_KINDS if item not in present]


def _row_blockers(spec: OptimizerLateStageSpec, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        kind = str(row.get("optimizer_kind") or "unknown")
        if not row.get("source"):
            blockers.append(f"v5_p{spec.stage_id}_{spec.token}_row_source_missing:{kind}")
        for field in ("ready", spec.row_ready_field, spec.later_field, *_row_required_fields(spec)):
            if row.get(field) is not True:
                blockers.append(f"v5_p{spec.stage_id}_{spec.token}_row_{field}_missing:{kind}")
        for field in _row_unsafe_fields():
            if row.get(field) is True:
                blockers.append(f"v5_p{spec.stage_id}_{spec.token}_row_unsafe_claim:{kind}:{field}")
        blockers.extend(f"v5_p{spec.stage_id}_{spec.token}_row_blocker:{kind}:{item}" for item in row.get("blocked_reasons") or [])
    return _dedupe(blockers)


def _row_required_fields(spec: OptimizerLateStageSpec) -> tuple[str, ...]:
    base = (
        "kernel_launch_boundary_ready",
        "tensor_transfer_boundary_ready",
        "parity_boundary_ready",
        "training_step_boundary_ready",
        "training_launch_boundary_ready",
        "request_schema_router_ui_boundary_ready",
        "default_rollout_boundary_ready",
        "rollback_policy_ready",
    )
    return tuple(field for field in base if field in spec.required_sections or field in spec.unsafe_true_fields or field.endswith("_ready"))


def _row_unsafe_fields() -> tuple[str, ...]:
    return (
        "kernel_launch_executed",
        "tensor_transfer_executed",
        "parity_execution_executed",
        "training_step_executed",
        "training_launch_executed",
        "request_fields_emitted",
        "default_rollout_allowed",
    )


def _unsafe_claims(spec: OptimizerLateStageSpec, value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p{spec.stage_id}_unsafe_claim:{label}:{field}" for field in spec.all_unsafe_true_fields if value.get(field) is True]


def _non_empty_claims(spec: OptimizerLateStageSpec, value: Mapping[str, Any], label: str) -> list[str]:
    return [f"v5_p{spec.stage_id}_unsafe_claim:{label}:{field}" for field in spec.all_unsafe_non_empty_fields if bool(value.get(field))]


def _unsafe_review_approval_fields(spec: OptimizerLateStageSpec) -> tuple[str, ...]:
    return tuple(f"approve_{field}" for field in spec.all_unsafe_true_fields)


def _review_template(spec: OptimizerLateStageSpec) -> dict[str, Any]:
    return {
        "requested_scope": spec.scope,
        spec.review_approval_field: True,
        **{field: True for field in spec.review_acks},
        **{field: False for field in _unsafe_review_approval_fields(spec)},
    }


def _allowed_next_actions(spec: OptimizerLateStageSpec, decision: str, blockers: Sequence[str]) -> list[str]:
    if decision == spec.ready_decision:
        return [f"prepare_{spec.next_contract}_default_off"]
    if decision == spec.rejected_decision:
        return ["keep_default_off", f"refresh_{spec.token}_evidence"]
    if decision == spec.hold_decision:
        return [f"collect_signed_{spec.token}_review"]
    return ["resolve_blockers", *list(blockers[:6])]


def _recommended_next_step(spec: OptimizerLateStageSpec, decision: str, blockers: Sequence[str]) -> str:
    if decision == spec.ready_decision:
        return f"draft {spec.next_contract} contract"
    if decision == spec.hold_decision:
        return f"collect signed owner review for {spec.title} contract"
    if decision == spec.rejected_decision:
        return f"keep {spec.title} default-off and refresh evidence"
    return blockers[0] if blockers else f"complete {spec.title} evidence"


__all__ = ["OptimizerLateStageSpec", "build_optimizer_late_stage_contract"]
