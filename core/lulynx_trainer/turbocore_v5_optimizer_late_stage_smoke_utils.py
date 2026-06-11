"""Shared smoke fixtures for optimizer late-stage contracts."""

from __future__ import annotations

from typing import Any, Callable

from core.turbocore_v5_optimizer_batch_validation_contract_p82 import DEFAULT_OPTIMIZER_KINDS
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec


GateFn = Callable[[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, list[Any] | None, list[Any] | None], dict[str, Any]]


def run_optimizer_late_stage_smoke(
    *,
    spec: OptimizerLateStageSpec,
    gate: GateFn,
    previous_ready: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    previous = previous_ready()
    ready = gate(previous, _evidence(spec), _review(spec), None, None)
    assert ready["ok"] is True, ready
    assert ready["decision"] == spec.ready_decision, ready
    assert ready[f"{spec.token}_contract_ready"] is True, ready
    _assert_default_off(spec, ready)

    missing_review = gate(previous, _evidence(spec), None, None, None)
    _assert_hold(spec, missing_review, f"signed_{spec.token}_review_missing")
    rejected = gate(previous, _evidence(spec), _review(spec, approve=False), None, None)
    assert rejected["ok"] is True, rejected
    assert rejected["decision"] == spec.rejected_decision, rejected
    assert rejected["rollback_required"] is True, rejected

    unsafe_field = _first_present(spec.unsafe_true_fields, "training_step_executed")
    payload_field = _first_present(spec.unsafe_non_empty_fields, "training_step_payload")
    row_unsafe = _row_unsafe_for(spec)
    cases = {
        "previous_missing": gate(None, _evidence(spec), _review(spec), None, None),
        "previous_not_ready": gate({**previous, spec.previous_ready_field: False}, _evidence(spec), _review(spec), None, None),
        "previous_unsafe": gate({**previous, unsafe_field: True}, _evidence(spec), _review(spec), None, None),
        "missing_source": gate(previous, _without(_evidence(spec), "source"), None, None, None),
        "missing_digest": gate(previous, _without_many(_evidence(spec), "sha256", "artifact_digest"), None, None, None),
        "missing_section": gate(previous, {**_evidence(spec), "available_sections": ["rollback_policy"]}, None, None, None),
        "invalid_intent": gate(previous, {**_evidence(spec), "review_intent": "launch_it"}, None, None, None),
        "default_on": gate(previous, {**_evidence(spec), "default_training_path_enabled": True}, None, None, None),
        "payload": gate(previous, {**_evidence(spec), payload_field: {"bad": True}}, None, None, None),
        "missing_lion": gate(
            previous,
            {**_evidence(spec), spec.row_keys[0]: [_row(spec, kind) for kind in DEFAULT_OPTIMIZER_KINDS if kind != "lion"]},
            None,
            None,
            None,
        ),
        "row_not_ready": gate(previous, {**_evidence(spec), spec.row_keys[0]: _rows_with(spec, "lion", ready=False)}, None, None, None),
        "row_unsafe": gate(previous, {**_evidence(spec), spec.row_keys[0]: _rows_with(spec, "lion", **{row_unsafe: True})}, None, None, None),
        "review_scope": gate(previous, _evidence(spec), {**_review(spec), "requested_scope": "wrong"}, None, None),
        "review_ack": gate(previous, _evidence(spec), {**_review(spec), spec.review_acks[0]: False}, None, None),
        "review_unsafe": gate(previous, _evidence(spec), {**_review(spec), f"approve_{unsafe_field}": True}, None, None),
        "failure_history": gate(previous, _evidence(spec), _review(spec), [{"reason": f"{spec.token}_gap", "status": "open"}], None),
        "rollback_history": gate(previous, _evidence(spec), _review(spec), None, [{"reason": "rollback_gap", "active": True}]),
    }
    fragments = {
        "previous_missing": f"{spec.previous_token}_not_ready",
        "previous_not_ready": f"{spec.previous_token}_not_ready",
        "previous_unsafe": unsafe_field,
        "missing_source": "source_missing",
        "missing_digest": "digest_missing",
        "missing_section": "required_section_missing",
        "invalid_intent": "review_intent_invalid",
        "default_on": "default_off",
        "payload": payload_field,
        "missing_lion": f"{spec.token}_row_missing:lion",
        "row_not_ready": f"{spec.token}_row_ready_missing:lion",
        "row_unsafe": f"{spec.token}_row_unsafe_claim:lion:{row_unsafe}",
        "review_scope": "review_scope_mismatch",
        "review_ack": "review_ack_missing",
        "review_unsafe": "unsafe_review_approval",
        "failure_history": f"{spec.token}_gap",
        "rollback_history": "rollback_gap",
    }
    for name, report in cases.items():
        _assert_blocked(spec, report, fragments[name])
    return {
        "schema_version": 1,
        "probe": f"turbocore_v5_p{spec.stage_id}_{spec.token}_smoke",
        "ok": True,
        "ready": _summary(spec, ready),
        "missing_review": _summary(spec, missing_review),
        "rejected": _summary(spec, rejected),
        **{name: _summary(spec, report) for name, report in cases.items()},
    }


def build_optimizer_late_stage_ready_report(
    *,
    spec: OptimizerLateStageSpec,
    gate: GateFn,
    previous_ready: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    report = gate(previous_ready(), _evidence(spec), _review(spec), None, None)
    assert report["ok"] is True, report
    assert report["decision"] == spec.ready_decision, report
    return report


def _evidence(spec: OptimizerLateStageSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": f"{spec.token}_contract_p{spec.stage_id}_v0",
        "ok": True,
        spec.package_ready_field: True,
        spec.policy_ready_field: True,
        spec.later_field: True,
        "review_intent": next(iter(spec.allowed_intents - {"hold_for_more_evidence"})),
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_off": True,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "source": f"devtools/docs/native_training_performance_roadmap_v5.md#v5-p{spec.stage_id}",
        "sha256": f"p{spec.stage_id}_{spec.token}_digest",
        "available_sections": list(spec.required_sections),
        spec.row_keys[0]: [_row(spec, kind) for kind in DEFAULT_OPTIMIZER_KINDS],
    }


def _row(spec: OptimizerLateStageSpec, kind: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "optimizer_kind": kind,
        "source": f"p{spec.stage_id}_{kind}_{spec.token}_row",
        "ready": True,
        spec.row_ready_field: True,
        spec.later_field: True,
        "kernel_launch_boundary_ready": True,
        "tensor_transfer_boundary_ready": True,
        "parity_boundary_ready": True,
        "training_step_boundary_ready": True,
        "training_launch_boundary_ready": True,
        "request_schema_router_ui_boundary_ready": True,
        "default_rollout_boundary_ready": True,
        "rollback_policy_ready": True,
        "kernel_launch_executed": False,
        "tensor_transfer_executed": False,
        "parity_execution_executed": False,
        "training_step_executed": False,
        "training_launch_executed": False,
        "request_fields_emitted": False,
        "default_rollout_allowed": False,
        "blocked_reasons": [],
    }
    row.update(overrides)
    return row


def _review(spec: OptimizerLateStageSpec, *, approve: bool = True) -> dict[str, Any]:
    return {
        "reviewer": "owner",
        "reviewed_at": "2026-06-03T00:00:00Z",
        "requested_scope": spec.scope,
        spec.review_approval_field: approve,
        **{field: True for field in spec.review_acks},
    }


def _rows_with(spec: OptimizerLateStageSpec, target: str, **overrides: Any) -> list[dict[str, Any]]:
    return [_row(spec, kind, **(overrides if kind == target else {})) for kind in DEFAULT_OPTIMIZER_KINDS]


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    out = dict(value)
    out.pop(key, None)
    return out


def _without_many(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    out = dict(value)
    for key in keys:
        out.pop(key, None)
    return out


def _first_present(values: tuple[str, ...], fallback: str) -> str:
    return values[0] if values else fallback


def _row_unsafe_for(spec: OptimizerLateStageSpec) -> str:
    for field in (
        "training_launch_executed",
        "training_step_executed",
        "parity_execution_executed",
        "tensor_transfer_executed",
        "kernel_launch_executed",
    ):
        if field in spec.unsafe_true_fields:
            return field
    return "request_fields_emitted"


def _assert_hold(spec: OptimizerLateStageSpec, report: dict[str, Any], fragment: str) -> None:
    assert report["ok"] is False, report
    assert report["decision"] == spec.hold_decision, report
    _assert_fragment(report, fragment)
    _assert_default_off(spec, report)


def _assert_blocked(spec: OptimizerLateStageSpec, report: dict[str, Any], fragment: str) -> None:
    assert report["ok"] is False, report
    assert report["decision"] == spec.blocked_decision, report
    _assert_fragment(report, fragment)
    _assert_default_off(spec, report)


def _assert_fragment(report: dict[str, Any], fragment: str) -> None:
    reasons = " ".join(str(item) for item in report.get("blocked_reasons", []))
    assert fragment in reasons, report


def _assert_default_off(spec: OptimizerLateStageSpec, report: dict[str, Any]) -> None:
    for field in spec.all_unsafe_true_fields:
        assert report.get(field) is False, (field, report)
    assert report.get(spec.post_fields) == {}, report


def _summary(spec: OptimizerLateStageSpec, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get(f"{spec.token}_contract_ready")),
        "review_signed": bool(report.get(f"{spec.token}_signed")),
        "rollback_required": bool(report.get("rollback_required")),
        "blocked_reasons": list(report.get("blocked_reasons") or []),
    }


__all__ = ["build_optimizer_late_stage_ready_report", "run_optimizer_late_stage_smoke"]
