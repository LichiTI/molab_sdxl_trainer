"""Smoke checks for V5-P80 training-step execution contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(CORE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_training_step_execution_contract_p80 import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P80_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_training_step_execution_contract_p80,
)
from lulynx_trainer.turbocore_v5_tensor_transfer_execution_contract_p79_smoke import (  # noqa: E402
    _gate as _p79_gate,
    _p78_ready,
)


READY_DECISION = "training_step_execution_contract_p80_recorded_default_off"
BLOCKED_DECISION = "training_step_execution_contract_p80_blocked_default_off"
HOLD_DECISION = "training_step_execution_contract_p80_hold_for_signed_review_default_off"
REJECTED_DECISION = "training_step_execution_contract_p80_rejected_default_off"
INVENTORY_SPECS = (
    ("step_plan", "training_step_execution_plan_inventory", "training_step_execution_plan", "p80_step_plan"),
    ("step_precondition", "training_step_execution_precondition_inventory", "training_step_execution_precondition", "p80_step_precondition"),
    ("authorization", "training_step_authorization_boundary", "training_step_authorization_boundary", "p80_authorization"),
    ("gradient", "gradient_boundary", "gradient_boundary", "p80_gradient"),
    ("optimizer_step", "optimizer_step_boundary", "optimizer_step_boundary", "p80_optimizer_step"),
    ("parameter_update", "parameter_update_boundary", "parameter_update_boundary", "p80_parameter_update"),
    ("state_update", "optimizer_state_update_boundary", "optimizer_state_update_boundary", "p80_state_update"),
    ("loss_scale", "loss_scale_boundary", "loss_scale_boundary", "p80_loss_scale"),
    ("training_launch", "training_launch_boundary", "training_launch_boundary", "p80_training_launch"),
    ("operator", "operator_training_step_boundary", "operator_training_step_boundary", "p80_operator"),
    ("observability", "observability_boundary", "observability_boundary", "p80_observability"),
    ("rollback", "rollback_policy_boundary", "rollback_policy", "p80_rollback"),
)


def run_smoke() -> dict[str, Any]:
    p79_ready = _p79_ready()
    ready = _gate(p79_ready)
    assert ready["ok"] is True, ready
    assert ready["training_step_execution_contract_ready"] is True, ready
    assert ready["training_step_execution_evidence_recorded"] is True, ready
    assert ready["training_step_execution_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p79_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p79_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p79_missing = _gate(None)
    _assert_blocked(p79_missing, "p79", "missing")
    p79_not_ready = _gate({**p79_ready, "ok": False, "tensor_transfer_execution_contract_ready": False})
    _assert_blocked(p79_not_ready, "p79", "not_ready")
    p79_decision_mismatch = _gate({**p79_ready, "decision": "wrong"})
    _assert_blocked(p79_decision_mismatch, "p79", "not_ready")
    p79_post_fields = _gate({**p79_ready, "post_p79_request_fields": {"bad": True}})
    _assert_blocked(p79_post_fields, "post_p79_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p80_training_step_execution_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p79_missing": _summary(p79_missing),
        "p79_not_ready": _summary(p79_not_ready),
        "p79_decision_mismatch": _summary(p79_decision_mismatch),
        "p79_post_fields": _summary(p79_post_fields),
        "p79_unsafe_cases": _unsafe_p79_cases(p79_ready),
        **_evidence_cases(p79_ready),
        **_inventory_cases(p79_ready),
        **_review_cases(p79_ready),
        **_history_cases(p79_ready),
    }


def _unsafe_p79_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_step_executed", "tensor_transfer_executed", "parity_executed",
        "kernel_launch_executed", "native_dispatch_executed", "runtime_dispatch_executed",
        "runtime_execution_executed", "runtime_state_refreshed", "runtime_adapter_enabled",
    ):
        report = _gate({**p79_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p79_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p79_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p79_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p79_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p79_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p79_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_step_contract": _gate(
            p79_ready, evidence={**_evidence(), "training_step_execution_contract_only": False}
        ),
        "not_manual_only": _gate(p79_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p79_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_launch_contract": _gate(
            p79_ready, evidence={**_evidence(), "requires_later_training_launch_execution_contract": False}
        ),
        "evidence_not_ready": _gate(
            p79_ready, evidence={**_evidence(), "training_step_execution_contract_ready": False}
        ),
        "missing_section": _gate(p79_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p79_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p79_ready, evidence={**_evidence(), "request_fields_emitted": True}),
        "evidence_blocker": _gate(p79_ready, evidence={**_evidence(), "blocked_reasons": ["step_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",), "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",), "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",), "not_records_only": ("records_evidence_only",),
        "not_step_contract": ("training_step_execution_contract_only",),
        "not_manual_only": ("manual_only",), "not_internal_only": ("internal_only",),
        "missing_later_launch_contract": ("training_launch",), "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",), "default_on": ("default_off",),
        "request_adapter_on": ("request_fields_emitted",), "evidence_blocker": ("step_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for name, field, kind, item_id in INVENTORY_SPECS:
        missing = _gate(p79_ready, evidence=_without(_evidence(), field))
        not_ready = _gate(p79_ready, evidence={**_evidence(), field: [{**_row(item_id), "ready": False}]})
        source_missing = _gate(p79_ready, evidence={**_evidence(), field: [_without(_row(item_id), "source")]})
        _assert_blocked(missing, f"{kind}_inventory_missing")
        _assert_blocked(not_ready, f"{kind}_not_ready")
        _assert_blocked(source_missing, f"{kind}_source_missing")
        cases[f"missing_{name}"] = _summary(missing)
        cases[f"{name}_not_ready"] = _summary(not_ready)
        cases[f"{name}_missing_source"] = _summary(source_missing)
    return {
        **cases,
        "unsafe_evidence_cases": _unsafe_evidence_cases(p79_ready),
        "unsafe_payload_cases": _unsafe_payload_cases(p79_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p79_ready),
    }


def _unsafe_evidence_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_step_approved", "training_step_enabled", "training_step_requested",
        "training_step_executed", "gradients_materialized", "optimizer_step_executed",
        "parameters_updated", "optimizer_state_updated", "loss_scale_updated",
        "training_launch_executed",
    ):
        report = _gate(p79_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_payload_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_step_execution_contract_request", "training_step_execution_contract_payload",
        "training_step_execution_payload", "training_step_authorization_payload",
        "gradient_payload", "optimizer_step_payload", "parameter_update_payload",
        "optimizer_state_update_payload", "loss_scale_payload", "training_launch_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p79_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_plan": _unsafe_inventory_claims(
            p79_ready, "training_step_execution_plan_inventory", _row("p80_step_plan"),
            ("training_step_allowed", "training_step_requested", "training_step_executed"),
        ),
        "gradient": _unsafe_inventory_claims(
            p79_ready, "gradient_boundary", _row("p80_gradient"),
            ("gradients_materialized", "training_step_executed", "training_launch_executed"),
        ),
        "optimizer": _unsafe_inventory_claims(
            p79_ready, "optimizer_step_boundary", _row("p80_optimizer_step"),
            ("optimizer_step_executed", "parameters_updated", "training_step_executed"),
        ),
        "launch": _unsafe_inventory_claims(
            p79_ready, "training_launch_boundary", _row("p80_training_launch"),
            ("training_launch_allowed", "training_launch_executed", "training_step_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p79_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p79_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p79_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p79_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p79_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(
            p79_ready, review={**_review(), "acknowledge_no_training_step_executed": False}
        ),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",), "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",), "review_missing_ack": ("ack_missing", "training_step"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in ("training_step_executed", "training_launch_executed"):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p79_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p79_ready,
        failure_history=[{"reason": "training_step_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p79_ready,
        rollback_history=[{"kind": "training_step_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p79_ready,
        failure_history=[{"reason": "closed_step_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)
    return {
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p79: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_training_step_execution_contract_p80(
        p79_tensor_transfer_execution_contract=p79,
        training_step_execution_evidence=_evidence() if evidence is None else evidence,
        training_step_execution_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p79_ready() -> dict[str, Any]:
    return _p79_gate(_p78_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    payload = {
        "evidence_id": "training_step_execution_contract_p80_v0",
        "evidence_version": "v0",
        "ok": True,
        "training_step_execution_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "training_step_execution_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_training_launch_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "sha256": "sha256:p80:training-step-execution-contract:ready",
        "artifact_digest": "sha256:p80:training-step-execution-contract:ready",
        "source": "temp/turbocore_v5_p80_training_step_execution_contract.json",
        **_safe_flags(),
    }
    for _name, field, _kind, item_id in INVENTORY_SPECS:
        payload[field] = [_row(item_id)]
    return payload


def _row(item_id: str) -> dict[str, Any]:
    return {"check_id": item_id, "ready": True, "source": f"temp/turbocore_v5_{item_id}.json", **_safe_flags()}


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-03T00:00:00Z",
        "requested_scope": P80_SCOPE,
        "approve_training_step_execution_contract": approve,
    }
    for field in UNSAFE_TRUE_FIELDS:
        review[f"approve_{field}"] = False
    for field in REQUIRED_REVIEW_ACKS:
        review[field] = True
    return review


def _safe_flags() -> dict[str, Any]:
    flags = {field: False for field in UNSAFE_TRUE_FIELDS}
    flags.update({field: {} for field in UNSAFE_NON_EMPTY_FIELDS})
    flags.update({"blocked_reasons": [], "promotion_blockers": []})
    return flags


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p79_ready(), review={**_review(), field: True})


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    copied = dict(value)
    copied.pop(key, None)
    return copied


def _without_many(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    copied = dict(value)
    for key in keys:
        copied.pop(key, None)
    return copied


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in UNSAFE_TRUE_FIELDS:
        assert report[field] is False, report
    assert report["post_p80_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["training_step_execution_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["training_step_execution_contract_ready"] is False, report
    assert report["decision"] == HOLD_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_reason_fragments(report: dict[str, Any], *fragments: str) -> None:
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "contract_ready": bool(report.get("training_step_execution_contract_ready", False)),
        "review_signed": bool(report.get("training_step_execution_review_signed", False)),
        "rollback_required": bool(report.get("rollback_required", False)),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
