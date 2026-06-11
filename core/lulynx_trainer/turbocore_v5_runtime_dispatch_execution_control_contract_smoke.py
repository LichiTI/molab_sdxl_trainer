"""Smoke checks for V5-P60 runtime dispatch execution control contract."""

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

from core.turbocore_v5_runtime_dispatch_execution_control_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P60_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_runtime_dispatch_execution_control_contract,
)
from lulynx_trainer.turbocore_v5_runtime_execution_contract_smoke import (  # noqa: E402
    _gate as _p59_gate,
    _p58_ready,
)


READY_DECISION = "runtime_dispatch_execution_control_recorded_default_off"
BLOCKED_DECISION = "runtime_dispatch_execution_control_blocked_default_off"
HOLD_DECISION = "runtime_dispatch_execution_control_hold_for_signed_review_default_off"
REJECTED_DECISION = "runtime_dispatch_execution_control_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p59_ready = _p59_ready()
    ready = _gate(p59_ready)
    assert ready["ok"] is True, ready
    assert ready["runtime_dispatch_execution_control_ready"] is True, ready
    assert ready["runtime_dispatch_execution_control_evidence_recorded"] is True, ready
    assert ready["runtime_dispatch_execution_control_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p59_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p59_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p59_missing = _gate(None)
    _assert_blocked(p59_missing, "p59", "missing")
    p59_not_ready = _gate({**p59_ready, "ok": False, "runtime_execution_contract_ready": False})
    _assert_blocked(p59_not_ready, "p59", "not_ready")
    p59_decision_mismatch = _gate({**p59_ready, "decision": "wrong"})
    _assert_blocked(p59_decision_mismatch, "p59", "not_ready")
    p59_post_fields = _gate({**p59_ready, "post_p59_request_fields": {"bad": True}})
    _assert_blocked(p59_post_fields, "post_p59_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p60_runtime_dispatch_execution_control_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p59_missing": _summary(p59_missing),
        "p59_not_ready": _summary(p59_not_ready),
        "p59_decision_mismatch": _summary(p59_decision_mismatch),
        "p59_post_fields": _summary(p59_post_fields),
        "p59_unsafe_cases": _unsafe_p59_cases(p59_ready),
        **_evidence_cases(p59_ready),
        **_inventory_cases(p59_ready),
        **_review_cases(p59_ready),
        **_history_cases(p59_ready),
    }


def _unsafe_p59_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "runtime_dispatch_executed", "runtime_execution_executed", "runtime_adapter_enabled",
        "runtime_state_refreshed", "native_dispatch_executed", "kernel_launch_executed",
        "training_step_executed", "request_adapter_mapping_allowed",
    ):
        report = _gate({**p59_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p59_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p59_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p59_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p59_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p59_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p59_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_control_only": _gate(p59_ready, evidence={**_evidence(), "runtime_dispatch_execution_control_only": False}),
        "not_manual_only": _gate(p59_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p59_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_native_contract": _gate(
            p59_ready,
            evidence={**_evidence(), "requires_later_native_dispatch_execution_contract": False},
        ),
        "evidence_not_ready": _gate(
            p59_ready,
            evidence={**_evidence(), "runtime_dispatch_execution_control_ready": False},
        ),
        "missing_section": _gate(p59_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p59_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p59_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p59_ready, evidence={**_evidence(), "blocked_reasons": ["dispatch_control_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_control_only": ("runtime_dispatch_execution_control_only",),
        "not_manual_only": ("manual_only",),
        "not_internal_only": ("internal_only",),
        "missing_later_native_contract": ("later_native_dispatch",),
        "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",),
        "default_on": ("default_off",),
        "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("dispatch_control_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_plan": _gate(p59_ready, evidence=_without(_evidence(), "runtime_dispatch_execution_control_plan_inventory")),
        "plan_not_ready": _gate(p59_ready, evidence={**_evidence(), "runtime_dispatch_execution_control_plan_inventory": [{**_row("p60_plan"), "ready": False}]}),
        "missing_authorization": _gate(p59_ready, evidence=_without(_evidence(), "runtime_dispatch_authorization_boundary")),
        "authorization_missing_source": _gate(p59_ready, evidence={**_evidence(), "runtime_dispatch_authorization_boundary": [_without(_row("p60_authorization"), "source")]}),
        "missing_precondition": _gate(p59_ready, evidence=_without(_evidence(), "runtime_dispatch_precondition_inventory")),
        "precondition_not_ready": _gate(p59_ready, evidence={**_evidence(), "runtime_dispatch_precondition_inventory": [{**_row("p60_precondition"), "ready": False}]}),
        "missing_adapter_lock": _gate(p59_ready, evidence=_without(_evidence(), "runtime_adapter_lock_boundary")),
        "adapter_lock_not_ready": _gate(p59_ready, evidence={**_evidence(), "runtime_adapter_lock_boundary": [{**_row("p60_adapter_lock"), "ready": False}]}),
        "missing_state_lock": _gate(p59_ready, evidence=_without(_evidence(), "runtime_state_lock_boundary")),
        "state_lock_not_ready": _gate(p59_ready, evidence={**_evidence(), "runtime_state_lock_boundary": [{**_row("p60_state_lock"), "ready": False}]}),
        "missing_native_boundary": _gate(p59_ready, evidence=_without(_evidence(), "native_dispatch_boundary")),
        "native_boundary_not_ready": _gate(p59_ready, evidence={**_evidence(), "native_dispatch_boundary": [{**_row("p60_native_boundary"), "ready": False}]}),
        "missing_kernel_boundary": _gate(p59_ready, evidence=_without(_evidence(), "kernel_launch_boundary")),
        "kernel_boundary_not_ready": _gate(p59_ready, evidence={**_evidence(), "kernel_launch_boundary": [{**_row("p60_kernel_boundary"), "ready": False}]}),
        "missing_rollback": _gate(p59_ready, evidence=_without(_evidence(), "rollback_preflight_inventory")),
        "rollback_not_ready": _gate(p59_ready, evidence={**_evidence(), "rollback_preflight_inventory": [{**_row("p60_rollback"), "ready": False}]}),
        "missing_observability": _gate(p59_ready, evidence=_without(_evidence(), "observability_preflight_inventory")),
        "observability_not_ready": _gate(p59_ready, evidence={**_evidence(), "observability_preflight_inventory": [{**_row("p60_observability"), "ready": False}]}),
    }
    fragments = {
        "missing_plan": ("control_plan_inventory_missing",),
        "plan_not_ready": ("control_plan_not_ready",),
        "missing_authorization": ("authorization_inventory_missing",),
        "authorization_missing_source": ("authorization_source_missing",),
        "missing_precondition": ("precondition_inventory_missing",),
        "precondition_not_ready": ("precondition_not_ready",),
        "missing_adapter_lock": ("adapter_lock_inventory_missing",),
        "adapter_lock_not_ready": ("adapter_lock_not_ready",),
        "missing_state_lock": ("state_lock_inventory_missing",),
        "state_lock_not_ready": ("state_lock_not_ready",),
        "missing_native_boundary": ("native_dispatch_boundary_inventory_missing",),
        "native_boundary_not_ready": ("native_dispatch_boundary_not_ready",),
        "missing_kernel_boundary": ("kernel_launch_boundary_inventory_missing",),
        "kernel_boundary_not_ready": ("kernel_launch_boundary_not_ready",),
        "missing_rollback": ("rollback_preflight_inventory_missing",),
        "rollback_not_ready": ("rollback_preflight_not_ready",),
        "missing_observability": ("observability_preflight_inventory_missing",),
        "observability_not_ready": ("observability_preflight_not_ready",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {
        **{name: _summary(report) for name, report in cases.items()},
        "unsafe_evidence_cases": _unsafe_evidence_cases(p59_ready),
        "unsafe_request_cases": _unsafe_request_cases(p59_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p59_ready),
    }


def _unsafe_evidence_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "runtime_dispatch_approved", "runtime_dispatch_executed", "runtime_dispatch_execution_allowed",
        "native_dispatch_enabled", "native_dispatch_executed", "kernel_launch_executed",
        "parity_check_executed", "training_step_executed", "runtime_execution_executed",
        "runtime_state_refreshed", "runtime_adapter_enabled",
    ):
        report = _gate(p59_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "api_route_registration", "ui_route_registration", "request_fields", "post_p60_request_fields",
        "runtime_dispatch_execution_control_request", "runtime_dispatch_execution_plan_payload",
        "runtime_dispatch_authorization_payload", "runtime_adapter_handoff_payload",
        "native_dispatch_handoff_payload", "kernel_launch_payload", "training_step_payload",
    ):
        report = _gate(p59_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": _unsafe_inventory_claims(
            p59_ready,
            "runtime_dispatch_execution_control_plan_inventory",
            _row("p60_plan"),
            ("runtime_dispatch_executed", "native_dispatch_executed", "training_step_executed"),
        ),
        "native": _unsafe_inventory_claims(
            p59_ready,
            "native_dispatch_boundary",
            _row("p60_native_boundary"),
            ("native_dispatch_enabled", "native_dispatch_executed", "kernel_launch_executed"),
        ),
        "kernel": _unsafe_inventory_claims(
            p59_ready,
            "kernel_launch_boundary",
            _row("p60_kernel_boundary"),
            ("kernel_launch_executed", "parity_check_executed", "training_step_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p59_ready: dict[str, Any],
    inventory: str,
    row: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p59_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p59_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p59_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p59_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p59_ready, review={**_review(), "acknowledge_no_runtime_dispatch_executed": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "runtime_dispatch"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "runtime_dispatch_approved", "runtime_dispatch_executed", "runtime_dispatch_execution_allowed",
        "native_dispatch_enabled", "native_dispatch_executed", "kernel_launch_executed",
        "parity_check_executed", "training_step_executed", "training_launch_allowed",
        "request_adapter_mapping_allowed",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p59_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p59_ready,
        failure_history=[{"reason": "runtime_dispatch_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p59_ready,
        rollback_history=[{"kind": "runtime_dispatch_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p59_ready,
        failure_history=[{"reason": "closed_runtime_dispatch_warning", "status": "closed", "severity": "high"}],
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
    p59: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_runtime_dispatch_execution_control_contract(
        p59_runtime_execution_contract=p59,
        runtime_dispatch_execution_control_evidence=_evidence() if evidence is None else evidence,
        runtime_dispatch_execution_control_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p59_ready() -> dict[str, Any]:
    return _p59_gate(_p58_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    return {
        "evidence_id": "runtime_dispatch_execution_control_v0",
        "evidence_version": "v0",
        "ok": True,
        "runtime_dispatch_execution_control_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "runtime_dispatch_execution_control_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_native_dispatch_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "runtime_dispatch_execution_control_plan_inventory": [_row("p60_plan")],
        "runtime_dispatch_authorization_boundary": [_row("p60_authorization")],
        "runtime_dispatch_precondition_inventory": [_row("p60_precondition")],
        "runtime_adapter_lock_boundary": [_row("p60_adapter_lock")],
        "runtime_state_lock_boundary": [_row("p60_state_lock")],
        "native_dispatch_boundary": [_row("p60_native_boundary")],
        "kernel_launch_boundary": [_row("p60_kernel_boundary")],
        "rollback_preflight_inventory": [_row("p60_rollback")],
        "observability_preflight_inventory": [_row("p60_observability")],
        "sha256": "sha256:p60:runtime-dispatch-execution-control:ready",
        "artifact_digest": "sha256:p60:runtime-dispatch-execution-control:ready",
        "source": "temp/turbocore_v5_p60_runtime_dispatch_execution_control.json",
        **_safe_flags(),
    }


def _row(item_id: str) -> dict[str, Any]:
    return {"check_id": item_id, "ready": True, "source": f"temp/turbocore_v5_{item_id}.json", **_safe_flags()}


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": P60_SCOPE,
        "approve_runtime_dispatch_execution_control_contract": approve,
    }
    for field in UNSAFE_TRUE_FIELDS:
        if field != "default_behavior_changed":
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
    return _gate(_p59_ready(), review={**_review(), field: True})


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
    assert report["post_p60_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["runtime_dispatch_execution_control_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["runtime_dispatch_execution_control_ready"] is False, report
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
        "contract_ready": bool(report.get("runtime_dispatch_execution_control_ready", False)),
        "review_signed": bool(report.get("runtime_dispatch_execution_control_review_signed", False)),
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
