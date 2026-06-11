"""Smoke checks for V5-P58 operator execution boundary contract."""

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

from core.turbocore_v5_operator_execution_boundary_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P58_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_operator_execution_boundary_contract,
)
from lulynx_trainer.turbocore_v5_execution_readiness_review_contract_smoke import (  # noqa: E402
    _gate as _p57_gate,
    _p56_ready,
)


READY_DECISION = "operator_execution_boundary_recorded_default_off"
BLOCKED_DECISION = "operator_execution_boundary_blocked_default_off"
HOLD_DECISION = "operator_execution_boundary_hold_for_signed_review_default_off"
REJECTED_DECISION = "operator_execution_boundary_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p57_ready = _p57_ready()
    ready = _gate(p57_ready)
    assert ready["ok"] is True, ready
    assert ready["operator_execution_boundary_contract_ready"] is True, ready
    assert ready["operator_execution_boundary_evidence_recorded"] is True, ready
    assert ready["operator_execution_boundary_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p57_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p57_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p57_missing = _gate(None)
    _assert_blocked(p57_missing, "p57", "missing")
    p57_not_ready = _gate({**p57_ready, "ok": False, "execution_readiness_review_contract_ready": False})
    _assert_blocked(p57_not_ready, "p57", "not_ready")
    p57_decision_mismatch = _gate({**p57_ready, "decision": "wrong"})
    _assert_blocked(p57_decision_mismatch, "p57", "not_ready")
    p57_post_fields = _gate({**p57_ready, "post_p57_request_fields": {"bad": True}})
    _assert_blocked(p57_post_fields, "post_p57_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p58_operator_execution_boundary_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p57_missing": _summary(p57_missing),
        "p57_not_ready": _summary(p57_not_ready),
        "p57_decision_mismatch": _summary(p57_decision_mismatch),
        "p57_post_fields": _summary(p57_post_fields),
        "p57_unsafe_cases": _unsafe_p57_cases(p57_ready),
        **_evidence_cases(p57_ready),
        **_inventory_cases(p57_ready),
        **_review_cases(p57_ready),
        **_history_cases(p57_ready),
    }


def _unsafe_p57_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "operator_execution_executed", "manual_execution_executed", "runtime_state_refreshed",
        "execution_replay_executed", "artifact_replay_executed", "native_dispatch_executed",
        "kernel_launch_executed", "training_step_executed",
    ):
        report = _gate({**p57_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p57_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p57_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p57_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p57_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p57_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p57_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_boundary_contract": _gate(p57_ready, evidence={**_evidence(), "operator_execution_boundary_only": False}),
        "not_manual_only": _gate(p57_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p57_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_runtime_contract": _gate(
            p57_ready,
            evidence={**_evidence(), "requires_later_runtime_execution_contract": False},
        ),
        "evidence_not_ready": _gate(p57_ready, evidence={**_evidence(), "operator_execution_boundary_contract_ready": False}),
        "missing_section": _gate(p57_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p57_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p57_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p57_ready, evidence={**_evidence(), "blocked_reasons": ["operator_boundary_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_boundary_contract": ("boundary_only",),
        "not_manual_only": ("manual_only",),
        "not_internal_only": ("internal_only",),
        "missing_later_runtime_contract": ("later_runtime",),
        "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",),
        "default_on": ("default_off",),
        "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("operator_boundary_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_plan": _gate(p57_ready, evidence=_without(_evidence(), "operator_execution_plan_inventory")),
        "plan_not_ready": _gate(p57_ready, evidence={**_evidence(), "operator_execution_plan_inventory": [{**_plan_row(), "ready": False}]}),
        "missing_identity": _gate(p57_ready, evidence=_without(_evidence(), "operator_identity_boundary")),
        "identity_missing_source": _gate(p57_ready, evidence={**_evidence(), "operator_identity_boundary": [_without(_identity_row(), "source")]}),
        "missing_scope": _gate(p57_ready, evidence=_without(_evidence(), "operator_scope_boundary")),
        "scope_not_ready": _gate(p57_ready, evidence={**_evidence(), "operator_scope_boundary": [{**_scope_row(), "ready": False}]}),
        "missing_precondition": _gate(p57_ready, evidence=_without(_evidence(), "execution_precondition_inventory")),
        "precondition_not_ready": _gate(p57_ready, evidence={**_evidence(), "execution_precondition_inventory": [{**_precondition_row(), "ready": False}]}),
        "missing_rollback": _gate(p57_ready, evidence=_without(_evidence(), "rollback_preflight_inventory")),
        "rollback_not_ready": _gate(p57_ready, evidence={**_evidence(), "rollback_preflight_inventory": [{**_rollback_row(), "ready": False}]}),
        "missing_observability": _gate(p57_ready, evidence=_without(_evidence(), "observability_preflight_inventory")),
        "observability_not_ready": _gate(p57_ready, evidence={**_evidence(), "observability_preflight_inventory": [{**_observability_row(), "ready": False}]}),
    }
    fragments = {
        "missing_plan": ("operator_execution_plan_inventory_missing",),
        "plan_not_ready": ("operator_execution_plan_not_ready",),
        "missing_identity": ("operator_identity_inventory_missing",),
        "identity_missing_source": ("operator_identity_source_missing",),
        "missing_scope": ("operator_scope_inventory_missing",),
        "scope_not_ready": ("operator_scope_not_ready",),
        "missing_precondition": ("execution_precondition_inventory_missing",),
        "precondition_not_ready": ("execution_precondition_not_ready",),
        "missing_rollback": ("rollback_preflight_inventory_missing",),
        "rollback_not_ready": ("rollback_preflight_not_ready",),
        "missing_observability": ("observability_preflight_inventory_missing",),
        "observability_not_ready": ("observability_preflight_not_ready",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {
        **{name: _summary(report) for name, report in cases.items()},
        "unsafe_evidence_cases": _unsafe_evidence_cases(p57_ready),
        "unsafe_request_cases": _unsafe_request_cases(p57_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p57_ready),
    }


def _unsafe_evidence_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "operator_execution_approved", "operator_execution_executed", "manual_execution_executed",
        "manual_operator_execution_executed", "runtime_execution_executed", "runtime_state_refreshed",
        "execution_replay_executed", "artifact_replay_executed", "native_dispatch_executed",
        "kernel_launch_executed", "training_step_executed",
    ):
        report = _gate(p57_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "api_route_registration", "ui_route_registration", "request_fields", "post_p58_request_fields",
        "operator_execution_boundary_request", "operator_execution_plan_payload",
        "manual_operator_execution_payload", "runtime_execution_payload",
    ):
        report = _gate(p57_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": _unsafe_inventory_claims(
            p57_ready,
            "operator_execution_plan_inventory",
            _plan_row(),
            ("operator_execution_executed", "manual_operator_execution_executed", "runtime_execution_executed"),
        ),
        "identity": _unsafe_inventory_claims(
            p57_ready,
            "operator_identity_boundary",
            _identity_row(),
            ("operator_execution_approved", "operator_execution_requested"),
        ),
        "precondition": _unsafe_inventory_claims(
            p57_ready,
            "execution_precondition_inventory",
            _precondition_row(),
            ("runtime_state_refreshed", "execution_replay_executed", "kernel_launch_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p57_ready: dict[str, Any],
    inventory: str,
    row: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p57_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p57_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p57_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p57_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p57_ready, review={**_review(), "acknowledge_no_operator_execution_executed": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "operator_execution"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])

    unsafe_cases = {}
    for base_field in (
        "operator_execution_approved", "operator_execution_executed", "manual_execution_executed",
        "manual_operator_execution_executed", "runtime_execution_executed", "runtime_state_refreshed",
        "execution_replay_executed", "artifact_replay_executed", "native_dispatch_executed",
        "kernel_launch_executed", "training_step_executed", "training_launch_allowed",
        "request_adapter_mapping_allowed",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p57_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p57_ready,
        failure_history=[{"reason": "operator_execution_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p57_ready,
        rollback_history=[{"kind": "operator_execution_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p57_ready,
        failure_history=[{"reason": "closed_operator_execution_warning", "status": "closed", "severity": "high"}],
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
    p57: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_operator_execution_boundary_contract(
        p57_execution_readiness_review=p57,
        operator_execution_evidence=_evidence() if evidence is None else evidence,
        operator_execution_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p57_ready() -> dict[str, Any]:
    return _p57_gate(_p56_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    return {
        "evidence_id": "operator_execution_boundary_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "operator_execution_boundary_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "operator_execution_boundary_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_runtime_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "operator_execution_plan_inventory": [_plan_row()],
        "operator_identity_boundary": [_identity_row()],
        "operator_scope_boundary": [_scope_row()],
        "execution_precondition_inventory": [_precondition_row()],
        "rollback_preflight_inventory": [_rollback_row()],
        "observability_preflight_inventory": [_observability_row()],
        "sha256": "sha256:p58:operator-execution-boundary:ready",
        "artifact_digest": "sha256:p58:operator-execution-boundary:ready",
        "source": "temp/turbocore_v5_p58_operator_execution_boundary.json",
        **_safe_flags(),
    }


def _plan_row() -> dict[str, Any]:
    return {"plan_id": "p58_operator_plan_ready", "ready": True, "source": "temp/turbocore_v5_p58_plan.json", **_safe_flags()}


def _identity_row() -> dict[str, Any]:
    return {"operator_id": "manual_owner_operator", "ready": True, "source": "temp/turbocore_v5_p58_identity.json", **_safe_flags()}


def _scope_row() -> dict[str, Any]:
    return {"scope_id": "p58_operator_scope_default_off", "ready": True, "source": "temp/turbocore_v5_p58_scope.json", **_safe_flags()}


def _precondition_row() -> dict[str, Any]:
    return {"check_id": "p58_execution_precondition_ready", "ready": True, "source": "temp/turbocore_v5_p58_precondition.json", **_safe_flags()}


def _rollback_row() -> dict[str, Any]:
    return {"check_id": "p58_rollback_preflight_ready", "ready": True, "source": "temp/turbocore_v5_p58_rollback.json", **_safe_flags()}


def _observability_row() -> dict[str, Any]:
    return {"check_id": "p58_observability_preflight_ready", "ready": True, "source": "temp/turbocore_v5_p58_observability.json", **_safe_flags()}


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": P58_SCOPE,
        "approve_operator_execution_boundary_contract": approve,
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
    return _gate(_p57_ready(), review={**_review(), field: True})


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
    assert report["post_p58_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["operator_execution_boundary_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["operator_execution_boundary_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("operator_execution_boundary_contract_ready", False)),
        "review_signed": bool(report.get("operator_execution_boundary_review_signed", False)),
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
