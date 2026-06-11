"""Smoke checks for V5-P57 execution readiness review contract."""

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

from core.turbocore_v5_execution_readiness_review_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P57_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_execution_readiness_review_contract,
)
from lulynx_trainer.turbocore_v5_execution_replay_package_refresh_contract_smoke import (  # noqa: E402
    _gate as _p56_gate,
    _p55_ready,
)


READY_DECISION = "execution_readiness_review_recorded_default_off"
BLOCKED_DECISION = "execution_readiness_review_blocked_default_off"
HOLD_DECISION = "execution_readiness_review_hold_for_signed_review_default_off"
REJECTED_DECISION = "execution_readiness_review_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p56_ready = _p56_ready()
    ready = _gate(p56_ready)
    assert ready["ok"] is True, ready
    assert ready["execution_readiness_review_contract_ready"] is True, ready
    assert ready["execution_readiness_evidence_recorded"] is True, ready
    assert ready["execution_readiness_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p56_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p56_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p56_missing = _gate(None)
    _assert_blocked(p56_missing, "p56", "missing")
    p56_not_ready = _gate({**p56_ready, "ok": False, "execution_replay_package_refresh_contract_ready": False})
    _assert_blocked(p56_not_ready, "p56", "not_ready")
    p56_decision_mismatch = _gate({**p56_ready, "decision": "wrong"})
    _assert_blocked(p56_decision_mismatch, "p56", "not_ready")
    p56_post_fields = _gate({**p56_ready, "post_p56_request_fields": {"bad": True}})
    _assert_blocked(p56_post_fields, "post_p56_request_fields")

    p56_unsafe_cases = _unsafe_p56_cases(p56_ready)
    evidence_cases = _evidence_cases(p56_ready)
    inventory_cases = _inventory_cases(p56_ready)
    review_cases = _review_cases(p56_ready)
    history_cases = _history_cases(p56_ready)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p57_execution_readiness_review_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p56_missing": _summary(p56_missing),
        "p56_not_ready": _summary(p56_not_ready),
        "p56_decision_mismatch": _summary(p56_decision_mismatch),
        "p56_post_fields": _summary(p56_post_fields),
        "p56_unsafe_cases": p56_unsafe_cases,
        **evidence_cases,
        **inventory_cases,
        **review_cases,
        **history_cases,
    }


def _unsafe_p56_cases(p56_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "execution_readiness_approved", "runtime_state_refreshed", "execution_replay_executed",
        "execution_replay_package_refreshed", "artifact_replay_executed", "native_dispatch_executed",
        "kernel_launch_executed", "training_step_executed",
    ):
        report = _gate({**p56_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p56_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p56_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p56_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p56_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p56_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p56_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p56_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_review_only": _gate(p56_ready, evidence={**_evidence(), "execution_readiness_review_only": False}),
        "not_manual_only": _gate(p56_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p56_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_operator_contract": _gate(
            p56_ready,
            evidence={**_evidence(), "requires_later_operator_execution_contract": False},
        ),
        "evidence_not_ready": _gate(
            p56_ready,
            evidence={**_evidence(), "execution_readiness_review_contract_ready": False},
        ),
        "missing_section": _gate(p56_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p56_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p56_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p56_ready, evidence={**_evidence(), "blocked_reasons": ["operator_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_review_only": ("review_only",),
        "not_manual_only": ("manual_only",),
        "not_internal_only": ("internal_only",),
        "missing_later_operator_contract": ("later_operator",),
        "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",),
        "default_on": ("default_off",),
        "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("operator_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p56_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_checklist": _gate(p56_ready, evidence=_without(_evidence(), "execution_readiness_checklist")),
        "check_not_ready": _gate(
            p56_ready,
            evidence={**_evidence(), "execution_readiness_checklist": [{**_check_row(), "ready": False}]},
        ),
        "check_missing_source": _gate(
            p56_ready,
            evidence={**_evidence(), "execution_readiness_checklist": [_without(_check_row(), "source")]},
        ),
        "missing_rollback_preflight": _gate(
            p56_ready,
            evidence=_without(_evidence(), "rollback_preflight_inventory"),
        ),
        "rollback_not_ready": _gate(
            p56_ready,
            evidence={**_evidence(), "rollback_preflight_inventory": [{**_rollback_row(), "ready": False}]},
        ),
        "missing_observability_preflight": _gate(
            p56_ready,
            evidence=_without(_evidence(), "observability_preflight_inventory"),
        ),
        "observability_not_ready": _gate(
            p56_ready,
            evidence={**_evidence(), "observability_preflight_inventory": [{**_observability_row(), "ready": False}]},
        ),
    }
    fragments = {
        "missing_checklist": ("checklist_missing",),
        "check_not_ready": ("check_not_ready",),
        "check_missing_source": ("source_missing",),
        "missing_rollback_preflight": ("rollback_preflight_inventory_missing",),
        "rollback_not_ready": ("rollback_preflight_not_ready",),
        "missing_observability_preflight": ("observability_preflight_inventory_missing",),
        "observability_not_ready": ("observability_preflight_not_ready",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])

    unsafe_evidence_cases = _unsafe_evidence_cases(p56_ready)
    unsafe_request_cases = _unsafe_request_cases(p56_ready)
    unsafe_inventory_cases = {
        "checklist": _unsafe_inventory_claims(
            p56_ready,
            "execution_readiness_checklist",
            _check_row(),
            ("operator_execution_executed", "runtime_state_refreshed", "kernel_launch_executed"),
        ),
        "rollback_preflight": _unsafe_inventory_claims(
            p56_ready,
            "rollback_preflight_inventory",
            _rollback_row(),
            ("manual_execution_executed", "execution_replay_executed", "training_step_executed"),
        ),
        "observability_preflight": _unsafe_inventory_claims(
            p56_ready,
            "observability_preflight_inventory",
            _observability_row(),
            ("execution_readiness_approved", "native_dispatch_executed", "artifact_replay_executed"),
        ),
    }
    return {
        **{name: _summary(report) for name, report in cases.items()},
        "unsafe_evidence_cases": unsafe_evidence_cases,
        "unsafe_request_cases": unsafe_request_cases,
        "unsafe_inventory_cases": unsafe_inventory_cases,
    }


def _unsafe_evidence_cases(p56_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "execution_readiness_approved", "operator_execution_executed", "manual_execution_executed",
        "execution_replay_executed", "runtime_state_refreshed", "artifact_replay_executed",
        "native_dispatch_executed", "kernel_launch_executed", "training_step_executed",
    ):
        report = _gate(p56_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p56_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "api_route_registration", "ui_route_registration", "request_fields", "post_p57_request_fields",
        "execution_readiness_review_request", "operator_execution_payload", "manual_execution_payload",
    ):
        report = _gate(p56_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_claims(
    p56_ready: dict[str, Any],
    inventory: str,
    row: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p56_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p56_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p56_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p56_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p56_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(
            p56_ready,
            review={**_review(), "acknowledge_no_execution_approved": False},
        ),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "execution_approved"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])

    unsafe_cases = {}
    for base_field in (
        "execution_readiness_approved", "operator_execution_executed", "manual_execution_executed",
        "runtime_state_refreshed", "execution_replay_executed", "artifact_replay_executed",
        "native_dispatch_executed", "kernel_launch_executed", "training_step_executed",
        "training_launch_allowed", "request_adapter_mapping_allowed",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p56_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p56_ready,
        failure_history=[{"reason": "execution_readiness_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p56_ready,
        rollback_history=[{"kind": "execution_readiness_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p56_ready,
        failure_history=[{"reason": "closed_readiness_warning", "status": "closed", "severity": "high"}],
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
    p56: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_execution_readiness_review_contract(
        p56_execution_replay_package_refresh=p56,
        execution_readiness_evidence=_evidence() if evidence is None else evidence,
        execution_readiness_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p56_ready() -> dict[str, Any]:
    return _p56_gate(_p55_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    return {
        "evidence_id": "execution_readiness_review_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "execution_readiness_review_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "execution_readiness_review_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_operator_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "execution_readiness_checklist": [_check_row()],
        "rollback_preflight_inventory": [_rollback_row()],
        "observability_preflight_inventory": [_observability_row()],
        "sha256": "sha256:p57:execution-readiness-review:ready",
        "artifact_digest": "sha256:p57:execution-readiness-review:ready",
        "source": "temp/turbocore_v5_p57_execution_readiness_review.json",
        **_safe_flags(),
    }


def _check_row() -> dict[str, Any]:
    return {
        "check_id": "p57_manual_operator_boundary_ready",
        "ready": True,
        "source": "temp/turbocore_v5_p57_readiness_checklist.json",
        **_safe_flags(),
    }


def _rollback_row() -> dict[str, Any]:
    return {
        "preflight_id": "p57_rollback_preflight_ready",
        "ready": True,
        "source": "temp/turbocore_v5_p57_rollback_preflight.json",
        **_safe_flags(),
    }


def _observability_row() -> dict[str, Any]:
    return {
        "preflight_id": "p57_observability_preflight_ready",
        "ready": True,
        "source": "temp/turbocore_v5_p57_observability_preflight.json",
        **_safe_flags(),
    }


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": P57_SCOPE,
        "approve_execution_readiness_review_contract": approve,
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
    return _gate(_p56_ready(), review={**_review(), field: True})


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
    assert report["post_p57_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["execution_readiness_review_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["execution_readiness_review_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("execution_readiness_review_contract_ready", False)),
        "review_signed": bool(report.get("execution_readiness_review_signed", False)),
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
