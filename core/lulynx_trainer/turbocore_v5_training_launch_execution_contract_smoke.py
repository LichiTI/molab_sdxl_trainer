"""Smoke checks for V5-P73 training-launch execution contract."""

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

from core.turbocore_v5_training_launch_execution_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P73_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_training_launch_execution_contract,
)
from lulynx_trainer.turbocore_v5_run_dispatch_contract_smoke import (  # noqa: E402
    _gate as _p72_gate,
    _p71_ready,
)


READY_DECISION = "training_launch_execution_contract_recorded_default_off"
BLOCKED_DECISION = "training_launch_execution_contract_blocked_default_off"
HOLD_DECISION = "training_launch_execution_contract_hold_for_signed_review_default_off"
REJECTED_DECISION = "training_launch_execution_contract_rejected_default_off"
INVENTORY_SPECS = (
    ("launch_plan", "training_launch_execution_plan_inventory", "training_launch_execution_plan", "p73_launch_plan"),
    ("launch_precondition", "training_launch_execution_precondition_inventory", "training_launch_execution_precondition", "p73_launch_precondition"),
    ("authorization", "training_launch_authorization_boundary", "training_launch_authorization_boundary", "p73_authorization"),
    ("launch_request", "launch_request_boundary", "launch_request_boundary", "p73_launch_request"),
    ("launch_config", "launch_config_boundary", "launch_config_boundary", "p73_launch_config"),
    ("launch_job", "launch_job_boundary", "launch_job_boundary", "p73_launch_job"),
    ("launch_run", "launch_run_boundary", "launch_run_boundary", "p73_launch_run"),
    ("runtime", "runtime_dispatch_boundary", "runtime_dispatch_boundary", "p73_runtime"),
    ("training_runtime", "training_runtime_boundary", "training_runtime_boundary", "p73_training_runtime"),
    ("operator", "operator_launch_boundary", "operator_launch_boundary", "p73_operator"),
    ("observability", "observability_boundary", "observability_boundary", "p73_observability"),
    ("rollback", "rollback_policy_boundary", "rollback_policy", "p73_rollback"),
)


def run_smoke() -> dict[str, Any]:
    p72_ready = _p72_ready()
    ready = _gate(p72_ready)
    assert ready["ok"] is True, ready
    assert ready["training_launch_execution_contract_ready"] is True, ready
    assert ready["training_launch_execution_evidence_recorded"] is True, ready
    assert ready["training_launch_execution_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p72_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p72_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p72_missing = _gate(None)
    _assert_blocked(p72_missing, "p72", "missing")
    p72_not_ready = _gate({**p72_ready, "ok": False, "run_dispatch_contract_ready": False})
    _assert_blocked(p72_not_ready, "p72", "not_ready")
    p72_decision_mismatch = _gate({**p72_ready, "decision": "wrong"})
    _assert_blocked(p72_decision_mismatch, "p72", "not_ready")
    p72_post_fields = _gate({**p72_ready, "post_p72_request_fields": {"bad": True}})
    _assert_blocked(p72_post_fields, "post_p72_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p73_training_launch_execution_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p72_missing": _summary(p72_missing),
        "p72_not_ready": _summary(p72_not_ready),
        "p72_decision_mismatch": _summary(p72_decision_mismatch),
        "p72_post_fields": _summary(p72_post_fields),
        "p72_unsafe_cases": _unsafe_p72_cases(p72_ready),
        **_evidence_cases(p72_ready),
        **_inventory_cases(p72_ready),
        **_review_cases(p72_ready),
        **_history_cases(p72_ready),
    }


def _unsafe_p72_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_launch_executed", "training_runtime_started", "runtime_dispatch_executed",
        "native_dispatch_executed", "kernel_launch_executed", "parity_executed", "training_step_executed",
    ):
        report = _gate({**p72_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p72_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p72_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p72_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p72_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p72_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p72_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_launch_execution_contract": _gate(
            p72_ready, evidence={**_evidence(), "training_launch_execution_contract_only": False}
        ),
        "not_manual_only": _gate(p72_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p72_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_runtime_contract": _gate(
            p72_ready, evidence={**_evidence(), "requires_later_runtime_execution_contract": False}
        ),
        "evidence_not_ready": _gate(
            p72_ready, evidence={**_evidence(), "training_launch_execution_contract_ready": False}
        ),
        "missing_section": _gate(p72_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p72_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p72_ready, evidence={**_evidence(), "request_fields_emitted": True}),
        "evidence_blocker": _gate(p72_ready, evidence={**_evidence(), "blocked_reasons": ["launch_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",), "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",), "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",), "not_records_only": ("records_evidence_only",),
        "not_launch_execution_contract": ("training_launch_execution_contract_only",),
        "not_manual_only": ("manual_only",), "not_internal_only": ("internal_only",),
        "missing_later_runtime_contract": ("runtime_execution",), "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",), "default_on": ("default_off",),
        "request_adapter_on": ("request_fields_emitted",), "evidence_blocker": ("launch_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for name, field, kind, item_id in INVENTORY_SPECS:
        missing = _gate(p72_ready, evidence=_without(_evidence(), field))
        not_ready = _gate(p72_ready, evidence={**_evidence(), field: [{**_row(item_id), "ready": False}]})
        source_missing = _gate(p72_ready, evidence={**_evidence(), field: [_without(_row(item_id), "source")]})
        _assert_blocked(missing, f"{kind}_inventory_missing")
        _assert_blocked(not_ready, f"{kind}_not_ready")
        _assert_blocked(source_missing, f"{kind}_source_missing")
        cases[f"missing_{name}"] = _summary(missing)
        cases[f"{name}_not_ready"] = _summary(not_ready)
        cases[f"{name}_missing_source"] = _summary(source_missing)
    return {
        **cases,
        "unsafe_evidence_cases": _unsafe_evidence_cases(p72_ready),
        "unsafe_payload_cases": _unsafe_payload_cases(p72_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p72_ready),
    }


def _unsafe_evidence_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_launch_allowed", "training_launch_enabled", "training_launch_executed",
        "training_runtime_started", "training_process_started", "launch_request_executed",
        "launch_job_started", "launch_run_started", "runtime_dispatch_executed",
        "runtime_execution_executed", "native_dispatch_executed", "kernel_launch_executed",
        "parity_executed", "tensor_transfer_executed", "training_step_executed",
    ):
        report = _gate(p72_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_payload_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_launch_execution_contract_request", "training_launch_execution_contract_payload",
        "training_launch_execution_payload", "launch_request_payload", "launch_config_payload",
        "launch_job_payload", "launch_run_payload", "runtime_dispatch_payload", "training_runtime_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p72_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch_plan": _unsafe_inventory_claims(
            p72_ready, "training_launch_execution_plan_inventory", _row("p73_launch_plan"),
            ("training_launch_allowed", "training_launch_enabled", "training_launch_executed"),
        ),
        "launch_job": _unsafe_inventory_claims(
            p72_ready, "launch_job_boundary", _row("p73_launch_job"),
            ("launch_job_started", "training_runtime_started", "runtime_dispatch_executed"),
        ),
        "runtime": _unsafe_inventory_claims(
            p72_ready, "runtime_dispatch_boundary", _row("p73_runtime"),
            ("runtime_dispatch_executed", "native_dispatch_executed", "kernel_launch_executed"),
        ),
        "training_runtime": _unsafe_inventory_claims(
            p72_ready, "training_runtime_boundary", _row("p73_training_runtime"),
            ("training_runtime_started", "training_step_executed", "parity_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p72_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p72_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p72_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p72_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p72_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(
            p72_ready, review={**_review(), "acknowledge_no_training_launch_executed": False}
        ),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",), "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",), "review_missing_ack": ("ack_missing", "training_launch"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "training_launch_executed", "training_runtime_started", "runtime_dispatch_executed",
        "native_dispatch_executed", "kernel_launch_executed", "parity_executed", "training_step_executed",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p72_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p72_ready,
        failure_history=[{"reason": "training_launch_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p72_ready,
        rollback_history=[{"kind": "training_launch_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p72_ready,
        failure_history=[{"reason": "closed_launch_warning", "status": "closed", "severity": "high"}],
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
    p72: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_training_launch_execution_contract(
        p72_run_dispatch_contract=p72,
        training_launch_execution_evidence=_evidence() if evidence is None else evidence,
        training_launch_execution_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p72_ready() -> dict[str, Any]:
    return _p72_gate(_p71_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    payload = {
        "evidence_id": "training_launch_execution_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "training_launch_execution_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "training_launch_execution_contract_only": True,
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
        "sha256": "sha256:p73:training-launch-execution-contract:ready",
        "artifact_digest": "sha256:p73:training-launch-execution-contract:ready",
        "source": "temp/turbocore_v5_p73_training_launch_execution_contract.json",
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
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": P73_SCOPE,
        "approve_training_launch_execution_contract": approve,
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
    return _gate(_p72_ready(), review={**_review(), field: True})


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
    assert report["post_p73_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["training_launch_execution_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["training_launch_execution_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("training_launch_execution_contract_ready", False)),
        "review_signed": bool(report.get("training_launch_execution_review_signed", False)),
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
