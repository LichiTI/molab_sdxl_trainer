"""Smoke checks for V5-P81 training-launch execution contract."""

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

from core.turbocore_v5_training_launch_execution_contract_p81 import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P81_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_training_launch_execution_contract_p81,
)
from lulynx_trainer.turbocore_v5_training_step_execution_contract_p80_smoke import (  # noqa: E402
    _gate as _p80_gate,
    _p79_ready,
)


READY_DECISION = "training_launch_execution_contract_p81_recorded_default_off"
BLOCKED_DECISION = "training_launch_execution_contract_p81_blocked_default_off"
HOLD_DECISION = "training_launch_execution_contract_p81_hold_for_signed_review_default_off"
REJECTED_DECISION = "training_launch_execution_contract_p81_rejected_default_off"
INVENTORY_SPECS = (
    ("launch_plan", "training_launch_execution_plan_inventory", "training_launch_execution_plan", "p81_launch_plan"),
    ("launch_precondition", "training_launch_execution_precondition_inventory", "training_launch_execution_precondition", "p81_launch_precondition"),
    ("authorization", "training_launch_authorization_boundary", "training_launch_authorization_boundary", "p81_authorization"),
    ("runtime", "training_runtime_boundary", "training_runtime_boundary", "p81_runtime"),
    ("process", "training_process_boundary", "training_process_boundary", "p81_process"),
    ("run_dispatch", "run_dispatch_boundary", "run_dispatch_boundary", "p81_run_dispatch"),
    ("scheduler", "scheduler_dispatch_boundary", "scheduler_dispatch_boundary", "p81_scheduler"),
    ("request", "request_execution_boundary", "request_execution_boundary", "p81_request"),
    ("job", "job_creation_boundary", "job_creation_boundary", "p81_job"),
    ("operator", "operator_training_launch_boundary", "operator_training_launch_boundary", "p81_operator"),
    ("observability", "observability_boundary", "observability_boundary", "p81_observability"),
    ("rollback", "rollback_policy_boundary", "rollback_policy", "p81_rollback"),
)


def run_smoke() -> dict[str, Any]:
    p80_ready = _p80_ready()
    ready = _gate(p80_ready)
    assert ready["ok"] is True, ready
    assert ready["training_launch_execution_contract_ready"] is True, ready
    assert ready["training_launch_execution_evidence_recorded"] is True, ready
    assert ready["training_launch_execution_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p80_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p80_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p80_missing = _gate(None)
    _assert_blocked(p80_missing, "p80", "missing")
    p80_not_ready = _gate({**p80_ready, "ok": False, "training_step_execution_contract_ready": False})
    _assert_blocked(p80_not_ready, "p80", "not_ready")
    p80_decision_mismatch = _gate({**p80_ready, "decision": "wrong"})
    _assert_blocked(p80_decision_mismatch, "p80", "not_ready")
    p80_post_fields = _gate({**p80_ready, "post_p80_request_fields": {"bad": True}})
    _assert_blocked(p80_post_fields, "post_p80_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p81_training_launch_execution_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p80_missing": _summary(p80_missing),
        "p80_not_ready": _summary(p80_not_ready),
        "p80_decision_mismatch": _summary(p80_decision_mismatch),
        "p80_post_fields": _summary(p80_post_fields),
        "p80_unsafe_cases": _unsafe_p80_cases(p80_ready),
        **_evidence_cases(p80_ready),
        **_inventory_cases(p80_ready),
        **_review_cases(p80_ready),
        **_history_cases(p80_ready),
    }


def _unsafe_p80_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_launch_executed", "training_step_executed", "tensor_transfer_executed",
        "parity_executed", "kernel_launch_executed", "native_dispatch_executed",
        "runtime_dispatch_executed", "runtime_execution_executed",
        "runtime_state_refreshed", "runtime_adapter_enabled",
    ):
        report = _gate({**p80_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p80_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p80_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p80_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p80_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p80_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p80_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_launch_contract": _gate(
            p80_ready, evidence={**_evidence(), "training_launch_execution_contract_only": False}
        ),
        "not_manual_only": _gate(p80_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p80_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_runtime_contract": _gate(
            p80_ready, evidence={**_evidence(), "requires_later_runtime_execution_contract": False}
        ),
        "evidence_not_ready": _gate(
            p80_ready, evidence={**_evidence(), "training_launch_execution_contract_ready": False}
        ),
        "missing_section": _gate(p80_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p80_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p80_ready, evidence={**_evidence(), "request_fields_emitted": True}),
        "evidence_blocker": _gate(p80_ready, evidence={**_evidence(), "blocked_reasons": ["launch_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",), "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",), "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",), "not_records_only": ("records_evidence_only",),
        "not_launch_contract": ("training_launch_execution_contract_only",),
        "not_manual_only": ("manual_only",), "not_internal_only": ("internal_only",),
        "missing_later_runtime_contract": ("runtime",), "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",), "default_on": ("default_off",),
        "request_adapter_on": ("request_fields_emitted",), "evidence_blocker": ("launch_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for name, field, kind, item_id in INVENTORY_SPECS:
        missing = _gate(p80_ready, evidence=_without(_evidence(), field))
        not_ready = _gate(p80_ready, evidence={**_evidence(), field: [{**_row(item_id), "ready": False}]})
        source_missing = _gate(p80_ready, evidence={**_evidence(), field: [_without(_row(item_id), "source")]})
        _assert_blocked(missing, f"{kind}_inventory_missing")
        _assert_blocked(not_ready, f"{kind}_not_ready")
        _assert_blocked(source_missing, f"{kind}_source_missing")
        cases[f"missing_{name}"] = _summary(missing)
        cases[f"{name}_not_ready"] = _summary(not_ready)
        cases[f"{name}_missing_source"] = _summary(source_missing)
    return {
        **cases,
        "unsafe_evidence_cases": _unsafe_evidence_cases(p80_ready),
        "unsafe_payload_cases": _unsafe_payload_cases(p80_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p80_ready),
    }


def _unsafe_evidence_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_launch_approved", "training_launch_enabled", "training_launch_requested",
        "training_launch_started", "training_launch_executed", "training_runtime_started",
        "training_process_started", "run_dispatch_executed", "scheduler_dispatch_executed",
        "request_execution_executed", "job_created",
    ):
        report = _gate(p80_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_payload_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "training_launch_execution_contract_request", "training_launch_execution_contract_payload",
        "training_launch_execution_payload", "training_launch_authorization_payload",
        "training_runtime_payload", "training_process_payload", "run_dispatch_payload",
        "scheduler_dispatch_payload", "request_execution_payload", "job_creation_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p80_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch_plan": _unsafe_inventory_claims(
            p80_ready, "training_launch_execution_plan_inventory", _row("p81_launch_plan"),
            ("training_launch_allowed", "training_launch_requested", "training_launch_executed"),
        ),
        "runtime": _unsafe_inventory_claims(
            p80_ready, "training_runtime_boundary", _row("p81_runtime"),
            ("training_runtime_started", "runtime_execution_executed", "training_launch_executed"),
        ),
        "run_dispatch": _unsafe_inventory_claims(
            p80_ready, "run_dispatch_boundary", _row("p81_run_dispatch"),
            ("run_dispatch_allowed", "run_dispatch_executed", "scheduler_dispatch_executed"),
        ),
        "job": _unsafe_inventory_claims(
            p80_ready, "job_creation_boundary", _row("p81_job"),
            ("job_created", "job_record_written", "queue_enqueue_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p80_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p80_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p80_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p80_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p80_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(
            p80_ready, review={**_review(), "acknowledge_no_training_launch_executed": False}
        ),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",), "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",), "review_missing_ack": ("ack_missing", "training_launch"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in ("training_launch_executed", "run_dispatch_executed", "job_created"):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p80_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p80_ready,
        failure_history=[{"reason": "training_launch_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p80_ready,
        rollback_history=[{"kind": "training_launch_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p80_ready,
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
    p80: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_training_launch_execution_contract_p81(
        p80_training_step_execution_contract=p80,
        training_launch_execution_evidence=_evidence() if evidence is None else evidence,
        training_launch_execution_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p80_ready() -> dict[str, Any]:
    return _p80_gate(_p79_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    payload = {
        "evidence_id": "training_launch_execution_contract_p81_v0",
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
        "sha256": "sha256:p81:training-launch-execution-contract:ready",
        "artifact_digest": "sha256:p81:training-launch-execution-contract:ready",
        "source": "temp/turbocore_v5_p81_training_launch_execution_contract.json",
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
        "requested_scope": P81_SCOPE,
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
    return _gate(_p80_ready(), review={**_review(), field: True})


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
    assert report["post_p81_request_fields"] == {}, report


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
