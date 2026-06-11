"""Smoke checks for V5-P71 request-execution/job-creation contract."""

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

from core.turbocore_v5_request_execution_job_creation_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P71_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_request_execution_job_creation_contract,
)
from lulynx_trainer.turbocore_v5_request_submission_contract_smoke import (  # noqa: E402
    _gate as _p70_gate,
    _p69_ready,
)


READY_DECISION = "request_execution_job_creation_contract_recorded_default_off"
BLOCKED_DECISION = "request_execution_job_creation_contract_blocked_default_off"
HOLD_DECISION = "request_execution_job_creation_contract_hold_for_signed_review_default_off"
REJECTED_DECISION = "request_execution_job_creation_contract_rejected_default_off"
INVENTORY_SPECS = (
    ("execution_plan", "request_execution_plan_inventory", "request_execution_plan", "p71_execution_plan"),
    ("execution_precondition", "request_execution_precondition_inventory", "request_execution_precondition", "p71_execution_precondition"),
    ("job_plan", "job_creation_plan_inventory", "job_creation_plan", "p71_job_plan"),
    ("job_precondition", "job_creation_precondition_inventory", "job_creation_precondition", "p71_job_precondition"),
    ("store", "request_store_boundary", "request_store_boundary", "p71_store"),
    ("job_record", "job_record_boundary", "job_record_boundary", "p71_job_record"),
    ("queue", "queue_enqueue_boundary", "queue_enqueue_boundary", "p71_queue"),
    ("run", "run_dispatch_boundary", "run_dispatch_boundary", "p71_run"),
    ("launch", "training_launch_boundary", "training_launch_boundary", "p71_launch"),
    ("resolution", "execution_resolution_boundary", "execution_resolution_boundary", "p71_resolution"),
    ("operator", "operator_execution_boundary", "operator_execution_boundary", "p71_operator"),
    ("observability", "observability_boundary", "observability_boundary", "p71_observability"),
    ("rollback", "rollback_policy_boundary", "rollback_policy", "p71_rollback"),
)


def run_smoke() -> dict[str, Any]:
    p70_ready = _p70_ready()
    ready = _gate(p70_ready)
    assert ready["ok"] is True, ready
    assert ready["request_execution_job_creation_contract_ready"] is True, ready
    assert ready["request_execution_job_creation_evidence_recorded"] is True, ready
    assert ready["request_execution_job_creation_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p70_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p70_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p70_missing = _gate(None)
    _assert_blocked(p70_missing, "p70", "missing")
    p70_not_ready = _gate({**p70_ready, "ok": False, "request_submission_contract_ready": False})
    _assert_blocked(p70_not_ready, "p70", "not_ready")
    p70_decision_mismatch = _gate({**p70_ready, "decision": "wrong"})
    _assert_blocked(p70_decision_mismatch, "p70", "not_ready")
    p70_post_fields = _gate({**p70_ready, "post_p70_request_fields": {"bad": True}})
    _assert_blocked(p70_post_fields, "post_p70_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p71_request_execution_job_creation_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p70_missing": _summary(p70_missing),
        "p70_not_ready": _summary(p70_not_ready),
        "p70_decision_mismatch": _summary(p70_decision_mismatch),
        "p70_post_fields": _summary(p70_post_fields),
        "p70_unsafe_cases": _unsafe_p70_cases(p70_ready),
        **_evidence_cases(p70_ready),
        **_inventory_cases(p70_ready),
        **_review_cases(p70_ready),
        **_history_cases(p70_ready),
    }


def _unsafe_p70_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "request_execution_executed", "job_creation_executed", "job_record_written",
        "queue_enqueued", "training_job_created", "runs_dispatched", "training_launch_allowed",
        "training_launch_executed", "runtime_dispatch_executed",
    ):
        report = _gate({**p70_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p70_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p70_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p70_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p70_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p70_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p70_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_execution_contract": _gate(p70_ready, evidence={**_evidence(), "request_execution_job_creation_contract_only": False}),
        "not_manual_only": _gate(p70_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p70_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_run_dispatch_contract": _gate(
            p70_ready,
            evidence={**_evidence(), "requires_later_run_dispatch_contract": False},
        ),
        "evidence_not_ready": _gate(p70_ready, evidence={**_evidence(), "request_execution_job_creation_contract_ready": False}),
        "missing_section": _gate(p70_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p70_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p70_ready, evidence={**_evidence(), "request_fields_emitted": True}),
        "evidence_blocker": _gate(p70_ready, evidence={**_evidence(), "blocked_reasons": ["job_creation_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",), "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",), "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",), "not_records_only": ("records_evidence_only",),
        "not_execution_contract": ("request_execution_job_creation_contract_only",),
        "not_manual_only": ("manual_only",), "not_internal_only": ("internal_only",),
        "missing_later_run_dispatch_contract": ("run_dispatch",),
        "evidence_not_ready": ("not_ready",), "missing_section": ("section_missing",),
        "default_on": ("default_off",), "request_adapter_on": ("request_fields_emitted",),
        "evidence_blocker": ("job_creation_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for name, field, kind, item_id in INVENTORY_SPECS:
        missing = _gate(p70_ready, evidence=_without(_evidence(), field))
        not_ready = _gate(p70_ready, evidence={**_evidence(), field: [{**_row(item_id), "ready": False}]})
        source_missing = _gate(p70_ready, evidence={**_evidence(), field: [_without(_row(item_id), "source")]})
        _assert_blocked(missing, f"{kind}_inventory_missing")
        _assert_blocked(not_ready, f"{kind}_not_ready")
        _assert_blocked(source_missing, f"{kind}_source_missing")
        cases[f"missing_{name}"] = _summary(missing)
        cases[f"{name}_not_ready"] = _summary(not_ready)
        cases[f"{name}_missing_source"] = _summary(source_missing)
    return {
        **cases,
        "unsafe_evidence_cases": _unsafe_evidence_cases(p70_ready),
        "unsafe_payload_cases": _unsafe_payload_cases(p70_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p70_ready),
    }


def _unsafe_evidence_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "request_execution_enabled", "request_execution_executed", "job_creation_enabled",
        "job_creation_executed", "job_record_written", "job_store_written", "queue_enqueued",
        "scheduler_enqueued", "run_dispatch_executed", "runs_dispatched", "training_job_created",
        "training_launch_allowed", "training_launch_executed", "runtime_dispatch_executed",
    ):
        report = _gate(p70_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_payload_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "request_execution_job_creation_contract_request",
        "request_execution_job_creation_contract_payload", "request_execution_payload",
        "job_creation_payload", "job_record_payload", "job_store_payload", "task_record_payload",
        "queue_enqueue_payload", "scheduler_payload", "execution_resolution_payload",
        "run_dispatch_payload", "run_record_payload", "training_launch_payload", "training_job_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p70_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_record": _unsafe_inventory_claims(
            p70_ready, "job_record_boundary", _row("p71_job_record"),
            ("job_record_written", "training_job_created", "job_store_written"),
        ),
        "queue": _unsafe_inventory_claims(
            p70_ready, "queue_enqueue_boundary", _row("p71_queue"),
            ("queue_enqueued", "scheduler_enqueued", "run_dispatch_allowed"),
        ),
        "run": _unsafe_inventory_claims(
            p70_ready, "run_dispatch_boundary", _row("p71_run"),
            ("run_dispatch_executed", "runs_dispatched", "training_launch_allowed"),
        ),
        "launch": _unsafe_inventory_claims(
            p70_ready, "training_launch_boundary", _row("p71_launch"),
            ("training_launch_allowed", "training_launch_executed", "runtime_dispatch_executed"),
        ),
        "resolution": _unsafe_inventory_claims(
            p70_ready, "execution_resolution_boundary", _row("p71_resolution"),
            ("execution_resolution_invoked", "request_execution_executed", "job_creation_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p70_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p70_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p70_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p70_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p70_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p70_ready, review={**_review(), "acknowledge_no_job_created": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",), "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",), "review_missing_ack": ("ack_missing", "job_created"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "request_execution_executed", "job_creation_executed", "job_record_written",
        "queue_enqueued", "runs_dispatched", "training_job_created", "training_launch_allowed",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p70_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p70_ready,
        failure_history=[{"reason": "job_creation_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p70_ready,
        rollback_history=[{"kind": "job_creation_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p70_ready,
        failure_history=[{"reason": "closed_job_warning", "status": "closed", "severity": "high"}],
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
    p70: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_request_execution_job_creation_contract(
        p70_request_submission_contract=p70,
        request_execution_job_creation_evidence=_evidence() if evidence is None else evidence,
        request_execution_job_creation_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p70_ready() -> dict[str, Any]:
    return _p70_gate(_p69_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    payload = {
        "evidence_id": "request_execution_job_creation_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "request_execution_job_creation_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "request_execution_job_creation_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_run_dispatch_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "sha256": "sha256:p71:request-execution-job-creation-contract:ready",
        "artifact_digest": "sha256:p71:request-execution-job-creation-contract:ready",
        "source": "temp/turbocore_v5_p71_request_execution_job_creation_contract.json",
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
        "requested_scope": P71_SCOPE,
        "approve_request_execution_job_creation_contract": approve,
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
    return _gate(_p70_ready(), review={**_review(), field: True})


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
    assert report["post_p71_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["request_execution_job_creation_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["request_execution_job_creation_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("request_execution_job_creation_contract_ready", False)),
        "review_signed": bool(report.get("request_execution_job_creation_review_signed", False)),
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
