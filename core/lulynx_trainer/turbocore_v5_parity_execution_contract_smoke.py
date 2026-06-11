"""Smoke checks for V5-P63 parity execution contract."""

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

from core.turbocore_v5_parity_execution_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P63_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_parity_execution_contract,
)
from lulynx_trainer.turbocore_v5_kernel_launch_execution_contract_smoke import (  # noqa: E402
    _gate as _p62_gate,
    _p61_ready,
)


READY_DECISION = "parity_execution_contract_recorded_default_off"
BLOCKED_DECISION = "parity_execution_contract_blocked_default_off"
HOLD_DECISION = "parity_execution_contract_hold_for_signed_review_default_off"
REJECTED_DECISION = "parity_execution_contract_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p62_ready = _p62_ready()
    ready = _gate(p62_ready)
    assert ready["ok"] is True, ready
    assert ready["parity_execution_contract_ready"] is True, ready
    assert ready["parity_execution_evidence_recorded"] is True, ready
    assert ready["parity_execution_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p62_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p62_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p62_missing = _gate(None)
    _assert_blocked(p62_missing, "p62", "missing")
    p62_not_ready = _gate({**p62_ready, "ok": False, "kernel_launch_execution_contract_ready": False})
    _assert_blocked(p62_not_ready, "p62", "not_ready")
    p62_decision_mismatch = _gate({**p62_ready, "decision": "wrong"})
    _assert_blocked(p62_decision_mismatch, "p62", "not_ready")
    p62_post_fields = _gate({**p62_ready, "post_p62_request_fields": {"bad": True}})
    _assert_blocked(p62_post_fields, "post_p62_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p63_parity_execution_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p62_missing": _summary(p62_missing),
        "p62_not_ready": _summary(p62_not_ready),
        "p62_decision_mismatch": _summary(p62_decision_mismatch),
        "p62_post_fields": _summary(p62_post_fields),
        "p62_unsafe_cases": _unsafe_p62_cases(p62_ready),
        **_evidence_cases(p62_ready),
        **_inventory_cases(p62_ready),
        **_review_cases(p62_ready),
        **_history_cases(p62_ready),
    }


def _unsafe_p62_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "parity_approved", "parity_check_executed", "parity_execution_executed",
        "parity_tensor_transfer_executed", "tensor_transfer_executed", "training_step_executed",
        "kernel_launch_executed", "native_dispatch_executed", "runtime_dispatch_executed",
        "runtime_execution_executed", "runtime_adapter_enabled", "request_adapter_mapping_allowed",
    ):
        report = _gate({**p62_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p62_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p62_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p62_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p62_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p62_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p62_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_parity_contract": _gate(p62_ready, evidence={**_evidence(), "parity_execution_contract_only": False}),
        "not_manual_only": _gate(p62_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p62_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_training_step_contract": _gate(
            p62_ready,
            evidence={**_evidence(), "requires_later_training_step_execution_contract": False},
        ),
        "evidence_not_ready": _gate(
            p62_ready,
            evidence={**_evidence(), "parity_execution_contract_ready": False},
        ),
        "missing_section": _gate(p62_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p62_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p62_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p62_ready, evidence={**_evidence(), "blocked_reasons": ["parity_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_parity_contract": ("parity_execution_contract_only",),
        "not_manual_only": ("manual_only",),
        "not_internal_only": ("internal_only",),
        "missing_later_training_step_contract": ("training_step",),
        "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",),
        "default_on": ("default_off",),
        "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("parity_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_plan": _gate(p62_ready, evidence=_without(_evidence(), "parity_execution_plan_inventory")),
        "plan_not_ready": _gate(p62_ready, evidence={**_evidence(), "parity_execution_plan_inventory": [{**_row("p63_plan"), "ready": False}]}),
        "missing_authorization": _gate(p62_ready, evidence=_without(_evidence(), "parity_authorization_boundary")),
        "authorization_missing_source": _gate(p62_ready, evidence={**_evidence(), "parity_authorization_boundary": [_without(_row("p63_authorization"), "source")]}),
        "missing_precondition": _gate(p62_ready, evidence=_without(_evidence(), "parity_precondition_inventory")),
        "precondition_not_ready": _gate(p62_ready, evidence={**_evidence(), "parity_precondition_inventory": [{**_row("p63_precondition"), "ready": False}]}),
        "missing_pairing": _gate(p62_ready, evidence=_without(_evidence(), "parity_pairing_boundary")),
        "pairing_not_ready": _gate(p62_ready, evidence={**_evidence(), "parity_pairing_boundary": [{**_row("p63_pairing"), "ready": False}]}),
        "missing_tolerance": _gate(p62_ready, evidence=_without(_evidence(), "parity_tolerance_boundary")),
        "tolerance_not_ready": _gate(p62_ready, evidence={**_evidence(), "parity_tolerance_boundary": [{**_row("p63_tolerance"), "ready": False}]}),
        "missing_sample": _gate(p62_ready, evidence=_without(_evidence(), "parity_sample_boundary")),
        "sample_not_ready": _gate(p62_ready, evidence={**_evidence(), "parity_sample_boundary": [{**_row("p63_sample"), "ready": False}]}),
        "missing_tensor": _gate(p62_ready, evidence=_without(_evidence(), "parity_tensor_boundary")),
        "tensor_not_ready": _gate(p62_ready, evidence={**_evidence(), "parity_tensor_boundary": [{**_row("p63_tensor"), "ready": False}]}),
        "missing_kernel_result": _gate(p62_ready, evidence=_without(_evidence(), "kernel_result_boundary")),
        "kernel_result_not_ready": _gate(p62_ready, evidence={**_evidence(), "kernel_result_boundary": [{**_row("p63_kernel_result"), "ready": False}]}),
        "missing_transfer": _gate(p62_ready, evidence=_without(_evidence(), "tensor_transfer_boundary")),
        "transfer_not_ready": _gate(p62_ready, evidence={**_evidence(), "tensor_transfer_boundary": [{**_row("p63_transfer"), "ready": False}]}),
        "missing_rollback": _gate(p62_ready, evidence=_without(_evidence(), "rollback_preflight_inventory")),
        "rollback_not_ready": _gate(p62_ready, evidence={**_evidence(), "rollback_preflight_inventory": [{**_row("p63_rollback"), "ready": False}]}),
        "missing_observability": _gate(p62_ready, evidence=_without(_evidence(), "observability_preflight_inventory")),
        "observability_not_ready": _gate(p62_ready, evidence={**_evidence(), "observability_preflight_inventory": [{**_row("p63_observability"), "ready": False}]}),
    }
    fragments = {
        "missing_plan": ("parity_execution_plan_inventory_missing",),
        "plan_not_ready": ("parity_execution_plan_not_ready",),
        "missing_authorization": ("parity_authorization_inventory_missing",),
        "authorization_missing_source": ("parity_authorization_source_missing",),
        "missing_precondition": ("parity_precondition_inventory_missing",),
        "precondition_not_ready": ("parity_precondition_not_ready",),
        "missing_pairing": ("parity_pairing_inventory_missing",),
        "pairing_not_ready": ("parity_pairing_not_ready",),
        "missing_tolerance": ("parity_tolerance_inventory_missing",),
        "tolerance_not_ready": ("parity_tolerance_not_ready",),
        "missing_sample": ("parity_sample_inventory_missing",),
        "sample_not_ready": ("parity_sample_not_ready",),
        "missing_tensor": ("parity_tensor_inventory_missing",),
        "tensor_not_ready": ("parity_tensor_not_ready",),
        "missing_kernel_result": ("kernel_result_inventory_missing",),
        "kernel_result_not_ready": ("kernel_result_not_ready",),
        "missing_transfer": ("tensor_transfer_boundary_inventory_missing",),
        "transfer_not_ready": ("tensor_transfer_boundary_not_ready",),
        "missing_rollback": ("rollback_preflight_inventory_missing",),
        "rollback_not_ready": ("rollback_preflight_not_ready",),
        "missing_observability": ("observability_preflight_inventory_missing",),
        "observability_not_ready": ("observability_preflight_not_ready",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {
        **{name: _summary(report) for name, report in cases.items()},
        "unsafe_evidence_cases": _unsafe_evidence_cases(p62_ready),
        "unsafe_request_cases": _unsafe_request_cases(p62_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p62_ready),
    }


def _unsafe_evidence_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "parity_approved", "parity_check_executed", "parity_execution_executed",
        "parity_tensor_transfer_executed", "tensor_transfer_executed", "training_step_executed",
        "kernel_launch_executed", "native_dispatch_executed", "runtime_dispatch_executed",
        "runtime_execution_executed", "runtime_state_refreshed", "runtime_adapter_enabled",
    ):
        report = _gate(p62_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "api_route_registration", "ui_route_registration", "request_fields", "post_p63_request_fields",
        "parity_execution_contract_request", "parity_execution_contract_payload",
        "parity_execution_plan_payload", "parity_authorization_payload", "parity_pairing_payload",
        "parity_tolerance_payload", "parity_sample_payload", "parity_tensor_payload",
        "kernel_result_payload", "parity_result_payload", "training_step_payload",
    ):
        if field not in UNSAFE_NON_EMPTY_FIELDS:
            continue
        report = _gate(p62_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": _unsafe_inventory_claims(
            p62_ready,
            "parity_execution_plan_inventory",
            _row("p63_plan"),
            ("parity_plan_executed", "parity_check_executed", "training_step_executed"),
        ),
        "pairing": _unsafe_inventory_claims(
            p62_ready,
            "parity_pairing_boundary",
            _row("p63_pairing"),
            ("parity_pairing_applied", "parity_execution_executed", "training_step_executed"),
        ),
        "tolerance": _unsafe_inventory_claims(
            p62_ready,
            "parity_tolerance_boundary",
            _row("p63_tolerance"),
            ("parity_tolerance_applied", "parity_execution_executed", "training_step_executed"),
        ),
        "tensor": _unsafe_inventory_claims(
            p62_ready,
            "parity_tensor_boundary",
            _row("p63_tensor"),
            ("parity_tensor_transfer_executed", "tensor_transfer_executed", "training_step_executed"),
        ),
        "kernel": _unsafe_inventory_claims(
            p62_ready,
            "kernel_result_boundary",
            _row("p63_kernel_result"),
            ("kernel_result_loaded", "parity_result_recorded", "training_step_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p62_ready: dict[str, Any], inventory: str, row: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p62_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p62_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p62_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p62_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p62_ready, review={**_review(), "acknowledge_no_parity_executed": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "parity"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "parity_approved", "parity_execution_executed", "parity_tensor_transfer_executed",
        "tensor_transfer_executed", "training_step_executed", "kernel_launch_executed",
        "training_launch_allowed", "request_adapter_mapping_allowed",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p62_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p62_ready,
        failure_history=[{"reason": "parity_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p62_ready,
        rollback_history=[{"kind": "parity_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p62_ready,
        failure_history=[{"reason": "closed_parity_warning", "status": "closed", "severity": "high"}],
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
    p62: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_parity_execution_contract(
        p62_kernel_launch_execution=p62,
        parity_execution_evidence=_evidence() if evidence is None else evidence,
        parity_execution_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p62_ready() -> dict[str, Any]:
    return _p62_gate(_p61_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    return {
        "evidence_id": "parity_execution_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "parity_execution_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "parity_execution_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_training_step_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "parity_execution_plan_inventory": [_row("p63_plan")],
        "parity_authorization_boundary": [_row("p63_authorization")],
        "parity_precondition_inventory": [_row("p63_precondition")],
        "parity_pairing_boundary": [_row("p63_pairing")],
        "parity_tolerance_boundary": [_row("p63_tolerance")],
        "parity_sample_boundary": [_row("p63_sample")],
        "parity_tensor_boundary": [_row("p63_tensor")],
        "kernel_result_boundary": [_row("p63_kernel_result")],
        "tensor_transfer_boundary": [_row("p63_transfer")],
        "rollback_preflight_inventory": [_row("p63_rollback")],
        "observability_preflight_inventory": [_row("p63_observability")],
        "sha256": "sha256:p63:parity-execution-contract:ready",
        "artifact_digest": "sha256:p63:parity-execution-contract:ready",
        "source": "temp/turbocore_v5_p63_parity_execution_contract.json",
        **_safe_flags(),
    }


def _row(item_id: str) -> dict[str, Any]:
    return {"check_id": item_id, "ready": True, "source": f"temp/turbocore_v5_{item_id}.json", **_safe_flags()}


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": P63_SCOPE,
        "approve_parity_execution_contract": approve,
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
    return _gate(_p62_ready(), review={**_review(), field: True})


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
    assert report["post_p63_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["parity_execution_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["parity_execution_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("parity_execution_contract_ready", False)),
        "review_signed": bool(report.get("parity_execution_review_signed", False)),
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
