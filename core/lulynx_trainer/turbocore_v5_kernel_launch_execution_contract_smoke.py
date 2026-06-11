"""Smoke checks for V5-P62 kernel launch execution contract."""

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

from core.turbocore_v5_kernel_launch_execution_contract import (  # noqa: E402
    DEFAULT_REQUIRED_SECTIONS,
    P62_SCOPE,
    REQUIRED_REVIEW_ACKS,
    UNSAFE_NON_EMPTY_FIELDS,
    UNSAFE_TRUE_FIELDS,
    build_v5_kernel_launch_execution_contract,
)
from lulynx_trainer.turbocore_v5_native_dispatch_execution_contract_smoke import (  # noqa: E402
    _gate as _p61_gate,
    _p60_ready,
)


READY_DECISION = "kernel_launch_execution_contract_recorded_default_off"
BLOCKED_DECISION = "kernel_launch_execution_contract_blocked_default_off"
HOLD_DECISION = "kernel_launch_execution_contract_hold_for_signed_review_default_off"
REJECTED_DECISION = "kernel_launch_execution_contract_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p61_ready = _p61_ready()
    ready = _gate(p61_ready)
    assert ready["ok"] is True, ready
    assert ready["kernel_launch_execution_contract_ready"] is True, ready
    assert ready["kernel_launch_execution_evidence_recorded"] is True, ready
    assert ready["kernel_launch_execution_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p61_ready, review=None)
    _assert_hold(missing_review, "review", "missing")
    rejected_review = _gate(p61_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p61_missing = _gate(None)
    _assert_blocked(p61_missing, "p61", "missing")
    p61_not_ready = _gate({**p61_ready, "ok": False, "native_dispatch_execution_contract_ready": False})
    _assert_blocked(p61_not_ready, "p61", "not_ready")
    p61_decision_mismatch = _gate({**p61_ready, "decision": "wrong"})
    _assert_blocked(p61_decision_mismatch, "p61", "not_ready")
    p61_post_fields = _gate({**p61_ready, "post_p61_request_fields": {"bad": True}})
    _assert_blocked(p61_post_fields, "post_p61_request_fields")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p62_kernel_launch_execution_contract_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p61_missing": _summary(p61_missing),
        "p61_not_ready": _summary(p61_not_ready),
        "p61_decision_mismatch": _summary(p61_decision_mismatch),
        "p61_post_fields": _summary(p61_post_fields),
        "p61_unsafe_cases": _unsafe_p61_cases(p61_ready),
        **_evidence_cases(p61_ready),
        **_inventory_cases(p61_ready),
        **_review_cases(p61_ready),
        **_history_cases(p61_ready),
    }


def _unsafe_p61_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "kernel_launch_executed", "parity_check_executed", "tensor_transfer_executed",
        "training_step_executed", "native_dispatch_executed", "runtime_dispatch_executed",
        "runtime_execution_executed", "runtime_adapter_enabled", "request_adapter_mapping_allowed",
    ):
        report = _gate({**p61_ready, field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _evidence_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_source": _gate(p61_ready, evidence=_without(_evidence(), "source")),
        "missing_digest": _gate(p61_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest")),
        "not_report_only": _gate(p61_ready, evidence={**_evidence(), "report_only": False}),
        "not_boundary_only": _gate(p61_ready, evidence={**_evidence(), "boundary_only": False}),
        "not_contract_only": _gate(p61_ready, evidence={**_evidence(), "contract_only": False}),
        "not_records_only": _gate(p61_ready, evidence={**_evidence(), "records_evidence_only": False}),
        "not_kernel_contract": _gate(p61_ready, evidence={**_evidence(), "kernel_launch_execution_contract_only": False}),
        "not_manual_only": _gate(p61_ready, evidence={**_evidence(), "manual_only": False}),
        "not_internal_only": _gate(p61_ready, evidence={**_evidence(), "internal_only": False}),
        "missing_later_parity_contract": _gate(
            p61_ready,
            evidence={**_evidence(), "requires_later_parity_execution_contract": False},
        ),
        "evidence_not_ready": _gate(
            p61_ready,
            evidence={**_evidence(), "kernel_launch_execution_contract_ready": False},
        ),
        "missing_section": _gate(p61_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]}),
        "default_on": _gate(p61_ready, evidence={**_evidence(), "default_training_path_enabled": True}),
        "request_adapter_on": _gate(p61_ready, evidence={**_evidence(), "request_adapter_mapping_allowed": True}),
        "evidence_blocker": _gate(p61_ready, evidence={**_evidence(), "blocked_reasons": ["kernel_launch_gap"]}),
    }
    fragments = {
        "missing_source": ("source_missing",),
        "missing_digest": ("digest_missing",),
        "not_report_only": ("report_only",),
        "not_boundary_only": ("boundary_only",),
        "not_contract_only": ("contract_only",),
        "not_records_only": ("records_evidence_only",),
        "not_kernel_contract": ("kernel_launch_execution_contract_only",),
        "not_manual_only": ("manual_only",),
        "not_internal_only": ("internal_only",),
        "missing_later_parity_contract": ("later_parity",),
        "evidence_not_ready": ("not_ready",),
        "missing_section": ("section_missing",),
        "default_on": ("default_off",),
        "request_adapter_on": ("request_adapter",),
        "evidence_blocker": ("kernel_launch_gap",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {name: _summary(report) for name, report in cases.items()}


def _inventory_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "missing_plan": _gate(p61_ready, evidence=_without(_evidence(), "kernel_launch_execution_plan_inventory")),
        "plan_not_ready": _gate(p61_ready, evidence={**_evidence(), "kernel_launch_execution_plan_inventory": [{**_row("p62_plan"), "ready": False}]}),
        "missing_authorization": _gate(p61_ready, evidence=_without(_evidence(), "kernel_launch_authorization_boundary")),
        "authorization_missing_source": _gate(p61_ready, evidence={**_evidence(), "kernel_launch_authorization_boundary": [_without(_row("p62_authorization"), "source")]}),
        "missing_precondition": _gate(p61_ready, evidence=_without(_evidence(), "kernel_launch_precondition_inventory")),
        "precondition_not_ready": _gate(p61_ready, evidence={**_evidence(), "kernel_launch_precondition_inventory": [{**_row("p62_precondition"), "ready": False}]}),
        "missing_artifact": _gate(p61_ready, evidence=_without(_evidence(), "kernel_build_artifact_boundary")),
        "artifact_not_ready": _gate(p61_ready, evidence={**_evidence(), "kernel_build_artifact_boundary": [{**_row("p62_artifact"), "ready": False}]}),
        "missing_parameter": _gate(p61_ready, evidence=_without(_evidence(), "kernel_launch_parameter_boundary")),
        "parameter_not_ready": _gate(p61_ready, evidence={**_evidence(), "kernel_launch_parameter_boundary": [{**_row("p62_parameter"), "ready": False}]}),
        "missing_stream": _gate(p61_ready, evidence=_without(_evidence(), "kernel_stream_synchronization_boundary")),
        "stream_not_ready": _gate(p61_ready, evidence={**_evidence(), "kernel_stream_synchronization_boundary": [{**_row("p62_stream"), "ready": False}]}),
        "missing_parity": _gate(p61_ready, evidence=_without(_evidence(), "parity_boundary")),
        "parity_not_ready": _gate(p61_ready, evidence={**_evidence(), "parity_boundary": [{**_row("p62_parity"), "ready": False}]}),
        "missing_tensor": _gate(p61_ready, evidence=_without(_evidence(), "tensor_transfer_boundary")),
        "tensor_not_ready": _gate(p61_ready, evidence={**_evidence(), "tensor_transfer_boundary": [{**_row("p62_tensor"), "ready": False}]}),
        "missing_rollback": _gate(p61_ready, evidence=_without(_evidence(), "rollback_preflight_inventory")),
        "rollback_not_ready": _gate(p61_ready, evidence={**_evidence(), "rollback_preflight_inventory": [{**_row("p62_rollback"), "ready": False}]}),
        "missing_observability": _gate(p61_ready, evidence=_without(_evidence(), "observability_preflight_inventory")),
        "observability_not_ready": _gate(p61_ready, evidence={**_evidence(), "observability_preflight_inventory": [{**_row("p62_observability"), "ready": False}]}),
    }
    fragments = {
        "missing_plan": ("kernel_launch_execution_plan_inventory_missing",),
        "plan_not_ready": ("kernel_launch_execution_plan_not_ready",),
        "missing_authorization": ("kernel_launch_authorization_inventory_missing",),
        "authorization_missing_source": ("kernel_launch_authorization_source_missing",),
        "missing_precondition": ("kernel_launch_precondition_inventory_missing",),
        "precondition_not_ready": ("kernel_launch_precondition_not_ready",),
        "missing_artifact": ("kernel_build_artifact_inventory_missing",),
        "artifact_not_ready": ("kernel_build_artifact_not_ready",),
        "missing_parameter": ("kernel_launch_parameter_inventory_missing",),
        "parameter_not_ready": ("kernel_launch_parameter_not_ready",),
        "missing_stream": ("kernel_stream_synchronization_inventory_missing",),
        "stream_not_ready": ("kernel_stream_synchronization_not_ready",),
        "missing_parity": ("parity_boundary_inventory_missing",),
        "parity_not_ready": ("parity_boundary_not_ready",),
        "missing_tensor": ("tensor_transfer_boundary_inventory_missing",),
        "tensor_not_ready": ("tensor_transfer_boundary_not_ready",),
        "missing_rollback": ("rollback_preflight_inventory_missing",),
        "rollback_not_ready": ("rollback_preflight_not_ready",),
        "missing_observability": ("observability_preflight_inventory_missing",),
        "observability_not_ready": ("observability_preflight_not_ready",),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    return {
        **{name: _summary(report) for name, report in cases.items()},
        "unsafe_evidence_cases": _unsafe_evidence_cases(p61_ready),
        "unsafe_request_cases": _unsafe_request_cases(p61_ready),
        "unsafe_inventory_cases": _unsafe_inventory_cases(p61_ready),
    }


def _unsafe_evidence_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "kernel_launch_approved", "kernel_launch_executed", "kernel_launch_execution_allowed",
        "parity_check_executed", "parity_execution_executed", "tensor_transfer_executed",
        "training_step_executed", "native_dispatch_executed", "runtime_dispatch_executed",
        "runtime_execution_executed", "runtime_state_refreshed", "runtime_adapter_enabled",
    ):
        report = _gate(p61_ready, evidence={**_evidence(), field: True})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_request_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for field in (
        "api_route_registration", "ui_route_registration", "request_fields", "post_p62_request_fields",
        "kernel_launch_execution_contract_request", "kernel_launch_execution_plan_payload",
        "kernel_launch_authorization_payload", "kernel_build_artifact_payload",
        "kernel_launch_parameter_payload", "kernel_stream_synchronization_payload",
        "kernel_execution_result_payload", "parity_execution_payload", "tensor_transfer_payload",
        "training_step_payload",
    ):
        report = _gate(p61_ready, evidence={**_evidence(), field: {"bad": True}})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _unsafe_inventory_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": _unsafe_inventory_claims(
            p61_ready,
            "kernel_launch_execution_plan_inventory",
            _row("p62_plan"),
            ("kernel_launch_executed", "parity_check_executed", "training_step_executed"),
        ),
        "artifact": _unsafe_inventory_claims(
            p61_ready,
            "kernel_build_artifact_boundary",
            _row("p62_artifact"),
            ("kernel_build_artifact_loaded", "kernel_launch_executed", "training_step_executed"),
        ),
        "stream": _unsafe_inventory_claims(
            p61_ready,
            "kernel_stream_synchronization_boundary",
            _row("p62_stream"),
            ("kernel_stream_synchronization_executed", "kernel_launch_executed", "parity_check_executed"),
        ),
        "parity": _unsafe_inventory_claims(
            p61_ready,
            "parity_boundary",
            _row("p62_parity"),
            ("parity_check_executed", "parity_execution_executed", "training_step_executed"),
        ),
        "tensor": _unsafe_inventory_claims(
            p61_ready,
            "tensor_transfer_boundary",
            _row("p62_tensor"),
            ("tensor_transfer_executed", "kernel_launch_executed", "training_step_executed"),
        ),
    }


def _unsafe_inventory_claims(
    p61_ready: dict[str, Any],
    inventory: str,
    row: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    cases = {}
    for field in fields:
        report = _gate(p61_ready, evidence={**_evidence(), inventory: [{**row, field: True}]})
        _assert_blocked(report, field)
        cases[field] = _summary(report)
    return cases


def _review_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    cases = {
        "review_missing_reviewer": _gate(p61_ready, review={**_review(), "reviewer": ""}),
        "review_missing_reviewed_at": _gate(p61_ready, review={**_review(), "reviewed_at": ""}),
        "review_scope_mismatch": _gate(p61_ready, review={**_review(), "requested_scope": "wrong"}),
        "review_missing_ack": _gate(p61_ready, review={**_review(), "acknowledge_no_kernel_launch_executed": False}),
    }
    fragments = {
        "review_missing_reviewer": ("reviewer",),
        "review_missing_reviewed_at": ("reviewed_at",),
        "review_scope_mismatch": ("scope",),
        "review_missing_ack": ("ack_missing", "kernel_launch"),
    }
    for name, report in cases.items():
        _assert_blocked(report, *fragments[name])
    unsafe_cases = {}
    for base_field in (
        "kernel_launch_approved", "kernel_launch_executed", "kernel_launch_execution_allowed",
        "parity_check_executed", "parity_execution_executed", "tensor_transfer_executed",
        "training_step_executed", "training_launch_allowed", "request_adapter_mapping_allowed",
    ):
        if base_field not in UNSAFE_TRUE_FIELDS:
            continue
        field = f"approve_{base_field}"
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        unsafe_cases[field] = _summary(report)
    return {**{name: _summary(report) for name, report in cases.items()}, "review_unsafe_cases": unsafe_cases}


def _history_cases(p61_ready: dict[str, Any]) -> dict[str, Any]:
    failure_history = _gate(
        p61_ready,
        failure_history=[{"reason": "kernel_launch_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(
        p61_ready,
        rollback_history=[{"kind": "kernel_launch_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(
        p61_ready,
        failure_history=[{"reason": "closed_kernel_launch_warning", "status": "closed", "severity": "high"}],
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
    p61: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_kernel_launch_execution_contract(
        p61_native_dispatch_execution=p61,
        kernel_launch_execution_evidence=_evidence() if evidence is None else evidence,
        kernel_launch_execution_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p61_ready() -> dict[str, Any]:
    return _p61_gate(_p60_ready())


def _evidence() -> dict[str, Any]:
    sections = list(DEFAULT_REQUIRED_SECTIONS)
    return {
        "evidence_id": "kernel_launch_execution_contract_v0",
        "evidence_version": "v0",
        "ok": True,
        "kernel_launch_execution_contract_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "kernel_launch_execution_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_parity_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "kernel_launch_execution_plan_inventory": [_row("p62_plan")],
        "kernel_launch_authorization_boundary": [_row("p62_authorization")],
        "kernel_launch_precondition_inventory": [_row("p62_precondition")],
        "kernel_build_artifact_boundary": [_row("p62_artifact")],
        "kernel_launch_parameter_boundary": [_row("p62_parameter")],
        "kernel_stream_synchronization_boundary": [_row("p62_stream")],
        "parity_boundary": [_row("p62_parity")],
        "tensor_transfer_boundary": [_row("p62_tensor")],
        "rollback_preflight_inventory": [_row("p62_rollback")],
        "observability_preflight_inventory": [_row("p62_observability")],
        "sha256": "sha256:p62:kernel-launch-execution-contract:ready",
        "artifact_digest": "sha256:p62:kernel-launch-execution-contract:ready",
        "source": "temp/turbocore_v5_p62_kernel_launch_execution_contract.json",
        **_safe_flags(),
    }


def _row(item_id: str) -> dict[str, Any]:
    return {"check_id": item_id, "ready": True, "source": f"temp/turbocore_v5_{item_id}.json", **_safe_flags()}


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": P62_SCOPE,
        "approve_kernel_launch_execution_contract": approve,
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
    return _gate(_p61_ready(), review={**_review(), field: True})


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
    assert report["post_p62_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["kernel_launch_execution_contract_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["kernel_launch_execution_contract_ready"] is False, report
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
        "contract_ready": bool(report.get("kernel_launch_execution_contract_ready", False)),
        "review_signed": bool(report.get("kernel_launch_execution_review_signed", False)),
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
