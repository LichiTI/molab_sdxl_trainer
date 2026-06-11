"""Smoke checks for V5-P46 operator activation execution contract boundary."""

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

from core.turbocore_v5_operator_activation_execution_contract_boundary import (  # noqa: E402
    build_v5_operator_activation_execution_contract_boundary,
)
from lulynx_trainer.turbocore_v5_operator_activation_request_boundary_smoke import (  # noqa: E402
    _gate as _p45_gate,
    _p44_ready,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "auto_launch_allowed",
    "runs_dispatched",
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "ui_exposure_allowed",
    "product_ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "request_adapter_registered",
    "request_adapter_enabled",
    "runtime_adapter_registered",
    "operator_activation_request_submitted",
    "operator_activation_requested",
    "operator_activation_execution_allowed",
    "operator_activation_execution_executed",
    "operator_activation_executed",
    "operator_request_executed",
    "activation_request_executed",
    "activation_execution_performed",
    "runtime_activation_allowed",
    "runtime_activation_enabled",
    "runtime_enablement_allowed",
    "runtime_enablement_enabled",
    "runtime_adapter_enabled",
    "native_runtime_enabled",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed",
    "training_manager_patch_allowed",
    "rollout_authorization_allowed",
)
READY_DECISION = "operator_activation_execution_contract_boundary_recorded_default_off"
BLOCKED_DECISION = "operator_activation_execution_contract_boundary_blocked_default_off"
HOLD_DECISION = "operator_activation_execution_contract_boundary_hold_for_signed_review_default_off"
REJECTED_DECISION = "operator_activation_execution_contract_boundary_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p45_ready = _p45_ready()
    ready = _gate(p45_ready)
    assert ready["ok"] is True, ready
    assert ready["operator_activation_execution_contract_boundary_ready"] is True, ready
    assert ready["operator_activation_execution_review_signed"] is True, ready
    assert ready["operator_activation_execution_evidence_recorded"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p45_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p45_ready, review=_operator_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p45_missing = _gate(None)
    _assert_blocked(p45_missing, "p45", "missing")

    p45_not_ready = _gate({**p45_ready, "ok": False, "operator_activation_request_boundary_ready": False})
    _assert_blocked(p45_not_ready, "p45", "not_ready")

    p45_decision_mismatch = _gate({**p45_ready, "decision": "wrong"})
    _assert_blocked(p45_decision_mismatch, "p45", "not_ready")

    p45_request_submitted = _gate({**p45_ready, "operator_activation_request_submitted": True})
    _assert_blocked(p45_request_submitted, "operator_activation_request_submitted")

    p45_activation_requested = _gate({**p45_ready, "operator_activation_requested": True})
    _assert_blocked(p45_activation_requested, "operator_activation_requested")

    p45_native_dispatch = _gate({**p45_ready, "native_dispatch_enabled": True})
    _assert_blocked(p45_native_dispatch, "native_dispatch_enabled")

    p45_unsigned_review = _gate({**p45_ready, "operator_activation_request_review_signed": False})
    _assert_blocked(p45_unsigned_review, "p45", "not_ready")

    p45_post_fields = _gate({**p45_ready, "post_p45_request_fields": {"bad": True}})
    _assert_blocked(p45_post_fields, "post_p45_request_fields")

    missing_source = _gate(p45_ready, evidence=_without(_operator_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")

    missing_digest = _gate(p45_ready, evidence=_without_many(_operator_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")

    not_report_only = _gate(p45_ready, evidence={**_operator_evidence(), "report_only": False})
    _assert_blocked(not_report_only, "report_only")

    not_execution_only = _gate(p45_ready, evidence={**_operator_evidence(), "execution_only": False})
    _assert_blocked(not_execution_only, "execution_only")

    not_contract_only = _gate(p45_ready, evidence={**_operator_evidence(), "contract_only": False})
    _assert_blocked(not_contract_only, "contract_only")

    not_records_evidence_only = _gate(p45_ready, evidence={**_operator_evidence(), "records_evidence_only": False})
    _assert_blocked(not_records_evidence_only, "records_evidence_only")

    missing_later_enablement = _gate(
        p45_ready,
        evidence={**_operator_evidence(), "requires_later_runtime_enablement_contract": False},
    )
    _assert_blocked(missing_later_enablement, "later_runtime_enablement_contract")

    missing_section = _gate(p45_ready, evidence={**_operator_evidence(), "available_sections": ["rollback_policy"]})
    _assert_blocked(missing_section, "section_missing")

    missing_execution_inventory = _gate(
        p45_ready,
        evidence=_without(_operator_evidence(), "operator_activation_execution_plan_inventory"),
    )
    _assert_blocked(missing_execution_inventory, "execution_plan_inventory")

    missing_precondition_inventory = _gate(
        p45_ready,
        evidence=_without(_operator_evidence(), "execution_precondition_inventory"),
    )
    _assert_blocked(missing_precondition_inventory, "execution_precondition_inventory")

    execution_cases = {}
    for field in (
        "operator_activation_execution_executed",
        "operator_activation_executed",
        "operator_request_executed",
        "activation_request_executed",
        "activation_execution_performed",
        "runtime_activation_enabled",
        "runtime_enablement_enabled",
        "runtime_adapter_enabled",
        "native_dispatch_allowed",
        "native_dispatch_enabled",
        "request_adapter_enabled",
    ):
        report = _plan_claim(field)
        _assert_blocked(report, field)
        execution_cases[field] = _summary(report)

    precondition_cases = {}
    for field in ("execution_precondition_active", "execution_check_registered", "execution_check_enabled"):
        report = _precondition_claim(field)
        _assert_blocked(report, field)
        precondition_cases[field] = _summary(report)

    review_missing_reviewer = _gate(p45_ready, review={**_operator_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p45_ready, review={**_operator_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p45_ready, review={**_operator_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(
        p45_ready,
        review={**_operator_review(), "acknowledge_no_operator_activation_execution": False},
    )
    _assert_blocked(review_missing_ack, "ack_missing", "operator_activation")

    review_unsafe_cases = {}
    for field in (
        "approve_operator_activation_execution_executed",
        "approve_activation_request_executed",
        "approve_runtime_activation_enabled",
        "approve_native_dispatch_enabled",
        "approve_training_launch_allowed",
    ):
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        review_unsafe_cases[field] = _summary(report)

    failure_history = _gate(
        p45_ready,
        failure_history=[{"reason": "operator_execution_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(
        p45_ready,
        rollback_history=[{"kind": "operator_execution_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p45_ready,
        failure_history=[{"reason": "closed_operator_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p46_operator_activation_execution_contract_boundary_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p45_missing": _summary(p45_missing),
        "p45_not_ready": _summary(p45_not_ready),
        "p45_decision_mismatch": _summary(p45_decision_mismatch),
        "p45_request_submitted": _summary(p45_request_submitted),
        "p45_activation_requested": _summary(p45_activation_requested),
        "p45_native_dispatch": _summary(p45_native_dispatch),
        "p45_post_fields": _summary(p45_post_fields),
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "p45_unsigned_review": _summary(p45_unsigned_review),
        "missing_execution_inventory": _summary(missing_execution_inventory),
        "missing_precondition_inventory": _summary(missing_precondition_inventory),
        "execution_cases": execution_cases,
        "precondition_cases": precondition_cases,
        "review_unsafe_cases": review_unsafe_cases,
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p45: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _operator_review() if review is ... else review
    return build_v5_operator_activation_execution_contract_boundary(
        p45_operator_activation_request_boundary=p45,
        operator_activation_execution_evidence=_operator_evidence() if evidence is None else evidence,
        operator_activation_execution_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p45_ready() -> dict[str, Any]:
    return _p45_gate(_p44_ready())


def _operator_evidence() -> dict[str, Any]:
    sections = [
        "p45_operator_activation_request_boundary_reference",
        "operator_activation_execution_plan_inventory",
        "execution_precondition_inventory",
        "operator_identity_boundary",
        "activation_scope_boundary",
        "no_operator_request_execution_boundary",
        "no_activation_request_execution_boundary",
        "no_runtime_activation_boundary",
        "no_runtime_adapter_enabled_boundary",
        "no_request_fields_boundary",
        "no_training_launch_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    return {
        "evidence_id": "operator_activation_execution_contract_boundary_v0",
        "evidence_version": "v0",
        "ok": True,
        "operator_activation_execution_contract_boundary_ready": True,
        "report_only": True,
        "boundary_only": True,
        "execution_only": True,
        "contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_runtime_enablement_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "operator_activation_execution_plan_inventory": [
            {"plan_id": "native_update_operator_execution", **_safe_row_flags()},
            {"plan_id": "lora_fused_operator_execution", **_safe_row_flags()},
        ],
        "execution_precondition_inventory": [
            {"check_id": "native_update_preconditions", **_safe_row_flags()},
            {"check_id": "lora_fused_preconditions", **_safe_row_flags()},
        ],
        "sha256": "sha256:p46:operator-execution:ready",
        "artifact_digest": "sha256:p46:operator-execution:ready",
        "source": "temp/turbocore_v5_p46_operator_activation_execution.json",
        **_safe_row_flags(),
    }


def _operator_review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "operator_activation_execution_contract_boundary",
        "approve_operator_activation_execution_boundary": approve,
    }
    for field in DEFAULT_OFF_FIELDS:
        if field != "default_behavior_changed":
            review[f"approve_{field}"] = False
    for field in (
        "acknowledge_p45_operator_activation_request_boundary_recorded",
        "acknowledge_default_off_boundary",
        "acknowledge_no_training_launch",
        "acknowledge_no_ui_exposure",
        "acknowledge_no_operator_activation_execution",
        "acknowledge_no_activation_request_executed",
        "acknowledge_no_runtime_activation",
        "acknowledge_no_runtime_adapter_enabled",
        "acknowledge_no_request_fields_emitted",
        "acknowledge_no_default_or_auto_rollout",
        "acknowledge_execution_evidence_replayable",
        "acknowledge_later_runtime_enablement_contract_required",
        "acknowledge_manual_review_only",
    ):
        review[field] = True
    return review


def _safe_row_flags() -> dict[str, Any]:
    flags = {field: False for field in DEFAULT_OFF_FIELDS}
    flags.update(
        {
            "blocked_reasons": [],
            "promotion_blockers": [],
            "post_p45_request_fields": {},
            "post_p46_request_fields": {},
            "post_operator_activation_execution_fields": {},
            "operator_activation_execution_request": {},
            "activation_execution_payload": {},
            "operator_execution_payload": {},
            "operator_activation_request": {},
            "activation_request_payload": {},
            "request_adapter_fields": {},
            "request_fields": {},
            "api_route_registration": {},
            "launch_request": {},
            "training_request": {},
        }
    )
    return flags


def _plan_claim(field: str) -> dict[str, Any]:
    row = {"plan_id": "native", **_safe_row_flags(), field: True}
    return _gate(_p45_ready(), evidence={**_operator_evidence(), "operator_activation_execution_plan_inventory": [row]})


def _precondition_claim(field: str) -> dict[str, Any]:
    row = {"check_id": "native", **_safe_row_flags(), field: True}
    return _gate(_p45_ready(), evidence={**_operator_evidence(), "execution_precondition_inventory": [row]})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p45_ready(), review={**_operator_review(), field: True})


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
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report
    assert report["post_p46_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["operator_activation_execution_contract_boundary_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["operator_activation_execution_contract_boundary_ready"] is False, report
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
        "boundary_ready": bool(report.get("operator_activation_execution_contract_boundary_ready", False)),
        "review_signed": bool(report.get("operator_activation_execution_review_signed", False)),
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
