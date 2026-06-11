"""Smoke checks for V5-P50 explicit dispatch execution / kernel launch boundary."""

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

from core.turbocore_v5_explicit_dispatch_execution_kernel_launch_boundary import (  # noqa: E402
    build_v5_explicit_dispatch_execution_kernel_launch_boundary,
)
from lulynx_trainer.turbocore_v5_runtime_dispatch_contract_boundary_smoke import (  # noqa: E402
    _gate as _p49_gate,
    _p48_ready,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed", "training_launch_allowed", "auto_launch_allowed", "runs_dispatched",
    "default_training_path_enabled", "training_path_enabled", "default_rollout_allowed", "auto_rollout_allowed",
    "ui_exposure_allowed", "product_ui_exposure_allowed", "launcher_exposure_allowed", "webui_exposure_allowed",
    "request_adapter_mapping_allowed", "request_fields_emitted", "request_adapter_registered", "request_adapter_enabled",
    "runtime_adapter_registered", "runtime_adapter_enabled", "runtime_enablement_allowed", "runtime_enablement_enabled",
    "runtime_enablement_executed", "runtime_enablement_applied", "runtime_enablement_execution_allowed",
    "runtime_enablement_execution_enabled", "runtime_enablement_execution_executed", "runtime_enablement_execution_applied",
    "runtime_execution_allowed", "runtime_execution_executed", "runtime_dispatch_allowed", "runtime_dispatch_enabled",
    "runtime_dispatch_executed", "explicit_dispatch_execution_allowed", "explicit_dispatch_execution_enabled",
    "explicit_dispatch_execution_executed", "runtime_activation_allowed", "runtime_activation_enabled",
    "native_runtime_enabled", "native_dispatch_allowed", "native_dispatch_enabled", "native_dispatch_started",
    "native_dispatch_executed", "kernel_launch_allowed", "kernel_launch_enabled", "kernel_launch_executed",
    "training_step_allowed", "training_step_enabled", "training_step_executed", "generation_request_patch_allowed",
    "config_adapter_patch_allowed", "runtime_resolver_patch_allowed", "execution_resolver_patch_allowed",
    "training_manager_patch_allowed", "rollout_authorization_allowed",
)
READY_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_recorded_default_off"
BLOCKED_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_blocked_default_off"
HOLD_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_hold_for_signed_review_default_off"
REJECTED_DECISION = "explicit_dispatch_execution_kernel_launch_boundary_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p49_ready = _p49_ready()
    ready = _gate(p49_ready)
    assert ready["ok"] is True, ready
    assert ready["explicit_dispatch_execution_kernel_launch_boundary_ready"] is True, ready
    assert ready["dispatch_execution_kernel_launch_review_signed"] is True, ready
    assert ready["dispatch_execution_kernel_launch_evidence_recorded"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p49_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p49_ready, review=_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p49_missing = _gate(None)
    _assert_blocked(p49_missing, "p49", "missing")
    p49_not_ready = _gate({**p49_ready, "ok": False, "runtime_dispatch_contract_boundary_ready": False})
    _assert_blocked(p49_not_ready, "p49", "not_ready")
    p49_decision_mismatch = _gate({**p49_ready, "decision": "wrong"})
    _assert_blocked(p49_decision_mismatch, "p49", "not_ready")
    p49_unsigned_review = _gate({**p49_ready, "runtime_dispatch_review_signed": False})
    _assert_blocked(p49_unsigned_review, "p49", "not_ready")
    p49_post_fields = _gate({**p49_ready, "post_p49_request_fields": {"bad": True}})
    _assert_blocked(p49_post_fields, "post_p49_request_fields")

    p49_unsafe_cases = {}
    for field in ("runtime_dispatch_executed", "native_dispatch_started", "native_dispatch_executed", "kernel_launch_executed", "training_step_executed"):
        report = _gate({**p49_ready, field: True})
        _assert_blocked(report, field)
        p49_unsafe_cases[field] = _summary(report)

    missing_source = _gate(p49_ready, evidence=_without(_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")
    missing_digest = _gate(p49_ready, evidence=_without_many(_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")
    not_contract_only = _gate(p49_ready, evidence={**_evidence(), "contract_only": False})
    _assert_blocked(not_contract_only, "contract_only")
    not_records_only = _gate(p49_ready, evidence={**_evidence(), "records_evidence_only": False})
    _assert_blocked(not_records_only, "records_evidence_only")
    not_execution_only = _gate(p49_ready, evidence={**_evidence(), "execution_only": False})
    _assert_blocked(not_execution_only, "execution_only")
    not_kernel_boundary_only = _gate(p49_ready, evidence={**_evidence(), "kernel_launch_boundary_only": False})
    _assert_blocked(not_kernel_boundary_only, "kernel_launch_boundary_only")
    missing_later_contract = _gate(p49_ready, evidence={**_evidence(), "requires_later_native_execution_readiness_contract": False})
    _assert_blocked(missing_later_contract, "later_contract")
    missing_section = _gate(p49_ready, evidence={**_evidence(), "available_sections": ["rollback_policy"]})
    _assert_blocked(missing_section, "section_missing")
    missing_plan = _gate(p49_ready, evidence=_without(_evidence(), "explicit_dispatch_execution_plan_inventory"))
    _assert_blocked(missing_plan, "plan_inventory")
    missing_preconditions = _gate(p49_ready, evidence=_without(_evidence(), "dispatch_execution_precondition_inventory"))
    _assert_blocked(missing_preconditions, "precondition_inventory")

    plan_cases = {}
    for field in (
        "runtime_dispatch_executed", "native_dispatch_started", "native_dispatch_enabled", "native_dispatch_executed",
        "kernel_launch_allowed", "kernel_launch_enabled", "kernel_launch_executed", "training_step_allowed",
        "training_step_executed", "request_fields_emitted",
    ):
        report = _plan_claim(field)
        _assert_blocked(report, field)
        plan_cases[field] = _summary(report)

    precondition_cases = {}
    for field in ("dispatch_execution_precondition_active", "kernel_launch_check_registered", "kernel_launch_check_enabled"):
        report = _precondition_claim(field)
        _assert_blocked(report, field)
        precondition_cases[field] = _summary(report)

    route_registration = _gate(p49_ready, evidence={**_evidence(), "api_route_registration": {"bad": True}})
    _assert_blocked(route_registration, "api_route_registration")
    review_missing_reviewer = _gate(p49_ready, review={**_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")
    review_missing_reviewed_at = _gate(p49_ready, review={**_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")
    review_scope_mismatch = _gate(p49_ready, review={**_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")
    review_missing_ack = _gate(p49_ready, review={**_review(), "acknowledge_no_kernel_launch_executed": False})
    _assert_blocked(review_missing_ack, "ack_missing", "kernel_launch")

    review_unsafe_cases = {}
    for field in (
        "approve_runtime_dispatch_executed", "approve_native_dispatch_executed", "approve_kernel_launch_executed",
        "approve_training_step_executed", "approve_training_launch_allowed",
    ):
        report = _unsafe_review(field)
        _assert_blocked(report, field)
        review_unsafe_cases[field] = _summary(report)

    failure_history = _gate(p49_ready, failure_history=[{"reason": "dispatch_execution_gap", "open": True, "severity": "high"}])
    _assert_blocked(failure_history, "failure_history")
    rollback_history = _gate(p49_ready, rollback_history=[{"kind": "kernel_launch_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")
    closed_failure = _gate(p49_ready, failure_history=[{"reason": "closed_kernel_warning", "status": "closed", "severity": "high"}])
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p50_explicit_dispatch_execution_kernel_launch_boundary_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p49_missing": _summary(p49_missing),
        "p49_not_ready": _summary(p49_not_ready),
        "p49_decision_mismatch": _summary(p49_decision_mismatch),
        "p49_unsigned_review": _summary(p49_unsigned_review),
        "p49_post_fields": _summary(p49_post_fields),
        "p49_unsafe_cases": p49_unsafe_cases,
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "not_contract_only": _summary(not_contract_only),
        "not_records_only": _summary(not_records_only),
        "not_execution_only": _summary(not_execution_only),
        "not_kernel_boundary_only": _summary(not_kernel_boundary_only),
        "missing_later_contract": _summary(missing_later_contract),
        "missing_section": _summary(missing_section),
        "missing_plan": _summary(missing_plan),
        "missing_preconditions": _summary(missing_preconditions),
        "plan_cases": plan_cases,
        "precondition_cases": precondition_cases,
        "route_registration": _summary(route_registration),
        "review_missing_reviewer": _summary(review_missing_reviewer),
        "review_missing_reviewed_at": _summary(review_missing_reviewed_at),
        "review_scope_mismatch": _summary(review_scope_mismatch),
        "review_missing_ack": _summary(review_missing_ack),
        "review_unsafe_cases": review_unsafe_cases,
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p49: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _review() if review is ... else review
    return build_v5_explicit_dispatch_execution_kernel_launch_boundary(
        p49_runtime_dispatch_contract_boundary=p49,
        dispatch_execution_kernel_launch_evidence=_evidence() if evidence is None else evidence,
        dispatch_execution_kernel_launch_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p49_ready() -> dict[str, Any]:
    return _p49_gate(_p48_ready())


def _evidence() -> dict[str, Any]:
    sections = [
        "p49_runtime_dispatch_boundary_reference", "explicit_dispatch_execution_plan_inventory",
        "dispatch_execution_precondition_inventory", "runtime_dispatch_execution_boundary",
        "native_dispatch_execution_boundary", "kernel_launch_boundary", "training_step_boundary",
        "request_adapter_boundary", "no_runtime_dispatch_execution_boundary", "no_native_dispatch_execution_boundary",
        "no_kernel_launch_execution_boundary", "no_training_step_boundary", "no_request_fields_boundary",
        "no_training_launch_boundary", "rollback_policy", "observability_policy",
    ]
    return {
        "evidence_id": "explicit_dispatch_execution_kernel_launch_boundary_v0",
        "evidence_version": "v0",
        "ok": True,
        "explicit_dispatch_execution_kernel_launch_boundary_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "execution_only": True,
        "kernel_launch_boundary_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_native_execution_readiness_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "explicit_dispatch_execution_plan_inventory": [{"plan_id": "native_update_execution", **_safe_flags()}],
        "dispatch_execution_precondition_inventory": [{"check_id": "native_update_kernel_preconditions", **_safe_flags()}],
        "sha256": "sha256:p50:dispatch-execution-kernel-launch:ready",
        "artifact_digest": "sha256:p50:dispatch-execution-kernel-launch:ready",
        "source": "temp/turbocore_v5_p50_dispatch_execution_kernel_launch.json",
        **_safe_flags(),
    }


def _review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "explicit_dispatch_execution_kernel_launch_boundary",
        "approve_explicit_dispatch_execution_kernel_launch_boundary": approve,
    }
    for field in DEFAULT_OFF_FIELDS:
        if field != "default_behavior_changed":
            review[f"approve_{field}"] = False
    for field in (
        "acknowledge_p49_runtime_dispatch_boundary_recorded", "acknowledge_default_off_boundary",
        "acknowledge_no_training_launch", "acknowledge_no_ui_exposure", "acknowledge_no_runtime_dispatch_executed",
        "acknowledge_no_native_dispatch_executed", "acknowledge_no_kernel_launch_executed",
        "acknowledge_no_training_step_executed", "acknowledge_no_request_adapter_enabled",
        "acknowledge_no_request_fields_emitted", "acknowledge_no_default_or_auto_rollout",
        "acknowledge_dispatch_execution_kernel_launch_evidence_replayable",
        "acknowledge_later_native_execution_readiness_contract_required", "acknowledge_manual_review_only",
    ):
        review[field] = True
    return review


def _safe_flags() -> dict[str, Any]:
    flags = {field: False for field in DEFAULT_OFF_FIELDS}
    flags.update(
        {
            "blocked_reasons": [], "promotion_blockers": [], "post_p49_request_fields": {},
            "post_p50_request_fields": {}, "post_dispatch_execution_kernel_launch_fields": {},
            "runtime_dispatch_request": {}, "runtime_dispatch_payload": {}, "dispatch_execution_request": {},
            "dispatch_execution_payload": {}, "explicit_dispatch_execution_request": {},
            "explicit_dispatch_execution_payload": {}, "native_dispatch_request": {}, "native_dispatch_payload": {},
            "kernel_launch_request": {}, "kernel_launch_payload": {}, "training_step_request": {},
            "training_step_payload": {}, "request_adapter_fields": {}, "request_fields": {},
            "api_route_registration": {}, "backend_route_registration": {}, "ui_route_registration": {},
            "launch_request": {}, "training_request": {}, "generation_request_patch": {}, "config_adapter_patch": {},
            "runtime_resolver_patch": {}, "execution_resolver_patch": {}, "entry_train_patch": {},
            "training_manager_patch": {}, "auto_launch_plan": {},
        }
    )
    return flags


def _plan_claim(field: str) -> dict[str, Any]:
    row = {"plan_id": "native", **_safe_flags(), field: True}
    return _gate(_p49_ready(), evidence={**_evidence(), "explicit_dispatch_execution_plan_inventory": [row]})


def _precondition_claim(field: str) -> dict[str, Any]:
    row = {"check_id": "native", **_safe_flags(), field: True}
    return _gate(_p49_ready(), evidence={**_evidence(), "dispatch_execution_precondition_inventory": [row]})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p49_ready(), review={**_review(), field: True})


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
    assert report["post_p50_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["explicit_dispatch_execution_kernel_launch_boundary_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["explicit_dispatch_execution_kernel_launch_boundary_ready"] is False, report
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
        "boundary_ready": bool(report.get("explicit_dispatch_execution_kernel_launch_boundary_ready", False)),
        "review_signed": bool(report.get("dispatch_execution_kernel_launch_review_signed", False)),
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
