"""Smoke checks for native-update runtime execution contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_native_update_runtime_execution_contract import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_runtime_execution_contract,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_started",
    "native_dispatch_executed",
    "runtime_execution_allowed",
    "runtime_execution_enabled",
    "runtime_execution_started",
    "runtime_execution_executed",
    "operator_execution_allowed",
    "operator_execution_enabled",
    "operator_execution_executed",
    "manual_execution_allowed",
    "manual_execution_executed",
    "runtime_state_refresh_allowed",
    "runtime_state_refreshed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_executed",
    "artifact_loaded",
    "execution_replay_executed",
    "parity_executed",
    "kernel_launch_allowed",
    "kernel_launch_executed",
    "training_step_executed",
    "ready_for_ui",
    "ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)


def run_smoke() -> dict[str, Any]:
    pending = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence=_runtime_execution_evidence(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_runtime_execution_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_runtime_execution_request_fields"] == {}, pending
    assert "native_update_runtime_execution_review_missing" in pending["blocked_reasons"], pending
    _assert_default_off(pending)

    signed = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence=_runtime_execution_evidence(),
        runtime_execution_review=_runtime_execution_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["runtime_execution_contract_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_activation = build_native_update_runtime_execution_contract(
        activation_contract={},
        runtime_execution_evidence=_runtime_execution_evidence(),
    )
    _assert_evidence_blocked(missing_activation, "activation_contract_missing")

    missing_evidence = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence={},
    )
    _assert_evidence_blocked(missing_evidence, "runtime_execution_evidence_missing")

    unsafe_activation = build_native_update_runtime_execution_contract(
        activation_contract={**_activation_contract(), "runtime_dispatch_enabled": True},
        runtime_execution_evidence=_runtime_execution_evidence(),
    )
    _assert_evidence_blocked(unsafe_activation, "runtime_dispatch_enabled")

    unsafe_evidence = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence={**_runtime_execution_evidence(), "kernel_launch_executed": True},
    )
    _assert_evidence_blocked(unsafe_evidence, "kernel_launch_executed")

    bad_plan = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence=_runtime_execution_evidence(plan_patch={"runtime_execution_executed": True}),
    )
    _assert_evidence_blocked(bad_plan, "runtime_execution_plan_not_default_off")

    bad_precondition = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence=_runtime_execution_evidence(precondition_patch={"runtime_execution_check_enabled": True}),
    )
    _assert_evidence_blocked(bad_precondition, "runtime_execution_precondition_not_default_off")

    unsafe_review = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence=_runtime_execution_evidence(),
        runtime_execution_review={**_runtime_execution_review(approve=True), "approve_kernel_launch_executed": True},
    )
    _assert_review_blocked(unsafe_review, "approve_kernel_launch_executed")

    missing_ack = build_native_update_runtime_execution_contract(
        activation_contract=_activation_contract(),
        runtime_execution_evidence=_runtime_execution_evidence(),
        runtime_execution_review={**_runtime_execution_review(approve=True), "acknowledge_no_training_step": False},
    )
    _assert_review_blocked(missing_ack, "ack_missing", "training_step")

    real_artifact = _optional_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_runtime_execution_contract_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _optional_real_artifact_case() -> dict[str, Any]:
    activation_path = REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_activation_contract.json"
    evidence_path = REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_runtime_execution_evidence.json"
    if not activation_path.exists() or not evidence_path.exists():
        return {}
    package = build_native_update_runtime_execution_contract(
        activation_contract=load_json(activation_path),
        runtime_execution_evidence=load_json(evidence_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_runtime_execution_request_fields"] == {}, package
    _assert_default_off(package)
    return package


def _assert_default_off(package: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert package[field] is False, (field, package)


def _assert_evidence_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_ready"] is False, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _assert_review_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def _activation_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_activation_contract_v0",
        "ok": True,
        "evidence_ready": True,
        "ready_for_activation_review": True,
        "activation_contract_recorded": False,
        "decision": "native_update_activation_contract_hold_for_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "native_dispatch_enabled": False,
        "runtime_dispatch_enabled": False,
        "runtime_activation_enabled": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "post_activation_request_fields": {},
    }


def _runtime_execution_evidence(
    *,
    plan_patch: dict[str, Any] | None = None,
    precondition_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sections = [
        "activation_contract_reference",
        "runtime_execution_plan_inventory",
        "runtime_execution_precondition_inventory",
        "runtime_isolation_boundary",
        "runtime_state_boundary",
        "no_runtime_execution_boundary",
        "no_operator_execution_boundary",
        "no_manual_execution_boundary",
        "no_runtime_state_refresh_boundary",
        "no_runtime_dispatch_boundary",
        "no_native_dispatch_boundary",
        "no_kernel_launch_boundary",
        "no_training_step_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    plan = {
        "plan_id": "native_update_runtime_execution_default_off",
        "runtime_execution_allowed": False,
        "runtime_execution_enabled": False,
        "runtime_execution_executed": False,
        "operator_execution_executed": False,
        "manual_execution_executed": False,
        "runtime_state_refreshed": False,
        "runtime_dispatch_executed": False,
        "native_dispatch_executed": False,
        "kernel_launch_executed": False,
        "training_step_executed": False,
        "request_fields_emitted": False,
    }
    plan.update(plan_patch or {})
    precondition = {
        "check_id": "native_update_runtime_execution_preconditions",
        "precondition_registered": False,
        "precondition_active": False,
        "runtime_execution_check_enabled": False,
    }
    precondition.update(precondition_patch or {})
    return {
        "schema_version": 1,
        "evidence": "native_update_runtime_execution_evidence_v0",
        "ok": True,
        "runtime_execution_contract_ready": True,
        "report_only": True,
        "contract_only": True,
        "runtime_execution_contract_only": True,
        "records_evidence_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_later_runtime_dispatch_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": "smoke://native_update_runtime_execution_evidence",
        "artifact_digest": "smoke-runtime-execution-digest",
        "sections": sections,
        "runtime_execution_plan_inventory": [plan],
        "runtime_execution_precondition_inventory": [precondition],
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
    }


def _runtime_execution_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_runtime_execution_contract",
        "approve_native_update_runtime_execution_contract": bool(approve),
    }
    for field in (
        "acknowledge_activation_contract_ready",
        "acknowledge_runtime_execution_plan_default_off",
        "acknowledge_runtime_execution_preconditions_default_off",
        "acknowledge_no_runtime_execution",
        "acknowledge_no_operator_or_manual_execution",
        "acknowledge_no_runtime_state_refresh",
        "acknowledge_no_native_dispatch_or_kernel_launch",
        "acknowledge_no_training_step",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_later_runtime_dispatch_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
