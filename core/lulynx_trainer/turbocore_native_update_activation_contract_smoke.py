"""Smoke checks for native-update default-off activation contract."""

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

from core.turbocore_native_update_activation_contract import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_activation_contract,
    load_json,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "auto_launch_allowed",
    "runs_dispatched",
    "default_training_path_enabled",
    "training_path_enabled",
    "training_dispatch",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_started",
    "native_mutation_allowed",
    "training_parameter_mutation_allowed",
    "runtime_activation_allowed",
    "runtime_activation_enabled",
    "runtime_activation_executed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_executed",
    "kernel_launch_allowed",
    "kernel_launch_executed",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "ui_exposure_allowed",
    "product_ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "ui_entry_enabled",
    "ready_for_ui",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)


def run_smoke() -> dict[str, Any]:
    pending = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence=_activation_evidence(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_activation_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_activation_request_fields"] == {}, pending
    assert "native_update_activation_review_missing" in pending["blocked_reasons"], pending
    _assert_default_off(pending)

    signed = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence=_activation_evidence(),
        activation_review=_activation_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["activation_contract_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_integration = build_native_update_activation_contract(
        integration_contract={},
        activation_evidence=_activation_evidence(),
    )
    _assert_evidence_blocked(missing_integration, "integration_contract_missing")

    missing_evidence = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence={},
    )
    _assert_evidence_blocked(missing_evidence, "activation_evidence_missing")

    unsafe_integration = build_native_update_activation_contract(
        integration_contract={**_integration_contract(), "native_dispatch_enabled": True},
        activation_evidence=_activation_evidence(),
    )
    _assert_evidence_blocked(unsafe_integration, "native_dispatch_enabled")

    unsafe_evidence = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence={**_activation_evidence(), "runtime_activation_enabled": True},
    )
    _assert_evidence_blocked(unsafe_evidence, "runtime_activation_enabled")

    bad_plan = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence=_activation_evidence(plan_patch={"activation_enabled": True}),
    )
    _assert_evidence_blocked(bad_plan, "activation_plan_not_default_off")

    bad_precondition = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence=_activation_evidence(precondition_patch={"precondition_active": True}),
    )
    _assert_evidence_blocked(bad_precondition, "activation_precondition_not_default_off")

    unsafe_review = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence=_activation_evidence(),
        activation_review={**_activation_review(approve=True), "approve_runtime_dispatch_enabled": True},
    )
    _assert_review_blocked(unsafe_review, "approve_runtime_dispatch_enabled")

    missing_ack = build_native_update_activation_contract(
        integration_contract=_integration_contract(),
        activation_evidence=_activation_evidence(),
        activation_review={**_activation_review(approve=True), "acknowledge_no_runtime_activation": False},
    )
    _assert_review_blocked(missing_ack, "ack_missing", "runtime_activation")

    real_artifact = _optional_real_artifact_case()
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_activation_contract_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _optional_real_artifact_case() -> dict[str, Any]:
    integration_path = REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_training_dispatch_integration_contract.json"
    evidence_path = REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_activation_evidence.json"
    if not integration_path.exists() or not evidence_path.exists():
        return {}
    package = build_native_update_activation_contract(
        integration_contract=load_json(integration_path),
        activation_evidence=load_json(evidence_path),
    )
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_activation_request_fields"] == {}, package
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


def _integration_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_training_dispatch_integration_contract_v0",
        "ok": True,
        "evidence_ready": True,
        "ready_for_integration_review": True,
        "integration_contract_recorded": False,
        "decision": "native_update_training_dispatch_integration_contract_hold_for_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "training_path_enabled": False,
        "training_dispatch": False,
        "native_dispatch_enabled": False,
        "runtime_dispatch_enabled": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "post_integration_request_fields": {},
        "dispatch_contract_summary": {
            "component_boundary_count": 7,
            "component_default_off_count": 7,
        },
    }


def _activation_evidence(
    *,
    plan_patch: dict[str, Any] | None = None,
    precondition_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sections = [
        "integration_contract_reference",
        "activation_plan_inventory",
        "activation_precondition_inventory",
        "no_product_activation_boundary",
        "no_runtime_dispatch_boundary",
        "no_request_ui_schema_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    plan = {
        "plan_id": "native_update_activation_default_off",
        "activation_allowed": False,
        "activation_enabled": False,
        "runtime_dispatch_enabled": False,
        "training_dispatch": False,
        "request_fields_emitted": False,
    }
    plan.update(plan_patch or {})
    precondition = {
        "check_id": "native_update_activation_preconditions",
        "precondition_registered": False,
        "precondition_active": False,
        "activation_check_enabled": False,
    }
    precondition.update(precondition_patch or {})
    return {
        "schema_version": 1,
        "evidence": "native_update_activation_evidence_v0",
        "ok": True,
        "activation_contract_ready": True,
        "report_only": True,
        "contract_only": True,
        "manual_only": True,
        "internal_only": True,
        "default_off": True,
        "requires_later_operator_opt_in_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "source": "smoke://native_update_activation_evidence",
        "artifact_digest": "smoke-digest",
        "sections": sections,
        "activation_plan_inventory": [plan],
        "activation_precondition_inventory": [precondition],
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
    }


def _activation_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_activation_contract",
        "approve_native_update_activation_contract": bool(approve),
    }
    for field in (
        "acknowledge_integration_contract_ready",
        "acknowledge_activation_plan_default_off",
        "acknowledge_activation_preconditions_default_off",
        "acknowledge_no_product_training_dispatch",
        "acknowledge_no_runtime_activation",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_later_operator_opt_in_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
