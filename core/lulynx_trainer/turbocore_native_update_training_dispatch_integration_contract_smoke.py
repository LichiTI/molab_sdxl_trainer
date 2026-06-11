"""Smoke checks for native-update training dispatch integration contract."""

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

from core.turbocore_native_update_dispatch_contract import build_native_update_dispatch_contract  # noqa: E402
from core.turbocore_native_update_training_dispatch_integration_contract import (  # noqa: E402
    BLOCKED_DECISION,
    HOLD_DECISION,
    READY_DECISION,
    build_native_update_training_dispatch_integration_contract,
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
    "native_mutation_allowed",
    "training_parameter_mutation_allowed",
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
    pending = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract=_default_off_dispatch_contract(),
    )
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_integration_review"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["post_integration_request_fields"] == {}, pending
    assert "native_update_training_dispatch_integration_review_missing" in pending["blocked_reasons"], pending
    assert pending["dispatch_contract_summary"]["component_boundary_count"] == 7, pending
    assert pending["dispatch_contract_summary"]["component_default_off_count"] == 7, pending
    _assert_default_off(pending)

    signed = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract=_default_off_dispatch_contract(),
        integration_review=_integration_review(approve=True),
    )
    assert signed["ok"] is True, signed
    assert signed["integration_contract_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    missing_rollout = build_native_update_training_dispatch_integration_contract(
        rollout_review_package={},
        dispatch_contract=_default_off_dispatch_contract(),
    )
    _assert_evidence_blocked(missing_rollout, "rollout_review_package_missing")

    missing_dispatch = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract={},
    )
    _assert_evidence_blocked(missing_dispatch, "dispatch_contract_missing")

    unsafe_rollout = build_native_update_training_dispatch_integration_contract(
        rollout_review_package={**_rollout_package(), "request_fields_emitted": True},
        dispatch_contract=_default_off_dispatch_contract(),
    )
    _assert_evidence_blocked(unsafe_rollout, "request_fields_emitted")

    unsafe_dispatch = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract={**_default_off_dispatch_contract(), "training_dispatch": True},
    )
    _assert_evidence_blocked(unsafe_dispatch, "training_dispatch")

    missing_component = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract=_dispatch_without_flat_owner_boundary(),
    )
    _assert_evidence_blocked(missing_component, "training_flat_owner")

    positive_dispatch = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract=_enabled_dispatch_contract(),
    )
    _assert_evidence_blocked(positive_dispatch, "training_dispatch", "training_executor")

    unsafe_review = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract=_default_off_dispatch_contract(),
        integration_review={**_integration_review(approve=True), "approve_request_fields_emitted": True},
    )
    _assert_review_blocked(unsafe_review, "approve_request_fields_emitted")

    review_missing_ack = build_native_update_training_dispatch_integration_contract(
        rollout_review_package=_rollout_package(),
        dispatch_contract=_default_off_dispatch_contract(),
        integration_review={**_integration_review(approve=True), "acknowledge_training_kernel_default_off": False},
    )
    _assert_review_blocked(review_missing_ack, "ack_missing", "training_kernel")

    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_training_dispatch_integration_contract_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "recommended_next_step": pending["recommended_next_step"],
    }


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


def _rollout_package() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_rollout_review_package_v0",
        "ok": True,
        "evidence_package_ready": True,
        "ready_for_owner_review": True,
        "native_update_rollout_review_recorded": False,
        "decision": "native_update_rollout_review_hold_for_owner_review_default_off",
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "post_native_update_request_fields": {},
    }


def _default_off_dispatch_contract() -> dict[str, Any]:
    return build_native_update_dispatch_contract(
        mode="native_experimental",
        requested=True,
        readiness_report={
            "native_kernel_present": True,
            "training_dispatch_kernel_present": True,
            "stream_lifetime_bound": True,
            "stream_lifetime_ownership_bound": True,
            "stream_ordering_verified": True,
            "performance_test_ready": True,
            "native_checks": {
                "flat_owner_contract_ready": True,
                "reference_flat_owner_ready": True,
                "training_flat_owner_promoted": True,
                "training_dispatch_kernel_contract_ready": True,
                "training_dispatch_kernel_present": True,
            },
            "owner_checks": {
                "direct_gradient_write_boundary_ready": True,
                "direct_gradient_write_native_supported": True,
                "direct_gradient_write_training_integrated": True,
                "owner_gradient_sync_boundary_ready": True,
                "owner_gradient_sync_supported": True,
                "owner_gradient_sync_training_integrated": True,
            },
        },
        shadow_report={
            "copyback_dispatch_probe": {
                "copyback_dispatch_validated": True,
                "copyback_dispatch_target": "training_parameters",
            },
            "native_binding_probe": {
                "stream_lifetime_bound": True,
                "event_chain_verified": True,
            },
            "owner_native_launch_probe": {
                "ok": True,
                "kernel_executed": True,
                "parity_ok": True,
                "event_chain_verified": True,
            },
        },
        dispatch_preflight={
            "dispatch_preflight_passed": True,
            "native_kernel_present": True,
            "stream_lifetime_bound": True,
            "stream_lifetime_ownership_bound": True,
            "stream_ordering_verified": True,
            "performance_test_ready": True,
            "evidence": {"performance": {"representative_performance_gate_ready": True}},
            "blocked_reasons": [],
        },
        fallback_policy={
            "fallback_to_pytorch_enabled": True,
            "runtime_recovery": {
                "default_off_recovery_bridge_ready": True,
                "recovery_observation_bridge_ready": True,
                "training_dispatch_recovery_ready": False,
                "training_dispatch_recovery_blocked": False,
                "actions": [],
                "blocked_reasons": ["training_dispatch_recovery_default_off"],
            },
        },
        runtime_context={},
    )


def _enabled_dispatch_contract() -> dict[str, Any]:
    report = _default_off_dispatch_contract()
    for key in ("training_dispatch", "training_path_enabled", "would_allow_native_dispatch"):
        report[key] = True
    report["pytorch_optimizer_authoritative"] = False
    report["training_executor"] = {**report["training_executor"], "default_off": False, "bound_to_training_path": True}
    return report


def _dispatch_without_flat_owner_boundary() -> dict[str, Any]:
    report = _default_off_dispatch_contract()
    report["training_flat_owner"] = {**report["training_flat_owner"], "owner_boundary_ready": False}
    return report


def _integration_review(*, approve: bool) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-04",
        "requested_scope": "native_update_training_dispatch_integration_contract",
        "approve_native_update_training_dispatch_integration_contract": bool(approve),
    }
    for field in (
        "acknowledge_rollout_review_package_ready",
        "acknowledge_recovery_boundary_default_off",
        "acknowledge_owner_gradient_sync_default_off",
        "acknowledge_flat_owner_default_off",
        "acknowledge_training_kernel_default_off",
        "acknowledge_stream_lifetime_ownership_default_off",
        "acknowledge_runtime_executor_default_off",
        "acknowledge_training_path_request_default_off",
        "acknowledge_no_product_training_dispatch",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_later_activation_contract_required",
    ):
        review[field] = True
    return review


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
