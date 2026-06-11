"""Smoke checks for TurboCore native update dispatch contract."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_dispatch_contract import build_native_update_dispatch_contract  # noqa: E402


def test_contract_without_preflight_stays_hard_blocked() -> None:
    report = build_native_update_dispatch_contract(
        mode="off",
        requested=False,
        readiness_report=None,
        shadow_report=None,
    )
    reasons = set(report["blocked_reasons"])
    assert report["contract"] == "turbocore_native_update_dispatch_contract_v0", report
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["dispatch_rehearsal_ready"] is False, report
    assert report["pytorch_optimizer_authoritative"] is True, report
    assert "dispatch_preflight_missing" in reasons, report
    assert "native_update_not_requested" in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report
    assert "native_dispatch_training_path_disabled" in reasons, report


def test_contract_maps_owner_copyback_and_recovery_evidence() -> None:
    shadow = {
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
    }
    fallback = {
        "fallback_to_pytorch_enabled": True,
        "runtime_recovery": {
            "policy_defined": True,
            "dispatch_integration_ready": True,
            "default_off_recovery_bridge_ready": True,
            "recovery_observation_bridge_ready": True,
            "training_dispatch_recovery_ready": False,
            "training_dispatch_recovery_blocked": True,
            "actions": ["keep_training_dispatch_disabled_until_recovery_integrated"],
            "blocked_reasons": ["native_runtime_recovery_training_dispatch_disabled"],
        },
    }
    preflight = {
        "dispatch_preflight_passed": False,
        "native_kernel_present": True,
        "stream_lifetime_bound": True,
        "performance_test_ready": False,
        "evidence": {
            "performance": {
                "representative_performance_gate_ready": False,
                "blocked_reasons": ["representative_training_matrix_missing"],
            }
        },
        "blocked_reasons": ["representative_performance_gate_missing"],
    }
    report = build_native_update_dispatch_contract(
        mode="native_experimental",
        requested=True,
        readiness_report={
            "native_kernel_present": False,
            "training_dispatch_kernel_present": False,
            "native_checks": {
                "flat_owner_contract_ready": True,
                "reference_flat_owner_ready": True,
                "training_flat_owner_promoted": False,
                "training_dispatch_kernel_contract_ready": True,
                "training_dispatch_kernel_present": False,
            },
            "owner_checks": {
                "direct_gradient_write_boundary_ready": True,
                "direct_gradient_write_native_supported": False,
                "direct_gradient_write_training_integrated": True,
            },
        },
        shadow_report=shadow,
        dispatch_preflight=preflight,
        fallback_policy=fallback,
    )
    rehearsal = report["rehearsal"]
    assert report["native_kernel_present"] is True, report
    assert report["stream_lifetime_bound"] is True, report
    assert report["stream_lifetime_ownership_bound"] is True, report
    assert report["stream_ordering_verified"] is True, report
    assert rehearsal["would_use_owner_buffers"] is True, report
    assert rehearsal["would_bind_training_tensors"] is True, report
    assert rehearsal["would_launch_native_kernel"] is True, report
    assert rehearsal["would_copyback_to_training_parameters"] is True, report
    assert rehearsal["would_zero_owner_grad"] is True, report
    assert rehearsal["stream_ordering_ready"] is True, report
    assert report["evidence"]["event_chain_verified"] is True, report
    assert report["recovery"]["recovery_observation_bridge_ready"] is True, report
    assert report["evidence"]["runtime_recovery_observation_bridge_ready"] is True, report
    assert report["recovery"]["training_dispatch_recovery_blocked"] is True, report
    direct_grad = report["direct_gradient_write"]
    assert direct_grad["contract"] == "turbocore_native_update_direct_gradient_write_contract_v0", report
    assert direct_grad["write_boundary_ready"] is True, report
    assert direct_grad["native_supported"] is False, report
    assert direct_grad["training_lifecycle_integrated"] is True, report
    assert direct_grad["bound_to_training_path"] is False, report
    assert direct_grad["default_off"] is True, report
    assert "direct_gradient_write_default_off" in direct_grad["blocked_reasons"], report
    assert "direct_gradient_write_not_native_supported" not in direct_grad["blocked_reasons"], report
    flat_owner = report["training_flat_owner"]
    assert flat_owner["contract"] == "turbocore_native_update_training_flat_owner_contract_v0", report
    assert flat_owner["owner_boundary_ready"] is True, report
    assert flat_owner["reference_owner_ready"] is True, report
    assert flat_owner["bound_to_training_path"] is False, report
    assert flat_owner["default_off"] is True, report
    assert "native_training_flat_owner_default_off" in flat_owner["blocked_reasons"], report
    dispatch_kernel = report["training_dispatch_kernel"]
    assert dispatch_kernel["contract"] == "turbocore_native_update_training_dispatch_kernel_contract_v0", report
    assert dispatch_kernel["kernel_boundary_ready"] is True, report
    assert dispatch_kernel["kernel_present_evidence"] is True, report
    assert dispatch_kernel["bound_to_training_path"] is False, report
    assert dispatch_kernel["default_off"] is True, report
    assert "native_training_dispatch_kernel_default_off" in dispatch_kernel["blocked_reasons"], report
    executor = report["training_executor"]
    assert executor["contract"] == "turbocore_native_update_training_executor_contract_v0", report
    assert executor["callable_slot_integrated"] is True, report
    assert executor["executor_boundary_ready"] is True, report
    assert executor["bound_to_training_path"] is False, report
    assert executor["default_off"] is True, report
    assert "native_dispatch_training_runtime_executor_default_off" in executor["blocked_reasons"], report
    stream_lifetime = report["stream_lifetime_ownership"]
    assert stream_lifetime["contract"] == "turbocore_native_update_stream_lifetime_ownership_contract_v0", report
    assert stream_lifetime["ownership_boundary_ready"] is True, report
    assert stream_lifetime["ordering_verified"] is True, report
    assert stream_lifetime["ownership_bound_evidence"] is True, report
    assert stream_lifetime["bound_to_training_path"] is False, report
    assert stream_lifetime["default_off"] is True, report
    assert "stream_lifetime_ownership_default_off" in stream_lifetime["blocked_reasons"], report
    assert "native_recovery_keeps_dispatch_disabled" in report["blocked_reasons"], report
    assert "collect_representative_native_dispatch_benchmark_matrix" in report["actions_required"], report
    assert report["training_dispatch"] is False, report


def test_contract_keeps_ordering_separate_from_stream_lifetime_ownership() -> None:
    shadow = {
        "copyback_dispatch_probe": {
            "copyback_dispatch_validated": True,
            "copyback_dispatch_target": "training_parameters",
        },
        "native_binding_probe": {
            "stream_lifetime_bound": False,
            "event_chain_verified": False,
        },
        "owner_native_launch_probe": {
            "ok": True,
            "kernel_executed": True,
            "parity_ok": True,
            "event_chain_verified": True,
            "pre_launch_ordering_verified": True,
            "post_launch_ordering_verified": True,
            "stream_wait_event_verified": True,
        },
    }
    preflight = {
        "dispatch_preflight_passed": False,
        "native_kernel_present": True,
        "stream_lifetime_bound": True,
        "stream_lifetime_ownership_bound": False,
        "stream_ordering_verified": True,
        "performance_test_ready": False,
        "evidence": {"performance": {"representative_performance_gate_ready": False}},
        "blocked_reasons": ["stream_lifetime_ownership_not_promoted"],
    }
    report = build_native_update_dispatch_contract(
        mode="native_experimental",
        requested=True,
        readiness_report={
            "native_kernel_present": False,
            "stream_lifetime_bound": True,
            "stream_lifetime_ownership_bound": False,
            "stream_ordering_verified": True,
            "native_checks": {
                "flat_owner_contract_ready": True,
                "reference_flat_owner_ready": True,
                "training_flat_owner_promoted": False,
                "training_dispatch_kernel_contract_ready": True,
                "training_dispatch_kernel_present": False,
            },
            "owner_checks": {
                "direct_gradient_write_boundary_ready": True,
                "direct_gradient_write_native_supported": False,
                "direct_gradient_write_training_integrated": True,
            },
        },
        shadow_report=shadow,
        dispatch_preflight=preflight,
        fallback_policy={"fallback_to_pytorch_enabled": True},
    )
    assert report["stream_lifetime_bound"] is True, report
    assert report["stream_lifetime_ownership_bound"] is False, report
    assert report["stream_ordering_verified"] is True, report
    assert report["rehearsal"]["stream_ordering_ready"] is True, report
    assert report["evidence"]["event_chain_verified"] is True, report
    assert report["evidence"]["stream_ordering_verified"] is True, report
    assert report["evidence"]["stream_lifetime_ownership_bound"] is False, report
    stream_lifetime = report["stream_lifetime_ownership"]
    assert stream_lifetime["ownership_boundary_ready"] is True, report
    assert stream_lifetime["ordering_verified"] is True, report
    assert stream_lifetime["ownership_bound_evidence"] is False, report
    assert stream_lifetime["bound_to_training_path"] is False, report
    assert "stream_lifetime_ownership_default_off" in stream_lifetime["blocked_reasons"], report
    assert "stream_lifetime_ownership_not_promoted" in report["blocked_reasons"], report
    assert "validate_stream_lifetime_and_event_chain" in report["actions_required"], report


def test_contract_reports_not_promoted_after_explicit_training_context() -> None:
    report = build_native_update_dispatch_contract(
        mode="native_experimental",
        requested=True,
        readiness_report={
            "native_kernel_present": False,
            "training_dispatch_kernel_present": False,
            "native_checks": {
                "flat_owner_contract_ready": True,
                "reference_flat_owner_ready": True,
                "training_flat_owner_promoted": False,
                "training_dispatch_kernel_contract_ready": True,
                "training_dispatch_kernel_present": False,
            },
            "owner_checks": {
                "direct_gradient_write_boundary_ready": True,
                "direct_gradient_write_native_supported": False,
                "direct_gradient_write_training_integrated": True,
            },
        },
        shadow_report={"owner_native_launch_probe": {"kernel_executed": False}},
        runtime_context={
            "native_update_training_dispatch_enabled": True,
            "training_path_enabled": True,
        },
    )
    flat_owner = report["training_flat_owner"]
    dispatch_kernel = report["training_dispatch_kernel"]
    direct_grad = report["direct_gradient_write"]
    assert direct_grad["explicit_training_context_requested"] is True, report
    assert "direct_gradient_write_default_off" not in direct_grad["blocked_reasons"], report
    assert "direct_gradient_write_not_native_supported" in direct_grad["blocked_reasons"], report
    assert flat_owner["explicit_training_context_requested"] is True, report
    assert "native_training_flat_owner_default_off" not in flat_owner["blocked_reasons"], report
    assert "native_training_flat_owner_not_promoted" in flat_owner["blocked_reasons"], report
    assert dispatch_kernel["explicit_training_context_requested"] is True, report
    assert "native_training_dispatch_kernel_default_off" not in dispatch_kernel["blocked_reasons"], report
    assert "native_training_dispatch_kernel_not_promoted" in dispatch_kernel["blocked_reasons"], report


def test_contract_can_allow_dispatch_when_all_training_preconditions_are_bound() -> None:
    report = build_native_update_dispatch_contract(
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
                "policy_defined": True,
                "dispatch_integration_ready": True,
                "default_off_recovery_bridge_ready": True,
                "recovery_observation_bridge_ready": True,
                "training_dispatch_recovery_ready": True,
                "training_dispatch_recovery_blocked": False,
                "actions": [],
                "blocked_reasons": [],
            },
        },
        runtime_context={
            "native_update_training_dispatch_enabled": True,
            "training_path_enabled": True,
            "native_update_runtime_dispatch_available": True,
            "native_update_owner_gradient_sync_guard_enabled": True,
            "native_update_owner_gradient_sync_bound": True,
            "native_update_direct_gradient_write_guard_enabled": True,
            "native_update_direct_gradient_write_bound": True,
            "native_update_flat_owner_training_guard_enabled": True,
            "native_update_flat_owner_bound": True,
            "native_update_training_dispatch_kernel_guard_enabled": True,
            "native_update_training_dispatch_kernel_bound": True,
            "native_update_executor_present": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
            "native_update_stream_lifetime_ownership_guard_enabled": True,
            "native_update_stream_lifetime_ownership_bound": True,
        },
    )
    assert report["training_dispatch"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["dispatch_rehearsal_ready"] is True, report
    assert report["would_allow_native_dispatch"] is True, report
    assert report["pytorch_optimizer_authoritative"] is False, report
    assert report["native_mutation_allowed"] is True, report
    assert report["training_parameter_mutation_allowed"] is True, report
    assert report["blocked_reasons"] == [], report
    assert report["actions_required"] == [], report
    assert report["direct_gradient_write"]["direct_gradient_write_preconditions_ready"] is True, report
    assert report["training_flat_owner"]["training_flat_owner_preconditions_ready"] is True, report
    assert report["training_dispatch_kernel"]["training_dispatch_kernel_preconditions_ready"] is True, report
    assert report["training_executor"]["training_executor_preconditions_ready"] is True, report
    assert report["stream_lifetime_ownership"]["stream_lifetime_ownership_preconditions_ready"] is True, report
    assert report["recovery"]["training_dispatch_recovery_ready"] is True, report


def main() -> int:
    test_contract_without_preflight_stays_hard_blocked()
    test_contract_maps_owner_copyback_and_recovery_evidence()
    test_contract_keeps_ordering_separate_from_stream_lifetime_ownership()
    test_contract_reports_not_promoted_after_explicit_training_context()
    test_contract_can_allow_dispatch_when_all_training_preconditions_are_bound()
    print("turbocore_native_update_dispatch_contract_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
