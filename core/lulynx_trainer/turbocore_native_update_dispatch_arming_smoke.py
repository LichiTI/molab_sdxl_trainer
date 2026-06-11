"""Smoke checks for TurboCore native update pre-step arming."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_update_dispatch_arming import TurboCoreNativeUpdateDispatchArmer  # noqa: E402


def test_first_step_uses_pytorch_without_previous_gate() -> None:
    armer = TurboCoreNativeUpdateDispatchArmer()
    report = armer.prepare_before_optimizer(step=0)
    reasons = set(report["blocked_reasons"])
    assert report["decision"] == "turbocore_native_update_dispatch_arming_v0", report
    assert report["previous_gate_present"] is False, report
    assert report["execute_native_step"] is False, report
    assert report["call_pytorch_optimizer_step"] is True, report
    assert report["native_dispatch_rehearsal_evidence_ready"] is False, report
    assert report["native_dispatch_training_promotion_preconditions_ready"] is False, report
    assert "previous_gate_report_missing" in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report


def test_previous_gate_only_arms_report_not_execution() -> None:
    armer = TurboCoreNativeUpdateDispatchArmer()
    gate = {
        "dispatch_request": {"requested": True, "dispatch_allowed": False},
        "dispatch_contract": {"dispatch_rehearsal_ready": False},
        "kernel_launch_plan": {"launch_allowed": False},
    }
    observation = armer.observe_after_optimizer(gate)
    report = armer.prepare_before_optimizer(step=1)
    reasons = set(report["blocked_reasons"])
    assert observation["observed_gate_report"] is True, observation
    assert observation["next_step_can_consider_native_dispatch"] is False, observation
    assert report["previous_gate_present"] is True, report
    assert report["previous_request_requested"] is True, report
    assert report["previous_request_allowed"] is False, report
    assert report["armed_for_native_dispatch"] is False, report
    assert report["previous_kernel_launch_plan_present"] is True, report
    assert report["previous_kernel_launch_allowed"] is False, report
    assert report["execute_native_step"] is False, report
    assert report["call_pytorch_optimizer_step"] is True, report
    assert "previous_dispatch_request_not_allowed" in reasons, report
    assert "previous_dispatch_contract_not_ready" in reasons, report
    assert "previous_kernel_launch_not_allowed" in reasons, report


def test_rehearsal_evidence_ready_still_does_not_arm_training_dispatch() -> None:
    armer = TurboCoreNativeUpdateDispatchArmer()
    gate = {
        "dispatch_request": {
            "requested": True,
            "dispatch_allowed": False,
            "training_path_request": {
                "request_boundary_ready": True,
                "explicit_training_path_requested": False,
            },
        },
        "dispatch_contract": {
            "dispatch_rehearsal_ready": False,
            "rehearsal": {"would_launch_native_kernel": True},
            "evidence": {
                "owner_native_launch_ok": True,
                "copyback_dispatch_validated": True,
                "event_chain_verified": False,
                "stream_ordering_verified": False,
                "stream_lifetime_ownership_bound": False,
                "representative_performance_gate_ready": False,
            },
            "recovery": {
                "default_off_recovery_bridge_ready": True,
                "recovery_observation_bridge_ready": True,
                "training_dispatch_recovery_ready": False,
            },
            "direct_gradient_write": {
                "write_boundary_ready": True,
                "native_supported": False,
                "training_lifecycle_integrated": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["direct_gradient_write_default_off"],
            },
            "owner_gradient_sync": {
                "sync_boundary_ready": True,
                "native_supported": True,
                "training_lifecycle_integrated": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["owner_gradient_sync_default_off"],
            },
            "training_executor": {
                "executor_boundary_ready": True,
                "callable_slot_integrated": True,
                "bound_to_training_path": False,
                "training_executor_preconditions_ready": False,
            },
            "training_flat_owner": {
                "owner_boundary_ready": True,
                "reference_owner_ready": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["native_training_flat_owner_default_off"],
            },
            "training_dispatch_kernel": {
                "kernel_boundary_ready": True,
                "kernel_present_evidence": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["native_training_dispatch_kernel_default_off"],
            },
            "stream_lifetime_ownership": {
                "ownership_boundary_ready": True,
                "ordering_verified": False,
                "ownership_bound_evidence": False,
                "bound_to_training_path": False,
                "blocked_reasons": ["stream_event_chain_validation_missing"],
            },
        },
        "kernel_launch_plan": {
            "launch_allowed": False,
            "evidence": {
                "diagnostic_kernel_executed": True,
                "diagnostic_parity_ok": True,
            },
        },
    }
    armer.observe_after_optimizer(gate)
    report = armer.prepare_before_optimizer(step=3)
    reasons = set(report["blocked_reasons"])
    preconditions = report["promotion_preconditions"]
    missing = set(preconditions["missing_for_training_promotion"])
    assert report["previous_request_requested"] is True, report
    assert report["native_dispatch_rehearsal_evidence_ready"] is True, report
    assert report["native_dispatch_training_promotion_preconditions_ready"] is False, report
    assert report["armed_for_native_dispatch"] is False, report
    assert report["execute_native_step"] is False, report
    assert preconditions["owner_native_kernel_ready"] is True, report
    assert preconditions["copyback_dispatch_ready"] is True, report
    assert preconditions["recovery_observation_bridge_ready"] is True, report
    assert preconditions["training_dispatch_recovery_ready"] is False, report
    assert preconditions["training_dispatch_recovery_blocked"] is True, report
    assert preconditions["direct_gradient_write_boundary_ready"] is True, report
    assert preconditions["direct_gradient_write_native_supported"] is False, report
    assert preconditions["direct_gradient_write_lifecycle_ready"] is True, report
    assert preconditions["direct_gradient_write_bound"] is False, report
    assert preconditions["direct_gradient_write_default_off"] is True, report
    assert preconditions["training_flat_owner_boundary_ready"] is True, report
    assert preconditions["training_flat_owner_reference_ready"] is True, report
    assert preconditions["training_flat_owner_bound"] is False, report
    assert preconditions["training_flat_owner_default_off"] is True, report
    assert preconditions["training_dispatch_kernel_boundary_ready"] is True, report
    assert preconditions["training_dispatch_kernel_evidence_present"] is True, report
    assert preconditions["training_dispatch_kernel_bound"] is False, report
    assert preconditions["training_dispatch_kernel_default_off"] is True, report
    assert preconditions["training_runtime_executor_boundary_ready"] is True, report
    assert preconditions["training_runtime_executor_bound"] is False, report
    assert preconditions["training_runtime_executor_default_off"] is True, report
    assert preconditions["training_path_request_boundary_ready"] is True, report
    assert preconditions["explicit_training_path_requested"] is False, report
    assert preconditions["training_path_default_off"] is True, report
    assert "native_dispatch_rehearsal_evidence_only" in reasons, report
    assert "stream_event_chain_validation_missing" in missing, report
    assert "training_dispatch_recovery_missing" not in missing, report
    assert "training_dispatch_recovery_default_off" in missing, report
    assert "direct_gradient_write_default_off" not in missing, report
    assert "direct_gradient_write_not_native_supported" not in missing, report
    assert "owner_gradient_sync_default_off" in missing, report
    assert "native_training_flat_owner_default_off" in missing, report
    assert "native_training_flat_owner_not_promoted" not in missing, report
    assert "native_training_dispatch_kernel_default_off" in missing, report
    assert "native_training_dispatch_kernel_not_promoted" not in missing, report
    assert "representative_performance_gate_missing" in missing, report
    assert "native_dispatch_training_runtime_executor_missing" not in missing, report
    assert "native_dispatch_training_runtime_executor_default_off" in missing, report
    assert "native_dispatch_training_path_disabled" not in missing, report
    assert "native_dispatch_training_path_default_off" in missing, report


def test_ordering_ready_without_stream_lifetime_ownership_still_blocks_promotion() -> None:
    armer = TurboCoreNativeUpdateDispatchArmer()
    gate = {
        "dispatch_request": {
            "requested": True,
            "dispatch_allowed": False,
            "training_path_request": {
                "request_boundary_ready": True,
                "explicit_training_path_requested": False,
            },
        },
        "dispatch_contract": {
            "dispatch_rehearsal_ready": False,
            "rehearsal": {"would_launch_native_kernel": True},
            "evidence": {
                "owner_native_launch_ok": True,
                "copyback_dispatch_validated": True,
                "event_chain_verified": True,
                "stream_ordering_verified": True,
                "stream_lifetime_ownership_bound": False,
                "representative_performance_gate_ready": False,
            },
            "recovery": {
                "default_off_recovery_bridge_ready": True,
                "recovery_observation_bridge_ready": True,
                "training_dispatch_recovery_ready": False,
            },
            "direct_gradient_write": {
                "write_boundary_ready": True,
                "native_supported": False,
                "training_lifecycle_integrated": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["direct_gradient_write_default_off"],
            },
            "owner_gradient_sync": {
                "sync_boundary_ready": True,
                "native_supported": True,
                "training_lifecycle_integrated": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["owner_gradient_sync_default_off"],
            },
            "training_executor": {
                "executor_boundary_ready": True,
                "callable_slot_integrated": True,
                "bound_to_training_path": False,
                "training_executor_preconditions_ready": False,
            },
            "training_flat_owner": {
                "owner_boundary_ready": True,
                "reference_owner_ready": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["native_training_flat_owner_default_off"],
            },
            "training_dispatch_kernel": {
                "kernel_boundary_ready": True,
                "kernel_present_evidence": True,
                "bound_to_training_path": False,
                "blocked_reasons": ["native_training_dispatch_kernel_default_off"],
            },
            "stream_lifetime_ownership": {
                "ownership_boundary_ready": True,
                "ordering_verified": True,
                "ownership_bound_evidence": False,
                "bound_to_training_path": False,
                "blocked_reasons": ["stream_lifetime_ownership_default_off"],
            },
        },
        "kernel_launch_plan": {
            "launch_allowed": False,
            "evidence": {
                "diagnostic_kernel_executed": True,
                "diagnostic_parity_ok": True,
            },
        },
    }
    armer.observe_after_optimizer(gate)
    report = armer.prepare_before_optimizer(step=4)
    preconditions = report["promotion_preconditions"]
    missing = set(preconditions["missing_for_training_promotion"])
    assert report["native_dispatch_rehearsal_evidence_ready"] is True, report
    assert preconditions["stream_event_chain_ready"] is True, report
    assert preconditions["stream_ordering_ready"] is True, report
    assert preconditions["stream_lifetime_ownership_boundary_ready"] is True, report
    assert preconditions["stream_lifetime_ownership_evidence_bound"] is False, report
    assert preconditions["stream_lifetime_ownership_default_off"] is True, report
    assert preconditions["stream_lifetime_ownership_ready"] is False, report
    assert "stream_event_chain_validation_missing" not in missing, report
    assert "stream_lifetime_ownership_default_off" in missing, report
    assert "stream_lifetime_ownership_not_promoted" not in missing, report
    assert "direct_gradient_write_default_off" not in missing, report
    assert "owner_gradient_sync_default_off" in missing, report
    assert "native_training_flat_owner_default_off" in missing, report
    assert "native_training_dispatch_kernel_default_off" in missing, report
    assert report["execute_native_step"] is False, report


def test_runtime_disable_latch_blocks_arming() -> None:
    armer = TurboCoreNativeUpdateDispatchArmer()
    armer.observe_after_optimizer(
        {
            "dispatch_request": {"requested": True, "dispatch_allowed": False},
            "dispatch_contract": {"dispatch_rehearsal_ready": False},
            "kernel_launch_plan": {"launch_allowed": False},
        }
    )
    report = armer.prepare_before_optimizer(
        step=2,
        runtime_state={"disabled_for_run": True, "disable_reason": "native_runtime_error_observed"},
    )
    assert report["runtime_disabled_for_run"] is True, report
    assert report["runtime_disable_reason"] == "native_runtime_error_observed", report
    assert "native_dispatch_disabled_for_run" in report["blocked_reasons"], report


def test_full_training_promotion_preconditions_can_arm_dispatch() -> None:
    armer = TurboCoreNativeUpdateDispatchArmer()
    gate = {
        "dispatch_request": {
            "requested": True,
            "dispatch_allowed": True,
            "training_path_request": {
                "request_boundary_ready": True,
                "explicit_training_path_requested": True,
            },
        },
        "dispatch_contract": {
            "dispatch_rehearsal_ready": True,
            "rehearsal": {"would_launch_native_kernel": True},
            "evidence": {
                "owner_native_launch_ok": True,
                "copyback_dispatch_validated": True,
                "event_chain_verified": True,
                "stream_ordering_verified": True,
                "stream_lifetime_ownership_bound": True,
                "representative_performance_gate_ready": True,
            },
            "recovery": {
                "default_off_recovery_bridge_ready": True,
                "recovery_observation_bridge_ready": True,
                "training_dispatch_recovery_ready": True,
            },
            "direct_gradient_write": {
                "write_boundary_ready": True,
                "native_supported": True,
                "training_lifecycle_integrated": True,
                "bound_to_training_path": True,
                "blocked_reasons": [],
            },
            "owner_gradient_sync": {
                "sync_boundary_ready": True,
                "native_supported": True,
                "training_lifecycle_integrated": True,
                "bound_to_training_path": True,
                "owner_gradient_sync_preconditions_ready": True,
                "blocked_reasons": [],
            },
            "training_executor": {
                "executor_boundary_ready": True,
                "callable_slot_integrated": True,
                "bound_to_training_path": True,
                "training_executor_preconditions_ready": True,
            },
            "training_flat_owner": {
                "owner_boundary_ready": True,
                "reference_owner_ready": True,
                "bound_to_training_path": True,
                "blocked_reasons": [],
            },
            "training_dispatch_kernel": {
                "kernel_boundary_ready": True,
                "kernel_present_evidence": True,
                "bound_to_training_path": True,
                "blocked_reasons": [],
            },
            "stream_lifetime_ownership": {
                "ownership_boundary_ready": True,
                "ordering_verified": True,
                "ownership_bound_evidence": True,
                "bound_to_training_path": True,
                "blocked_reasons": [],
            },
        },
        "kernel_launch_plan": {
            "launch_allowed": True,
            "evidence": {
                "diagnostic_kernel_executed": True,
                "diagnostic_parity_ok": True,
            },
        },
    }
    armer.observe_after_optimizer(gate)
    report = armer.prepare_before_optimizer(step=5)
    preconditions = report["promotion_preconditions"]
    assert preconditions["training_promotion_ready"] is True, report
    assert preconditions["missing_for_training_promotion"] == [], report
    assert report["native_dispatch_training_promotion_preconditions_ready"] is True, report
    assert report["armed_for_native_dispatch"] is True, report
    assert report["execute_native_step"] is True, report
    assert report["call_pytorch_optimizer_step"] is False, report
    assert report["native_mutation_allowed"] is True, report


def test_shadow_autostop_keeps_last_promoted_gate_for_perf_latch() -> None:
    armer = TurboCoreNativeUpdateDispatchArmer()
    promoted_gate = {
        "mode": "native_experimental",
        "config": {"mode": "native_experimental", "dispatch_enabled": True},
        "dispatch_contract": {
            "dispatch_rehearsal_ready": True,
            "rehearsal": {"would_launch_native_kernel": True},
            "evidence": {
                "owner_native_launch_ok": True,
                "copyback_dispatch_validated": True,
                "event_chain_verified": True,
                "stream_ordering_verified": True,
                "stream_lifetime_ownership_bound": True,
                "representative_performance_gate_ready": True,
            },
            "recovery": {
                "default_off_recovery_bridge_ready": True,
                "recovery_observation_bridge_ready": True,
                "training_dispatch_recovery_ready": True,
            },
            "owner_gradient_sync": {
                "sync_boundary_ready": True,
                "native_supported": True,
                "training_lifecycle_integrated": True,
                "bound_to_training_path": True,
                "owner_gradient_sync_preconditions_ready": True,
            },
            "training_executor": {
                "executor_boundary_ready": True,
                "bound_to_training_path": True,
                "training_executor_preconditions_ready": True,
            },
            "training_flat_owner": {
                "owner_boundary_ready": True,
                "reference_owner_ready": True,
                "bound_to_training_path": True,
            },
            "training_dispatch_kernel": {
                "kernel_boundary_ready": True,
                "kernel_present_evidence": True,
                "bound_to_training_path": True,
            },
            "stream_lifetime_ownership": {
                "ownership_boundary_ready": True,
                "ordering_verified": True,
                "ownership_bound_evidence": True,
                "bound_to_training_path": True,
            },
        },
        "dispatch_request": {
            "requested": True,
            "dispatch_allowed": True,
            "training_path_enabled": True,
            "training_path_request": {
                "request_boundary_ready": True,
                "explicit_training_path_requested": True,
            },
        },
        "kernel_launch_plan": {"launch_allowed": True, "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True}},
    }
    armer.observe_after_optimizer(promoted_gate)
    observation = armer.observe_after_optimizer(
        {
            "mode": "native_experimental",
            "config": {"mode": "native_experimental", "dispatch_enabled": True},
            "shadow": {"reason": "auto_stopped_after_consecutive_passes", "blocked_reasons": ["shadow_not_compared"]},
            "readiness": {"blocked_reasons": ["owner_native_launch_probe_missing"]},
            "blocked_reasons": ["shadow_not_compared", "owner_native_launch_probe_missing"],
            "fallback_policy": {"runtime_recovery": {"disable_native_update_for_run": False}},
        }
    )
    report = armer.prepare_before_optimizer(step=6)
    assert observation["retained_previous_gate_after_shadow_autostop"] is True, observation
    assert observation["next_step_can_consider_native_dispatch"] is True, observation
    assert observation["retained_probe_evidence"] is True, observation
    assert observation["probe_cache_reused_steps"] == 1, observation
    assert report["armed_for_native_dispatch"] is True, report
    assert report["execute_native_step"] is True, report
    assert report["retained_probe_evidence"] is True, report
    assert report["probe_cache_source"] == "previous_step_gate_report", report


def main() -> int:
    test_first_step_uses_pytorch_without_previous_gate()
    test_previous_gate_only_arms_report_not_execution()
    test_rehearsal_evidence_ready_still_does_not_arm_training_dispatch()
    test_ordering_ready_without_stream_lifetime_ownership_still_blocks_promotion()
    test_runtime_disable_latch_blocks_arming()
    test_full_training_promotion_preconditions_can_arm_dispatch()
    test_shadow_autostop_keeps_last_promoted_gate_for_perf_latch()
    print("turbocore_native_update_dispatch_arming_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
