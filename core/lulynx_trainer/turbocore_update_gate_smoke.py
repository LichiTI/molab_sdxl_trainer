"""Smoke checks for TurboCore native update gate policy."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_update_gate import TurboCoreNativeUpdateGate, build_native_update_gate_config  # noqa: E402


def _shadow_report(max_abs: float = 1e-7, mean_abs: float = 1e-8) -> dict[str, object]:
    return {
        "schema_version": 1,
        "training_path_enabled": False,
        "after_optimizer": {
            "compared": True,
            "parity_ok_loose": True,
            "max_abs_param_diff": max_abs,
            "mean_abs_param_diff": mean_abs,
        },
    }


def _shadow_report_with_copyback(*, ok: bool = True, mutated: bool = False) -> dict[str, object]:
    report = _shadow_report()
    report["copyback_probe"] = {
        "scratch_copyback_validated": bool(ok),
        "real_parameters_mutated": bool(mutated),
        "elapsed_ms": 1.25,
    }
    return report


def _shadow_report_with_copyback_dispatch(*, ok: bool = True, restored: bool = True) -> dict[str, object]:
    report = _shadow_report_with_copyback(ok=True)
    report["copyback_dispatch_probe"] = {
        "copyback_dispatch_enabled": True,
        "copyback_dispatch_validated": bool(ok),
        "copyback_dispatch_target": "training_parameters",
        "real_parameters_mutated": True,
        "real_parameters_restored": bool(restored),
        "elapsed_ms": 0.5,
    }
    return report


def _shadow_report_with_native_binding() -> dict[str, object]:
    report = _shadow_report_with_copyback(ok=True)
    report["native_binding_probe"] = {
        "request_shape_ready": True,
        "tensor_object_binding_ready": True,
        "launch_plan_ready": True,
        "stream_lifetime_bound": False,
        "stream_lease_id": 7,
        "stream_guard_present": True,
        "stream_guard_ready": False,
        "stream_identity_ready": True,
        "stream_guard_level": "identity_verified_sync_blocked",
        "stream_handle_kind": "external_cuda_stream_handle",
        "stream_handle_reported": True,
        "stream_handle_nonzero": True,
        "synchronization_guard_ready": False,
        "synchronization_strategy": "borrowed_cuda_stream_event_chain_required",
        "event_chain_contract": "turbocore_stream_event_chain_guard_v2",
        "event_chain_state": "not_attempted",
        "event_chain_probe_requested": False,
        "event_chain_probe_attempted": False,
        "event_chain_verified": False,
        "pre_launch_ordering_verified": False,
        "post_launch_ordering_verified": False,
        "stream_wait_event_verified": False,
        "native_launch_candidate": False,
        "borrowed_external_stream": True,
        "stream_device_match": True,
        "stream_contract": {
            "contract": "turbocore_tensor_binding_stream_lifetime_v0",
            "stream_kind": "default",
            "stream_lifetime_bound": False,
        },
        "elapsed_ms": 2.5,
    }
    return report


def _shadow_report_with_native_binding_and_owner_event_chain() -> dict[str, object]:
    report = _shadow_report_with_native_binding()
    report["owner_native_launch_probe"] = {
        "ok": True,
        "attempted": True,
        "kernel_executed": True,
        "parity_ok": True,
        "persistent_owner_mutated": False,
        "max_abs_diff": 2.0e-8,
        "max_rel_diff": 1.0e-7,
        "elapsed_ms": 0.75,
        "event_chain_probe_requested": True,
        "event_chain_probe_attempted": True,
        "event_chain_verified": True,
        "pre_launch_ordering_verified": True,
        "post_launch_ordering_verified": True,
        "stream_wait_event_verified": True,
        "training_dispatch": False,
        "training_path_enabled": False,
    }
    return report


def _shadow_report_with_owner_native_launch(*, ok: bool = True, event_chain: bool = False) -> dict[str, object]:
    report = _shadow_report_with_copyback_dispatch(ok=True, restored=True)
    report["owner_native_launch_probe"] = {
        "ok": bool(ok),
        "attempted": True,
        "kernel_executed": bool(ok),
        "parity_ok": bool(ok),
        "persistent_owner_mutated": False,
        "max_abs_diff": 2.0e-8 if ok else 1.0,
        "max_rel_diff": 1.0e-7 if ok else 1.0,
        "elapsed_ms": 0.75,
        "event_chain_probe_requested": bool(event_chain),
        "event_chain_probe_attempted": bool(event_chain),
        "event_chain_verified": bool(event_chain and ok),
        "pre_launch_ordering_verified": bool(event_chain and ok),
        "post_launch_ordering_verified": bool(event_chain and ok),
        "stream_wait_event_verified": bool(event_chain and ok),
        "training_dispatch": False,
        "training_path_enabled": False,
    }
    return report


def _promotion_performance_report() -> dict[str, object]:
    return {
        "optimizer_performance_gate": {
            "gate": "turbocore_optimizer_performance_gate",
            "ok": True,
            "promotion_gate_ok": True,
            "runtime_dispatch_allowed": False,
            "evidence_quality": "promotion_benchmark",
            "best_candidate": {
                "optimizer": "turbocore_cuda_adamw_v0",
                "speedup_vs_baseline": 1.35,
            },
        },
        "benchmark_matrix": {
            "matrix": "turbocore_update_benchmark_matrix_v0",
            "run": True,
            "summary": {
                "executed_count": 2,
                "all_success": True,
                "mean_step_ms_by_case": {
                    "baseline_phase": 1000.0,
                    "native_update_dispatch": 950.0,
                },
            },
            "cases": [
                {"case": {"name": "baseline_phase"}, "summary": {"steps_completed": 24, "mean_step_ms": 1000.0}},
                {
                    "case": {"name": "native_update_dispatch"},
                    "summary": {
                        "steps_completed": 24,
                        "mean_step_ms": 950.0,
                        "native_dispatch_executed": True,
                    },
                },
            ],
        },
    }


def _promotion_shadow_report() -> dict[str, object]:
    report = _shadow_report_with_native_binding_and_owner_event_chain()
    report["copyback_dispatch_probe"] = {
        "copyback_dispatch_enabled": True,
        "copyback_dispatch_validated": True,
        "copyback_dispatch_target": "training_parameters",
        "real_parameters_mutated": True,
        "real_parameters_restored": True,
        "elapsed_ms": 0.5,
    }
    report["native_binding_probe"].update(  # type: ignore[index,union-attr]
        {
            "stream_lifetime_bound": True,
            "stream_guard_ready": True,
            "event_chain_verified": True,
            "pre_launch_ordering_verified": True,
            "post_launch_ordering_verified": True,
            "stream_wait_event_verified": True,
        }
    )
    report["performance_report"] = _promotion_performance_report()
    return report


def _promotion_readiness_ok() -> dict[str, object]:
    return {
        "schema_version": 1,
        "ok": True,
        "training_path_enabled": False,
        "native_kernel_present": True,
        "training_dispatch_kernel_present": True,
        "performance_test_ready": True,
        "stream_lifetime_bound": True,
        "stream_lifetime_ownership_bound": True,
        "stream_ordering_verified": True,
        "event_chain_verified": True,
        "owner_checks": {
            "direct_gradient_write_boundary_ready": True,
            "direct_gradient_write_native_supported": True,
            "direct_gradient_write_training_integrated": True,
            "owner_gradient_sync_boundary_ready": True,
            "owner_gradient_sync_supported": True,
            "owner_gradient_sync_training_integrated": True,
        },
        "native_checks": {
            "flat_owner_contract_ready": True,
            "reference_flat_owner_ready": True,
            "training_flat_owner_promoted": True,
            "training_dispatch_kernel_contract_ready": True,
            "training_dispatch_kernel_present": True,
        },
        "blocked_reasons": [],
    }


def _promotion_runtime_context() -> dict[str, object]:
    return {
        "multi_gpu": False,
        "num_processes": 1,
        "num_machines": 1,
        "deepspeed": False,
        "gradient_release_active": False,
        "native_update_training_dispatch_enabled": True,
        "training_path_enabled": True,
        "native_update_runtime_dispatch_available": True,
        "native_update_executor_present": True,
        "native_update_runtime_execution_guard_enabled": True,
        "native_update_training_mutation_guard_enabled": True,
        "native_update_owner_gradient_sync_guard_enabled": True,
        "native_update_owner_gradient_sync_bound": True,
        "native_update_direct_gradient_write_guard_enabled": True,
        "native_update_direct_gradient_write_bound": True,
        "native_update_flat_owner_training_guard_enabled": True,
        "native_update_flat_owner_bound": True,
        "native_update_training_dispatch_kernel_guard_enabled": True,
        "native_update_training_dispatch_kernel_bound": True,
        "native_update_stream_lifetime_ownership_guard_enabled": True,
        "native_update_stream_lifetime_ownership_bound": True,
    }


def _readiness_ok() -> dict[str, object]:
    return {
        "schema_version": 1,
        "ok": True,
        "training_path_enabled": False,
        "native_kernel_present": False,
        "performance_test_ready": False,
        "stream_lifetime_bound": False,
        "blocked_reasons": [],
    }


def _assert_dispatch_preflight_hard_blocked(report: dict[str, object]) -> dict[str, object]:
    preflight = report["dispatch_preflight"]
    assert isinstance(preflight, dict), report
    reasons = set(preflight["blocked_reasons"])
    runtime_recovery = preflight["evidence"]["runtime_recovery"]
    performance = preflight["evidence"]["performance"]
    assert preflight["preflight"] == "turbocore_native_update_dispatch_preflight_v0", preflight
    assert preflight["dispatch_preflight_passed"] is False, preflight
    assert preflight["would_allow_native_dispatch"] is False, preflight
    assert preflight["training_dispatch"] is False, preflight
    assert preflight["training_path_enabled"] is False, preflight
    assert runtime_recovery["policy_defined"] is True, preflight
    assert runtime_recovery["runtime_recovery_ready"] is False, preflight
    assert runtime_recovery["dispatch_integration_ready"] is True, preflight
    assert runtime_recovery["default_off_recovery_bridge_ready"] is True, preflight
    assert runtime_recovery["recovery_observation_bridge_ready"] is True, preflight
    assert runtime_recovery["training_dispatch_recovery_ready"] is False, preflight
    assert runtime_recovery["training_dispatch_recovery_blocked"] is True, preflight
    assert performance["policy_defined"] is True, preflight
    assert performance["performance_test_ready"] is False, preflight
    assert "native_dispatch_runtime_not_implemented" in reasons, preflight
    assert "native_dispatch_training_path_disabled" in reasons, preflight
    assert "native_runtime_recovery_dispatch_not_integrated" not in runtime_recovery["blocked_reasons"], preflight
    assert "native_runtime_recovery_training_dispatch_disabled" in runtime_recovery["blocked_reasons"], preflight
    assert "representative_training_matrix_missing" in performance["blocked_reasons"], preflight
    return preflight


def _assert_dispatch_contract_hard_blocked(report: dict[str, object]) -> dict[str, object]:
    contract = report["dispatch_contract"]
    assert isinstance(contract, dict), report
    reasons = set(contract["blocked_reasons"])
    rehearsal = contract["rehearsal"]
    assert contract["contract"] == "turbocore_native_update_dispatch_contract_v0", contract
    assert contract["training_dispatch"] is False, contract
    assert contract["training_path_enabled"] is False, contract
    assert contract["dispatch_rehearsal_ready"] is False, contract
    assert contract["would_allow_native_dispatch"] is False, contract
    assert contract["pytorch_optimizer_authoritative"] is True, contract
    assert contract["native_mutation_allowed"] is False, contract
    assert contract["training_parameter_mutation_allowed"] is False, contract
    assert contract["scheduler_stays_python_side"] is True, contract
    assert rehearsal["stage"] == "report_only_rehearsal", contract
    assert rehearsal["would_step_scheduler_after_successful_dispatch"] is False, contract
    assert "native_dispatch_runtime_not_implemented" in reasons, contract
    assert "native_dispatch_training_path_disabled" in reasons, contract
    assert "implement_native_update_dispatch_runtime" in contract["actions_required"], contract
    assert "add_explicit_default_off_training_dispatch_flag" in contract["actions_required"], contract
    assert all(item["enabled"] is False for item in contract["dispatch_sequence"]), contract
    return contract


def _assert_dispatch_request_hard_blocked(report: dict[str, object]) -> dict[str, object]:
    request = report["dispatch_request"]
    assert isinstance(request, dict), report
    reasons = set(request["blocked_reasons"])
    assert request["request"] == "turbocore_native_update_dispatch_request_v0", request
    assert request["training_dispatch"] is False, request
    assert request["training_path_enabled"] is False, request
    assert request["dispatch_allowed"] is False, request
    assert request["runtime_dispatch_available"] is False, request
    assert request["pytorch_optimizer_authoritative"] is True, request
    assert request["plan"]["execute_native_step"] is False, request
    assert request["plan"]["call_pytorch_optimizer_step"] is True, request
    assert "native_dispatch_runtime_not_implemented" in reasons, request
    return request


def _assert_kernel_launch_plan_hard_blocked(report: dict[str, object]) -> dict[str, object]:
    plan = report["kernel_launch_plan"]
    assert isinstance(plan, dict), report
    reasons = set(plan["blocked_reasons"])
    assert plan["launcher"] == "turbocore_native_update_kernel_launcher_v0", plan
    assert plan["kernel"] == "adamw_flat_fp32_cuda_kernel_v0", plan
    assert plan["training_dispatch"] is False, plan
    assert plan["training_path_enabled"] is False, plan
    assert plan["launch_allowed"] is False, plan
    assert plan["launch_attempted"] is False, plan
    assert plan["kernel_executed"] is False, plan
    assert plan["mutates_training_parameters"] is False, plan
    assert "native_dispatch_runtime_not_implemented" in reasons, plan
    assert "native_dispatch_training_path_disabled" in reasons, plan
    assert all(item["enabled"] is False for item in plan["sequence"]), plan
    return plan


def test_gate_requires_shadow_warmup_and_kernel_promotion() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config("native_experimental", required_shadow_passes=2)
    )

    first = gate.update(
        shadow_report=_shadow_report(),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert first["would_enable_native_update"] is False, first
    assert "shadow_warmup_not_satisfied" in first["blocked_reasons"], first
    assert "native_kernel_promotion_not_enabled" in first["blocked_reasons"], first
    assert first["training_path_enabled"] is False, first
    assert first["native_kernel_present"] is False, first

    second = gate.update(
        shadow_report=_shadow_report(),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert second["consecutive_shadow_passes"] == 2, second
    assert second["would_enable_native_update"] is False, second
    assert second["blocked_reasons"] == ["native_kernel_promotion_not_enabled"], second
    _assert_dispatch_preflight_hard_blocked(second)
    _assert_dispatch_contract_hard_blocked(second)
    request = _assert_dispatch_request_hard_blocked(second)
    assert request["dispatch_enabled"] is False, request
    assert request["requested"] is False, request
    _assert_kernel_launch_plan_hard_blocked(second)


def test_gate_dev_escape_hatch_is_still_report_only() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report(),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert report["would_enable_native_update"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["native_kernel_present"] is False, report
    assert report["performance_test_ready"] is False, report
    preflight = _assert_dispatch_preflight_hard_blocked(report)
    assert preflight["requested"] is True, preflight
    contract = _assert_dispatch_contract_hard_blocked(report)
    assert contract["requested"] is True, contract
    request = _assert_dispatch_request_hard_blocked(report)
    assert request["requested"] is False, request
    _assert_kernel_launch_plan_hard_blocked(report)


def test_gate_dispatch_request_flag_is_still_blocked() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
            dispatch_enabled=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_owner_native_launch(ok=True),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    request = _assert_dispatch_request_hard_blocked(report)
    reasons = set(request["blocked_reasons"])
    assert request["dispatch_enabled"] is True, request
    assert request["requested"] is True, request
    assert request["evidence"]["gate_would_enable_native_update"] is True, request
    assert "native_dispatch_contract_not_allowing_dispatch" in reasons, request
    assert "native_dispatch_training_path_disabled" in reasons, request
    plan = _assert_kernel_launch_plan_hard_blocked(report)
    assert plan["requested"] is True, plan


def test_gate_surfaces_copyback_probe_status() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_copyback(ok=True),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert report["shadow"]["copyback_probe_present"] is True, report
    assert report["shadow"]["copyback_scratch_validated"] is True, report
    assert report["shadow"]["copyback_real_parameters_mutated"] is False, report
    assert report["fallback_policy"]["copyback_scratch_validated"] is True, report
    assert report["fallback_policy"]["copyback_dispatch_probe_present"] is False, report
    assert "keep_pytorch_optimizer_due_to_copyback_dispatch_disabled" in report["fallback_policy"]["actions"], report


def test_gate_surfaces_copyback_dispatch_probe_status() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_copyback_dispatch(ok=True, restored=True),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert report["shadow"]["copyback_dispatch_probe_present"] is True, report
    assert report["shadow"]["copyback_dispatch_enabled"] is True, report
    assert report["shadow"]["copyback_dispatch_validated"] is True, report
    assert report["shadow"]["copyback_dispatch_target"] == "training_parameters", report
    assert report["shadow"]["copyback_dispatch_real_parameters_mutated"] is True, report
    assert report["shadow"]["copyback_dispatch_real_parameters_restored"] is True, report
    assert report["fallback_policy"]["copyback_dispatch_validated"] is True, report
    assert "keep_pytorch_optimizer_due_to_copyback_dispatch_disabled" not in report["fallback_policy"]["actions"], report


def test_gate_surfaces_native_binding_probe_status() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_native_binding(),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert report["shadow"]["native_binding_probe_present"] is True, report
    assert report["shadow"]["native_binding_request_shape_ready"] is True, report
    assert report["shadow"]["native_binding_launch_plan_ready"] is True, report
    assert report["shadow"]["native_binding_stream_lifetime_bound"] is False, report
    assert report["shadow"]["native_binding_stream_contract_present"] is True, report
    assert report["shadow"]["native_binding_stream_kind"] == "default", report
    assert report["shadow"]["native_binding_stream_lease_id"] == 7, report
    assert report["shadow"]["native_binding_stream_guard_present"] is True, report
    assert report["shadow"]["native_binding_stream_guard_ready"] is False, report
    assert report["shadow"]["native_binding_stream_identity_ready"] is True, report
    assert report["shadow"]["native_binding_stream_guard_level"] == "identity_verified_sync_blocked", report
    assert report["shadow"]["native_binding_stream_handle_kind"] == "external_cuda_stream_handle", report
    assert report["shadow"]["native_binding_stream_handle_reported"] is True, report
    assert report["shadow"]["native_binding_stream_handle_nonzero"] is True, report
    assert report["shadow"]["native_binding_synchronization_guard_ready"] is False, report
    assert report["shadow"]["native_binding_synchronization_strategy"] == "borrowed_cuda_stream_event_chain_required", report
    assert report["shadow"]["native_binding_event_chain_contract"] == "turbocore_stream_event_chain_guard_v2", report
    assert report["shadow"]["native_binding_event_chain_state"] == "not_attempted", report
    assert report["shadow"]["native_binding_event_chain_probe_requested"] is False, report
    assert report["shadow"]["native_binding_event_chain_probe_attempted"] is False, report
    assert report["shadow"]["native_binding_event_chain_verified"] is False, report
    assert report["shadow"]["native_binding_pre_launch_ordering_verified"] is False, report
    assert report["shadow"]["native_binding_post_launch_ordering_verified"] is False, report
    assert report["shadow"]["native_binding_stream_wait_event_verified"] is False, report
    assert report["shadow"]["native_binding_native_launch_candidate"] is False, report
    assert report["shadow"]["native_binding_borrowed_external_stream"] is True, report
    assert report["shadow"]["native_binding_stream_device_match"] is True, report
    assert report["fallback_policy"]["native_binding_probe_present"] is True, report
    assert report["fallback_policy"]["native_binding_stream_contract_present"] is True, report
    assert "keep_pytorch_optimizer_due_to_unbound_stream_lifetime" in report["fallback_policy"]["actions"], report
    assert "keep_pytorch_optimizer_due_to_stream_guard_not_ready" in report["fallback_policy"]["actions"], report
    assert "keep_pytorch_optimizer_due_to_event_chain_not_verified" in report["fallback_policy"]["actions"], report
    assert "keep_pytorch_optimizer_due_to_stream_identity_not_ready" not in report["fallback_policy"]["actions"], report


def test_gate_refines_stream_lifetime_when_owner_event_chain_is_verified() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_native_binding_and_owner_event_chain(),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    preflight = _assert_dispatch_preflight_hard_blocked(report)
    native_binding = preflight["evidence"]["native_binding"]
    contract = _assert_dispatch_contract_hard_blocked(report)
    missing = set(
        report["dispatch_request"]
        .get("training_path_request", {})
        .get("blocked_reasons", [])
    )
    assert preflight["stream_lifetime_bound"] is True, preflight
    assert preflight["stream_lifetime_ownership_bound"] is False, preflight
    assert preflight["stream_ordering_verified"] is True, preflight
    assert preflight["event_chain_verified"] is True, preflight
    assert native_binding["stream_lifetime_bound"] is True, native_binding
    assert native_binding["stream_lifetime_ownership_bound"] is False, native_binding
    assert native_binding["stream_ordering_verified"] is True, native_binding
    assert native_binding["owner_native_event_chain_verified"] is True, native_binding
    assert "stream_lifetime_unbound" not in preflight["blocked_reasons"], preflight
    assert "stream_lifetime_ownership_not_promoted" in preflight["blocked_reasons"], preflight
    assert "event_chain_not_verified" not in preflight["blocked_reasons"], preflight
    assert contract["stream_lifetime_bound"] is True, contract
    assert contract["stream_lifetime_ownership_bound"] is False, contract
    assert contract["stream_ordering_verified"] is True, contract
    assert contract["evidence"]["stream_ordering_verified"] is True, contract
    assert contract["evidence"]["stream_lifetime_ownership_bound"] is False, contract
    stream_lifetime = contract["stream_lifetime_ownership"]
    assert stream_lifetime["ownership_boundary_ready"] is True, contract
    assert stream_lifetime["ordering_verified"] is True, contract
    assert stream_lifetime["bound_to_training_path"] is False, contract
    assert stream_lifetime["default_off"] is True, contract
    assert "stream_lifetime_ownership_default_off" in stream_lifetime["blocked_reasons"], contract
    assert "keep_pytorch_optimizer_due_to_stream_lifetime_ownership_not_promoted" in report["fallback_policy"]["actions"], report
    assert "keep_pytorch_optimizer_due_to_unbound_stream_lifetime" not in report["fallback_policy"]["actions"], report
    assert "native_dispatch_training_path_default_off" in missing, report


def test_gate_surfaces_owner_native_launch_probe_status() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_owner_native_launch(ok=True),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert report["shadow"]["owner_native_launch_probe_present"] is True, report
    assert report["shadow"]["owner_native_launch_attempted"] is True, report
    assert report["shadow"]["owner_native_launch_ok"] is True, report
    assert report["shadow"]["owner_native_launch_kernel_executed"] is True, report
    assert report["shadow"]["owner_native_launch_parity_ok"] is True, report
    assert report["shadow"]["owner_native_launch_persistent_owner_mutated"] is False, report
    assert report["shadow"]["owner_native_launch_event_chain_requested"] is False, report
    assert report["fallback_policy"]["owner_native_launch_probe_present"] is True, report
    assert report["fallback_policy"]["owner_native_launch_ok"] is True, report
    assert report["fallback_policy"]["runtime_recovery"]["policy_defined"] is True, report
    assert report["fallback_policy"]["runtime_recovery"]["runtime_recovery_ready"] is False, report
    assert "owner_native_launch_probe_failed" not in report["blocked_reasons"], report
    assert "keep_pytorch_optimizer_due_to_owner_native_launch_probe" not in report["fallback_policy"]["actions"], report
    preflight = _assert_dispatch_preflight_hard_blocked(report)
    owner_native = preflight["evidence"]["owner_native_kernel"]
    assert owner_native["kernel_executed"] is True, preflight
    assert owner_native["parity_ok"] is True, preflight
    assert owner_native["event_chain_probe_requested"] is False, preflight
    contract = _assert_dispatch_contract_hard_blocked(report)
    assert contract["native_kernel_present"] is True, contract
    assert contract["rehearsal"]["would_use_owner_buffers"] is True, contract
    assert contract["rehearsal"]["would_launch_native_kernel"] is True, contract
    assert contract["rehearsal"]["would_copyback_to_training_parameters"] is True, contract
    assert contract["rehearsal"]["would_zero_owner_grad"] is True, contract
    _assert_dispatch_request_hard_blocked(report)
    plan = _assert_kernel_launch_plan_hard_blocked(report)
    assert plan["evidence"]["diagnostic_kernel_executed"] is True, plan
    assert plan["evidence"]["diagnostic_parity_ok"] is True, plan


def test_gate_surfaces_owner_native_event_chain_status() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_owner_native_launch(ok=True, event_chain=True),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert report["shadow"]["owner_native_launch_event_chain_requested"] is True, report
    assert report["shadow"]["owner_native_launch_event_chain_attempted"] is True, report
    assert report["shadow"]["owner_native_launch_event_chain_verified"] is True, report
    preflight = _assert_dispatch_preflight_hard_blocked(report)
    owner_native = preflight["evidence"]["owner_native_kernel"]
    assert owner_native["event_chain_probe_requested"] is True, preflight
    assert owner_native["event_chain_verified"] is True, preflight
    contract = _assert_dispatch_contract_hard_blocked(report)
    assert contract["evidence"]["event_chain_verified"] is True, contract


def test_gate_blocks_failed_copyback_probe() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_copyback(ok=False, mutated=True),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    reasons = set(report["blocked_reasons"])
    assert "copyback_scratch_validation_failed" in reasons, report
    assert "copyback_probe_mutated_training_parameters" in reasons, report
    assert "keep_pytorch_optimizer_due_to_copyback_validation" in report["fallback_policy"]["actions"], report
    assert "disable_native_update_due_to_copyback_mutation" in report["fallback_policy"]["actions"], report


def test_gate_blocks_failed_copyback_dispatch_probe() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_copyback_dispatch(ok=False, restored=False),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    reasons = set(report["blocked_reasons"])
    assert "copyback_dispatch_validation_failed" in reasons, report
    assert "copyback_dispatch_left_training_parameters_mutated" in reasons, report
    assert "keep_pytorch_optimizer_due_to_copyback_dispatch_validation" in report["fallback_policy"]["actions"], report
    assert "disable_native_update_due_to_unrestored_copyback_dispatch_mutation" in report["fallback_policy"]["actions"], report


def test_gate_blocks_failed_owner_native_launch_probe() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report_with_owner_native_launch(ok=False),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report=_readiness_ok(),
    )
    assert "owner_native_launch_probe_failed" in set(report["blocked_reasons"]), report
    assert "keep_pytorch_optimizer_due_to_owner_native_launch_probe" in report["fallback_policy"]["actions"], report
    recovery = report["fallback_policy"]["runtime_recovery"]
    assert recovery["runtime"]["runtime_error_observed"] is True, report
    assert recovery["disable_native_update_for_run"] is True, report
    assert "disable_native_update_for_run_on_runtime_error" in recovery["actions"], report


def test_gate_blocks_unsupported_cases() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(build_native_update_gate_config("profile", required_shadow_passes=1))
    report = gate.update(
        shadow_report=_shadow_report(max_abs=1.0, mean_abs=1.0),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={"num_processes": 2, "deepspeed": True, "gradient_release_active": True},
    )
    reasons = set(report["blocked_reasons"])
    assert "optimizer_not_adamw" in reasons, report
    assert "distributed_not_supported" in reasons, report
    assert "deepspeed_not_supported" in reasons, report
    assert "gradient_release_not_supported" in reasons, report
    assert "shadow_max_abs_diff_too_high" in reasons, report
    assert "shadow_mean_abs_diff_too_high" in reasons, report
    assert report["would_enable_native_update"] is False, report


def test_gate_includes_readiness_blockers() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
        )
    )
    report = gate.update(
        shadow_report=_shadow_report(),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context={},
        readiness_report={
            "ok": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "performance_test_ready": False,
            "stream_lifetime_bound": False,
            "blocked_reasons": ["stream_lifetime_unbound"],
        },
    )
    assert report["would_enable_native_update"] is False, report
    assert "stream_lifetime_unbound" in report["blocked_reasons"], report
    assert report["readiness"]["present"] is True, report
    preflight = _assert_dispatch_preflight_hard_blocked(report)
    assert "stream_lifetime_unbound" in preflight["blocked_reasons"], preflight
    contract = _assert_dispatch_contract_hard_blocked(report)
    assert "stream_lifetime_unbound" in contract["blocked_reasons"], contract


def test_gate_can_allow_dispatch_when_full_training_context_is_ready() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            "native_experimental",
            required_shadow_passes=1,
            allow_missing_native_kernel=True,
            dispatch_enabled=True,
        )
    )
    report = gate.update(
        shadow_report=_promotion_shadow_report(),
        optimizer=optimizer,
        trainable_param_count=1,
        runtime_context=_promotion_runtime_context(),
        readiness_report=_promotion_readiness_ok(),
    )
    preflight = report["dispatch_preflight"]
    contract = report["dispatch_contract"]
    request = report["dispatch_request"]
    launch = report["kernel_launch_plan"]
    assert report["would_enable_native_update"] is True, report
    assert report["blocked_reasons"] == [], report
    assert preflight["dispatch_preflight_passed"] is True, report
    assert preflight["would_allow_native_dispatch"] is True, report
    assert preflight["blocked_reasons"] == [], report
    assert contract["dispatch_rehearsal_ready"] is True, report
    assert contract["would_allow_native_dispatch"] is True, report
    assert contract["pytorch_optimizer_authoritative"] is False, report
    assert contract["blocked_reasons"] == [], report
    assert request["dispatch_allowed"] is True, report
    assert request["plan"]["execute_native_step"] is True, report
    assert request["plan"]["call_pytorch_optimizer_step"] is False, report
    assert launch["launch_allowed"] is True, report
    assert launch["mutates_training_parameters"] is True, report
    assert launch["blocked_reasons"] == [], report


def main() -> int:
    test_gate_requires_shadow_warmup_and_kernel_promotion()
    test_gate_dev_escape_hatch_is_still_report_only()
    test_gate_dispatch_request_flag_is_still_blocked()
    test_gate_surfaces_copyback_probe_status()
    test_gate_surfaces_copyback_dispatch_probe_status()
    test_gate_surfaces_native_binding_probe_status()
    test_gate_refines_stream_lifetime_when_owner_event_chain_is_verified()
    test_gate_surfaces_owner_native_launch_probe_status()
    test_gate_surfaces_owner_native_event_chain_status()
    test_gate_blocks_failed_copyback_probe()
    test_gate_blocks_failed_copyback_dispatch_probe()
    test_gate_blocks_failed_owner_native_launch_probe()
    test_gate_blocks_unsupported_cases()
    test_gate_includes_readiness_blockers()
    test_gate_can_allow_dispatch_when_full_training_context_is_ready()
    print("turbocore_update_gate_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
