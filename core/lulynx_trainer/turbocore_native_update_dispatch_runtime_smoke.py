"""Smoke checks for TurboCore native update dispatch runtime facade."""

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

from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime  # noqa: E402
from core.turbocore_native_update_dispatch_diagnostic_executor import (  # noqa: E402
    build_shadow_owner_native_diagnostic_executor,
)
from core.turbocore_native_update_recovery import build_native_update_runtime_recovery_policy  # noqa: E402
from core.turbocore_native_update_training_executor import build_native_update_training_executor  # noqa: E402


def test_runtime_without_arming_keeps_pytorch_step() -> None:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    report = runtime.prepare_step(step=0)
    reasons = set(report["blocked_reasons"])
    assert report["runtime"] == "turbocore_native_update_dispatch_runtime_v0", report
    assert report["native_step_executed"] is False, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert report["should_call_python_scheduler"] is True, report
    assert report["fallback_to_pytorch_required"] is True, report
    assert "dispatch_arming_report_missing" in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report


def test_runtime_explicit_request_is_still_not_executed() -> None:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    report = runtime.prepare_step(
        step=1,
        arming_report={"previous_request_requested": True, "execute_native_step": False},
        kernel_launch_plan={"launch_allowed": False},
    )
    reasons = set(report["blocked_reasons"])
    assert report["requested"] is True, report
    assert report["native_step_executed"] is False, report
    assert report["native_kernel_launched"] is False, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert report["state"]["native_dispatch_attempts"] == 1, report
    plan = report["execution_plan"]
    assert plan["execution_allowed"] is False, report
    assert plan["should_call_pytorch_optimizer_step"] is True, report
    assert "dispatch_not_armed" in reasons, report
    assert "kernel_launch_not_allowed" in reasons, report
    assert "native_step_execution_disabled" in reasons, report
    assert "native_dispatch_training_runtime_executor_default_off" in reasons, report
    assert "native_dispatch_training_path_default_off" in reasons, report
    assert "native_dispatch_runtime_executor_missing" not in reasons, report
    assert "native_dispatch_training_mutation_guard_disabled" not in reasons, report


def test_runtime_execution_plan_reports_default_off_executor_boundary() -> None:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    report = runtime.prepare_step(
        step=4,
        arming_report={
            "previous_request_requested": True,
            "armed_for_native_dispatch": True,
            "execute_native_step": True,
        },
        kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
        runtime_context={
            "native_update_executor_present": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
            "training_path_enabled": True,
        },
        native_executor=lambda _request: {"ok": True},
    )
    plan = report["execution_plan"]
    reasons = set(report["blocked_reasons"])
    probe = report["executor_probe"]
    assert plan["executor_preconditions_ready"] is True, report
    assert plan["training_executor_preconditions_ready"] is True, report
    assert plan["diagnostic_executor_preconditions_ready"] is False, report
    assert "native_dispatch_diagnostic_clone_context_disabled" in plan["diagnostic_executor_blocked_reasons"], report
    assert plan["diagnostic_executor_probe_allowed"] is False, report
    assert plan["execution_allowed"] is False, report
    assert plan["would_call_native_executor"] is False, report
    assert probe["called"] is False, report
    assert report["native_step_executed"] is False, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert "native_dispatch_diagnostic_executor_call_disabled" in reasons, report
    assert "native_dispatch_runtime_default_off" in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report
    assert "native_dispatch_training_path_disabled" in reasons, report
    assert "native_dispatch_training_runtime_executor_default_off" in reasons, report


def test_runtime_diagnostic_executor_probe_can_be_called_without_training_dispatch() -> None:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    calls: list[dict] = []

    def _executor(request: dict) -> dict:
        calls.append(request)
        return {
            "ok": True,
            "reason": "diagnostic_probe_ok",
            "native_kernel_launched": True,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_step_executed": False,
            "training_parameters_mutated": False,
            "should_call_pytorch_optimizer_step": True,
        }

    report = runtime.prepare_step(
        step=5,
        arming_report={
            "previous_request_requested": True,
            "armed_for_native_dispatch": True,
            "execute_native_step": True,
        },
        kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
        runtime_context={
            "native_update_executor_present": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
            "native_update_diagnostic_executor_call_enabled": True,
            "native_update_diagnostic_clone_context_enabled": True,
            "training_path_enabled": False,
        },
        native_executor=_executor,
    )
    probe = report["executor_probe"]
    plan = report["execution_plan"]
    reasons = set(report["blocked_reasons"])
    assert len(calls) == 1, report
    assert plan["training_executor_preconditions_ready"] is False, report
    assert plan["diagnostic_executor_preconditions_ready"] is True, report
    assert probe["called"] is True, report
    assert probe["ok"] is True, report
    assert probe["native_kernel_launched"] is True, report
    assert probe["native_step_executed"] is False, report
    assert probe["should_call_pytorch_optimizer_step"] is True, report
    assert report["native_step_executed"] is False, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert "native_dispatch_runtime_default_off" in reasons, report
    assert "native_dispatch_runtime_not_implemented" in reasons, report
    assert "native_dispatch_training_path_default_off" in reasons, report


def test_runtime_blocks_diagnostic_executor_that_claims_training_mutation() -> None:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    report = runtime.prepare_step(
        step=6,
        arming_report={
            "previous_request_requested": True,
            "armed_for_native_dispatch": True,
            "execute_native_step": True,
        },
        kernel_launch_plan={"launch_allowed": True},
        runtime_context={
            "native_update_executor_present": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
            "native_update_diagnostic_executor_call_enabled": True,
            "native_update_diagnostic_clone_context_enabled": True,
            "training_path_enabled": False,
        },
        native_executor=lambda _request: {
            "ok": True,
            "native_step_executed": True,
            "training_parameters_mutated": True,
            "should_call_pytorch_optimizer_step": False,
        },
    )
    probe = report["executor_probe"]
    reasons = set(report["blocked_reasons"])
    assert probe["called"] is True, report
    assert probe["ok"] is False, report
    assert "native_dispatch_executor_reported_training_mutation" in reasons, report
    assert "native_dispatch_executor_attempted_to_skip_pytorch_optimizer" in reasons, report


def test_runtime_can_replay_shadow_owner_native_diagnostic_executor() -> None:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    executor = build_shadow_owner_native_diagnostic_executor(
        {
            "owner_native_launch_probe": {
                "ok": True,
                "kernel_executed": True,
                "parity_ok": True,
                "event_chain_verified": True,
                "persistent_owner_mutated": False,
            }
        }
    )
    report = runtime.prepare_step(
        step=7,
        arming_report={
            "previous_request_requested": True,
            "armed_for_native_dispatch": True,
            "execute_native_step": True,
        },
        kernel_launch_plan={"launch_allowed": True},
        runtime_context={
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_diagnostic_executor_call_enabled": True,
            "native_update_diagnostic_clone_context_enabled": True,
            "training_path_enabled": False,
        },
        native_executor=executor,
    )
    probe = report["executor_probe"]
    result = probe["result"]
    assert report["training_dispatch"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_step_executed"] is False, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert report["execution_plan"]["diagnostic_executor_preconditions_ready"] is True, report
    assert report["execution_plan"]["training_executor_preconditions_ready"] is False, report
    assert probe["called"] is True, report
    assert probe["ok"] is True, report
    assert result["diagnostic_replay"] is True, report
    assert result["shadow_owner_native_kernel_evidence"] is True, report
    assert result["native_step_executed"] is False, report


def test_runtime_can_execute_explicit_training_executor() -> None:
    torch.manual_seed(1701)
    ref_params = [
        torch.nn.Parameter(torch.randn(2, 3, dtype=torch.float32) * 0.1),
        torch.nn.Parameter(torch.randn(4, dtype=torch.float32) * 0.1),
    ]
    params = [torch.nn.Parameter(param.detach().clone()) for param in ref_params]
    optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    ref = torch.optim.AdamW(ref_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    for left, right in zip(ref_params, params):
        grad = torch.randn_like(left)
        left.grad = grad.detach().clone()
        right.grad = grad.detach().clone()
    ref.step()
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    executor = build_native_update_training_executor(
        optimizer=optimizer,
        params=params,
        config={"lr": 1e-3, "weight_decay": 0.01, "max_grad_norm": 0.0, "prefer_triton": False},
    )
    try:
        report = runtime.prepare_step(
            step=8,
            arming_report={
                "previous_request_requested": True,
                "armed_for_native_dispatch": True,
                "execute_native_step": True,
            },
            kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
            runtime_context={
                "native_update_executor_present": True,
                "native_update_runtime_execution_guard_enabled": True,
                "native_update_training_mutation_guard_enabled": True,
                "native_update_training_dispatch_enabled": True,
                "native_update_runtime_dispatch_available": True,
                "training_path_enabled": True,
            },
            native_executor=executor,
        )
    finally:
        executor.close()
    assert report["training_dispatch"] is True, report
    assert report["training_path_enabled"] is True, report
    assert report["native_step_executed"] is True, report
    assert report["should_call_pytorch_optimizer_step"] is False, report
    assert report["fallback_to_pytorch_required"] is False, report
    assert report["state"]["native_steps_executed"] == 1, report
    assert report["training_executor"]["called"] is True, report
    assert report["training_executor"]["ok"] is True, report
    assert report["training_executor"]["training_parameters_mutated"] is True, report
    assert not report["blocked_reasons"], report
    for actual, expected in zip(params, ref_params):
        diff = float((actual.detach() - expected.detach()).abs().max().item())
        assert diff <= 3e-7, {"diff": diff, "report": report}
    for param in params:
        state = optimizer.state[param]
        assert "exp_avg" in state and "exp_avg_sq" in state, state
        assert int(state["step"].detach().cpu().item()) == 1, state


def test_runtime_falls_back_when_required_native_cuda_is_unavailable() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    param.grad = torch.tensor([0.1, -0.2], dtype=torch.float32)
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    executor = build_native_update_training_executor(
        optimizer=optimizer,
        params=[param],
        config={
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 0.0,
            "prefer_native_cuda": True,
            "require_native_cuda": True,
            "prefer_triton": False,
        },
    )
    try:
        report = runtime.prepare_step(
            step=9,
            arming_report={
                "previous_request_requested": True,
                "armed_for_native_dispatch": True,
                "execute_native_step": True,
            },
            kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
            runtime_context={
                "native_update_executor_present": True,
                "native_update_runtime_execution_guard_enabled": True,
                "native_update_training_mutation_guard_enabled": True,
                "native_update_training_dispatch_enabled": True,
                "native_update_runtime_dispatch_available": True,
                "training_path_enabled": True,
            },
            native_executor=executor,
        )
    finally:
        executor.close()
    assert report["native_step_executed"] is False, report
    assert report["should_call_pytorch_optimizer_step"] is True, report
    assert report["fallback_to_pytorch_required"] is True, report
    assert "native_update_training_executor_error" in report["blocked_reasons"], report


def test_runtime_recovery_policy_disables_for_run() -> None:
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    observation = runtime.observe_recovery_policy(
        {
            "disable_native_update_for_run": True,
            "blocked_reasons": ["native_runtime_error_observed"],
        }
    )
    first = runtime.prepare_step(
        step=2,
        arming_report={"previous_request_requested": True, "execute_native_step": False},
        kernel_launch_plan={"launch_allowed": False},
    )
    second = runtime.prepare_step(
        step=3,
        arming_report={"previous_request_requested": True, "execute_native_step": False},
        kernel_launch_plan={"launch_allowed": False},
    )
    assert observation["disabled_for_run"] is True, observation
    assert observation["disable_reason"] == "native_runtime_error_observed", observation
    assert first["state"]["disabled_for_run"] is True, first
    assert first["state"]["disable_reason"] == "native_runtime_error_observed", first
    assert "native_dispatch_disabled_for_run" in first["blocked_reasons"], first
    assert second["state"]["disabled_for_run"] is True, second
    assert "native_dispatch_disabled_for_run" in second["blocked_reasons"], second


def test_recovery_policy_does_not_latch_shadow_autostop_skip() -> None:
    policy = build_native_update_runtime_recovery_policy(
        mode="native_experimental",
        shadow_report={
            "reason": "auto_stopped_after_consecutive_passes",
            "after_optimizer": {
                "skipped": True,
                "compared": False,
                "reason": "auto_stopped_after_consecutive_passes",
            },
        },
        runtime_context={
            "training_path_enabled": True,
            "native_update_training_dispatch_enabled": True,
            "native_update_runtime_dispatch_available": True,
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_training_mutation_guard_enabled": True,
        },
    )
    state = policy["state_safety"]
    assert state["shadow_auto_stopped_after_consecutive_passes"] is True, policy
    assert state["shadow_parity_known"] is False, policy
    assert state["state_mismatch_observed"] is False, policy
    assert policy["disable_native_update_for_run"] is False, policy
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    observation = runtime.observe_recovery_policy(policy)
    assert observation["disabled_for_run"] is False, observation
    assert observation["disable_reason"] == "", observation


def main() -> int:
    test_runtime_without_arming_keeps_pytorch_step()
    test_runtime_explicit_request_is_still_not_executed()
    test_runtime_execution_plan_reports_default_off_executor_boundary()
    test_runtime_diagnostic_executor_probe_can_be_called_without_training_dispatch()
    test_runtime_blocks_diagnostic_executor_that_claims_training_mutation()
    test_runtime_can_replay_shadow_owner_native_diagnostic_executor()
    test_runtime_can_execute_explicit_training_executor()
    test_runtime_falls_back_when_required_native_cuda_is_unavailable()
    test_runtime_recovery_policy_disables_for_run()
    test_recovery_policy_does_not_latch_shadow_autostop_skip()
    print("turbocore_native_update_dispatch_runtime_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
