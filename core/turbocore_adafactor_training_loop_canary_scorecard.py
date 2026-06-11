"""TrainingLoop canary for Adafactor native dispatch."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch
from transformers.optimization import Adafactor

from core.lulynx_trainer.training_loop import TrainingLoop


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_adafactor_training_loop_canary_scorecard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_adafactor_training_loop_canary")
    case = _run_case()
    ok = bool(case.get("ok", False))
    blockers = list(case.get("blocked_reasons", []) or [])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adafactor_training_loop_canary_scorecard_v0",
        "gate": "adafactor_training_loop_native_canary",
        "ok": ok,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "adafactor",
        "case": case,
        "summary": {
            "native_step_count": 1 if case.get("native_step_executed") is True else 0,
            "native_kernel_launch_count": 1 if case.get("native_kernel_launched") is True else 0,
            "primed_pytorch_state": bool(case.get("primed_pytorch_state", False)),
            "step_after_native": case.get("step_after_native"),
            "factored": case.get("factored"),
        },
        "promotion_blockers": blockers + ["product_rollout_review_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Adafactor e2e shadow matrix and rollout policy"
            if ok
            else "fix Adafactor TrainingLoop canary blockers"
        ),
    }


def _run_case() -> dict[str, Any]:
    loop, param, optimizer = _make_loop()
    _prime_optimizer_state(param, optimizer)
    _seed_previous_gate(loop)
    captured: list[dict[str, Any]] = []

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict[str, Any],
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ) -> Any:
        params = self._get_trainable_params()
        for item in params:
            item.grad = None
        loss = sum(((item.float() * 0.55) ** 2).mean() + item.float().mean() * 0.011 for item in params)
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    if not captured:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "adafactor_training_loop_native_canary_v0",
            "result": result,
            "captured_step_count": 0,
            "blocked_reasons": ["adafactor_training_loop_did_not_emit_step"],
        }
    runtime = _runtime_payload(captured[0])
    training_executor = _executor_payload(runtime)
    executor_result = _as_dict(training_executor.get("result")) if isinstance(training_executor, dict) else {}
    first_case = _first_executor_case(executor_result)
    native_step = bool(runtime.get("native_step_executed", False))
    native_kernel = bool(runtime.get("native_kernel_launched", False))
    state = optimizer.state[param]
    step_after = int(state.get("step", 0) or 0)
    factored = bool("exp_avg_sq_row" in state and "exp_avg_sq_col" in state)
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and not bool(runtime.get("should_call_pytorch_optimizer_step", True))
        and step_after == 2
        and factored
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "adafactor_training_loop_native_canary_v0",
        "result": result,
        "captured_step_count": len(captured),
        "primed_pytorch_state": True,
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": bool(runtime.get("should_call_pytorch_optimizer_step", True)),
        "fallback_to_pytorch_required": bool(runtime.get("fallback_to_pytorch_required", True)),
        "training_executor_called": bool(training_executor.get("called", False)) if isinstance(training_executor, dict) else False,
        "training_executor_ok": bool(training_executor.get("ok", False)) if isinstance(training_executor, dict) else False,
        "step_after_native": step_after,
        "factored": factored,
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "executor_case": first_case,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else _dedupe(list(runtime.get("blocked_reasons", []) or []) + ["adafactor_training_loop_native_step_missing"]),
    }


def _make_loop() -> tuple[TrainingLoop, torch.nn.Parameter, torch.optim.Optimizer]:
    device = torch.device("cuda")
    param = torch.nn.Parameter(torch.linspace(-0.08, 0.08, steps=128 * 128, device=device, dtype=torch.float32).reshape(128, 128))
    optimizer = Adafactor(
        [param],
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.01,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=_Injector([param]),
        optimizer=optimizer,
        lr_scheduler=None,
        device="cuda",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=0.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_native_update_mode="native_experimental",
        turbocore_native_update_required_shadow_passes=1,
        turbocore_native_update_allow_missing_kernel=True,
        turbocore_native_update_dispatch_enabled=True,
        turbocore_native_update_training_path_enabled=True,
        turbocore_native_update_require_native_cuda=True,
        turbocore_native_update_quantized_optimizer_kind="adafactor",
    )
    loop.total_steps = 1
    return loop, param, optimizer


def _prime_optimizer_state(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> None:
    param.grad = None
    loss = ((param.float() * 0.31) ** 2).mean() + param.float().mean() * 0.007
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _seed_previous_gate(loop: TrainingLoop) -> None:
    loop._turbocore_native_update_dispatch_armer._last_gate_report = _explicit_gate()


def _explicit_gate() -> dict[str, Any]:
    request = {
        "requested": True,
        "dispatch_allowed": True,
        "training_path_enabled": True,
        "training_path_request": {"request_boundary_ready": True, "explicit_training_path_requested": True},
    }
    contract = {
        "dispatch_rehearsal_ready": True,
        "would_allow_native_dispatch": True,
        "rehearsal": {"would_launch_native_kernel": True},
        "recovery": {"default_off_recovery_bridge_ready": True, "training_dispatch_recovery_ready": True},
        "owner_gradient_sync": _ready_contract("sync_boundary_ready", "owner_gradient_sync_preconditions_ready"),
        "training_flat_owner": _ready_contract("owner_boundary_ready", "training_flat_owner_preconditions_ready"),
        "training_dispatch_kernel": _ready_contract("kernel_boundary_ready", "training_dispatch_kernel_preconditions_ready"),
        "training_executor": {"executor_boundary_ready": True, "training_executor_preconditions_ready": True},
        "stream_lifetime_ownership": {
            "ownership_boundary_ready": True,
            "stream_lifetime_ownership_preconditions_ready": True,
        },
        "evidence": {
            "owner_native_launch_ok": True,
            "copyback_dispatch_validated": True,
            "event_chain_verified": True,
            "stream_ordering_verified": True,
            "representative_performance_gate_ready": True,
        },
        "blocked_reasons": [],
    }
    return {
        "dispatch_request": request,
        "dispatch_contract": contract,
        "kernel_launch_plan": {"launch_allowed": True, "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True}},
    }


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {boundary: True, precondition: True, "native_supported": True, "training_lifecycle_integrated": True}


def _runtime_payload(step_info: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(step_info.get("turbocore_native_update_dispatch_runtime"))


def _executor_payload(runtime: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(runtime.get("training_executor"))


def _first_executor_case(executor_result: dict[str, Any]) -> dict[str, Any]:
    cases = executor_result.get("cases")
    if isinstance(cases, list) and cases:
        return _as_dict(cases[0])
    return {}


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adafactor_training_loop_canary_scorecard_v0",
        "gate": "adafactor_training_loop_native_canary",
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "adafactor",
        "case": {},
        "summary": {"native_step_count": 0, "native_kernel_launch_count": 0, "primed_pytorch_state": False},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run Adafactor TrainingLoop canary on CUDA",
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_adafactor_training_loop_canary_scorecard"]
