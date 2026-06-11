"""TrainingLoop canary for fp32 PagedAdamW native dispatch."""

from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import patch

import torch

from core.configs import OptimizerType, UnifiedTrainingConfig
from core.lulynx_trainer.training_loop import TrainingLoop
from core.lulynx_trainer.trainer import LulynxTrainer


TARGETS = (
    (OptimizerType.PAGED_ADAMW, "paged_adamw"),
    (OptimizerType.PAGED_ADAMW_32BIT, "paged_adamw32bit"),
)


class _Injector:
    def __init__(self, value: torch.Tensor) -> None:
        self.param = torch.nn.Parameter(value.detach().clone())
        self.injected_layers = {}

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [self.param]


def build_paged_adamw32_training_loop_canary_scorecard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_paged_adamw32_training_loop_canary")
    if importlib.util.find_spec("bitsandbytes") is None:
        return _blocked("bitsandbytes_required_for_paged_adamw32_training_loop_canary")
    cases = [_run_case(optimizer, kind) for optimizer, kind in TARGETS]
    ok = all(bool(case.get("ok", False)) for case in cases)
    blockers = _dedupe([reason for case in cases for reason in case.get("blocked_reasons", [])])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw32_training_loop_canary_scorecard_v0",
        "gate": "paged_adamw32_training_loop_native_canary",
        "ok": ok,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kinds": [kind for _, kind in TARGETS],
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "native_step_count": sum(1 for case in cases if case.get("native_step_executed") is True),
            "native_kernel_launch_count": sum(1 for case in cases if case.get("native_kernel_launched") is True),
            "primed_pytorch_state_count": sum(1 for case in cases if case.get("primed_pytorch_state") is True),
        },
        "promotion_blockers": blockers + ["product_rollout_review_missing"],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "measure PagedAdamW/PagedAdamW32bit product canary overhead and decide rollout scope"
            if ok
            else "fix fp32 PagedAdamW TrainingLoop canary blockers"
        ),
    }


def _run_case(optimizer_type: OptimizerType, optimizer_kind: str) -> dict[str, Any]:
    loop, param, optimizer = _make_loop(optimizer_type, optimizer_kind)
    optimizer_name = type(optimizer).__name__
    if optimizer_name == "AdamW":
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "paged_adamw32_training_loop_native_canary_v0",
            "optimizer_type": optimizer_type.value,
            "optimizer_kind": optimizer_kind,
            "optimizer_class": optimizer_name,
            "blocked_reasons": [f"{optimizer_kind}_resolved_to_fallback_adamw"],
        }
    _prime_optimizer_state(param, optimizer)
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
        loss = sum(((item.float() * 0.43) ** 2).mean() + item.float().mean() * 0.007 for item in params)
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        _seed_previous_gate(self)
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    if not captured:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "paged_adamw32_training_loop_native_canary_v0",
            "optimizer_type": optimizer_type.value,
            "optimizer_kind": optimizer_kind,
            "result": result,
            "captured_step_count": 0,
            "blocked_reasons": [f"{optimizer_kind}_training_loop_did_not_emit_step"],
        }
    runtime = _runtime_payload(captured[0])
    training_executor = _executor_payload(runtime)
    executor_result = training_executor.get("result", {}) if isinstance(training_executor, dict) else {}
    state = optimizer.state[param]
    native_step = bool(runtime.get("native_step_executed", False))
    native_kernel = bool(runtime.get("native_kernel_launched", False))
    step_after = _step_to_int(state.get("step"))
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and not bool(runtime.get("should_call_pytorch_optimizer_step", True))
        and step_after == 2
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "paged_adamw32_training_loop_native_canary_v0",
        "optimizer_type": optimizer_type.value,
        "optimizer_kind": optimizer_kind,
        "optimizer_class": optimizer_name,
        "result": result,
        "captured_step_count": len(captured),
        "primed_pytorch_state": True,
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": bool(runtime.get("should_call_pytorch_optimizer_step", True)),
        "fallback_to_pytorch_required": bool(runtime.get("fallback_to_pytorch_required", True)),
        "training_executor_called": bool(training_executor.get("called", False)) if isinstance(training_executor, dict) else False,
        "training_executor_ok": bool(training_executor.get("ok", False)) if isinstance(training_executor, dict) else False,
        "executor_result_ok": bool(executor_result.get("ok", False)) if isinstance(executor_result, dict) else False,
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "step_after_native": step_after,
        "state1_dtype": str(state.get("state1").dtype).replace("torch.", "") if torch.is_tensor(state.get("state1")) else "",
        "state2_dtype": str(state.get("state2").dtype).replace("torch.", "") if torch.is_tensor(state.get("state2")) else "",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else _dedupe(list(runtime.get("blocked_reasons", []) or []) + [f"{optimizer_kind}_training_loop_native_step_missing"]),
    }


def _make_loop(
    optimizer_type: OptimizerType,
    optimizer_kind: str,
) -> tuple[TrainingLoop, torch.nn.Parameter, torch.optim.Optimizer]:
    values = torch.linspace(-1.0, 1.0, steps=4096, device="cuda", dtype=torch.float32)
    trainer = _make_trainer(values, optimizer_type)
    param = trainer.lora_injector.param
    optimizer = trainer._create_optimizer()
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=trainer.lora_injector,
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
        turbocore_native_update_quantized_optimizer_kind=optimizer_kind,
    )
    loop.total_steps = 1
    return loop, param, optimizer


def _make_trainer(value: torch.Tensor, optimizer_type: OptimizerType) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = UnifiedTrainingConfig(
        optimizer_type=optimizer_type,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args="",
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector(value)
    trainer.model = None
    trainer.trainable_params = []
    trainer._block_weight_manager = None
    trainer._easy_control = None
    trainer._ip_adapter = None
    trainer._repa_projector = None
    trainer._advanced_optimizer_strategy_profile = {}
    trainer._optimizer_backend_profile = {}
    trainer._log_messages = []
    trainer._log = lambda msg: trainer._log_messages.append(str(msg))
    trainer._attach_optimizer_profiles_to_training_loop = lambda: None
    return trainer


def _prime_optimizer_state(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> None:
    param.grad = None
    loss = ((param.float() * 0.33) ** 2).mean() + param.float().mean() * 0.009
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
        "training_path_request": {
            "request_boundary_ready": True,
            "explicit_training_path_requested": True,
        },
    }
    contract = {
        "dispatch_rehearsal_ready": True,
        "would_allow_native_dispatch": True,
        "rehearsal": {"would_launch_native_kernel": True},
        "recovery": {
            "default_off_recovery_bridge_ready": True,
            "training_dispatch_recovery_ready": True,
        },
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
        "kernel_launch_plan": {
            "launch_allowed": True,
            "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True},
        },
    }


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {
        boundary: True,
        precondition: True,
        "native_supported": True,
        "training_lifecycle_integrated": True,
    }


def _runtime_payload(step_info: dict[str, Any]) -> dict[str, Any]:
    value = step_info.get("turbocore_native_update_dispatch_runtime", {})
    return dict(value) if isinstance(value, dict) else {}


def _executor_payload(runtime: dict[str, Any]) -> dict[str, Any]:
    value = runtime.get("training_executor", {})
    return dict(value) if isinstance(value, dict) else {}


def _step_to_int(value: Any) -> int:
    if torch.is_tensor(value) and value.numel() > 0:
        return int(value.detach().reshape(-1)[0].cpu().item())
    return int(value or 0)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw32_training_loop_canary_scorecard_v0",
        "gate": "paged_adamw32_training_loop_native_canary",
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kinds": [kind for _, kind in TARGETS],
        "cases": [],
        "summary": {
            "case_count": 0,
            "native_step_count": 0,
            "native_kernel_launch_count": 0,
            "primed_pytorch_state_count": 0,
        },
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run fp32 PagedAdamW TrainingLoop canary on CUDA with bitsandbytes",
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["TARGETS", "build_paged_adamw32_training_loop_canary_scorecard"]
