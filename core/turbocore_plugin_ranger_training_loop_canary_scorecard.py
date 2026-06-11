"""TrainingLoop canary for selected plugin Ranger native dispatch."""

from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import patch

import torch

from core.configs import SchedulerType
from core.lulynx_trainer.config import LulynxConfig, OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import plugin_resume_case
from core.lulynx_trainer.trainer import LulynxTrainer
from core.lulynx_trainer.training_loop import TrainingLoop


SELECTED_OPTIMIZER = "ranger"
NATIVE_ROUTE = "rust_cuda_plugin_ranger_v0"


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_plugin_ranger_training_loop_canary_scorecard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_plugin_ranger_training_loop_canary")
    case = _run_case()
    ok = bool(case.get("ok", False))
    blockers = _strings(case.get("blocked_reasons"))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_ranger_training_loop_canary_scorecard_v0",
        "gate": "plugin_ranger_training_loop_native_canary",
        "ok": ok,
        "promotion_ready": False,
        "selected_native_canary_ready": ok,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "selector_type": OptimizerType.PYTORCH_OPTIMIZER.value,
        "selected_optimizer_name": SELECTED_OPTIMIZER,
        "optimizer_family": "adam_like_formula",
        "native_route": NATIVE_ROUTE,
        "case": case,
        "summary": {
            "native_step_count": 1 if case.get("native_step_executed") is True else 0,
            "native_kernel_launch_count": 1 if case.get("native_kernel_launched") is True else 0,
            "primed_pytorch_state": bool(case.get("primed_pytorch_state", False)),
            "step_after_native": case.get("step_after_native"),
            "optimizer_class": case.get("optimizer_class"),
            "executor_optimizer_kind": case.get("executor_optimizer_kind"),
            "lookahead_k": case.get("lookahead_k"),
        },
        "promotion_blockers": blockers
        + [
            "ranger_e2e_shadow_matrix_refresh_required",
            "ranger_canary_rollout_policy_refresh_required",
            "product_rollout_review_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "refresh selected Adam-like e2e shadow matrix and rollout policy for ranger"
            if ok
            else "fix selected plugin Ranger TrainingLoop canary blockers"
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
        loss = sum(((item.float() * 0.25) ** 2).mean() + item.float().mean() * 0.004 for item in params)
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
            "probe": "plugin_ranger_training_loop_native_canary_v0",
            "result": result,
            "captured_step_count": 0,
            "blocked_reasons": ["plugin_ranger_training_loop_did_not_emit_step"],
        }
    runtime = _runtime_payload(captured[0])
    training_executor = _executor_payload(runtime)
    executor_result = _as_dict(training_executor.get("result")) if isinstance(training_executor, Mapping) else {}
    native_step = bool(runtime.get("native_step_executed", False))
    native_kernel = bool(runtime.get("native_kernel_launched", False))
    group = optimizer.param_groups[0]
    step_after = int(group.get("step", 0) or 0)
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and not bool(runtime.get("should_call_pytorch_optimizer_step", True))
        and step_after == 2
        and type(optimizer).__name__ == "Ranger"
        and str(executor_result.get("optimizer_kind") or "") == "ranger"
        and int(group.get("k", 0) or 0) == 2
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "plugin_ranger_training_loop_native_canary_v0",
        "result": result,
        "captured_step_count": len(captured),
        "primed_pytorch_state": True,
        "optimizer_class": type(optimizer).__name__,
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "should_call_pytorch_optimizer_step": bool(runtime.get("should_call_pytorch_optimizer_step", True)),
        "fallback_to_pytorch_required": bool(runtime.get("fallback_to_pytorch_required", True)),
        "training_executor_called": bool(training_executor.get("called", False))
        if isinstance(training_executor, Mapping)
        else False,
        "training_executor_ok": bool(training_executor.get("ok", False))
        if isinstance(training_executor, Mapping)
        else False,
        "executor_result_ok": bool(executor_result.get("ok", False)),
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "step_after_native": step_after,
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "lookahead_k": int(group.get("k", 0) or 0),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": []
        if ok
        else _dedupe(_strings(runtime.get("blocked_reasons")) + ["plugin_ranger_training_loop_native_step_missing"]),
    }


def _make_loop() -> tuple[TrainingLoop, torch.nn.Parameter, torch.optim.Optimizer]:
    device = torch.device("cuda")
    param = torch.nn.Parameter(torch.linspace(-0.11, 0.21, steps=4096, device=device, dtype=torch.float32))
    optimizer = _create_selected_ranger_plugin_optimizer(param)
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
        turbocore_native_update_quantized_optimizer_kind=SELECTED_OPTIMIZER,
    )
    loop.total_steps = 1
    return loop, param, optimizer


def _create_selected_ranger_plugin_optimizer(param: torch.nn.Parameter) -> torch.optim.Optimizer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    case = plugin_resume_case(SELECTED_OPTIMIZER)
    trainer.config = LulynxConfig(
        optimizer_type=OptimizerType.PYTORCH_OPTIMIZER,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_args=f"{case.optimizer_args},k=2,use_gc=False",
        lr_scheduler=SchedulerType.COSINE,
        warmup_ratio=0.0,
    )
    trainer.config.semantic_tuner_enabled = False
    trainer.lora_injector = _Injector([param])
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
    return trainer._create_optimizer()


def _prime_optimizer_state(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> None:
    param.grad = None
    loss = ((param.float() * 0.17) ** 2).mean() + param.float().mean() * 0.003
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
        "kernel_launch_plan": {
            "launch_allowed": True,
            "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True},
        },
    }


def _ready_contract(boundary: str, precondition: str) -> dict[str, Any]:
    return {boundary: True, precondition: True, "native_supported": True, "training_lifecycle_integrated": True}


def _runtime_payload(step_info: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(step_info.get("turbocore_native_update_dispatch_runtime"))


def _executor_payload(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(runtime.get("training_executor"))


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_ranger_training_loop_canary_scorecard_v0",
        "gate": "plugin_ranger_training_loop_native_canary",
        "ok": False,
        "promotion_ready": False,
        "selected_native_canary_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "selector_type": OptimizerType.PYTORCH_OPTIMIZER.value,
        "selected_optimizer_name": SELECTED_OPTIMIZER,
        "optimizer_family": "adam_like_formula",
        "native_route": NATIVE_ROUTE,
        "case": {},
        "summary": {"native_step_count": 0, "native_kernel_launch_count": 0, "primed_pytorch_state": False},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run selected plugin Ranger TrainingLoop canary on CUDA",
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_plugin_ranger_training_loop_canary_scorecard"]
