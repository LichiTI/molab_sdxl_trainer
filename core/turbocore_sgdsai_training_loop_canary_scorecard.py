"""TrainingLoop canary for SGDSaI native state-adapter dispatch."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch
from pytorch_optimizer import SGDSaI

from core.lulynx_trainer.training_loop import TrainingLoop


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_sgdsai_training_loop_canary_scorecard() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked("cuda_required_for_sgdsai_training_loop_canary")
    case = _run_case()
    ok = case.get("ok") is True
    blockers = list(case.get("blocked_reasons", []) or [])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_sgdsai_training_loop_canary_scorecard_v0",
        "gate": "sgdsai_state_adapter_special_training_loop_canary",
        "ok": ok,
        "promotion_ready": False,
        "training_loop_canary_ready": ok,
        "training_loop_canary_hit": ok,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "optimizer_family": "plugin_state_adapter_special",
        "case": case,
        "summary": {
            "optimizer_count": 1,
            "training_loop_canary_ready_count": 1 if ok else 0,
            "native_step_count": 1 if case.get("native_step_executed") is True else 0,
            "native_kernel_launch_count": 1 if case.get("native_kernel_launched") is True else 0,
            "training_executor_called_count": 1 if case.get("training_executor_called") is True else 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "sgdsai_e2e_shadow_matrix_missing",
            "sgdsai_owner_release_approval_missing",
            "state_adapter_special_family_runtime_launch_incomplete",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "expand SGDSaI state-adapter shadow matrix with dispatch still default-off"
            if ok
            else "fix SGDSaI TrainingLoop canary blockers"
        ),
    }


def _run_case() -> dict[str, Any]:
    loop, param, optimizer = _make_loop()
    _prime_warmup(param, optimizer)
    _seed_previous_gate(loop)
    captured: list[dict[str, Any]] = []

    def _fake_train_step(
        self: TrainingLoop,
        _batch: dict[str, Any],
        accumulation_steps: int = 1,
        return_loss_tensor: bool = False,
    ) -> Any:
        for item in self._get_trainable_params():
            item.grad = None
        loss = ((param.float() * 0.37) ** 2).mean() + param.float().mean() * 0.009
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    before = param.detach().clone()
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    after = param.detach().clone()
    if not captured:
        return _case_blocked("sgdsai_training_loop_did_not_emit_step")
    runtime = _as_dict(captured[0].get("turbocore_native_update_dispatch_runtime"))
    training_executor = _as_dict(runtime.get("training_executor"))
    executor_result = _as_dict(training_executor.get("result"))
    state = optimizer.state[param]
    step_after = int(optimizer.param_groups[0].get("step", 0) or 0)
    gsnr_ready = "gsnr" in state
    momentum_ready = "momentum_buffer" in state
    native_step = runtime.get("native_step_executed") is True
    native_kernel = runtime.get("native_kernel_launched") is True
    mutated = _max_abs_diff(before, after) > 0.0
    ok = bool(
        result.get("steps") == 1
        and native_step
        and native_kernel
        and mutated
        and runtime.get("should_call_pytorch_optimizer_step") is False
        and executor_result.get("optimizer_kind") == "sgdsai"
        and step_after == 2
        and gsnr_ready
        and momentum_ready
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "probe": "sgdsai_training_loop_native_canary_v0",
        "optimizer_kind": "sgdsai",
        "result": result,
        "captured_step_count": len(captured),
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "training_parameters_mutated": mutated,
        "should_call_pytorch_optimizer_step": runtime.get("should_call_pytorch_optimizer_step") is True,
        "training_executor_called": training_executor.get("called") is True,
        "training_executor_ok": training_executor.get("ok") is True,
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "step_after_native": step_after,
        "has_warmup": bool(getattr(optimizer, "has_warmup", False)),
        "gsnr_ready": gsnr_ready,
        "momentum_ready": momentum_ready,
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "blocked_reasons": [] if ok else _dedupe(
            _strings(runtime.get("blocked_reasons")) + ["sgdsai_training_loop_native_step_missing"]
        ),
    }


def _make_loop() -> tuple[TrainingLoop, torch.nn.Parameter, torch.optim.Optimizer]:
    param = torch.nn.Parameter(
        torch.linspace(-0.12, 0.16, steps=64, device="cuda", dtype=torch.float32).view(8, 8).contiguous()
    )
    optimizer = SGDSaI([param], lr=1.0e-2, momentum=0.9, weight_decay=0.01, weight_decouple=True)
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
        turbocore_native_update_quantized_optimizer_kind="sgdsai",
    )
    loop.total_steps = 1
    return loop, param, optimizer


def _prime_warmup(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> None:
    param.grad = None
    loss = ((param.float() * 0.21) ** 2).mean() + param.float().mean() * 0.005
    loss.backward()
    optimizer.warmup_step()
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


def _case_blocked(reason: str) -> dict[str, Any]:
    return {"schema_version": 1, "ok": False, "probe": "sgdsai_training_loop_native_canary_v0", "blocked_reasons": [reason]}


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_sgdsai_training_loop_canary_scorecard_v0",
        "gate": "sgdsai_state_adapter_special_training_loop_canary",
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "case": {},
        "summary": {"native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "run SGDSaI TrainingLoop canary on CUDA",
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().item())


__all__ = ["build_sgdsai_training_loop_canary_scorecard"]
