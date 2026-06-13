"""Bridge-created per-selected plugin TrainingLoop native canaries."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest.mock import patch

import torch

from core.lulynx_trainer.optimizer_plugin_bridge import create_pytorch_optimizer
from core.lulynx_trainer.training_loop import TrainingLoop


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_NAME = "turbocore_plugin_bridge_training_loop_canary_scorecard.json"


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


@dataclass(frozen=True)
class _Spec:
    name: str
    family: str
    shape: tuple[int, ...]
    lr: float
    weight_decay: float
    optimizer_args: Mapping[str, Any]
    prime: str
    validate: Callable[[dict[str, Any], torch.optim.Optimizer, torch.nn.Parameter], list[str]]
    param_shapes: tuple[tuple[int, ...], ...] | None = None


def build_plugin_bridge_training_loop_canary_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        report = _blocked("cuda_required_for_plugin_bridge_training_loop_canary")
    else:
        cases = [_run_case(spec) for spec in _SPECS]
        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        ready = len(cases) == len(_SPECS) and all(case.get("ok") is True for case in cases)
        native_step_count = sum(1 for case in cases if case.get("native_step_executed") is True)
        native_kernel_count = sum(1 for case in cases if case.get("native_kernel_launched") is True)
        executor_count = sum(1 for case in cases if case.get("training_executor_called") is True)
        skip_pytorch_count = sum(1 for case in cases if case.get("should_call_pytorch_optimizer_step") is False)
        report = {
            "schema_version": 1,
            "scorecard": "turbocore_plugin_bridge_training_loop_canary_scorecard_v0",
            "gate": "plugin_bridge_selected_training_loop_native_canary",
            "roadmap": ROADMAP,
            "ok": ready,
            "promotion_ready": False,
            "selected_native_canary_ready": ready,
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "product_native_ready": False,
            "cases": cases,
            "summary": {
                "selected_optimizer_count": len(cases),
                "case_count": len(cases),
                "native_step_count": native_step_count,
                "native_kernel_launch_count": native_kernel_count,
                "training_executor_called_count": executor_count,
                "skip_pytorch_count": skip_pytorch_count,
                "plugin_bridge_training_loop_canary_case_count": len(cases),
                "plugin_bridge_training_loop_canary_native_step_count": native_step_count,
                "plugin_bridge_training_loop_canary_native_kernel_launch_count": native_kernel_count,
                "plugin_bridge_training_loop_canary_training_executor_called_count": executor_count,
                "plugin_bridge_training_loop_canary_skip_pytorch_count": skip_pytorch_count,
                "runtime_dispatch_ready_count": 0,
                "native_dispatch_allowed_count": 0,
                "training_path_enabled_count": 0,
                "product_native_ready_count": 0,
            },
            "promotion_blockers": _dedupe(
                blockers
                + [
                    "plugin_bridge_owner_release_review_missing",
                    "plugin_bridge_product_training_route_not_bound",
                ]
            ),
            "blocked_reasons": blockers,
            "recommended_next_step": (
                "expand bridge-created per-selected canaries to the remaining runtime-precondition families"
                if ready
                else "fix bridge-created per-selected TrainingLoop native canary blockers"
            ),
            "notes": [
                "Each case creates the selected plugin optimizer through create_pytorch_optimizer.",
                "This is CUDA/native canary evidence only; product dispatch remains default-off.",
            ],
        }
    if write_artifact:
        _write_artifact(report)
    return report


def _run_case(spec: _Spec) -> dict[str, Any]:
    loop, params, optimizer = _make_loop(spec)
    param = params[0]
    _prime_optimizer_state(spec, params, optimizer)
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
        loss = sum(((item.float() * 0.37) ** 2).mean() + item.float().mean() * 0.009 for item in self._get_trainable_params())
        loss = loss / max(int(accumulation_steps or 1), 1)
        loss.backward()
        return loss.detach() if return_loss_tensor else float(loss.detach().item())

    loop.on_step_end = lambda _step, _loss, info: captured.append(dict(info))
    before = [item.detach().clone() for item in params]
    with patch.object(TrainingLoop, "train_step", new=_fake_train_step):
        result = loop.train_epoch([{}], 0)
    after = [item.detach().clone() for item in params]
    if not captured:
        return _case_blocked(spec, "plugin_bridge_training_loop_did_not_emit_step")
    runtime = _as_dict(captured[0].get("turbocore_native_update_dispatch_runtime"))
    training_executor = _as_dict(runtime.get("training_executor"))
    executor_result = _as_dict(training_executor.get("result"))
    first_case = _first_executor_case(executor_result)
    native_step = runtime.get("native_step_executed") is True
    native_kernel = runtime.get("native_kernel_launched") is True
    mutated = bool(before) and all(_max_abs_diff(left, right) > 0.0 for left, right in zip(before, after))
    base_blockers = []
    if result.get("steps") != 1:
        base_blockers.append(f"{spec.name}_training_loop_step_count_invalid")
    if not native_step:
        base_blockers.append(f"{spec.name}_native_step_missing")
    if not native_kernel:
        base_blockers.append(f"{spec.name}_native_kernel_missing")
    if runtime.get("should_call_pytorch_optimizer_step") is not False:
        base_blockers.append(f"{spec.name}_pytorch_step_not_skipped")
    if training_executor.get("called") is not True or training_executor.get("ok") is not True:
        base_blockers.append(f"{spec.name}_training_executor_not_ok")
    if executor_result.get("optimizer_kind") != spec.name:
        base_blockers.append(f"{spec.name}_executor_kind_mismatch")
    if not mutated:
        base_blockers.append(f"{spec.name}_parameters_not_mutated")
    blockers = _dedupe(_strings(runtime.get("blocked_reasons")) + base_blockers + spec.validate(executor_result, optimizer, param))
    ok = not blockers
    return {
        "schema_version": 1,
        "ok": ok,
        "selected_optimizer_name": spec.name,
        "selected_optimizer_family": spec.family,
        "optimizer_class": type(getattr(optimizer, "_base", optimizer)).__name__,
        "result": result,
        "captured_step_count": len(captured),
        "native_step_executed": native_step,
        "native_kernel_launched": native_kernel,
        "training_parameters_mutated": mutated,
        "should_call_pytorch_optimizer_step": runtime.get("should_call_pytorch_optimizer_step") is True,
        "fallback_to_pytorch_required": runtime.get("fallback_to_pytorch_required") is True,
        "training_executor_called": training_executor.get("called") is True,
        "training_executor_ok": training_executor.get("ok") is True,
        "executor_result_ok": executor_result.get("ok") is True,
        "executor_optimizer_kind": str(executor_result.get("optimizer_kind") or ""),
        "executor_case": first_case,
        "param_dtype": str(param.dtype).replace("torch.", ""),
        "param_shape": [int(dim) for dim in param.shape],
        "param_count": len(params),
        "param_shapes": [[int(dim) for dim in item.shape] for item in params],
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "source_scorecard": "turbocore_plugin_bridge_training_loop_canary_scorecard_v0",
        "blocked_reasons": blockers,
    }


def _make_loop(spec: _Spec) -> tuple[TrainingLoop, list[torch.nn.Parameter], torch.optim.Optimizer]:
    shapes = spec.param_shapes or (spec.shape,)
    params: list[torch.nn.Parameter] = []
    for index, shape in enumerate(shapes):
        count = 1
        for dim in shape:
            count *= int(dim)
        start = -0.12 + index * 0.07
        stop = 0.16 + index * 0.05
        params.append(
            torch.nn.Parameter(
                torch.linspace(start, stop, steps=count, device="cuda", dtype=torch.float32).reshape(*shape).contiguous()
            )
        )
    optimizer = create_pytorch_optimizer(
        params,
        optimizer_name=spec.name,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        optimizer_args={"name": spec.name, **dict(spec.optimizer_args)},
    )
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=_Injector(params),
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
        turbocore_native_update_quantized_optimizer_kind=spec.name,
    )
    loop.total_steps = 1
    return loop, params, optimizer


def _prime_optimizer_state(spec: _Spec, params: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer) -> None:
    if spec.prime == "none":
        return
    for param in params:
        param.grad = None
    loss = sum(((param.float() * 0.19) ** 2).mean() + param.float().mean() * 0.005 for param in params)
    loss.backward()
    if spec.prime == "warmup" and hasattr(optimizer, "warmup_step"):
        optimizer.warmup_step()  # type: ignore[attr-defined]
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _seed_previous_gate(loop: TrainingLoop) -> None:
    loop._turbocore_native_update_dispatch_armer._last_gate_report = _explicit_gate()


def _explicit_gate() -> dict[str, Any]:
    ready_contract = {"native_supported": True, "training_lifecycle_integrated": True}
    return {
        "dispatch_request": {
            "requested": True,
            "dispatch_allowed": True,
            "training_path_enabled": True,
            "training_path_request": {"request_boundary_ready": True, "explicit_training_path_requested": True},
        },
        "dispatch_contract": {
            "dispatch_rehearsal_ready": True,
            "would_allow_native_dispatch": True,
            "rehearsal": {"would_launch_native_kernel": True},
            "recovery": {"default_off_recovery_bridge_ready": True, "training_dispatch_recovery_ready": True},
            "owner_gradient_sync": {**ready_contract, "sync_boundary_ready": True, "owner_gradient_sync_preconditions_ready": True},
            "training_flat_owner": {**ready_contract, "owner_boundary_ready": True, "training_flat_owner_preconditions_ready": True},
            "training_dispatch_kernel": {**ready_contract, "kernel_boundary_ready": True, "training_dispatch_kernel_preconditions_ready": True},
            "training_executor": {"executor_boundary_ready": True, "training_executor_preconditions_ready": True},
            "stream_lifetime_ownership": {"ownership_boundary_ready": True, "stream_lifetime_ownership_preconditions_ready": True},
            "evidence": {
                "owner_native_launch_ok": True,
                "copyback_dispatch_validated": True,
                "event_chain_verified": True,
                "stream_ordering_verified": True,
                "representative_performance_gate_ready": True,
            },
            "blocked_reasons": [],
        },
        "kernel_launch_plan": {"launch_allowed": True, "evidence": {"diagnostic_kernel_executed": True, "diagnostic_parity_ok": True}},
    }


def _validate_muon(result: dict[str, Any], _optimizer: torch.optim.Optimizer, _param: torch.nn.Parameter) -> list[str]:
    first_case = _first_executor_case(result)
    return [] if int(first_case.get("step_after", 0) or 0) >= 1 else ["muon_step_after_native_invalid"]


def _validate_muon_family_adamw(
    result: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    first_case = _first_executor_case(result)
    name = str(result.get("optimizer_kind") or "muon_family")
    blockers = []
    if group.get("use_muon") is not False:
        blockers.append(f"{name}_non_muon_group_not_selected")
    if int(group.get("step", 0) or 0) != 1:
        blockers.append(f"{name}_step_after_native_invalid")
    if first_case.get("use_muon") is not False:
        blockers.append(f"{name}_executor_non_muon_case_missing")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"{name}_{key}_state_missing")
    return blockers


def _validate_adafactor(result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    first_case = _first_executor_case(result)
    step_after = state.get("step", optimizer.param_groups[0].get("step", 0) if optimizer.param_groups else 0)
    blockers = []
    if _step_int(step_after) != 2:
        blockers.append("adafactor_step_after_native_invalid")
    if "exp_avg_sq_row" not in state or "exp_avg_sq_col" not in state:
        blockers.append("adafactor_factored_state_missing")
    if first_case.get("factored") is not True:
        blockers.append("adafactor_executor_factored_state_missing")
    return blockers


def _step_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return 0
        return int(value.detach().cpu().item())
    return int(value or 0)


def _validate_sgdsai(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 2:
        blockers.append("sgdsai_step_after_native_invalid")
    if "gsnr" not in state:
        blockers.append("sgdsai_gsnr_state_missing")
    if "momentum_buffer" not in state:
        blockers.append("sgdsai_momentum_state_missing")
    return blockers


def _validate_sm3(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("sm3_step_after_native_invalid")
    if not torch.is_tensor(state.get("accumulator_0")):
        blockers.append("sm3_accumulator_state_missing")
    if not torch.is_tensor(state.get("momentum_buffer")):
        blockers.append("sm3_momentum_state_missing")
    return blockers


def _validate_spam(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("spam_step_after_native_invalid")
    if int(optimizer.state.get("total_step", 0) or 0) != 1:
        blockers.append("spam_total_step_after_native_invalid")
    if int(optimizer.state.get("current_step", 0) or 0) != 3:
        blockers.append("spam_current_step_after_native_invalid")
    mask = state.get("mask")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or not bool(mask.all().detach().cpu().item()):
        blockers.append("spam_dense_bool_mask_missing")
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if not torch.is_tensor(exp_avg) or tuple(exp_avg.shape) != (int(param.numel()),):
        blockers.append("spam_exp_avg_state_missing")
    if not torch.is_tensor(exp_avg_sq) or tuple(exp_avg_sq.shape) != (int(param.numel()),):
        blockers.append("spam_exp_avg_sq_state_missing")
    return blockers


def _validate_stablespam(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("stablespam_step_after_native_invalid")
    if int(getattr(optimizer, "total_step", 0) or 0) != 1:
        blockers.append("stablespam_total_step_after_native_invalid")
    if getattr(optimizer, "t_max", None) is not None:
        blockers.append("stablespam_warmup_unexpected_for_canary")
    if abs(float(optimizer.param_groups[0].get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("stablespam_weight_decay_unexpected_for_canary")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"stablespam_{key}_state_missing")
    for key in ("m_norm_t", "v_norm_t", "m_max_t"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != (1,):
            blockers.append(f"stablespam_{key}_state_missing")
    return blockers


def _validate_adadelta(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adadelta_step_after_native_invalid")
    if not torch.is_tensor(state.get("square_avg")):
        blockers.append("adadelta_square_avg_state_missing")
    if not torch.is_tensor(state.get("acc_delta")):
        blockers.append("adadelta_acc_delta_state_missing")
    return blockers


def _validate_ftrl(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("ftrl_step_after_native_invalid")
    if not torch.is_tensor(state.get("z")):
        blockers.append("ftrl_z_state_missing")
    if not torch.is_tensor(state.get("n")):
        blockers.append("ftrl_n_state_missing")
    return blockers


def _validate_diffgrad(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("diffgrad_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq", "previous_grad"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"diffgrad_{key}_state_missing")
    return blockers


def _validate_adabelief(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adabelief_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_var"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adabelief_{key}_state_missing")
    return blockers


def _validate_adabound(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adabound_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adabound_{key}_state_missing")
    return blockers


def _validate_laprop(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("laprop_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"laprop_{key}_state_missing")
    for key in ("exp_avg_lr_1", "exp_avg_lr_2"):
        if not isinstance(state.get(key), float):
            blockers.append(f"laprop_{key}_state_missing")
    return blockers


def _validate_adai(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adai_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq", "beta1_prod"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adai_{key}_state_missing")
    return blockers


def _validate_adopt(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 2:
        blockers.append("adopt_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adopt_{key}_state_missing")
    return blockers


def _validate_msvag(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("msvag_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq", "s"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"msvag_{key}_state_missing")
    return blockers


def _validate_ademamix(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("ademamix_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq", "exp_avg_slow"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"ademamix_{key}_state_missing")
    return blockers


def _validate_simplifiedademamix(
    _result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter
) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("simplifiedademamix_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"simplifiedademamix_{key}_state_missing")
    for key in ("num_sum", "den_sum"):
        if not isinstance(state.get(key), float):
            blockers.append(f"simplifiedademamix_{key}_state_missing")
    return blockers


def _validate_a2grad(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("a2grad_step_after_native_invalid")
    for key in ("avg_grad", "x_k"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"a2grad_{key}_state_missing")
    value = state.get("v_k")
    if not torch.is_tensor(value) or tuple(value.shape) != (1,):
        blockers.append("a2grad_v_k_state_missing")
    if not isinstance(state.get("alpha_k"), float):
        blockers.append("a2grad_alpha_k_state_missing")
    return blockers


def _validate_avagrad(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 2:
        blockers.append("avagrad_step_after_native_invalid")
    if group.get("gamma") is None:
        blockers.append("avagrad_gamma_state_missing")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"avagrad_{key}_state_missing")
    return blockers


def _validate_adanorm(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adanorm_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_var"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adanorm_{key}_state_missing")
    value = state.get("exp_grad_norm")
    if not torch.is_tensor(value) or tuple(value.shape) != (1,):
        blockers.append("adanorm_exp_grad_norm_state_missing")
    return blockers


def _validate_bcos(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("bcos_step_after_native_invalid")
    value = state.get("v")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("bcos_v_state_missing")
    return blockers


def _validate_adagc(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adagc_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adagc_{key}_state_missing")
    value = state.get("gamma")
    if not torch.is_tensor(value) or tuple(value.shape) != (1,):
        blockers.append("adagc_gamma_state_missing")
    return blockers


def _validate_adasmooth(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adasmooth_step_after_native_invalid")
    for key in ("prev_param", "s", "n", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adasmooth_{key}_state_missing")
    return blockers


def _validate_adashift(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 2:
        blockers.append("adashift_step_after_native_invalid")
    grad_queue = state.get("grad_queue")
    if not isinstance(grad_queue, deque) or len(grad_queue) != int(group.get("keep_num", 0) or 0):
        blockers.append("adashift_grad_queue_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adashift_{key}_state_missing")
    return blockers


def _validate_adammini(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("adammini_step_after_native_invalid")
    if str(group.get("name") or "") != "params.0":
        blockers.append("adammini_unexpected_canary_group_name")
    value = state.get("m")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("adammini_m_state_missing")
    value = state.get("v_mean")
    if not torch.is_tensor(value) or tuple(value.shape) != ():
        blockers.append("adammini_v_mean_state_missing")
    value = state.get("dimension")
    if not torch.is_tensor(value) or tuple(value.shape) != ():
        blockers.append("adammini_dimension_state_missing")
    if state.get("reduced") is not False:
        blockers.append("adammini_reduced_state_unexpected")
    return blockers


def _validate_adapnm(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adapnm_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq", "neg_exp_avg"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adapnm_{key}_state_missing")
    return blockers


def _validate_adan(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adan_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq", "exp_avg_diff", "previous_grad"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adan_{key}_state_missing")
    return blockers


def _validate_ano(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("ano_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"ano_{key}_state_missing")
    return blockers


def _validate_amos(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("amos_step_after_native_invalid")
    for key in ("exp_avg_sq", "decay"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != (1,):
            blockers.append(f"amos_{key}_state_missing")
    return blockers


def _validate_apollo(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("apollo_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"apollo_{key}_state_missing")
    return blockers


def _validate_galore(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("galore_step_after_native_invalid")
    if "rank" in group:
        blockers.append("galore_rank_projector_unexpected_for_canary")
    if "projector" in state:
        blockers.append("galore_projector_state_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("galore_weight_decay_unexpected_for_canary")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"galore_{key}_state_missing")
    return blockers


def _validate_fira(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("fira_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"fira_{key}_state_missing")
    return blockers


def _validate_focus(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("focus_step_after_native_invalid")
    for key in ("exp_avg", "pbar"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"focus_{key}_state_missing")
    return blockers


def _validate_alig(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("alig_step_after_native_invalid")
    if float(group.get("step_size", 0.0) or 0.0) <= 0.0:
        blockers.append("alig_step_size_missing")
    if state:
        blockers.append("alig_unexpected_state_for_no_momentum_canary")
    base = getattr(optimizer, "_base", optimizer)
    if getattr(base, "projection_fn", None) is not None:
        blockers.append("alig_projection_fn_unexpected_for_canary")
    return blockers


def _validate_adahessian(
    _result: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("adahessian_step_after_native_invalid")
    for key in ("exp_avg", "exp_hessian_diag_sq", "hessian"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adahessian_{key}_state_missing")
    if abs(float(group.get("hessian_power", 1.0) or 1.0) - 1.0) > 1.0e-12:
        blockers.append("adahessian_hessian_power_unexpected_for_canary")
    return blockers


def _validate_alice(result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    first_case = _first_executor_case(result)
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("alice_step_after_native_invalid")
    for key in ("U", "Q", "m", "v", "p", "phi"):
        if not torch.is_tensor(state.get(key)):
            blockers.append(f"alice_{key}_state_missing")
    if first_case.get("rank") != 2 or first_case.get("leading_basis") != 1:
        blockers.append("alice_rank_leading_basis_canary_invalid")
    if tuple(param.shape) != (2, 2):
        blockers.append("alice_param_shape_unexpected_for_canary")
    return blockers


def _validate_kron(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("kron_step_after_native_invalid")
    value = state.get("momentum_buffer")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("kron_momentum_buffer_missing")
    q = state.get("Q")
    if not isinstance(q, list) or len(q) != 1 or not torch.is_tensor(q[0]):
        blockers.append("kron_identity_q_state_missing")
    if not isinstance(state.get("expressions"), tuple):
        blockers.append("kron_expressions_state_missing")
    if int(getattr(optimizer, "prob_step", 0) or 0) != 1:
        blockers.append("kron_prob_step_after_native_invalid")
    return blockers


def _validate_lorarite(result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    group = optimizer.param_groups[0]
    params = list(group.get("params", []))
    first_case = _first_executor_case(result)
    blockers: list[str] = []
    if len(params) != 2:
        blockers.append("lorarite_param_pair_count_invalid")
        return blockers
    left, right = params
    state = optimizer.state[left]
    right_state = optimizer.state.get(right, {})
    if left is not param:
        blockers.append("lorarite_primary_param_not_left_factor")
    if tuple(left.shape) != (2, 3) or tuple(right.shape) != (4, 2):
        blockers.append("lorarite_pair_shape_unexpected_for_canary")
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("lorarite_group_step_after_native_invalid")
    if int(state.get("step", 0) or 0) != 1:
        blockers.append("lorarite_pair_state_step_after_native_invalid")
    if right_state:
        blockers.append("lorarite_right_factor_state_unexpected")
    for key in ("v_l", "v_r", "m_l", "m_r", "basis_l", "basis_r", "escape_l", "escape_r"):
        if not torch.is_tensor(state.get(key)):
            blockers.append(f"lorarite_{key}_state_missing")
    for key in ("rotate_inv_l", "rotate_inv_r", "update_l", "update_r", "projection_l", "projection_r"):
        if key in state:
            blockers.append(f"lorarite_{key}_temporary_state_leaked")
    if first_case.get("left_parameters_mutated") is not True or first_case.get("right_parameters_mutated") is not True:
        blockers.append("lorarite_pair_mutation_missing")
    return blockers


def _validate_shampoo(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("shampoo_step_after_native_invalid")
    for key in ("pre_cond_0", "inv_pre_cond_0", "momentum_buffer"):
        value = state.get(key)
        if not torch.is_tensor(value):
            blockers.append(f"shampoo_{key}_state_missing")
    if abs(float(group.get("momentum", 0.0) or 0.0)) > 0.0:
        blockers.append("shampoo_momentum_unexpected_for_canary")
    return blockers


def _validate_scalableshampoo(
    _result: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("scalableshampoo_step_after_native_invalid")
    value = state.get("momentum")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("scalableshampoo_momentum_state_missing")
    if state.get("pre_conditioner") is None:
        blockers.append("scalableshampoo_pre_conditioner_state_missing")
    if state.get("graft") is None:
        blockers.append("scalableshampoo_graft_state_missing")
    if bool(group.get("nesterov", False)):
        blockers.append("scalableshampoo_nesterov_unexpected_for_canary")
    return blockers


def _validate_soap(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 2:
        blockers.append("soap_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"soap_{key}_state_missing")
    if not isinstance(state.get("GG"), list) or not isinstance(state.get("Q"), list):
        blockers.append("soap_preconditioner_state_missing")
    if bool(group.get("precondition_1d", False)):
        blockers.append("soap_precondition_1d_unexpected_for_canary")
    return blockers


def _validate_conda(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("conda_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"conda_{key}_state_missing")
    if "projector" in state:
        blockers.append("conda_projector_unexpected_for_1d_canary")
    return blockers


def _validate_grams(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("grams_step_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"grams_{key}_state_missing")
    return blockers


def _validate_srmm(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("srmm_step_after_native_invalid")
    for key in ("mov_avg_grad", "mov_avg_param"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"srmm_{key}_state_missing")
    return blockers


def _validate_splus(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("splus_step_after_native_invalid")
    for key in ("momentum", "ema"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"splus_{key}_state_missing")
    if "sides" in state or "q_sides" in state:
        blockers.append("splus_whitening_state_unexpected_for_1d_canary")
    return blockers


def _validate_tam(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("tam_step_after_native_invalid")
    for key in ("s", "momentum_buffer"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"tam_{key}_state_missing")
    return blockers


def _validate_swats(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("swats_step_after_native_invalid")
    if str(group.get("phase", "")) != "adam":
        blockers.append("swats_phase_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"swats_{key}_state_missing")
    exp_avg2 = state.get("exp_avg2")
    if not torch.is_tensor(exp_avg2) or tuple(exp_avg2.shape) != (1,):
        blockers.append("swats_exp_avg2_state_missing")
    return blockers


def _validate_sophiah(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("sophiah_step_after_native_invalid")
    for key in ("momentum", "hessian_moment"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"sophiah_{key}_state_missing")
    if "hessian" in state:
        blockers.append("sophiah_hessian_unexpected_for_first_step_canary")
    return blockers


def _validate_racs(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("racs_step_after_native_invalid")
    s = state.get("s")
    if not torch.is_tensor(s) or tuple(s.shape) != (int(param.numel()),):
        blockers.append("racs_s_state_missing")
    for key in ("q", "theta"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != (1,):
            blockers.append(f"racs_{key}_state_missing")
    return blockers


def _validate_kate(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("kate_step_after_native_invalid")
    for key in ("m", "b"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"kate_{key}_state_missing")
    return blockers


def _validate_rose(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("rose_step_after_native_invalid")
    if state:
        blockers.append("rose_unexpected_persistent_state")
    return blockers


def _validate_mars(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("mars_step_after_native_invalid")
    if bool(group.get("optimize_1d", False)):
        blockers.append("mars_optimize_1d_unexpected_for_fallback_canary")
    if str(group.get("mars_type", "")) != "adamw":
        blockers.append("mars_type_after_native_invalid")
    for key in ("exp_avg", "exp_avg_sq", "last_grad"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"mars_{key}_state_missing")
    return blockers


def _validate_aida(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("aida_step_after_native_invalid")
    if int(group.get("k", 0) or 0) != 0:
        blockers.append("aida_projection_loop_unexpected_for_canary")
    for key in ("exp_avg", "exp_avg_var"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"aida_{key}_state_missing")
    if "max_exp_avg_var" in state:
            blockers.append("aida_ams_bound_state_unexpected_for_canary")
    return blockers


def _validate_adatam(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 1:
        blockers.append("adatam_step_after_native_invalid")
    for key in ("s", "exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adatam_{key}_state_missing")
    return blockers


def _validate_came(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("came_step_after_native_invalid")
    if bool(group.get("ams_bound", False)):
        blockers.append("came_ams_bound_unexpected_for_canary")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"came_{key}_state_missing")
    if not torch.is_tensor(state.get("RMS")) and not isinstance(state.get("RMS"), float):
        blockers.append("came_rms_state_missing")
    return blockers


def _validate_adalite(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("adalite_step_after_native_invalid")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("adalite_weight_decay_unexpected_for_canary")
    for key in ("m_avg", "v_avg"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"adalite_{key}_state_missing")
    return blockers


def _validate_apollodqn(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("apollodqn_step_after_native_invalid")
    if str(group.get("rebound", "")) != "constant":
        blockers.append("apollodqn_rebound_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("apollodqn_weight_decay_unexpected_for_canary")
    for key in ("exp_avg_grad", "approx_hessian", "update"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"apollodqn_{key}_state_missing")
    return blockers


def _validate_scionlight(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("scionlight_step_after_native_invalid")
    if int(group.get("norm_type", 0)) != 0:
        blockers.append("scionlight_norm_type_unexpected_for_canary")
    if bool(group.get("constraint", False)):
        blockers.append("scionlight_constraint_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("scionlight_weight_decay_unexpected_for_canary")
    if state:
        blockers.append("scionlight_state_unexpected_for_canary")
    return blockers


def _validate_scion(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("scion_step_after_native_invalid")
    if int(group.get("norm_type", 0)) != 0:
        blockers.append("scion_norm_type_unexpected_for_canary")
    if bool(group.get("constraint", False)):
        blockers.append("scion_constraint_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("scion_weight_decay_unexpected_for_canary")
    value = state.get("d")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("scion_d_state_missing")
    return blockers


def _validate_spectralsphere(
    _result: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    param: torch.nn.Parameter,
) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("spectralsphere_step_after_native_invalid")
    if param.dim() != 2:
        blockers.append("spectralsphere_requires_2d_param")
    if bool(group.get("nesterov", False)):
        blockers.append("spectralsphere_nesterov_unexpected_for_canary")
    if abs(float(group.get("momentum", 0.0) or 0.0)) > 0.0:
        blockers.append("spectralsphere_momentum_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("spectralsphere_weight_decay_unexpected_for_canary")
    value = state.get("momentum_buffer")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("spectralsphere_momentum_buffer_state_missing")
    return blockers


def _validate_demo(result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    group = optimizer.param_groups[0]
    demo_state = getattr(optimizer, "demo_state", {})
    state = demo_state.get(param, {}) if isinstance(demo_state, dict) else {}
    first_case = _first_executor_case(result)
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("demo_step_after_native_invalid")
    if int(getattr(optimizer, "compression_top_k", 0) or 0) != int(param.numel()):
        blockers.append("demo_full_top_k_not_selected")
    if int(getattr(optimizer, "compression_chunk", 0) or 0) != int(param.numel()):
        blockers.append("demo_single_chunk_not_selected")
    if abs(float(getattr(optimizer, "compression_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("demo_compression_decay_unexpected_for_canary")
    value = state.get("delta") if isinstance(state, dict) else None
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("demo_delta_state_missing")
    elif _max_abs_diff(value, torch.zeros_like(value)) > 1.0e-6:
        blockers.append("demo_delta_not_cleared_after_full_top_k")
    expected_bytes = int(param.numel()) * (8 + 4)
    if int(first_case.get("data_transmit", 0) or 0) != expected_bytes:
        blockers.append("demo_data_transmit_invalid")
    if int(first_case.get("data_receive", 0) or 0) != expected_bytes:
        blockers.append("demo_data_receive_invalid")
    return blockers


def _validate_emolynx(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("emolynx_step_after_native_invalid")
    if bool(group.get("use_shadow", False)):
        blockers.append("emolynx_shadow_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("emolynx_weight_decay_unexpected_for_canary")
    value = state.get("exp_avg")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
        blockers.append("emolynx_exp_avg_state_missing")
    ema = optimizer.state.get("ema")
    if not isinstance(ema, dict) or set(ema) != {"short", "medium", "long"}:
        blockers.append("emolynx_ema_state_missing")
    return blockers


def _validate_emonavi(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("emonavi_step_after_native_invalid")
    if bool(group.get("use_shadow", False)):
        blockers.append("emonavi_shadow_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("emonavi_weight_decay_unexpected_for_canary")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"emonavi_{key}_state_missing")
    ema = optimizer.state.get("ema")
    if not isinstance(ema, dict) or set(ema) != {"short", "medium", "long"}:
        blockers.append("emonavi_ema_state_missing")
    return blockers


def _validate_emofact(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    group = optimizer.param_groups[0]
    blockers = []
    if int(group.get("step", 0) or 0) != 1:
        blockers.append("emofact_step_after_native_invalid")
    if bool(group.get("use_shadow", False)):
        blockers.append("emofact_shadow_unexpected_for_canary")
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("emofact_weight_decay_unexpected_for_canary")
    if param.dim() != 1:
        blockers.append("emofact_factored_path_unexpected_for_1d_canary")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(param.shape):
            blockers.append(f"emofact_{key}_state_missing")
    ema = optimizer.state.get("ema")
    if not isinstance(ema, dict) or set(ema) != {"short", "medium", "long"}:
        blockers.append("emofact_ema_state_missing")
    return blockers


def _validate_pnm(_result: dict[str, Any], optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> list[str]:
    state = optimizer.state[param]
    blockers = []
    if int(optimizer.param_groups[0].get("step", 0) or 0) != 2:
        blockers.append("pnm_step_after_native_invalid")
    if not torch.is_tensor(state.get("pos_momentum")):
        blockers.append("pnm_pos_momentum_missing")
    if not torch.is_tensor(state.get("neg_momentum")):
        blockers.append("pnm_neg_momentum_missing")
    return blockers


def _first_executor_case(result: Mapping[str, Any]) -> dict[str, Any]:
    cases = result.get("cases")
    if isinstance(cases, list) and cases and isinstance(cases[0], Mapping):
        return dict(cases[0])
    return {}


def _case_blocked(spec: _Spec, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "selected_optimizer_name": spec.name,
        "selected_optimizer_family": spec.family,
        "blocked_reasons": [reason],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_bridge_training_loop_canary_scorecard_v0",
        "gate": "plugin_bridge_selected_training_loop_native_canary",
        "roadmap": ROADMAP,
        "ok": False,
        "promotion_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "cases": [],
        "summary": {"case_count": 0, "native_step_count": 0, "native_kernel_launch_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().detach().cpu().item())


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / ARTIFACT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


_SPECS = (
    _Spec(
        name="adafactor",
        family="factored_memory_layout",
        shape=(128, 128),
        lr=1.0e-3,
        weight_decay=0.01,
        optimizer_args={
            "betas": (None, 0.999),
            "scale_parameter": False,
            "relative_step": False,
            "warmup_init": False,
            "clip_threshold": 1.0,
        },
        prime="step",
        validate=_validate_adafactor,
    ),
    _Spec(
        name="muon",
        family="model_or_shape_aware",
        shape=(4, 4),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={},
        prime="none",
        validate=_validate_muon,
    ),
    _Spec(
        name="distributedmuon",
        family="model_or_shape_aware",
        shape=(4, 4),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={},
        prime="none",
        validate=_validate_muon,
    ),
    _Spec(
        name="adamuon",
        family="model_or_shape_aware",
        shape=(64,),
        lr=3.0e-4,
        weight_decay=0.0,
        optimizer_args={"adamw_betas": (0.9, 0.999), "adamw_wd": 0.0, "eps": 1.0e-10},
        prime="none",
        validate=_validate_muon_family_adamw,
    ),
    _Spec(
        name="adago",
        family="model_or_shape_aware",
        shape=(64,),
        lr=3.0e-4,
        weight_decay=0.0,
        optimizer_args={"adamw_betas": (0.9, 0.95), "adamw_wd": 0.0, "adamw_eps": 1.0e-10},
        prime="none",
        validate=_validate_muon_family_adamw,
    ),
    _Spec(
        name="sgdsai",
        family="state_adapter_special",
        shape=(8, 8),
        lr=1.0e-2,
        weight_decay=0.01,
        optimizer_args={"momentum": 0.9, "weight_decouple": True},
        prime="warmup",
        validate=_validate_sgdsai,
    ),
    _Spec(
        name="sm3",
        family="factored_memory_layout",
        shape=(128,),
        lr=1.0e-1,
        weight_decay=0.0,
        optimizer_args={"momentum": 0.0, "beta": 0.0, "eps": 1.0e-30},
        prime="none",
        validate=_validate_sm3,
    ),
    _Spec(
        name="spam",
        family="state_adapter_special",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "density": 1.0,
            "threshold": 0,
            "warmup_epoch": 1,
            "update_proj_gap": 500,
        },
        prime="none",
        validate=_validate_spam,
    ),
    _Spec(
        name="stablespam",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "gamma1": 0.7,
            "gamma2": 0.9,
            "theta": 0.999,
            "t_max": None,
            "update_proj_gap": 1000,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_stablespam,
    ),
    _Spec(
        name="adadelta",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0,
        weight_decay=0.0,
        optimizer_args={"rho": 0.9, "eps": 1.0e-6, "weight_decouple": False, "fixed_decay": False},
        prime="none",
        validate=_validate_adadelta,
    ),
    _Spec(
        name="ftrl",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={"lr_power": -0.5, "beta": 0.0, "lambda_1": 0.0, "lambda_2": 0.0},
        prime="none",
        validate=_validate_ftrl,
    ),
    _Spec(
        name="diffgrad",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "rectify": False,
            "ams_bound": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_diffgrad,
    ),
    _Spec(
        name="adabelief",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "weight_decouple": True,
            "fixed_decay": False,
            "rectify": False,
            "ams_bound": False,
            "eps": 1.0e-16,
        },
        prime="none",
        validate=_validate_adabelief,
    ),
    _Spec(
        name="adabound",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "final_lr": 1.0e-1,
            "gamma": 1.0e-3,
            "weight_decouple": True,
            "fixed_decay": False,
            "ams_bound": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_adabound,
    ),
    _Spec(
        name="laprop",
        family="custom_formula",
        shape=(8, 8),
        lr=4.0e-4,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "centered": False,
            "weight_decouple": True,
            "fixed_decay": False,
            "ams_bound": False,
            "eps": 1.0e-15,
        },
        prime="none",
        validate=_validate_laprop,
    ),
    _Spec(
        name="adai",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.1, 0.99),
            "weight_decouple": False,
            "fixed_decay": False,
            "stable_weight_decay": False,
            "dampening": 1.0,
            "eps": 1.0e-3,
        },
        prime="none",
        validate=_validate_adai,
    ),
    _Spec(
        name="adopt",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.9999),
            "weight_decouple": False,
            "fixed_decay": False,
            "foreach": False,
            "eps": 1.0e-6,
        },
        prime="step",
        validate=_validate_adopt,
    ),
    _Spec(
        name="msvag",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={"beta": 0.9},
        prime="none",
        validate=_validate_msvag,
    ),
    _Spec(
        name="ademamix",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999, 0.9999),
            "alpha": 5.0,
            "t_alpha_beta3": None,
            "weight_decouple": False,
            "fixed_decay": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_ademamix,
    ),
    _Spec(
        name="simplifiedademamix",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-4,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.99, 0.95),
            "alpha": 0.0,
            "beta1_warmup": None,
            "min_beta1": 0.9,
            "weight_decouple": True,
            "fixed_decay": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_simplifiedademamix,
    ),
    _Spec(
        name="a2grad",
        family="custom_formula",
        shape=(8, 8),
        lr=0.0,
        weight_decay=0.0,
        optimizer_args={"variant": "uni", "beta": 10.0, "lips": 10.0},
        prime="none",
        validate=_validate_a2grad,
    ),
    _Spec(
        name="avagrad",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-1,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "weight_decouple": True,
            "fixed_decay": False,
            "eps": 1.0e-1,
            "adam_debias": False,
        },
        prime="step",
        validate=_validate_avagrad,
    ),
    _Spec(
        name="adanorm",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.99),
            "r": 0.95,
            "weight_decouple": True,
            "fixed_decay": False,
            "ams_bound": False,
            "eps": 1.0e-8,
            "adam_debias": False,
        },
        prime="none",
        validate=_validate_adanorm,
    ),
    _Spec(
        name="bcos",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "beta": 0.9,
            "beta2": None,
            "mode": "g",
            "simple_cond": False,
            "weight_decouple": True,
            "eps": 1.0e-6,
        },
        prime="none",
        validate=_validate_bcos,
    ),
    _Spec(
        name="adagc",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "beta": 0.98,
            "lambda_abs": 100.0,
            "lambda_rel": 1.05,
            "warmup_steps": 100,
            "weight_decouple": True,
            "fixed_decay": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_adagc,
    ),
    _Spec(
        name="adasmooth",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.5, 0.99),
            "weight_decouple": False,
            "fixed_decay": False,
            "eps": 1.0e-6,
        },
        prime="none",
        validate=_validate_adasmooth,
    ),
    _Spec(
        name="adashift",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "keep_num": 1, "reduce_func": None, "eps": 1.0e-10},
        prime="step",
        validate=_validate_adashift,
    ),
    _Spec(
        name="adammini",
        family="model_or_shape_aware",
        shape=(64,),
        lr=1.0,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "eps": 1.0e-8, "model_sharding": False},
        prime="none",
        validate=_validate_adammini,
    ),
    _Spec(
        name="adapnm",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999, 1.0),
            "weight_decouple": True,
            "fixed_decay": False,
            "ams_bound": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_adapnm,
    ),
    _Spec(
        name="adan",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.98, 0.92, 0.99),
            "weight_decouple": False,
            "max_grad_norm": 0.0,
            "foreach": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_adan,
    ),
    _Spec(
        name="ano",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-4,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.92, 0.99),
            "weight_decouple": True,
            "fixed_decay": False,
            "logarithmic_schedule": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_ano,
    ),
    _Spec(
        name="amos",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "beta": 0.999,
            "momentum": 0.0,
            "extra_l2": 0.0,
            "foreach": False,
            "eps": 1.0e-18,
        },
        prime="none",
        validate=_validate_amos,
    ),
    _Spec(
        name="apollo",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "scale_type": "tensor",
            "weight_decouple": True,
            "fixed_decay": False,
            "correct_bias": True,
            "eps": 1.0e-6,
        },
        prime="none",
        validate=_validate_apollo,
    ),
    _Spec(
        name="galore",
        family="model_or_shape_aware",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "eps": 1.0e-6},
        prime="none",
        validate=_validate_galore,
    ),
    _Spec(
        name="fira",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "eps": 1.0e-6},
        prime="none",
        validate=_validate_fira,
    ),
    _Spec(
        name="focus",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.0), "gamma": 0.1},
        prime="none",
        validate=_validate_focus,
    ),
    _Spec(
        name="alig",
        family="closure_or_second_order",
        shape=(64,),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={"max_lr": 1.0e-2, "momentum": 0.0, "adjusted_momentum": False},
        prime="none",
        validate=_validate_alig,
    ),
    _Spec(
        name="adahessian",
        family="closure_or_second_order",
        shape=(64,),
        lr=1.0e-1,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.0, 0.0),
            "hessian_power": 1.0,
            "update_period": 2,
            "num_samples": 1,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_adahessian,
    ),
    _Spec(
        name="alice",
        family="model_or_shape_aware",
        shape=(2, 2),
        lr=2.0e-2,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.9, 0.0),
            "alpha": 0.3,
            "alpha_c": 0.4,
            "rank": 2,
            "leading_basis": 1,
            "update_interval": 2,
            "gamma": 1.01,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_alice,
    ),
    _Spec(
        name="kron",
        family="closure_or_second_order",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "momentum": 0.0,
            "weight_decouple": True,
            "pre_conditioner_update_probability": 1.0e-12,
            "balance_prob": 0.0,
            "memory_save_mode": "all_diag",
            "precondition_dtype": torch.float32,
        },
        prime="none",
        validate=_validate_kron,
    ),
    _Spec(
        name="shampoo",
        family="factored_memory_layout",
        shape=(16,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"momentum": 0.0, "preconditioning_compute_steps": 1, "matrix_eps": 1.0e-6},
        prime="none",
        validate=_validate_shampoo,
    ),
    _Spec(
        name="scalableshampoo",
        family="factored_memory_layout",
        shape=(16,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.0, 0.999),
            "moving_average_for_momentum": False,
            "decoupled_weight_decay": False,
            "decoupled_learning_rate": True,
            "start_preconditioning_step": 25,
            "preconditioning_compute_steps": 1000,
            "statistics_compute_steps": 1,
            "graft_type": 0,
            "nesterov": False,
            "matrix_eps": 1.0e-6,
        },
        prime="none",
        validate=_validate_scalableshampoo,
    ),
    _Spec(
        name="soap",
        family="factored_memory_layout",
        shape=(16,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.95, 0.95),
            "precondition_frequency": 10,
            "max_precondition_dim": 10000,
            "merge_dims": False,
            "precondition_1d": False,
            "correct_bias": True,
            "normalize_gradient": False,
            "eps": 1.0e-8,
        },
        prime="step",
        validate=_validate_soap,
    ),
    _Spec(
        name="conda",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "eps": 1.0e-8},
        prime="none",
        validate=_validate_conda,
    ),
    _Spec(
        name="lorarite",
        family="custom_formula",
        shape=(2, 3),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.0, 0.0),
            "eps": 1.0e-6,
            "relative_epsilon": False,
            "clip_unmagnified_grad": 0.0,
            "update_capping": 0.0,
            "update_skipping": 0.0,
            "apply_escape": False,
            "balance_param": False,
            "lora_l_dim": 0,
            "lora_r_dim": -1,
            "maybe_inf_to_nan": True,
        },
        prime="none",
        validate=_validate_lorarite,
        param_shapes=((2, 3), (4, 2)),
    ),
    _Spec(
        name="grams",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "weight_decouple": True, "eps": 1.0e-6},
        prime="none",
        validate=_validate_grams,
    ),
    _Spec(
        name="srmm",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={"beta": 0.5, "memory_length": 100},
        prime="none",
        validate=_validate_srmm,
    ),
    _Spec(
        name="splus",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-1,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "weight_decouple": True,
            "ema_rate": 0.999,
            "nonstandard_constant": 1.0e-3,
        },
        prime="none",
        validate=_validate_splus,
    ),
    _Spec(
        name="tam",
        family="custom_formula",
        shape=(1,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"momentum": 0.9, "decay_rate": 0.9, "eps": 1.0e-8},
        prime="none",
        validate=_validate_tam,
    ),
    _Spec(
        name="swats",
        family="custom_formula",
        shape=(1,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "ams_bound": False, "nesterov": False, "eps": 1.0e-6},
        prime="none",
        validate=_validate_swats,
    ),
    _Spec(
        name="sophiah",
        family="closure_or_second_order",
        shape=(64,),
        lr=6.0e-2,
        weight_decay=0.0,
        optimizer_args={"betas": (0.96, 0.99), "p": 1.0e-2, "update_period": 10, "eps": 1.0e-12},
        prime="none",
        validate=_validate_sophiah,
    ),
    _Spec(
        name="racs",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"beta": 0.9, "alpha": 0.05, "gamma": 1.01, "eps": 1.0e-8},
        prime="none",
        validate=_validate_racs,
    ),
    _Spec(
        name="kate",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"delta": 0.0, "eps": 1.0e-8},
        prime="none",
        validate=_validate_kate,
    ),
    _Spec(
        name="rose",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "weight_decouple": False,
            "centralize": True,
            "stabilize": True,
            "bf16_sr": False,
            "compute_dtype": torch.float32,
        },
        prime="none",
        validate=_validate_rose,
    ),
    _Spec(
        name="mars",
        family="model_or_shape_aware",
        shape=(64,),
        lr=3.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "lr_1d": 3.0e-3,
            "betas_1d": (0.9, 0.95),
            "mars_type": "adamw",
            "optimize_1d": False,
            "ams_bound": False,
            "cautious": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_mars,
    ),
    _Spec(
        name="aida",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "k": 0,
            "rectify": False,
            "ams_bound": False,
            "adanorm": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_aida,
    ),
    _Spec(
        name="adatam",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={"betas": (0.9, 0.999), "decay_rate": 0.9, "eps": 1.0e-8},
        prime="none",
        validate=_validate_adatam,
    ),
    _Spec(
        name="came",
        family="factored_memory_layout",
        shape=(64,),
        lr=2.0e-4,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999, 0.9999),
            "clip_threshold": 1.0,
            "ams_bound": False,
            "eps1": 1.0e-30,
            "eps2": 1.0e-16,
        },
        prime="none",
        validate=_validate_came,
    ),
    _Spec(
        name="adalite",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "g_norm_min": 1.0e-10,
            "ratio_min": 1.0e-4,
            "tau": 1.0,
            "eps1": 1.0e-6,
            "eps2": 1.0e-10,
        },
        prime="none",
        validate=_validate_adalite,
    ),
    _Spec(
        name="apollodqn",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-2,
        weight_decay=0.0,
        optimizer_args={
            "init_lr": 1.0e-5,
            "beta": 0.9,
            "rebound": "constant",
            "weight_decay_type": "l2",
            "warmup_steps": 500,
            "eps": 1.0e-4,
        },
        prime="none",
        validate=_validate_apollodqn,
    ),
    _Spec(
        name="scionlight",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "momentum": 0.1,
            "constraint": False,
            "norm_type": 0,
            "norm_kwargs": {},
            "scale": 1.0,
            "weight_decouple": True,
            "foreach": False,
        },
        prime="none",
        validate=_validate_scionlight,
    ),
    _Spec(
        name="scion",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "momentum": 0.1,
            "constraint": False,
            "norm_type": 0,
            "norm_kwargs": {},
            "scale": 1.0,
            "weight_decouple": True,
            "foreach": False,
        },
        prime="none",
        validate=_validate_scion,
    ),
    _Spec(
        name="spectralsphere",
        family="model_or_shape_aware",
        shape=(4, 4),
        lr=3.0e-4,
        weight_decay=0.0,
        optimizer_args={
            "momentum": 0.0,
            "nesterov": False,
            "power_iteration_steps": 1,
            "msign_steps": 1,
            "solver_max_iterations": 10,
        },
        prime="none",
        validate=_validate_spectralsphere,
    ),
    _Spec(
        name="demo",
        family="state_adapter_special",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "compression_decay": 0.0,
            "compression_top_k": 64,
            "compression_chunk": 64,
        },
        prime="none",
        validate=_validate_demo,
    ),
    _Spec(
        name="emolynx",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.99),
            "use_shadow": False,
            "shadow_weight": 0.05,
            "weight_decouple": True,
            "fixed_decay": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_emolynx,
    ),
    _Spec(
        name="emonavi",
        family="custom_formula",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "use_shadow": False,
            "shadow_weight": 0.05,
            "weight_decouple": True,
            "fixed_decay": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_emonavi,
    ),
    _Spec(
        name="emofact",
        family="factored_memory_layout",
        shape=(64,),
        lr=1.0e-3,
        weight_decay=0.0,
        optimizer_args={
            "betas": (0.9, 0.999),
            "use_shadow": False,
            "shadow_weight": 0.05,
            "weight_decouple": True,
            "fixed_decay": False,
            "eps": 1.0e-8,
        },
        prime="none",
        validate=_validate_emofact,
    ),
    _Spec(
        name="pnm",
        family="custom_formula",
        shape=(8, 8),
        lr=1.0e-3,
        weight_decay=0.01,
        optimizer_args={"betas": (0.9, 1.0), "weight_decouple": True},
        prime="step",
        validate=_validate_pnm,
    ),
)


__all__ = ["ARTIFACT_NAME", "ROADMAP", "build_plugin_bridge_training_loop_canary_scorecard"]
