"""Default-off LoRARite Triton TrainingLoop executor for plugin canaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from core.turbocore_triton_optimizer import triton_adamw_flat_available, triton_adamw_flat_unavailable_reason


try:  # pragma: no cover - host-specific import availability
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoRARiteTrainingExecutorConfig:
    optimizer_kind: str = "lorarite"
    lr: float = 1.0e-2
    beta1: float = 0.0
    beta2: float = 0.0
    eps: float = 1.0e-6
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _add_update_kernel(param_ptr, update_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(param_ptr + offsets, p + u, mask=mask)


class LoRARiteTrainingExecutor:
    """Launch a paired LoRARite fp32 Triton canary against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: LoRARiteTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("LoRARiteTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("lorarite_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("lorarite_training_executor_requires_cuda_params")
        if len(self.params) != 2 or len(self.optimizer.param_groups) != 1:
            return _blocked("lorarite_training_executor_requires_single_pair_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"lorarite_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("lorarite_maximize_not_supported")

        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(group, group_cfg)
            if blockers:
                cases.append(_case_failed(blockers[0]))
                continue
            init_group = getattr(self.optimizer, "init_group", None)
            if callable(init_group):
                init_group(group)
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            pairs = list(getattr(self.optimizer, "iter_lora_pairs")(group))
            if len(pairs) != 1:
                cases.append(_case_failed("lorarite_training_executor_requires_exactly_one_pair"))
                continue
            left, right = pairs[0]
            cases.append(self._step_pair(left, right, group, group_cfg))

        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("lorarite_training_executor_no_pair_cases")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_lorarite_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "lorarite_native_step_failed"),
            "optimizer_kind": self.config.optimizer_kind,
            "training_dispatch": True,
            "training_path_enabled": True,
            "native_step_executed": ok,
            "native_kernel_launched": any(case.get("kernel_executed") is True for case in cases),
            "training_parameters_mutated": ok,
            "should_call_pytorch_optimizer_step": not ok,
            "pytorch_optimizer_state_synced": True,
            "parameter_step_count": len(cases),
            "cases": cases,
            "timing": {"elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4)},
            "blocked_reasons": blockers,
        }

    def close(self) -> None:
        return None

    def _step_pair(
        self,
        left: torch.nn.Parameter,
        right: torch.nn.Parameter,
        group: Mapping[str, Any],
        group_cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        blockers = _pair_blockers(left, right)
        if blockers:
            return _case_failed(blockers[0])
        if tuple(left.shape) != (2, 3) or tuple(right.shape) != (4, 2):
            return _case_failed("lorarite_training_executor_requires_test_pair_shapes")
        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("lorarite_training_executor_only_first_step_supported_for_canary")

        state = getattr(self.optimizer, "build_pair_info")(group, left, right)
        grad_norm = torch.linalg.norm(state["update_l"]).pow(2).add(torch.linalg.norm(state["update_r"]).pow(2)).sqrt()
        left_update, right_update = _compute_pair_updates(self.optimizer, group, left, right, grad_norm)
        before_left = left.detach().clone()
        before_right = right.detach().clone()
        _launch_add_update(left, left_update.contiguous(), int(group_cfg["block_size"]))
        _launch_add_update(right, right_update.contiguous(), int(group_cfg["block_size"]))
        mutated_left = _max_abs_diff(before_left, left.detach()) > 0.0
        mutated_right = _max_abs_diff(before_right, right.detach()) > 0.0
        ok = mutated_left and mutated_right and torch.isfinite(left).all().item() and torch.isfinite(right).all().item()
        return {
            "schema_version": 1,
            "ok": bool(ok),
            "left_shape": [int(dim) for dim in left.shape],
            "right_shape": [int(dim) for dim in right.shape],
            "param_dtype": str(left.dtype).replace("torch.", ""),
            "step_after": int(state.get("step", -1)),
            "kernel_executed": True,
            "training_parameters_mutated": bool(ok),
            "left_parameters_mutated": bool(mutated_left),
            "right_parameters_mutated": bool(mutated_right),
            "state_keys": sorted(str(key) for key in state.keys()),
            "blocked_reasons": [] if ok else ["lorarite_training_executor_pair_parameters_not_mutated"],
        }


def build_plugin_lorarite_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: LoRARiteTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> LoRARiteTrainingExecutor:
    return LoRARiteTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _compute_pair_updates(
    optimizer: torch.optim.Optimizer,
    group: Mapping[str, Any],
    left: torch.nn.Parameter,
    right: torch.nn.Parameter,
    grad_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    helper = optimizer.helper
    state = optimizer.state[left]
    update_left = state.pop("update_l")
    update_right = state.pop("update_r")
    rotate_inv_left = state.pop("rotate_inv_l")
    rotate_inv_right = state.pop("rotate_inv_r")
    projection_left = state.pop("projection_l")
    projection_right = state.pop("projection_r")
    beta1, beta2 = group["betas"]

    if group["clip_unmagnified_grad"] > 0.0 and grad_norm > group["clip_unmagnified_grad"]:
        scale = group["clip_unmagnified_grad"] / grad_norm
        update_left = update_left * scale
        update_right = update_right * scale

    second_left = helper.compute_second_moment(update_left)
    second_right = helper.compute_second_moment(update_right)
    transformed_v_left = helper.transform_second_moment_to_new_basis(state["v_l"], projection_left)
    transformed_v_right = helper.transform_second_moment_to_new_basis(state["v_r"], projection_right)
    escape_left = torch.zeros((), dtype=update_left.dtype, device=update_left.device)
    escape_right = torch.zeros((), dtype=update_right.dtype, device=update_right.device)
    v_left = helper.update_second_moment(state["step"], second_left, transformed_v_left, beta2)
    v_right = helper.update_second_moment(state["step"], second_right, transformed_v_right, beta2)
    update_left = helper.get_preconditioned_update(
        update_left,
        v_left,
        escape_left,
        group["eps"],
        group["eps_root"],
        group["relative_epsilon"],
        False,
    )
    update_right = helper.get_preconditioned_update(
        update_right,
        v_right,
        escape_right,
        group["eps"],
        group["eps_root"],
        group["relative_epsilon"],
        False,
    )
    m_left = helper.transform_first_moment_to_new_basis(state["m_l"], projection_left)
    m_right = helper.transform_first_moment_to_new_basis(state["m_r"], projection_right)
    m_left = helper.update_first_moment(state["step"], update_left, m_left, beta1)
    m_right = helper.update_first_moment(state["step"], update_right, m_right, beta1)
    update_left = helper.rotate_update(m_left, rotate_inv_right).mul(-group["lr"])
    update_right = helper.rotate_update(m_right, rotate_inv_left).mul(-group["lr"])

    state["step"] += 1
    state["v_l"] = v_left
    state["v_r"] = v_right
    state["m_l"] = m_left
    state["m_r"] = m_right
    state["escape_l"] = escape_left
    state["escape_r"] = escape_right
    left_update = helper.restore_param_shape(update_left, left, group["lora_l_dim"]).to(left.dtype)
    right_update = helper.restore_param_shape(update_right, right, group["lora_r_dim"]).to(right.dtype)
    return left_update, right_update


def _launch_add_update(param: torch.nn.Parameter, update: torch.Tensor, block_size: int) -> None:
    grid = (triton.cdiv(int(param.numel()), int(block_size)),)
    _add_update_kernel[grid](param, update, int(param.numel()), BLOCK_SIZE=int(block_size), num_warps=4)


def _normalize_config(
    value: LoRARiteTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> LoRARiteTrainingExecutorConfig:
    if isinstance(value, LoRARiteTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = payload.get("betas", group.get("betas", (0.0, 0.0)))
    return LoRARiteTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "lorarite"),
        lr=float(payload.get("lr", group.get("lr", 1.0e-2)) or 1.0e-2),
        beta1=float(betas[0]),
        beta2=float(betas[1]),
        eps=float(payload.get("eps", group.get("eps", 1.0e-6)) or 1.0e-6),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: LoRARiteTrainingExecutorConfig) -> dict[str, Any]:
    betas = group.get("betas", (config.beta1, config.beta2))
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "beta1": float(betas[0]),
        "beta2": float(betas[1]),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("lorarite_weight_decay_not_supported_for_canary")
    if abs(float(group_cfg["beta1"])) > 0.0 or abs(float(group_cfg["beta2"])) > 0.0:
        blockers.append("lorarite_requires_zero_betas_canary")
    for key in ("clip_unmagnified_grad", "update_capping", "update_skipping"):
        if abs(float(group.get(key, 0.0) or 0.0)) > 0.0:
            blockers.append(f"lorarite_{key}_not_supported_for_canary")
    if bool(group.get("apply_escape", False)):
        blockers.append("lorarite_apply_escape_not_supported_for_canary")
    if bool(group.get("balance_param", False)):
        blockers.append("lorarite_balance_param_not_supported_for_canary")
    if int(group.get("lora_l_dim", 0) or 0) != 0 or int(group.get("lora_r_dim", -1) or -1) != -1:
        blockers.append("lorarite_lora_dims_not_supported_for_canary")
    return blockers


def _pair_blockers(left: torch.nn.Parameter, right: torch.nn.Parameter) -> list[str]:
    blockers: list[str] = []
    for name, param in (("left", left), ("right", right)):
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            blockers.append(f"lorarite_{name}_requires_float32")
        if param.grad is None or param.grad.is_sparse:
            blockers.append(f"lorarite_{name}_sparse_or_missing_grad")
        if not param.is_contiguous() or (param.grad is not None and not param.grad.is_contiguous()):
            blockers.append(f"lorarite_{name}_requires_contiguous_tensor")
    return blockers


def _case_failed(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "kernel_executed": False,
        "training_parameters_mutated": False,
        "blocked_reasons": [reason],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "turbocore_plugin_lorarite_training_executor_v0",
        "ok": False,
        "reason": reason,
        "optimizer_kind": "lorarite",
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "blocked_reasons": [reason],
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().detach().cpu().item())


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["LoRARiteTrainingExecutor", "LoRARiteTrainingExecutorConfig", "build_plugin_lorarite_training_executor"]
