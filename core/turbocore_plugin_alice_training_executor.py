"""Default-off Alice Triton TrainingLoop executor for plugin canaries."""

from __future__ import annotations

import math
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
class AliceTrainingExecutorConfig:
    optimizer_kind: str = "alice"
    lr: float = 2.0e-2
    betas: tuple[float, float, float] = (0.9, 0.9, 0.0)
    alpha: float = 0.3
    alpha_c: float = 0.4
    rank: int = 2
    leading_basis: int = 1
    update_interval: int = 2
    gamma: float = 1.01
    eps: float = 1.0e-8
    block_size: int = 1024
    require_native_cuda: bool = True


if triton is not None and tl is not None:

    @triton.jit
    def _alice_apply_update_kernel(param_ptr, update_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(param_ptr + offsets, p - scale * u, mask=mask)


class AliceTrainingExecutor:
    """Launch a narrow Alice fp32 Triton canary against live optimizer state."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        config: AliceTrainingExecutorConfig | Mapping[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
        if not self.params:
            raise ValueError("AliceTrainingExecutor requires trainable parameters")
        self.config = _normalize_config(config, optimizer)
        self.workspace_root = Path(workspace_root or REPO_ROOT)

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request or {})
        if not bool(payload.get("training_dispatch", False)) or not bool(payload.get("training_path_enabled", False)):
            return _blocked("alice_training_executor_requires_training_dispatch")
        if self.config.require_native_cuda and any(param.device.type != "cuda" for param in self.params):
            return _blocked("alice_training_executor_requires_cuda_params")
        if len(self.params) != 1 or len(self.optimizer.param_groups) != 1:
            return _blocked("alice_training_executor_requires_single_param_group_canary")
        if not triton_adamw_flat_available() or triton is None:
            return _blocked(f"alice_triton_unavailable:{triton_adamw_flat_unavailable_reason()}")
        if bool(getattr(self.optimizer, "maximize", False)):
            return _blocked("alice_maximize_not_supported")

        started = time.perf_counter()
        cases: list[dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            group_cfg = _group_config(group, self.config)
            blockers = _unsupported_group_blockers(group, group_cfg)
            if blockers:
                cases.append(_case_failed(blockers[0]))
                continue
            group["step"] = int(group.get("step", 0) or 0) + 1  # type: ignore[index]
            for param in group["params"]:
                if param.grad is None:
                    continue
                cases.append(self._step_param(param, group, group_cfg))

        blockers = _dedupe(reason for case in cases for reason in case.get("blocked_reasons", []) or [])
        if not cases:
            blockers.append("alice_training_executor_no_grad_params")
        ok = bool(cases) and all(case.get("ok") is True for case in cases)
        return {
            "schema_version": 1,
            "executor": "turbocore_plugin_alice_training_executor_v0",
            "ok": ok,
            "reason": "called" if ok else (blockers[0] if blockers else "alice_native_step_failed"),
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

    def _step_param(
        self,
        param: torch.nn.Parameter,
        group: Mapping[str, Any],
        group_cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        if param.dtype != torch.float32 or (param.grad is not None and param.grad.dtype != torch.float32):
            return _case_failed("alice_training_executor_requires_float32")
        if param.grad is None or param.grad.is_sparse:
            return _case_failed("alice_training_executor_sparse_or_missing_grad")
        if not param.is_contiguous() or not param.grad.is_contiguous():
            return _case_failed("alice_training_executor_requires_contiguous_tensors")
        if tuple(param.shape) != (2, 2):
            return _case_failed("alice_training_executor_requires_2x2_canary")

        step = int(group.get("step", 0) or 0)
        if step != 1:
            return _case_failed("alice_training_executor_only_first_step_supported_for_canary")
        if int(group_cfg["rank"]) != 2 or int(group_cfg["leading_basis"]) != 1:
            return _case_failed("alice_training_executor_requires_rank2_leading1_canary")

        state = self.optimizer.state[param]
        _ensure_state(state, param, int(group_cfg["rank"]))
        update = _alice_update(param, state, group_cfg).contiguous()
        before = param.detach().clone()
        grid = (triton.cdiv(int(param.numel()), int(group_cfg["block_size"])),)
        _alice_apply_update_kernel[grid](
            param,
            update,
            int(param.numel()),
            float(group_cfg["lr"]) * float(group_cfg["alpha"]),
            BLOCK_SIZE=int(group_cfg["block_size"]),
            num_warps=4,
        )
        mutated = _max_abs_diff(before, param.detach()) > 0.0
        return {
            "schema_version": 1,
            "ok": mutated and torch.isfinite(param).all().item(),
            "param_shape": [int(dim) for dim in param.shape],
            "param_dtype": str(param.dtype).replace("torch.", ""),
            "step_after": step,
            "rank": int(group_cfg["rank"]),
            "leading_basis": int(group_cfg["leading_basis"]),
            "kernel_executed": True,
            "training_parameters_mutated": bool(mutated),
            "state_keys": sorted(str(key) for key in state.keys()),
            "blocked_reasons": [] if mutated else ["alice_training_executor_parameters_not_mutated"],
        }


def build_plugin_alice_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: AliceTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> AliceTrainingExecutor:
    return AliceTrainingExecutor(optimizer=optimizer, params=params, config=config, workspace_root=workspace_root)


def _normalize_config(
    value: AliceTrainingExecutorConfig | Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
) -> AliceTrainingExecutorConfig:
    if isinstance(value, AliceTrainingExecutorConfig):
        return value
    payload = dict(value or {})
    group = optimizer.param_groups[0] if optimizer.param_groups else {}
    betas = tuple(payload.get("betas", group.get("betas", (0.9, 0.9, 0.0))))
    beta3 = payload.get("beta3", betas[2] if len(betas) > 2 else 0.0)
    return AliceTrainingExecutorConfig(
        optimizer_kind=str(payload.get("optimizer_kind") or "alice"),
        lr=float(payload.get("lr", group.get("lr", 2.0e-2)) or 2.0e-2),
        betas=(float(betas[0]), float(betas[1]), float(beta3)),
        alpha=float(payload.get("alpha", group.get("alpha", 0.3)) or 0.3),
        alpha_c=float(payload.get("alpha_c", group.get("alpha_c", 0.4)) or 0.4),
        rank=int(payload.get("rank", group.get("rank", 2)) or 2),
        leading_basis=int(payload.get("leading_basis", group.get("leading_basis", 1)) or 1),
        update_interval=int(payload.get("update_interval", group.get("update_interval", 2)) or 2),
        gamma=float(payload.get("gamma", group.get("gamma", 1.01)) or 1.01),
        eps=float(payload.get("eps", group.get("eps", 1.0e-8)) or 1.0e-8),
        block_size=int(payload.get("block_size", 1024) or 1024),
        require_native_cuda=bool(payload.get("require_native_cuda", True)),
    )


def _group_config(group: Mapping[str, Any], config: AliceTrainingExecutorConfig) -> dict[str, Any]:
    betas = tuple(group.get("betas", config.betas))
    beta3 = betas[2] if len(betas) > 2 else config.betas[2]
    return {
        "lr": float(group.get("lr", config.lr) or config.lr),
        "betas": (float(betas[0]), float(betas[1]), float(beta3)),
        "alpha": float(group.get("alpha", config.alpha) or config.alpha),
        "alpha_c": float(group.get("alpha_c", config.alpha_c) or config.alpha_c),
        "rank": int(group.get("rank", config.rank) or config.rank),
        "leading_basis": int(group.get("leading_basis", config.leading_basis) or config.leading_basis),
        "update_interval": int(group.get("update_interval", config.update_interval) or config.update_interval),
        "gamma": float(group.get("gamma", config.gamma) or config.gamma),
        "eps": float(group.get("eps", config.eps) or config.eps),
        "block_size": int(config.block_size),
    }


def _unsupported_group_blockers(group: Mapping[str, Any], group_cfg: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if abs(float(group.get("weight_decay", 0.0) or 0.0)) > 0.0:
        blockers.append("alice_weight_decay_not_supported_for_canary")
    if not bool(group.get("weight_decouple", True)):
        blockers.append("alice_coupled_weight_decay_not_supported_for_canary")
    if int(group_cfg["rank"]) != 2 or int(group_cfg["leading_basis"]) != 1:
        blockers.append("alice_requires_rank2_leading1_canary")
    return blockers


def _ensure_state(state: dict[str, Any], param: torch.nn.Parameter, rank: int) -> None:
    rows, cols = int(param.shape[0]), int(param.shape[1])
    state.setdefault("U", torch.zeros((rows, rank), dtype=param.dtype, device=param.device))
    state.setdefault("Q", torch.zeros((rank, rank), dtype=param.dtype, device=param.device))
    state.setdefault("m", torch.zeros((rank, cols), dtype=param.dtype, device=param.device))
    state.setdefault("v", torch.zeros((rank, cols), dtype=param.dtype, device=param.device))
    state.setdefault("p", torch.zeros((cols,), dtype=param.dtype, device=param.device))
    state.setdefault("phi", torch.zeros((1,), dtype=param.dtype, device=param.device))


def _alice_update(param: torch.nn.Parameter, state: dict[str, Any], group_cfg: Mapping[str, Any]) -> torch.Tensor:
    grad = param.grad.detach()
    beta1, beta2, beta3 = group_cfg["betas"]
    rank = int(group_cfg["rank"])
    leading_basis = int(group_cfg["leading_basis"])
    u = state["U"]
    q = state["Q"]
    m = state["m"]
    v = state["v"]

    q_t = beta3 * (u @ q @ u.T) + (1.0 - beta3) * (grad @ grad.T)
    u = _switch(q_t, u, rank, leading_basis)
    sigma = u.T @ grad
    q.mul_(beta3).add_(sigma @ sigma.T, alpha=1.0 - beta3)
    m.mul_(beta1).add_(sigma, alpha=1.0 - beta1)
    v.mul_(beta2).add_(sigma.square(), alpha=1.0 - beta2)
    c_t, _phi = _compensation(grad, u, state["p"], state["phi"], float(group_cfg["gamma"]), beta1, rank)
    update = u @ (m / v.sqrt().clamp_min(float(group_cfg["eps"])))
    update.add_(c_t, alpha=float(group_cfg["alpha_c"]))
    state["U"] = u
    return update.view_as(param)


def _switch(q: torch.Tensor, u_prev: torch.Tensor, rank: int, leading_basis: int) -> torch.Tensor:
    u = u_prev
    for _ in range(1):
        u, _ = torch.linalg.qr(q.to(torch.float32) @ u.to(torch.float32))
    vals, vecs = torch.linalg.eigh(u.T @ q.to(torch.float32) @ u)
    leading_indices = torch.argsort(vals, descending=True)[:leading_basis]
    u_t1 = vecs[:, leading_indices]
    u_c, _ = torch.linalg.qr(torch.eye(q.shape[0], device=q.device, dtype=torch.float32) - u_t1 @ u_t1.T)
    u_t2 = u_c[:, : rank - leading_basis]
    return torch.cat([u_t1, u_t2], dim=1).to(q.dtype)


def _compensation(
    grad: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    phi: torch.Tensor,
    gamma: float,
    decay_rate: float,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = grad.shape
    sigma = u.T @ grad
    p.mul_(decay_rate).add_(grad.square().sum(dim=0) - sigma.square().sum(dim=0), alpha=1.0 - decay_rate).clamp_min_(
        1.0e-8
    )
    d = torch.zeros_like(grad)
    diag_len = min(rows, cols)
    d[torch.arange(diag_len, device=grad.device), torch.arange(diag_len, device=grad.device)] = 1.0 / p.sqrt()[
        :diag_len
    ]
    c_t = math.sqrt(rows - rank) * (grad - u @ sigma) * d if rows >= rank else torch.zeros_like(grad)
    scale = gamma / max(float(torch.norm(c_t).detach().cpu().item()) / float(phi.item()), gamma) if phi.item() > 0 else 1.0
    c_t.mul_(scale)
    return c_t, torch.norm(c_t).reshape_as(phi)


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().detach().cpu().item())


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
        "executor": "turbocore_plugin_alice_training_executor_v0",
        "ok": False,
        "reason": reason,
        "optimizer_kind": "alice",
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_step_executed": False,
        "native_kernel_launched": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": True,
        "blocked_reasons": [reason],
    }


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["AliceTrainingExecutor", "AliceTrainingExecutorConfig", "build_plugin_alice_training_executor"]
