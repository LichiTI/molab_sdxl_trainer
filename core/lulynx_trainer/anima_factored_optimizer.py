# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Anima-focused factored AdamW optimizer.

This is an experimental full-finetune optimizer for large DiT matrix weights.
For 2D parameters above a configurable size it stores factored second-moment
statistics (row + column) instead of a full tensor. Smaller tensors keep normal
AdamW moments for stability.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


class AnimaFactoredAdamW(torch.optim.Optimizer):
    """AdamW with factored second moments for large matrix parameters."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        min_dim: int = 128,
        min_numel: int = 65536,
        factored_eps: float = 1e-30,
    ) -> None:
        if lr <= 0:
            raise ValueError("AnimaFactoredAdamW requires lr > 0.")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("AnimaFactoredAdamW betas must be in [0, 1).")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            min_dim=int(min_dim),
            min_numel=int(min_numel),
            factored_eps=float(factored_eps),
        )
        super().__init__(params, defaults)
        self._factored_param_count = 0
        self._full_param_count = 0
        self._factored_numel = 0
        self._full_numel = 0

    @staticmethod
    def _should_factor(param: torch.Tensor, group: dict[str, Any]) -> bool:
        if param.grad is None or param.grad.is_sparse:
            return False
        if param.dim() != 2:
            return False
        if min(param.shape) < int(group["min_dim"]):
            return False
        return param.numel() >= int(group["min_numel"])

    @staticmethod
    def _factored_rms(row_state: torch.Tensor, col_state: torch.Tensor, eps: float) -> torch.Tensor:
        row_mean = row_state.mean(dim=0, keepdim=True).clamp_min(eps)
        return (row_state / row_mean).sqrt() * col_state.sqrt()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        factored_count = 0
        full_count = 0
        factored_numel = 0
        full_numel = 0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            factored_eps = float(group["factored_eps"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AnimaFactoredAdamW does not support sparse gradients.")

                grad = param.grad.detach()
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if self._should_factor(param, group):
                        state["factored"] = True
                        state["exp_avg_sq_row"] = torch.zeros(
                            (param.shape[0], 1),
                            device=param.device,
                            dtype=torch.float32,
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            (1, param.shape[1]),
                            device=param.device,
                            dtype=torch.float32,
                        )
                    else:
                        state["factored"] = False
                        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                state["step"] += 1
                step = int(state["step"].item())
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                if bool(state.get("factored", False)):
                    grad_sq = grad.float().pow(2)
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(grad_sq.mean(dim=1, keepdim=True), alpha=1.0 - beta2)
                    col.mul_(beta2).add_(grad_sq.mean(dim=0, keepdim=True), alpha=1.0 - beta2)
                    denom = self._factored_rms(row, col, factored_eps).to(dtype=param.dtype)
                    denom = denom / (bias_correction2**0.5)
                    factored_count += 1
                    factored_numel += param.numel()
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5)
                    full_count += 1
                    full_numel += param.numel()

                step_size = lr / bias_correction1
                param.addcdiv_(exp_avg, denom.add_(eps), value=-step_size)

        self._factored_param_count = factored_count
        self._full_param_count = full_count
        self._factored_numel = factored_numel
        self._full_numel = full_numel
        return loss

    def get_profile(self) -> dict[str, Any]:
        # AdamW normally keeps first and second moments at full tensor size.
        # This optimizer keeps the first moment full-size and factors only the
        # second moment for large matrices, so the estimate below is a lower
        # bound on optimizer-state savings.
        factored_second_moment_elems = 0
        full_second_moment_elems = 0
        for group in self.param_groups:
            for param in group.get("params", []):
                state = self.state.get(param, {})
                if not isinstance(state, dict):
                    continue
                if bool(state.get("factored", False)):
                    row = state.get("exp_avg_sq_row")
                    col = state.get("exp_avg_sq_col")
                    if isinstance(row, torch.Tensor):
                        factored_second_moment_elems += int(row.numel())
                    if isinstance(col, torch.Tensor):
                        factored_second_moment_elems += int(col.numel())
                    full_second_moment_elems += int(param.numel())

        saved_second_moment_elems = max(full_second_moment_elems - factored_second_moment_elems, 0)
        saved_second_moment_mb = saved_second_moment_elems * 4 / (1024 * 1024)
        return {
            "optimizer": "AnimaFactoredAdamW",
            "factored_param_tensors": int(self._factored_param_count),
            "full_param_tensors": int(self._full_param_count),
            "factored_numel": int(self._factored_numel),
            "full_numel": int(self._full_numel),
            "estimated_second_moment_saved_mb": round(saved_second_moment_mb, 3),
            "status": "active",
        }
