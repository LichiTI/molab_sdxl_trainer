# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Warehouse Automagic++ optimizer.

This file implements the optimizer from a product/algorithm specification, not
from third-party source code: factored second-moment preconditioning, local
per-element LR adaptation, and conservative update clipping.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Optional

import torch


class AutomagicPlusPlus(torch.optim.Optimizer):
    """Adafactor-like preconditioning with multiplicative per-element LR masks."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: Optional[float] = None,
        lr_up: float = 1.01,
        lr_down: float = 0.95,
        lr_adapt_mode: str = "multiplicative",
        eps: tuple[float, float] | float = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        beta2: float = 0.999,
        beta1: float = 0.0,
        weight_decay: float = 0.0,
        weight_decay_mode: str = "per_lr",
        max_update_rms_ratio: Optional[float] = 0.01,
        sign_eps: float = 0.0,
        lr_granularity: str = "standard",
        agreement_threshold: float = 0.6,
        do_parameter_swapping: bool = False,
        parameter_swapping_factor: float = 0.1,
        swap_interval: int = 0,
    ) -> None:
        lr = float(lr)
        if lr > 1e-3:
            lr = 1e-6
        min_lr = float(min_lr)
        max_lr = float(max_lr)
        if min_lr <= 0 or max_lr <= 0 or min_lr > max_lr:
            raise ValueError("Automagic++ requires 0 < min_lr <= max_lr.")
        if not 0.0 <= float(beta1) < 1.0:
            raise ValueError("Automagic++ beta1 must be in [0, 1).")
        if not 0.0 <= float(beta2) < 1.0:
            raise ValueError("Automagic++ beta2 must be in [0, 1).")

        adapt_mode = str(lr_adapt_mode or "multiplicative").strip().lower()
        if adapt_mode not in {"multiplicative", "additive"}:
            raise ValueError("Automagic++ lr_adapt_mode must be 'multiplicative' or 'additive'.")
        decay_mode = str(weight_decay_mode or "per_lr").strip().lower()
        if decay_mode not in {"per_lr", "adamw"}:
            raise ValueError("Automagic++ weight_decay_mode must be 'per_lr' or 'adamw'.")
        granularity = str(lr_granularity or "standard").strip().lower().replace("-", "_")
        granularity_aliases = {
            "standard": "standard",
            "element": "standard",
            "per_element": "standard",
            "low_overhead": "low_overhead",
            "low_cost": "low_overhead",
            "tensor": "low_overhead",
            "per_tensor": "low_overhead",
        }
        if granularity not in granularity_aliases:
            raise ValueError("Automagic++ lr_granularity must be 'standard' or 'low_overhead'.")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": float(clip_threshold),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = None if lr_bump is None else float(lr_bump)
        self.lr_up = float(lr_up)
        self.lr_down = float(lr_down)
        self.lr_adapt_mode = adapt_mode
        self.weight_decay_mode = decay_mode
        self.max_update_rms_ratio = (
            None if max_update_rms_ratio is None else float(max_update_rms_ratio)
        )
        self.sign_eps = float(sign_eps)
        self.lr_granularity = granularity_aliases[granularity]
        self.agreement_threshold = float(agreement_threshold)
        self.do_parameter_swapping = bool(do_parameter_swapping)
        self.parameter_swapping_factor = max(0.0, min(1.0, float(parameter_swapping_factor)))
        self.swap_interval = max(0, int(swap_interval or 0))
        self._swap_clock = 0

        self.base_lrs = [float(group["lr"]) for group in self.param_groups]
        self._total_numel = sum(int(p.numel()) for group in self.param_groups for p in group["params"])
        if self.do_parameter_swapping:
            self.rotate_trainable_parameters()

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / math.sqrt(max(tensor.numel(), 1))

    @staticmethod
    def _eps_values(value: tuple[float, float] | float) -> tuple[float, float]:
        if isinstance(value, (tuple, list)):
            if len(value) >= 2:
                return float(value[0]), float(value[1])
            if len(value) == 1:
                return float(value[0]), 1e-3
        return float(value), 1e-3

    @staticmethod
    def _factored_preconditioner(row_stat: torch.Tensor, col_stat: torch.Tensor) -> torch.Tensor:
        row_scale = row_stat / row_stat.mean(dim=-1, keepdim=True).clamp_min(1e-30)
        return row_scale.rsqrt().unsqueeze(-1) * col_stat.rsqrt().unsqueeze(-2)

    def rotate_trainable_parameters(self) -> None:
        """Optional tensor-level parameter rotation for experimentation."""

        params = [p for group in self.param_groups for p in group["params"]]
        for param in params:
            param.requires_grad_(False)
            param.grad = None

        random.shuffle(params)
        target = int(self._total_numel * self.parameter_swapping_factor)
        active = 0
        for param in params:
            if active > 0 and active >= target:
                break
            param.requires_grad_(True)
            active += int(param.numel())

    def _state_for(self, param: torch.nn.Parameter, group_lr: float) -> dict[str, Any]:
        state = self.state[param]
        if "local_lr" not in state:
            state["step"] = 0
            lr_shape = () if self.lr_granularity == "low_overhead" else param.shape
            state["local_lr"] = torch.full(
                lr_shape,
                float(group_lr),
                device=param.device,
                dtype=torch.float32,
            )
            state["prev_sign"] = torch.zeros(param.shape, device=param.device, dtype=torch.int8)
            state["has_prev_sign"] = False
            if param.ndim >= 2:
                state["row_var"] = torch.zeros(param.shape[:-1], device=param.device, dtype=torch.float32)
                state["col_var"] = torch.zeros(
                    param.shape[:-2] + param.shape[-1:],
                    device=param.device,
                    dtype=torch.float32,
                )
            else:
                state["full_var"] = torch.zeros(param.shape, device=param.device, dtype=torch.float32)
            state["avg_lr"] = state["local_lr"].mean()
        return state

    def _precondition(self, grad: torch.Tensor, state: dict[str, Any], beta2: float, eps1: float) -> torch.Tensor:
        squared = grad.square().add(eps1)
        if grad.ndim >= 2:
            row_var = state["row_var"].to(device=grad.device, dtype=torch.float32)
            col_var = state["col_var"].to(device=grad.device, dtype=torch.float32)
            row_var.mul_(beta2).add_(squared.mean(dim=-1), alpha=1.0 - beta2)
            col_var.mul_(beta2).add_(squared.mean(dim=-2), alpha=1.0 - beta2)
            state["row_var"] = row_var
            state["col_var"] = col_var
            return self._factored_preconditioner(row_var, col_var).mul(grad)

        full_var = state["full_var"].to(device=grad.device, dtype=torch.float32)
        full_var.mul_(beta2).add_(squared, alpha=1.0 - beta2)
        state["full_var"] = full_var
        return full_var.rsqrt().mul(grad)

    def _adapt_local_lr(
        self,
        update: torch.Tensor,
        state: dict[str, Any],
        group_lr: float,
    ) -> torch.Tensor:
        local_lr = state["local_lr"].to(device=update.device, dtype=torch.float32)
        prev_sign = state["prev_sign"].to(device=update.device, dtype=torch.int8)
        current_sign = torch.sign(update).to(torch.int8)
        if self.sign_eps > 0:
            current_sign = torch.where(update.abs() <= self.sign_eps, torch.zeros_like(current_sign), current_sign)

        if not bool(state["has_prev_sign"]):
            next_lr = local_lr
            state["has_prev_sign"] = True
        elif self.lr_granularity == "low_overhead":
            same = (current_sign == prev_sign) & (current_sign != 0) & (prev_sign != 0)
            agreement = same.to(torch.float32).mean()
            if self.lr_adapt_mode == "multiplicative":
                up_lr = local_lr * self.lr_up
                down_lr = local_lr * self.lr_down
            else:
                bump = float(group_lr if self.lr_bump is None else self.lr_bump)
                up_lr = local_lr + bump
                down_lr = local_lr - bump
            next_lr = torch.where(agreement >= self.agreement_threshold, up_lr, down_lr)
            next_lr = next_lr.clamp_(self.min_lr, self.max_lr)
        else:
            same = (current_sign == prev_sign) & (current_sign != 0)
            flipped = (current_sign != prev_sign) & (current_sign != 0) & (prev_sign != 0)
            if self.lr_adapt_mode == "multiplicative":
                next_lr = torch.where(same, local_lr * self.lr_up, local_lr)
                next_lr = torch.where(flipped, next_lr * self.lr_down, next_lr)
            else:
                bump = float(group_lr if self.lr_bump is None else self.lr_bump)
                next_lr = torch.where(same, local_lr + bump, local_lr)
                next_lr = torch.where(flipped, next_lr - bump, next_lr)
            next_lr = next_lr.clamp_(self.min_lr, self.max_lr)

        state["prev_sign"] = torch.where(current_sign != 0, current_sign, prev_sign)
        state["local_lr"] = next_lr.detach()
        state["avg_lr"] = next_lr.detach().mean()
        return next_lr

    def get_learning_rates(self) -> list[float]:
        result: list[float] = []
        for group in self.param_groups:
            values = []
            for param in group["params"]:
                avg_lr = self.state[param].get("avg_lr") if param in self.state else None
                if torch.is_tensor(avg_lr):
                    values.append(float(avg_lr.detach().cpu()))
            result.append(sum(values) / len(values) if values else float(group["lr"]))
        return result or list(self.base_lrs)

    def get_avg_learning_rate(self) -> float:
        rates = self.get_learning_rates()
        return sum(rates) / len(rates) if rates else float(self.base_lrs[0])

    @torch.no_grad()
    def step(self, closure: Any = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group_lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            eps1, eps2 = self._eps_values(group["eps"])

            for param in group["params"]:
                if param.grad is None or not param.requires_grad:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Automagic++ does not support sparse gradients.")

                state = self._state_for(param, group_lr)
                state["step"] += 1
                grad = param.grad.detach().to(dtype=torch.float32)
                update = self._precondition(grad, state, beta2, eps1)
                update.div_((self._rms(update) / float(group["clip_threshold"])).clamp_min(1.0))

                if beta1 > 0:
                    momentum = state.get("momentum")
                    if momentum is None:
                        momentum = torch.zeros_like(update, dtype=torch.float32)
                    momentum = momentum.to(device=update.device, dtype=torch.float32)
                    momentum.mul_(beta1).add_(update, alpha=1.0 - beta1)
                    state["momentum"] = momentum
                    update = momentum

                local_lr = self._adapt_local_lr(update, state, group_lr)
                update = update.mul(local_lr)

                param_fp32 = param.detach().to(dtype=torch.float32)
                if self.max_update_rms_ratio is not None:
                    max_update = self._rms(param_fp32).clamp_min(eps2) * self.max_update_rms_ratio
                    update.mul_((max_update / self._rms(update).clamp_min(1e-16)).clamp_max(1.0))

                weight_decay = float(group["weight_decay"])
                if weight_decay:
                    if self.weight_decay_mode == "adamw":
                        param_fp32.mul_(1.0 - weight_decay * float(state["avg_lr"].detach().cpu()))
                    else:
                        param_fp32.addcmul_(param_fp32, local_lr, value=-weight_decay)

                param_fp32.add_(update, alpha=-1.0)
                param.copy_(param_fp32.to(dtype=param.dtype))

        if self.do_parameter_swapping and self.swap_interval > 0:
            self._swap_clock += 1
            if self._swap_clock % self.swap_interval == 0:
                self.rotate_trainable_parameters()

        return loss

