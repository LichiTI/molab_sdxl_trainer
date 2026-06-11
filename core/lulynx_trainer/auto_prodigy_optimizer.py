# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Warehouse AutoProdigy optimizer.

This local optimizer combines a conservative global distance estimate with
Adam-style preconditioning, optional schedule-free averaging, and update RMS
caps.  It is intentionally independent from third-party Prodigy/Schedule-Free
implementations and keeps its state names local to this file.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


class AutoProdigy(torch.optim.Optimizer):
    """Conservative adaptive optimizer for experimental native training."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = 1.02,
        safeguard_warmup: bool = True,
        max_update_rms_ratio: float | None = 0.01,
        damping: float = 1.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("AutoProdigy requires lr > 0.")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or not 0 <= beta3 < 1:
            raise ValueError("AutoProdigy betas must be in [0, 1).")
        if d0 <= 0 or d_coef <= 0 or growth_rate < 1:
            raise ValueError("AutoProdigy requires d0 > 0, d_coef > 0, growth_rate >= 1.")
        defaults = {
            "lr": float(lr),
            "betas": (float(beta1), float(beta2)),
            "beta3": float(beta3),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "d0": float(d0),
            "d_coef": float(d_coef),
            "growth_rate": float(growth_rate),
            "safeguard_warmup": bool(safeguard_warmup),
            "max_update_rms_ratio": None if max_update_rms_ratio is None else float(max_update_rms_ratio),
            "damping": float(damping),
        }
        super().__init__(params, defaults)
        self._ap_global = {"distance": float(d0), "steps": 0}
        self._using_average_weights = False

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / math.sqrt(max(tensor.numel(), 1))

    def state_dict(self) -> dict[str, Any]:
        data = super().state_dict()
        data["auto_prodigy_global"] = dict(self._ap_global)
        data["auto_prodigy_eval_mode"] = bool(self._using_average_weights)
        return data

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        self._ap_global = dict(state_dict.pop("auto_prodigy_global", self._ap_global))
        self._using_average_weights = bool(state_dict.pop("auto_prodigy_eval_mode", False))
        super().load_state_dict(state_dict)

    def _state_for(self, param: torch.nn.Parameter) -> dict[str, Any]:
        state = self.state[param]
        if "tick" not in state:
            state["tick"] = 0
            state["m1"] = torch.zeros_like(param, dtype=torch.float32)
            state["m2"] = torch.zeros_like(param, dtype=torch.float32)
            state["origin"] = param.detach().clone().to(dtype=torch.float32)
            state["average"] = param.detach().clone().to(dtype=torch.float32)
        return state

    def train(self) -> "AutoProdigy":
        if self._using_average_weights:
            for group in self.param_groups:
                for param in group["params"]:
                    stash = self.state[param].pop("train_weight", None)
                    if torch.is_tensor(stash):
                        param.data.copy_(stash.to(device=param.device, dtype=param.dtype))
            self._using_average_weights = False
        return self

    def eval(self) -> "AutoProdigy":
        if not self._using_average_weights:
            for group in self.param_groups:
                for param in group["params"]:
                    state = self._state_for(param)
                    state["train_weight"] = param.detach().clone()
                    param.data.copy_(state["average"].to(device=param.device, dtype=param.dtype))
            self._using_average_weights = True
        return self

    @torch.no_grad()
    def step(self, closure: Any = None):
        if self._using_average_weights:
            self.train()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        estimate_num = 0.0
        estimate_den = 0.0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None or not param.requires_grad:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AutoProdigy does not support sparse gradients.")
                state = self._state_for(param)
                grad = param.grad.detach().to(dtype=torch.float32)
                origin = state["origin"].to(device=param.device, dtype=torch.float32)
                param_fp32 = param.detach().to(dtype=torch.float32)
                estimate_num += float((param_fp32 - origin).mul(grad).sum().abs().detach().cpu())
                estimate_den += float(grad.square().sum().detach().cpu())

        current_d = float(self._ap_global.get("distance", 1e-6))
        if estimate_den > 0:
            group0 = self.param_groups[0]
            d_coef = float(group0["d_coef"])
            candidate = d_coef * estimate_num / (math.sqrt(estimate_den) + float(group0["eps"]))
            if bool(group0["safeguard_warmup"]):
                candidate = max(candidate, current_d)
            capped = min(candidate, current_d * float(group0["growth_rate"]))
            if math.isfinite(capped):
                current_d = max(current_d, capped)
        self._ap_global["distance"] = current_d
        self._ap_global["steps"] = int(self._ap_global.get("steps", 0)) + 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta3 = float(group["beta3"])
            step_size = float(group["lr"]) * current_d / max(float(group["damping"]), 1e-12)
            for param in group["params"]:
                if param.grad is None or not param.requires_grad:
                    continue
                state = self._state_for(param)
                state["tick"] += 1
                grad = param.grad.detach().to(dtype=torch.float32)
                m1 = state["m1"].to(device=param.device, dtype=torch.float32)
                m2 = state["m2"].to(device=param.device, dtype=torch.float32)
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m2.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                state["m1"] = m1
                state["m2"] = m2

                bias1 = 1.0 - beta1 ** int(state["tick"])
                bias2 = 1.0 - beta2 ** int(state["tick"])
                update = (m1 / max(bias1, 1e-16)) / ((m2 / max(bias2, 1e-16)).sqrt() + float(group["eps"]))
                update.mul_(step_size)

                param_fp32 = param.detach().to(dtype=torch.float32)
                cap_ratio = group["max_update_rms_ratio"]
                if cap_ratio is not None:
                    param_rms = self._rms(param_fp32)
                    if float(param_rms.detach().cpu()) > float(group["eps"]):
                        cap = param_rms * float(cap_ratio)
                        update.mul_((cap / self._rms(update).clamp_min(1e-16)).clamp_max(1.0))

                if float(group["weight_decay"]):
                    param_fp32.mul_(1.0 - step_size * float(group["weight_decay"]))
                param_fp32.add_(update, alpha=-1.0)
                param.copy_(param_fp32.to(dtype=param.dtype))

                average = state["average"].to(device=param.device, dtype=torch.float32)
                average.mul_(beta3).add_(param.detach().to(dtype=torch.float32), alpha=1.0 - beta3)
                state["average"] = average

        return loss

