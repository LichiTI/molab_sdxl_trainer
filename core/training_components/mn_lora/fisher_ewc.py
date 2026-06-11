"""Fisher/EWC regularizer for MN-LoRA.

This controller implements Elastic Weight Consolidation as optimizer-local
gradient injection. After the main loss backward pass, it maintains a diagonal
Fisher proxy from squared gradients and adds:

    lambda * fisher_diag * (theta - theta0)

to trainable parameter gradients. Keeping this in the optimizer avoids invasive
changes to every training-loop loss path while producing the same first-order
effect as adding the EWC penalty to the loss.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import torch


class MNLoRAFisherEWCController:
    """Diagonal Fisher/EWC controller for trainable MN-LoRA parameters."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        lambda_ewc: float = 1e-4,
        fisher_beta: float = 0.95,
        fisher_floor: float = 1e-12,
        fisher_max: float = 1e4,
        start_step: int = 1,
        update_interval: int = 1,
        max_penalty_norm_ratio: float = 0.25,
        params: Optional[Iterable[torch.nn.Parameter]] = None,
        param_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.lambda_ewc = max(0.0, float(lambda_ewc))
        self.fisher_beta = max(0.0, min(0.9999, float(fisher_beta)))
        self.fisher_floor = max(0.0, float(fisher_floor))
        self.fisher_max = max(self.fisher_floor, float(fisher_max))
        self.start_step = max(1, int(start_step))
        self.update_interval = max(1, int(update_interval))
        self.max_penalty_norm_ratio = max(0.0, float(max_penalty_norm_ratio))
        self.param_names = dict(param_names or {})

        self.theta0: Dict[str, torch.Tensor] = {}
        self.fisher_diag: Dict[str, torch.Tensor] = {}
        self._registered = 0
        self._calls = 0
        self._fisher_updates = 0
        self._penalty_applications = 0
        self._penalty_norm_sum = 0.0
        self._penalty_norm_max = 0.0
        self._penalty_scale_min = 1.0
        self._last_penalty_loss = 0.0
        self._last_stats: Dict[str, Any] = {}

        if params:
            self.register_params(params, param_names=param_names)

    def _name_for(self, param: torch.nn.Parameter, fallback_index: int) -> str:
        return self.param_names.get(id(param), f"param_{fallback_index}")

    def register_params(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        param_names: Optional[Mapping[int, str]] = None,
    ) -> None:
        if param_names:
            self.param_names.update(dict(param_names))
        for idx, param in enumerate(params):
            if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
                continue
            name = self._name_for(param, idx)
            if name in self.theta0:
                continue
            self.theta0[name] = param.detach().float().cpu().clone()
            self._registered += 1

    def _iter_named_params(
        self,
        params: Iterable[torch.nn.Parameter],
    ) -> Iterable[tuple[str, torch.nn.Parameter]]:
        for idx, param in enumerate(params):
            if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
                continue
            name = self._name_for(param, idx)
            if name not in self.theta0:
                self.theta0[name] = param.detach().float().cpu().clone()
                self._registered += 1
            yield name, param

    def _update_fisher(self, name: str, grad: torch.Tensor) -> None:
        current = grad.detach().float().pow(2).clamp(self.fisher_floor, self.fisher_max).cpu()
        old = self.fisher_diag.get(name)
        if old is None or old.shape != current.shape:
            self.fisher_diag[name] = current
        else:
            self.fisher_diag[name] = old.mul(self.fisher_beta).add(current, alpha=1.0 - self.fisher_beta)
        self._fisher_updates += 1

    def build_penalty_grads(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        step: int,
        update_fisher: bool = True,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self._calls += 1
        stats: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "step": int(step),
            "registered_params": int(len(self.theta0)),
            "fisher_layers": int(len(self.fisher_diag)),
            "penalty_layers": 0,
            "penalty_norm": 0.0,
            "penalty_loss": 0.0,
            "scale": 1.0,
        }
        if not self.enabled or self.lambda_ewc <= 0 or int(step) < self.start_step:
            self._last_stats = stats
            return {}, stats

        penalty_grads: Dict[str, torch.Tensor] = {}
        penalty_norm_sq = 0.0
        penalty_loss = 0.0
        should_update = bool(update_fisher and int(step) % self.update_interval == 0)
        for name, param in self._iter_named_params(params):
            if param.grad is not None and should_update:
                self._update_fisher(name, param.grad)
            fisher = self.fisher_diag.get(name)
            theta0 = self.theta0.get(name)
            if fisher is None or theta0 is None:
                continue
            delta = param.detach().float().cpu() - theta0
            penalty_cpu = fisher * delta * self.lambda_ewc
            penalty_grads[name] = penalty_cpu.to(device=param.device, dtype=param.dtype)
            penalty_norm_sq += float(penalty_cpu.float().pow(2).sum().item())
            penalty_loss += float((0.5 * self.lambda_ewc * fisher * delta.pow(2)).sum().item())

        penalty_norm = penalty_norm_sq ** 0.5
        scale = 1.0
        if self.max_penalty_norm_ratio > 0 and penalty_grads:
            main_norm_sq = 0.0
            for _name, param in self._iter_named_params(params):
                if param.grad is not None:
                    main_norm_sq += float(param.grad.detach().float().pow(2).sum().item())
            main_norm = main_norm_sq ** 0.5
            limit = main_norm * self.max_penalty_norm_ratio
            if main_norm > 0 and penalty_norm > limit > 0:
                scale = limit / max(penalty_norm, 1e-16)
                for name in list(penalty_grads.keys()):
                    penalty_grads[name] = penalty_grads[name] * scale
                penalty_norm *= scale

        stats.update(
            {
                "registered_params": int(len(self.theta0)),
                "fisher_layers": int(len(self.fisher_diag)),
                "penalty_layers": int(len(penalty_grads)),
                "penalty_norm": float(penalty_norm),
                "penalty_loss": float(penalty_loss),
                "scale": float(scale),
            }
        )
        if penalty_grads and penalty_norm > 0:
            self._penalty_applications += 1
            self._penalty_norm_sum += float(penalty_norm)
            self._penalty_norm_max = max(self._penalty_norm_max, float(penalty_norm))
            self._penalty_scale_min = min(self._penalty_scale_min, float(scale))
        self._last_penalty_loss = float(penalty_loss)
        self._last_stats = stats
        return penalty_grads, stats

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "lambda_ewc": float(self.lambda_ewc),
            "fisher_beta": float(self.fisher_beta),
            "start_step": int(self.start_step),
            "update_interval": int(self.update_interval),
            "registered_params": int(len(self.theta0)),
            "fisher_layers": int(len(self.fisher_diag)),
            "calls": int(self._calls),
            "fisher_updates": int(self._fisher_updates),
            "penalty_applications": int(self._penalty_applications),
            "penalty_norm_avg": (
                float(self._penalty_norm_sum / self._penalty_applications)
                if self._penalty_applications
                else 0.0
            ),
            "penalty_norm_max": float(self._penalty_norm_max),
            "penalty_scale_min": float(self._penalty_scale_min if self._penalty_applications else 1.0),
            "last_penalty_loss": float(self._last_penalty_loss),
            "last": dict(self._last_stats),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lambda_ewc": self.lambda_ewc,
            "fisher_beta": self.fisher_beta,
            "fisher_floor": self.fisher_floor,
            "fisher_max": self.fisher_max,
            "start_step": self.start_step,
            "update_interval": self.update_interval,
            "max_penalty_norm_ratio": self.max_penalty_norm_ratio,
            "theta0": self.theta0,
            "fisher_diag": self.fisher_diag,
            "telemetry": self.get_telemetry_snapshot(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        self.lambda_ewc = float(state_dict.get("lambda_ewc", self.lambda_ewc))
        self.fisher_beta = float(state_dict.get("fisher_beta", self.fisher_beta))
        self.fisher_floor = float(state_dict.get("fisher_floor", self.fisher_floor))
        self.fisher_max = float(state_dict.get("fisher_max", self.fisher_max))
        self.start_step = int(state_dict.get("start_step", self.start_step))
        self.update_interval = int(state_dict.get("update_interval", self.update_interval))
        self.max_penalty_norm_ratio = float(
            state_dict.get("max_penalty_norm_ratio", self.max_penalty_norm_ratio)
        )
        theta0 = state_dict.get("theta0", {})
        fisher_diag = state_dict.get("fisher_diag", {})
        if isinstance(theta0, dict):
            self.theta0 = {str(k): v.detach().float().cpu().clone() for k, v in theta0.items() if isinstance(v, torch.Tensor)}
        if isinstance(fisher_diag, dict):
            self.fisher_diag = {
                str(k): v.detach().float().cpu().clone()
                for k, v in fisher_diag.items()
                if isinstance(v, torch.Tensor)
            }
