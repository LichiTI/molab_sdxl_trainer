# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Small optimizer wrappers used by optional plugin bridge routes."""

from __future__ import annotations

from typing import Any

import torch


class LossValueClosureOptimizer(torch.optim.Optimizer):
    """Adapter for optimizers whose closure only needs the current loss value."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._base = optimizer
        super().__init__([param for group in optimizer.param_groups for param in group.get("params", [])], defaults={})
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self._lulynx_loss_value_for_step: float | torch.Tensor | None = None

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self._base.zero_grad(set_to_none=set_to_none)
        self.state = self._base.state

    def step(self, closure=None):  # type: ignore[override]
        loss_value = closure() if closure is not None else self._lulynx_loss_value_for_step
        if loss_value is None:
            raise RuntimeError("AliG requires the current loss value before optimizer.step().")
        if isinstance(loss_value, torch.Tensor):
            loss_value = float(loss_value.detach().float().item())
        result = self._base.step(lambda: float(loss_value))
        self.param_groups = self._base.param_groups
        self.state = self._base.state
        self._lulynx_loss_value_for_step = None
        return result

    def state_dict(self):  # type: ignore[override]
        return self._base.state_dict()

    def load_state_dict(self, state_dict):  # type: ignore[override]
        result = self._base.load_state_dict(state_dict)
        self.param_groups = self._base.param_groups
        self.state = self._base.state
        return result


class FusedBackwardOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer, *, name: str) -> None:
        self._base = optimizer
        self._lulynx_fused_backward_name = str(name).strip().lower()
        super().__init__([param for group in optimizer.param_groups for param in group.get("params", [])], defaults={})
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self._lulynx_uses_fused_backward = True

    def _lulynx_fused_backward(self, loss: torch.Tensor, lr: float) -> None:
        self._base.fused_backward(loss, lr)
        self.param_groups = self._base.param_groups
        self.state = self._base.state

    def step(self, closure=None):  # type: ignore[override]
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self._base.zero_grad(set_to_none=set_to_none)
        self.state = self._base.state

    def state_dict(self):  # type: ignore[override]
        state_dict = self._base.state_dict()
        if self._lulynx_fused_backward_name == "adalomo":
            state_dict["lulynx_adalomo_state"] = {
                "num_steps": int(getattr(self._base, "num_steps", 0)),
                "exp_avg_sq": _clone_named_state_tensors(getattr(self._base, "exp_avg_sq", {})),
                "exp_avg_sq_row": _clone_named_state_tensors(getattr(self._base, "exp_avg_sq_row", {})),
                "exp_avg_sq_col": _clone_named_state_tensors(getattr(self._base, "exp_avg_sq_col", {})),
            }
        return state_dict

    def load_state_dict(self, state_dict):  # type: ignore[override]
        extra = state_dict.get("lulynx_adalomo_state") if isinstance(state_dict, dict) else None
        base_state_dict = dict(state_dict)
        base_state_dict.pop("lulynx_adalomo_state", None)
        result = self._base.load_state_dict(base_state_dict)
        if isinstance(extra, dict) and self._lulynx_fused_backward_name == "adalomo":
            self._base.num_steps = int(extra.get("num_steps", getattr(self._base, "num_steps", 0)) or 0)
            _restore_named_state_tensors(self._base.exp_avg_sq, extra.get("exp_avg_sq", {}))
            _restore_named_state_tensors(self._base.exp_avg_sq_row, extra.get("exp_avg_sq_row", {}))
            _restore_named_state_tensors(self._base.exp_avg_sq_col, extra.get("exp_avg_sq_col", {}))
        self.param_groups = self._base.param_groups
        self.state = self._base.state
        return result


def _clone_named_state_tensors(state: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def _restore_named_state_tensors(target: dict[str, Any], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if key not in target:
            target[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
            continue
        current = target[key]
        if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
            current.copy_(value.to(device=current.device, dtype=current.dtype))
        else:
            target[key] = value


class ClosureRequiredOptimizer(torch.optim.Optimizer):
    """Adapter for optimizers that require optimizer.step(closure)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        name: str,
        requires_initial_backward: bool = False,
    ) -> None:
        self._base = optimizer
        self._lulynx_closure_optimizer_name = str(name).strip().lower()
        super().__init__([param for group in optimizer.param_groups for param in group.get("params", [])], defaults={})
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self._lulynx_requires_step_closure = True
        self._lulynx_step_closure_requires_initial_backward = bool(requires_initial_backward)
        self._lulynx_step_closure = None

    def step(self, closure=None):  # type: ignore[override]
        step_closure = closure if closure is not None else self._lulynx_step_closure
        if step_closure is None:
            raise RuntimeError(f"{self._lulynx_closure_optimizer_name} requires a bound optimizer step closure.")
        result = self._base.step(step_closure)
        self.param_groups = self._base.param_groups
        self.state = self._base.state
        self._lulynx_step_closure = None
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self._base.zero_grad(set_to_none=set_to_none)
        self.state = self._base.state

    def state_dict(self):  # type: ignore[override]
        return self._base.state_dict()

    def load_state_dict(self, state_dict):  # type: ignore[override]
        result = self._base.load_state_dict(state_dict)
        self.param_groups = self._base.param_groups
        self.state = self._base.state
        return result
