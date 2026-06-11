# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Experimental optimizer-state paging wrapper.

The wrapper keeps the model weights on their normal device, but may park
optimizer state tensors on CPU between optimizer steps. It is deliberately
conservative: tensors are moved to the parameter device right before the base
optimizer step and returned to CPU immediately afterwards.
"""

from __future__ import annotations

from typing import Any

import torch


class OptimizerStatePagingWrapper(torch.optim.Optimizer):
    """Wrap a torch optimizer and page state tensors to CPU between steps."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        enabled: bool = True,
        min_tensor_bytes: int = 1 << 20,
        pin_memory: bool = False,
    ) -> None:
        self._base = base_optimizer
        self._initializing_optimizer = True
        torch.optim.Optimizer.__init__(self, base_optimizer.param_groups, base_optimizer.defaults)
        self._initializing_optimizer = False
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        self.enabled = bool(enabled)
        self.min_tensor_bytes = max(int(min_tensor_bytes), 0)
        self.pin_memory = bool(pin_memory)
        self._last_profile: dict[str, Any] = {
            "enabled": self.enabled,
            "status": "idle",
            "paged_tensors": 0,
            "paged_mb": 0.0,
        }

    def _param_device_map(self) -> dict[int, torch.device]:
        devices: dict[int, torch.device] = {}
        for group in self._base.param_groups:
            for param in group.get("params", []):
                if isinstance(param, torch.Tensor):
                    devices[id(param)] = param.device
        return devices

    @staticmethod
    def _tensor_bytes(value: torch.Tensor) -> int:
        return int(value.numel() * value.element_size())

    def _move_state(self, *, to_param_device: bool) -> tuple[int, int]:
        if not self.enabled:
            return 0, 0
        devices = self._param_device_map()
        moved = 0
        moved_bytes = 0
        for param, state in list(self._base.state.items()):
            if not isinstance(state, dict):
                continue
            target_device = devices.get(id(param), None)
            for key, value in list(state.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if self._tensor_bytes(value) < self.min_tensor_bytes:
                    continue
                if to_param_device:
                    if target_device is None or value.device == target_device:
                        continue
                    state[key] = value.to(device=target_device, non_blocking=True)
                else:
                    if value.device.type == "cpu":
                        continue
                    moved_value = value.detach().to(device="cpu", non_blocking=True)
                    if self.pin_memory and torch.cuda.is_available():
                        try:
                            moved_value = moved_value.pin_memory()
                        except RuntimeError:
                            pass
                    state[key] = moved_value
                moved += 1
                moved_bytes += self._tensor_bytes(value)
        return moved, moved_bytes

    def step(self, closure=None):
        to_device_count, to_device_bytes = self._move_state(to_param_device=True)
        loss = self._base.step(closure)
        to_cpu_count, to_cpu_bytes = self._move_state(to_param_device=False)
        self._last_profile = {
            "enabled": self.enabled,
            "status": "active" if self.enabled else "disabled",
            "to_param_device_tensors": int(to_device_count),
            "to_param_device_mb": round(to_device_bytes / (1024 * 1024), 3),
            "paged_tensors": int(to_cpu_count),
            "paged_mb": round(to_cpu_bytes / (1024 * 1024), 3),
            "min_tensor_bytes": int(self.min_tensor_bytes),
            "pin_memory": bool(self.pin_memory),
            "base_optimizer": type(self._base).__name__,
        }
        return loss

    def zero_grad(self, set_to_none: bool = True):
        return self._base.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        self._move_state(to_param_device=False)
        return self._base.state_dict()

    def load_state_dict(self, state_dict):
        result = self._base.load_state_dict(state_dict)
        self.param_groups = self._base.param_groups
        self.state = self._base.state
        self.defaults = self._base.defaults
        self._move_state(to_param_device=False)
        return result

    def add_param_group(self, param_group):
        if getattr(self, "_initializing_optimizer", False):
            return torch.optim.Optimizer.add_param_group(self, param_group)
        return self._base.add_param_group(param_group)

    def get_profile(self) -> dict[str, Any]:
        return dict(self._last_profile)


def maybe_wrap_optimizer_state_paging(
    optimizer: torch.optim.Optimizer,
    *,
    enabled: bool,
    min_tensor_mb: float = 1.0,
    pin_memory: bool = False,
) -> torch.optim.Optimizer:
    if not enabled:
        return optimizer
    return OptimizerStatePagingWrapper(
        optimizer,
        enabled=True,
        min_tensor_bytes=int(max(float(min_tensor_mb), 0.0) * 1024 * 1024),
        pin_memory=pin_memory,
    )

