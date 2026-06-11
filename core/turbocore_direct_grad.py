"""Direct-gradient binding prototype for TurboCore flat owners.

The binding is developer-only. It mirrors gradients produced by PyTorch
autograd into a ``PersistentFlatAdamW`` flat gradient buffer, while leaving the
normal ``param.grad`` path intact. It does not enable training dispatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class DirectGradBindingSnapshot:
    schema_version: int
    binding: str
    parameter_tensors: int
    total_numel: int
    hooks_installed: int
    active: bool
    accumulate: bool
    writes: int
    written_numel: int
    owner_grad_norm: float
    training_path_enabled: bool = False
    direct_grad_to_flat_owner: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TurboCoreDirectGradBinding:
    """Mirror autograd gradients into a persistent flat owner buffer."""

    def __init__(
        self,
        owner: Any,
        params: Iterable[torch.Tensor],
        *,
        accumulate: bool = True,
    ) -> None:
        self.owner = owner
        self.params = [param for param in params if isinstance(param, torch.Tensor)]
        if not self.params:
            raise ValueError("TurboCoreDirectGradBinding requires at least one tensor")
        self.accumulate = bool(accumulate)
        self._hooks: list[Any] = []
        self.active = True
        self._writes = 0
        self._written_numel = 0
        self._validate_layout()

    @property
    def installed(self) -> bool:
        return bool(self._hooks)

    def install(self) -> "TurboCoreDirectGradBinding":
        if self._hooks:
            return self
        for index, param in enumerate(self.params):
            if not bool(getattr(param, "requires_grad", False)):
                raise ValueError(f"Parameter {index} does not require gradients")
            offset = int(self.owner.layout.offsets[index])
            count = int(self.owner.layout.numels[index])
            self._hooks.append(param.register_hook(self._make_hook(offset, count)))
        return self

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def zero_owner_grad(self) -> None:
        self.owner.grad_flat.zero_()
        self._writes = 0
        self._written_numel = 0

    def set_active(self, active: bool) -> None:
        self.active = bool(active)

    def snapshot(self) -> dict[str, Any]:
        grad = getattr(self.owner, "grad_flat")
        norm = 0.0
        if isinstance(grad, torch.Tensor) and grad.numel() > 0:
            norm = float(torch.linalg.vector_norm(grad.detach().float(), ord=2).cpu().item())
        return DirectGradBindingSnapshot(
            schema_version=1,
            binding="turbocore_direct_grad_flat_owner_v0",
            parameter_tensors=len(self.params),
            total_numel=int(sum(param.numel() for param in self.params)),
            hooks_installed=len(self._hooks),
            active=bool(self.active),
            accumulate=bool(self.accumulate),
            writes=int(self._writes),
            written_numel=int(self._written_numel),
            owner_grad_norm=norm,
        ).as_dict()

    def _make_hook(self, offset: int, count: int):
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            if not self.active:
                return grad
            target = self.owner.grad_flat.narrow(0, int(offset), int(count))
            incoming = grad.detach().reshape(-1).to(device=target.device, dtype=target.dtype)
            with torch.no_grad():
                if self.accumulate:
                    target.add_(incoming)
                else:
                    target.copy_(incoming)
            self._writes += 1
            self._written_numel += int(count)
            return grad

        return _hook

    def _validate_layout(self) -> None:
        layout = getattr(self.owner, "layout", None)
        if layout is None:
            raise ValueError("Direct-grad binding requires a PersistentFlatAdamW-like owner")
        owner_numels = [int(item) for item in getattr(layout, "numels", ())]
        if len(owner_numels) != len(self.params):
            raise ValueError("Owner layout tensor count does not match parameters")
        for index, (param, numel) in enumerate(zip(self.params, owner_numels)):
            if int(param.numel()) != int(numel):
                raise ValueError(f"Parameter {index} numel does not match owner layout")
        devices = {str(param.device) for param in self.params}
        owner_device = str(getattr(self.owner, "grad_flat").device)
        if len(devices) != 1 or owner_device not in devices:
            raise ValueError("Direct-grad binding requires params and owner grad buffer on one device")


__all__ = ["DirectGradBindingSnapshot", "TurboCoreDirectGradBinding"]
