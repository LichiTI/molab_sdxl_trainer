# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Filtered saved-activation CPU offload (default-off).

Reuses the pinned-pool + async-stream machinery from
``offloaded_checkpointing.OffloadedCheckpointContext`` but adds the two filters
that make it safe to combine with the native Anima/Newbie block-checkpointing
path (where the saved tensors are block-boundary activations, NOT weights):

1. parameter guard — module parameters/buffers saved for backward (e.g. frozen
   Linear weights) are never offloaded, avoiding a PCIe round-trip per layer.
2. min-size threshold — small tensors (norm stats, modulation vectors) stay on
   device; only large activations pay the H2D/D2H cost.

Unlike ``cpu_offload_checkpointing`` (forced off under the anima faithful
forward), this context is driven by its own ``activation_cpu_offload_enabled``
config flag and wraps the regular forward, so it composes with native block
checkpointing: with interval-N checkpointing the non-recomputed blocks keep
full activations — exactly the tensors this offload targets.

NOTE: saved_tensors_hooks do not nest (innermost wins), so when both this and
activation compression are enabled the offload takes precedence.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .offloaded_checkpointing import OffloadedCheckpointContext


@dataclass
class ActivationOffloadStats:
    offloaded_tensors: int = 0
    restored_tensors: int = 0
    offloaded_bytes: int = 0
    skipped_small: int = 0
    skipped_parameters: int = 0
    skipped_other: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "offloaded_tensors": int(self.offloaded_tensors),
            "restored_tensors": int(self.restored_tensors),
            "offloaded_mb": round(self.offloaded_bytes / (1024 * 1024), 3),
            "skipped_small": int(self.skipped_small),
            "skipped_parameters": int(self.skipped_parameters),
            "skipped_other": int(self.skipped_other),
        }


class _OffloadedPayload:
    """Tags inner-context payloads so unpack can tell them from raw tensors."""

    __slots__ = ("packed",)

    def __init__(self, packed: Any) -> None:
        self.packed = packed


class ActivationCpuOffloadContext:
    """Autograd saved-tensor hook that offloads large activations to pinned CPU.

    ``register_parameter_guard(module)`` should be called once with the trained
    module so its parameters/buffers are excluded from offloading.
    """

    def __init__(
        self,
        enabled: bool = False,
        min_tensor_bytes: int = 1 << 20,
        pool_gb: float = 1.0,
        device: str = "cuda",
    ) -> None:
        self.enabled = bool(enabled)
        self.min_tensor_bytes = max(int(min_tensor_bytes), 0)
        self.pool_gb = max(float(pool_gb or 0.0), 0.0)
        self.stats = ActivationOffloadStats()
        self._guard_ptrs: set[int] = set()
        self._inner: Optional[OffloadedCheckpointContext] = None
        if self.enabled:
            self._inner = OffloadedCheckpointContext(pool_gb=self.pool_gb or 1.0, device=device)

    def register_parameter_guard(self, module: Any) -> int:
        """Record data_ptrs of ``module``'s parameters/buffers; returns count."""
        if module is None:
            return 0
        count = 0
        try:
            for param in module.parameters():
                self._guard_ptrs.add(param.data_ptr())
                count += 1
            for buffer in module.buffers():
                self._guard_ptrs.add(buffer.data_ptr())
                count += 1
        except Exception:
            pass
        return count

    def _pack(self, tensor: torch.Tensor):
        if self._inner is None or not tensor.is_cuda or not tensor.is_floating_point():
            self.stats.skipped_other += 1
            return tensor
        if isinstance(tensor, torch.nn.Parameter) or tensor.data_ptr() in self._guard_ptrs:
            self.stats.skipped_parameters += 1
            return tensor
        nbytes = int(tensor.numel() * tensor.element_size())
        if nbytes < self.min_tensor_bytes:
            self.stats.skipped_small += 1
            return tensor
        try:
            packed = self._inner._pack_hook(tensor)
        except Exception:
            self.stats.skipped_other += 1
            return tensor
        self.stats.offloaded_tensors += 1
        self.stats.offloaded_bytes += nbytes
        return _OffloadedPayload(packed)

    def _unpack(self, value):
        if not isinstance(value, _OffloadedPayload):
            return value
        self.stats.restored_tensors += 1
        return self._inner._unpack_hook(value.packed)

    def context(self):
        """Per-forward context: resets the pinned pool, installs the hooks."""
        if not self.enabled or self._inner is None:
            return nullcontext()
        hooks = getattr(torch.autograd.graph, "saved_tensors_hooks", None)
        if hooks is None:
            return nullcontext()
        # The pool hands out slices monotonically; backward of the previous
        # step has completed by the next forward, so the slices are free again.
        self._inner._pool.reset()
        self._inner._saved_tensor_events.clear()
        return hooks(self._pack, self._unpack)

    def as_dict(self) -> dict[str, Any]:
        payload = self.stats.as_dict()
        payload.update(
            {
                "enabled": bool(self.enabled),
                "min_tensor_bytes": int(self.min_tensor_bytes),
                "pool_gb": float(self.pool_gb),
                "guarded_tensors": len(self._guard_ptrs),
                "status": "active" if self.enabled else "disabled",
            }
        )
        if self._inner is not None:
            payload["pool"] = dict(self._inner._pool.stats)
        return payload


__all__ = ["ActivationCpuOffloadContext", "ActivationOffloadStats"]
