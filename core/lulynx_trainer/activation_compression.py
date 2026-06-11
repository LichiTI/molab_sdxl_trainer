# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""GPU-local saved-activation compression prototype."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ActivationCompressionStats:
    packed_tensors: int = 0
    restored_tensors: int = 0
    original_bytes: int = 0
    stored_bytes: int = 0
    skipped_tensors: int = 0

    def as_dict(self) -> dict[str, Any]:
        saved = max(self.original_bytes - self.stored_bytes, 0)
        return {
            "packed_tensors": int(self.packed_tensors),
            "restored_tensors": int(self.restored_tensors),
            "skipped_tensors": int(self.skipped_tensors),
            "original_mb": round(self.original_bytes / (1024 * 1024), 3),
            "stored_mb": round(self.stored_bytes / (1024 * 1024), 3),
            "estimated_saved_mb": round(saved / (1024 * 1024), 3),
        }


class _PackedActivation:
    def __init__(self, payload: torch.Tensor, original_dtype: torch.dtype, original_device: torch.device) -> None:
        self.payload = payload
        self.original_dtype = original_dtype
        self.original_device = original_device


class ActivationCompressionContext:
    """Autograd saved-tensor hook that stores eligible activations in lower precision."""

    def __init__(
        self,
        enabled: bool = False,
        storage_dtype: str = "fp16",
        min_tensor_bytes: int = 1 << 20,
    ) -> None:
        self.enabled = bool(enabled)
        self.storage_dtype_name = str(storage_dtype or "fp16").strip().lower()
        self.min_tensor_bytes = max(int(min_tensor_bytes), 0)
        self.stats = ActivationCompressionStats()

    def _target_dtype(self) -> torch.dtype | None:
        if self.storage_dtype_name in {"fp16", "float16", "half"}:
            return torch.float16
        if self.storage_dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if self.storage_dtype_name in {"fp8", "float8", "fp8_e4m3"} and hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
        return None

    @staticmethod
    def _nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _pack(self, tensor: torch.Tensor):
        target_dtype = self._target_dtype()
        original_bytes = self._nbytes(tensor)
        if (
            target_dtype is None
            or not tensor.is_floating_point()
            or original_bytes < self.min_tensor_bytes
            or tensor.dtype == target_dtype
        ):
            self.stats.skipped_tensors += 1
            return tensor
        try:
            payload = tensor.detach().to(dtype=target_dtype)
        except RuntimeError:
            self.stats.skipped_tensors += 1
            return tensor
        self.stats.packed_tensors += 1
        self.stats.original_bytes += original_bytes
        self.stats.stored_bytes += self._nbytes(payload)
        return _PackedActivation(payload, tensor.dtype, tensor.device)

    def _unpack(self, value):
        if not isinstance(value, _PackedActivation):
            return value
        self.stats.restored_tensors += 1
        return value.payload.to(device=value.original_device, dtype=value.original_dtype)

    def context(self):
        if not self.enabled:
            return nullcontext()
        hooks = getattr(torch.autograd.graph, "saved_tensors_hooks", None)
        if hooks is None:
            return nullcontext()
        return hooks(self._pack, self._unpack)

    def as_dict(self) -> dict[str, Any]:
        payload = self.stats.as_dict()
        payload.update(
            {
                "enabled": bool(self.enabled),
                "storage_dtype": self.storage_dtype_name,
                "min_tensor_bytes": int(self.min_tensor_bytes),
                "status": "active" if self.enabled else "disabled",
            }
        )
        return payload

