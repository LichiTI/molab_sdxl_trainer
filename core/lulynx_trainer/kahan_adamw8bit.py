# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""KahanAdamW8bit — 8-bit AdamW optimizer with Kahan compensated summation.

Stores optimizer moment states (exp_avg, exp_avg_sq) in 8-bit quantized form
to halve optimizer state memory.  Uses Kahan compensated summation to maintain
numerical accuracy when accumulating small updates into bf16/fp16 parameters.

## Why Kahan summation?

With bf16 parameters and small learning rates (e.g. 1e-4), the magnitude of
the parameter update is often below the bf16 precision threshold.  Standard
``param.add_(update)`` silently drops these tiny updates.  Kahan summation
maintains a compensation buffer that tracks the accumulated rounding error,
ensuring no gradient information is lost.

## 8-bit quantization

Each moment tensor is stored as (uint8 data, float32 absmax per block).
Block-wise dynamic quantization maps the tensor's range to [0, 255].
Dequantization is performed lazily only when the moment is needed for the
parameter update.

Warehouse implementation — no external dependencies beyond PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["KahanAdamW8bit"]

_QBLOCK = 256


def _quantize_blockwise(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize a float tensor to uint8 with per-block absmax scaling.

    Returns (quantized_uint8, absmax_fp32) where absmax has one entry per
    block of ``_QBLOCK`` elements.
    """
    flat = tensor.reshape(-1).float()
    n = flat.numel()
    n_blocks = (n + _QBLOCK - 1) // _QBLOCK

    if n % _QBLOCK != 0:
        flat = torch.nn.functional.pad(flat, (0, _QBLOCK * n_blocks - n))

    blocks = flat.reshape(n_blocks, _QBLOCK)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)

    scaled = (blocks / absmax.unsqueeze(1)) * 127.0 + 128.0
    quantized = scaled.clamp(0, 255).to(torch.uint8).reshape(-1)[:n]

    return quantized, absmax


def _dequantize_blockwise(
    quantized: Tensor,
    absmax: Tensor,
    shape: torch.Size,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Dequantize uint8 tensor back to float using per-block absmax."""
    n = quantized.numel()
    n_blocks = absmax.numel()

    flat = quantized.float()
    if n % _QBLOCK != 0:
        flat = torch.nn.functional.pad(flat, (0, _QBLOCK * n_blocks - n))

    blocks = flat.reshape(n_blocks, _QBLOCK)
    deq = (blocks - 128.0) / 127.0 * absmax.unsqueeze(1)
    return deq.reshape(-1)[:n].reshape(shape).to(dtype)


class _QuantizedState:
    """Container for a quantized optimizer state tensor."""

    __slots__ = ("data", "absmax", "shape", "numel")

    def __init__(self, tensor: Tensor) -> None:
        self.shape = tensor.shape
        self.numel = tensor.numel()
        self.data, self.absmax = _quantize_blockwise(tensor.detach())

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        return _dequantize_blockwise(self.data, self.absmax, self.shape, dtype)

    def update(self, tensor: Tensor) -> None:
        self.data, self.absmax = _quantize_blockwise(tensor.detach())

    @property
    def memory_bytes(self) -> int:
        return self.data.nelement() + self.absmax.nelement() * 4


class KahanAdamW8bit(torch.optim.Optimizer):
    """8-bit AdamW optimizer with Kahan compensated summation.

    Moment states are stored in 8-bit quantized form (per-block dynamic
    quantization).  The Kahan compensation buffer is stored in fp32 to
    accumulate rounding errors that would otherwise be lost in bf16/fp16
    parameter updates.

    API is drop-in compatible with ``torch.optim.AdamW``.
    """

    def __init__(
        self,
        params: Iterable[Tensor | Dict[str, Any]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("KahanAdamW8bit does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_q"] = _QuantizedState(
                        torch.zeros_like(p, dtype=torch.float32)
                    )
                    state["exp_avg_sq_q"] = _QuantizedState(
                        torch.zeros_like(p, dtype=torch.float32)
                    )
                    state["kahan_comp"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg_q"].dequantize(torch.float32).to(p.device)
                exp_avg_sq = state["exp_avg_sq_q"].dequantize(torch.float32).to(p.device)
                kahan_comp = state["kahan_comp"]

                grad_fp32 = grad.float()

                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                state["exp_avg_q"].update(exp_avg)
                state["exp_avg_sq_q"].update(exp_avg_sq)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg / denom * (-step_size)

                if weight_decay != 0.0:
                    update.add_(p.data.float(), alpha=-lr * weight_decay)

                kahan_y = update - kahan_comp
                param_fp32 = p.data.float()
                kahan_t = param_fp32 + kahan_y
                kahan_comp.copy_((kahan_t - param_fp32) - kahan_y)
                p.data.copy_(kahan_t.to(p.dtype))

        return loss

    def estimate_state_memory_mb(self) -> float:
        """Estimate total optimizer state memory in MB."""
        total = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state is None:
                    continue
                if "exp_avg_q" in state:
                    total += state["exp_avg_q"].memory_bytes
                if "exp_avg_sq_q" in state:
                    total += state["exp_avg_sq_q"].memory_bytes
                if "kahan_comp" in state:
                    total += state["kahan_comp"].nelement() * state["kahan_comp"].element_size()
        return total / (1024 * 1024)

