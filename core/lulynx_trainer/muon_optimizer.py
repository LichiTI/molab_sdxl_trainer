# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Muon optimizer for 2D LoRA factors (cleanroom Lulynx implementation).

Muon = *MomentUm Orthogonalized by Newton-schulz*.  For a 2D weight matrix it
orthogonalizes the momentum-smoothed gradient with a quintic Newton-Schulz
iteration — driving the singular values of the *update* toward 1 — which
empirically reaches Adam-class convergence while storing only a single
momentum buffer (no second moment) for the Muon group.

LoRA exposes its trainable weights as two 2D factors per layer
(``lora_down`` ``[rank, in]`` and ``lora_up`` ``[out, rank]``), which is exactly
the shape Muon is designed for.  Every non-2D parameter (bias / norm / DoRA
magnitude / scalars) and any param explicitly flagged ``use_muon=False`` falls
back to a built-in decoupled AdamW so a single optimizer instance covers the
whole model.

Reimplemented from the published algorithm and its Newton-Schulz coefficients;
shares no source with any reference implementation.  Note: this v1 does not do
the distributed cross-rank update sharding of the original — it orthogonalizes
each 2D param locally, which is correct for single-GPU / DDP-replicated runs.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

# Quintic Newton-Schulz coefficients (Jordan et al.).  The polynomial
# p(x) = a*x + b*x^3 + c*x^5 has p(1)=1 and steep slope near 0, so iterating
# it on a spectrally-normalized matrix pushes every singular value to ~1.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def _zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Return the orthogonal polar factor of a 2D matrix via Newton-Schulz.

    The result has the same shape as ``grad`` and singular values ≈ 1.  Runs in
    bf16 on CUDA (fast, and the iteration is robust to the reduced precision)
    and in fp32 on CPU (bf16 matmul is flaky there).  Scale-invariant after the
    initial spectral normalization, so the caller controls magnitude via lr.
    """
    if grad.ndim != 2:
        raise ValueError(f"Newton-Schulz orthogonalization needs a 2D tensor, got {grad.ndim}D")
    work_dtype = torch.bfloat16 if grad.is_cuda else torch.float32
    x = grad.to(work_dtype)
    transposed = x.size(0) > x.size(1)
    if transposed:  # iterate on the wider orientation for fewer flops
        x = x.t()
    x = x / (x.norm() + 1e-7)
    for _ in range(max(1, int(steps))):
        gram = x @ x.t()
        poly = _NS_B * gram + _NS_C * (gram @ gram)
        x = _NS_A * x + poly @ x
    if transposed:
        x = x.t()
    return x


class Muon(Optimizer):
    """Hybrid Muon (2D) + decoupled AdamW (everything else) optimizer.

    Param groups carry a ``use_muon`` flag.  ``use_muon=True`` groups must hold
    2D tensors and are updated with momentum + Newton-Schulz orthogonalization;
    any non-2D tensor that slips into such a group is transparently routed to
    the AdamW path so the step never crashes.  Both paths use ``group['lr']``
    and decoupled weight decay, so the trainer sets per-group learning rates.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            use_muon=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon", True):
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    # ── Muon path (2D matrices) ──────────────────────────────────────────
    def _step_muon(self, group) -> None:
        momentum = float(group["momentum"])
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.ndim != 2:  # safety: non-2D in a muon group → adamw
                self._adamw_update(p, grad, group)
                continue
            state = self.state[p]
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = state["momentum_buffer"] = torch.zeros_like(grad)
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
            ortho = _zeropower_via_newtonschulz5(update, group["ns_steps"])
            # RMS-match scaling so the orthogonal update has Adam-like norm
            # regardless of the matrix aspect ratio.
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(ortho.to(p.dtype), alpha=-lr * scale)

    # ── AdamW fallback path (1D / scalars / use_muon=False) ──────────────
    def _step_adamw(self, group) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            self._adamw_update(p, p.grad, group)

    def _adamw_update(self, p, grad, group) -> None:
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] += 1
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        eps = float(group["eps"])
        wd = float(group["weight_decay"])
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias1 = 1.0 - beta1 ** state["step"]
        bias2 = 1.0 - beta2 ** state["step"]
        denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        p.addcdiv_(exp_avg, denom, value=-lr / bias1)


__all__ = ["Muon", "_zeropower_via_newtonschulz5"]
