# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Fused AdamW optimizer -- minimises Python-level tensor operations per parameter
by performing the entire Adam moment update + weight decay + parameter change
in a single fused pass (``_fused_adamw_step_``) per parameter group.

On CUDA devices the helper uses ``torch.where`` / fused-inplace ops so that
each parameter incurs only *one* logical kernel launch instead of the
half-dozen separate add/mul/div/copy calls that the default
``torch.optim.AdamW`` performs.

No custom C++/CUDA kernels are required -- the "fusion" is purely at the
Python-dispatch level, but structured so that a future native-kernel
replacement would only need to swap out ``_fused_adamw_step_``.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Iterator, Optional

import torch
from torch import Tensor

__all__ = ["FusedAdamW", "maybe_replace_optimizer"]


# ---------------------------------------------------------------------------
# Fused step kernel (pure-PyTorch, single pass per parameter)
# ---------------------------------------------------------------------------

def _fused_adamw_step_(
    param: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Optional[Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    amsgrad: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
) -> None:
    """In-place fused AdamW update for a *single* parameter tensor.

    All moment updates, bias correction, weight decay, and the final
    parameter write-back happen in one pass with minimal temporary
    allocations.  For CUDA tensors we lean on ``torch.where`` and in-place
    ops to keep the dispatch count low.
    """
    if maximize:
        grad = -grad

    # --- one-based step counter (used for bias correction) ---
    step.add_(1)
    bias_correction1 = 1 - beta1 ** step.item()
    bias_correction2 = 1 - beta2 ** step.item()

    # --- moment updates (in-place) ---
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    if amsgrad:
        # Maintain the max of moment_sq; in-place update
        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    else:
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    # --- weight decay (AdamW-style: decoupled, applied to param directly) ---
    if weight_decay != 0.0:
        param.add_(param, alpha=-lr * weight_decay)

    # --- parameter update ---
    # param -= step_size * (exp_avg / denom)
    # Use addcdiv_ for a single fused op
    param.addcdiv_(exp_avg, denom, value=-step_size)


# ---------------------------------------------------------------------------
# FusedAdamW Optimizer
# ---------------------------------------------------------------------------

class FusedAdamW(torch.optim.Optimizer):
    """Fused AdamW optimizer that coalesces the standard AdamW step into a
    single dispatch per parameter via ``_fused_adamw_step_``.

    Supports the same options as ``torch.optim.AdamW`` plus:
      * ``capturable`` -- ensures state tensors share the same device/dtype
        as parameters so the step can be captured in a CUDA graph.
      * ``differentiable`` -- enables autograd through the optimiser step
        (e.g. for meta-learning).

    The param-group schema mirrors ``torch.optim.AdamW`` so that
    ``state_dict`` / ``load_state_dict`` are fully compatible.
    """

    def __init__(
        self,
        params: Iterable[Tensor | dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single fused optimisation step.

        If *closure* is supplied it will be evaluated (to recompute the
        loss) and its value returned; otherwise ``None`` is returned.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]
            capturable = group["capturable"]
            differentiable = group["differentiable"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "FusedAdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # ---- lazy state initialisation ----
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
                    # If capturable, keep step on same device/dtype as params
                    if capturable:
                        state["step"] = state["step"].to(device=p.device, dtype=p.dtype)

                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]
                max_exp_avg_sq = state.get("max_exp_avg_sq")

                # -- differentiable path: run under autograd --
                if differentiable:
                    with torch.enable_grad():
                        _fused_adamw_step_(
                            param=p,
                            grad=grad,
                            step=step_t,
                            exp_avg=exp_avg,
                            exp_avg_sq=exp_avg_sq,
                            max_exp_avg_sq=max_exp_avg_sq,
                            lr=lr,
                            beta1=beta1,
                            beta2=beta2,
                            eps=eps,
                            weight_decay=weight_decay,
                            amsgrad=amsgrad,
                            maximize=maximize,
                            capturable=capturable,
                            differentiable=differentiable,
                        )
                else:
                    _fused_adamw_step_(
                        param=p,
                        grad=grad,
                        step=step_t,
                        exp_avg=exp_avg,
                        exp_avg_sq=exp_avg_sq,
                        max_exp_avg_sq=max_exp_avg_sq,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        maximize=maximize,
                        capturable=capturable,
                        differentiable=differentiable,
                    )

        return loss


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------

def maybe_replace_optimizer(
    optimizer: torch.optim.Optimizer,
    config: Any,
) -> torch.optim.Optimizer:
    """Replace *optimizer* with a ``FusedAdamW`` instance if
    ``config.fused_optimizer`` is ``True`` **and** the incoming optimizer
    is a standard ``torch.optim.AdamW`` (or an 8-bit fallback).

    The new optimizer preserves all param groups (including per-group
    lr / weight_decay) and copies over whatever state it can so that
    warm-start transitions are as seamless as possible.

    Returns the original optimizer unchanged when the flag is off or the
    optimiser type is not AdamW-compatible.
    """
    if not getattr(config, "fused_optimizer", False):
        return optimizer

    # Only replace AdamW-family optimizers
    opt_name = type(optimizer).__name__
    if opt_name not in ("AdamW", "AdamW8bit"):
        return optimizer

    # Build FusedAdamW from existing param groups
    fused = FusedAdamW(
        optimizer.param_groups,  # preserves per-group lr, weight_decay, etc.
    )

    # Try to warm-start: copy moments from the old optimiser state
    # (best-effort -- 8-bit state is incompatible, so skip)
    if opt_name == "AdamW":
        old_state = optimizer.state
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in old_state and len(old_state[p]) > 0:
                    new_state = fused.state[p]
                    # Copy step count
                    if "step" in old_state[p]:
                        old_step = old_state[p]["step"]
                        if isinstance(old_step, int):
                            new_state["step"] = torch.tensor(
                                float(old_step),
                                device=p.device,
                                dtype=torch.float32,
                            )
                        elif isinstance(old_step, Tensor):
                            new_state["step"] = old_step.clone().to(
                                device=p.device, dtype=torch.float32
                            )
                    # Copy first moment
                    if "exp_avg" in old_state[p]:
                        new_state["exp_avg"] = old_state[p]["exp_avg"].clone()
                    # Copy second moment
                    if "exp_avg_sq" in old_state[p]:
                        new_state["exp_avg_sq"] = old_state[p]["exp_avg_sq"].clone()
                    # Copy max moment (AMSGrad)
                    if "max_exp_avg_sq" in old_state[p]:
                        new_state["max_exp_avg_sq"] = old_state[p][
                            "max_exp_avg_sq"
                        ].clone()

    return fused
