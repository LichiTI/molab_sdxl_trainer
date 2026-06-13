# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Fused AdamW optimizer -- batches the AdamW step across parameters with
``torch._foreach_*`` multi-tensor ops, so kernel-launch count per step is
O(buckets), not O(parameters).

Parameters are bucketed by ``(device, dtype, step_count)`` inside each param
group and each bucket runs the full moment update + weight decay + parameter
write-back as a short sequence of multi-tensor launches. Step counters live on
CPU (matching ``torch.optim`` convention for the non-capturable path), so bias
correction needs no GPU->CPU sync. ``capturable`` / ``differentiable`` groups
fall back to the per-parameter ``_fused_adamw_step_`` path, which stays
autograd- and graph-capture-friendly.

No custom C++/CUDA kernels are required -- a future native-kernel replacement
would only need to swap out the bucket update in ``_foreach_adamw_step_`` (or
``_fused_adamw_step_`` for the fallback path).
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


def _foreach_adamw_step_(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: Optional[list[Tensor]],
    new_step: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    amsgrad: bool,
    maximize: bool,
) -> None:
    """Multi-tensor AdamW update for one ``(device, dtype, step)`` bucket.

    Op-for-op the same element-wise sequence as ``_fused_adamw_step_`` (mul/
    add moments, sqrt/div/add denom, decoupled decay, addcdiv), so results
    match the per-parameter path bit-for-bit on deterministic kernels -- only
    the launch granularity changes.
    """
    bias_correction1 = 1 - beta1 ** new_step
    bias_correction2 = 1 - beta2 ** new_step

    if maximize:
        grads = torch._foreach_neg(grads)

    # --- moment updates (in-place) ---
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    if amsgrad:
        torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
        denom = torch._foreach_sqrt(max_exp_avg_sqs)
    else:
        denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, math.sqrt(bias_correction2))
    torch._foreach_add_(denom, eps)

    # --- weight decay (AdamW-style: decoupled, applied to param directly) ---
    if weight_decay != 0.0:
        torch._foreach_add_(params, params, alpha=-lr * weight_decay)

    # --- parameter update ---
    torch._foreach_addcdiv_(params, exp_avgs, denom, value=-(lr / bias_correction1))


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
        """Perform a single batched optimisation step.

        If *closure* is supplied it will be evaluated (to recompute the
        loss) and its value returned; otherwise ``None`` is returned.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["capturable"] or group["differentiable"]:
                # Graph-capture / autograd paths keep the per-parameter kernel.
                self._step_group_per_param(group)
                continue
            self._step_group_foreach(group)

        return loss

    def _init_param_state(self, p: Tensor, *, amsgrad: bool, capturable: bool) -> dict:
        state = self.state[p]
        if len(state) == 0:
            if capturable:
                # Step must live with the params so the update is graph-capturable.
                state["step"] = torch.tensor(0.0, device=p.device, dtype=p.dtype)
            else:
                # CPU step: bias correction reads it without a GPU sync.
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    def _step_group_foreach(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]
        maximize = group["maximize"]

        # (device, dtype, step_count) -> parallel tensor lists
        buckets: dict[tuple, tuple[list, list, list, list, list, list]] = {}
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("FusedAdamW does not support sparse gradients")
            state = self._init_param_state(p, amsgrad=amsgrad, capturable=False)
            step_t = state["step"]
            if step_t.device.type != "cpu":
                # Legacy / warm-started checkpoints kept step on the param
                # device; normalise once so later reads stay sync-free.
                step_t = step_t.detach().to(device="cpu", dtype=torch.float32)
                state["step"] = step_t
            key = (p.device, p.dtype, float(step_t.item()))
            bucket = buckets.setdefault(key, ([], [], [], [], [], []))
            bucket[0].append(p)
            bucket[1].append(p.grad)
            bucket[2].append(state["exp_avg"])
            bucket[3].append(state["exp_avg_sq"])
            if amsgrad:
                bucket[4].append(state["max_exp_avg_sq"])
            bucket[5].append(step_t)

        for (_device, _dtype, step_count), (params, grads, exp_avgs, exp_avg_sqs, max_sqs, steps) in buckets.items():
            for step_t in steps:
                step_t.add_(1)
            _foreach_adamw_step_(
                params=params,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                max_exp_avg_sqs=max_sqs if amsgrad else None,
                new_step=step_count + 1.0,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                maximize=maximize,
            )

    def _step_group_per_param(self, group: dict[str, Any]) -> None:
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

            state = self._init_param_state(p, amsgrad=amsgrad, capturable=capturable)

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
