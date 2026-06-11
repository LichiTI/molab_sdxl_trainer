# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Gradient Release — reduce peak gradient memory by releasing each parameter's
gradient immediately after the optimizer processes it.

Two modes:
- **post_step**: After the standard optimizer.step(), iterate parameters and
  release each gradient individually.  Saves memory only when combined with
  custom per-param optimizers (e.g. FusedAdamW).  Simple and compatible.
- **during_backward**: Register ``register_post_accumulate_grad_hook`` on each
  trainable parameter.  A real per-parameter optimizer instance is created for
  each param (same class as the user's configured optimizer).  Hooks fire on
  every backward — each micro-batch's gradient is consumed and freed immediately.
  For gradient accumulation (gas > 1), betas are rescaled as beta^(1/gas) so
  effective momentum matches single-step training.
  Peak gradient memory = max(single param grad) instead of sum(all).
  Requires PyTorch >= 2.1.

Warehouse implementation using only public PyTorch APIs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import inspect

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "GradientReleaseManager",
    "is_gradient_release_available",
]


def is_gradient_release_available() -> bool:
    """Check if during_backward mode is supported (needs register_post_accumulate_grad_hook)."""
    return hasattr(torch.Tensor, "register_post_accumulate_grad_hook")


def _filter_constructor_args(opt_class: type, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Filter defaults to only keys accepted by the optimizer constructor."""
    try:
        sig = inspect.signature(opt_class.__init__)
        params = sig.parameters
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if has_var_keyword:
            return defaults
        valid_names = {
            name for name, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in ("self", "params")
        }
        return {k: v for k, v in defaults.items() if k in valid_names}
    except (ValueError, TypeError):
        return defaults


def _create_per_param_optimizer(
    opt_class: type,
    param: torch.nn.Parameter,
    defaults: Dict[str, Any],
) -> torch.optim.Optimizer:
    filtered = _filter_constructor_args(opt_class, defaults)
    try:
        return opt_class([param], **filtered)
    except TypeError:
        reduced = {k: v for k, v in filtered.items() if k != "weight_decay"}
        try:
            return opt_class([param], **reduced) if reduced else None
        except TypeError:
            pass
    raise RuntimeError(
        f"GradientRelease: cannot create per-parameter {opt_class.__name__}. "
        f"Use gradient_release_mode='post_step' for this optimizer."
    )


def _rescale_betas(defaults: Dict[str, Any], gas: int) -> Dict[str, Any]:
    """Adjust betas for per-micro-batch stepping: beta_adj = beta^(1/gas)."""
    if gas <= 1 or "betas" not in defaults:
        return defaults
    out = dict(defaults)
    beta1, beta2 = out["betas"]
    out["betas"] = (beta1 ** (1.0 / gas), beta2 ** (1.0 / gas))
    return out


class GradientReleaseManager:
    """Manage gradient release across the training loop.

    Usage::

        mgr = GradientReleaseManager(mode="during_backward")
        mgr.register_parameters(model.parameters(), optimizer)

        # In training loop — hooks fire automatically on every backward:
        loss.backward()
        # No need to call optimizer.step() or zero_grad — handled internally

    For ``post_step`` mode, call ``release_gradients_after_step()`` after
    the normal optimizer.step()/zero_grad() cycle.
    """

    def __init__(
        self,
        mode: str = "post_step",
        accumulation_steps: int = 1,
    ) -> None:
        if mode not in ("post_step", "during_backward"):
            raise ValueError(f"Unknown gradient_release_mode: {mode!r}")
        if mode == "during_backward" and not is_gradient_release_available():
            logger.warning(
                "during_backward mode requires PyTorch >= 2.1; falling back to post_step"
            )
            mode = "post_step"

        self.mode = mode
        self.accumulation_steps = max(accumulation_steps, 1)

        self._params: List[torch.nn.Parameter] = []
        self._hooks: List[Any] = []
        self._released_count = 0
        self._peak_grad_bytes = 0

        self._original_optimizer: Optional[torch.optim.Optimizer] = None
        self._per_param_optimizers: Dict[int, torch.optim.Optimizer] = {}

    def register_parameters(
        self,
        params,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Register trainable parameters and their optimizer."""
        self._original_optimizer = optimizer
        count = 0
        for p in params:
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                self._params.append(p)
                count += 1

        if self.mode == "during_backward":
            self._build_per_param_optimizers(optimizer)
            self._install_hooks()

        logger.info(
            "GradientRelease registered %d params in %s mode",
            count, self.mode,
        )
        return count

    def _build_per_param_optimizers(self, optimizer: torch.optim.Optimizer) -> None:
        opt_class = type(optimizer)
        defaults = _rescale_betas(dict(optimizer.defaults), self.accumulation_steps)

        param_ids = {id(p) for p in self._params}
        has_train = callable(getattr(opt_class, "train", None))

        for p in self._params:
            per_opt = _create_per_param_optimizer(opt_class, p, defaults)
            if has_train:
                per_opt.train()
            self._per_param_optimizers[id(p)] = per_opt

        created = len(self._per_param_optimizers)
        gas = self.accumulation_steps
        if gas > 1 and "betas" in defaults:
            logger.info(
                "GradientRelease: created %d per-param %s instances "
                "(betas rescaled for gas=%d: %s)",
                created, opt_class.__name__, gas, defaults["betas"],
            )
        else:
            logger.info(
                "GradientRelease: created %d per-param %s instances",
                created, opt_class.__name__,
            )

    def _install_hooks(self) -> None:
        """Install post-accumulate-grad hooks for during_backward mode."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        for p in self._params:
            hook = p.register_post_accumulate_grad_hook(
                self._make_release_hook(p)
            )
            self._hooks.append(hook)

    def _make_release_hook(self, param: torch.nn.Parameter) -> Callable:
        per_opt = self._per_param_optimizers.get(id(param))

        def _hook(p: torch.nn.Parameter) -> None:
            if p.grad is None or p.grad.is_sparse:
                return

            grad_bytes = p.grad.nelement() * p.grad.element_size()
            self._peak_grad_bytes = max(self._peak_grad_bytes, grad_bytes)

            if per_opt is not None:
                per_opt.step()
                per_opt.zero_grad(set_to_none=True)

            p.grad = None
            self._released_count += 1

        return _hook

    @contextmanager
    def step_context(self, is_accumulation_boundary: bool = True):
        """Context for a single micro-batch backward pass.

        Retained for backward compatibility. In during_backward mode hooks
        fire on every backward regardless of the boundary flag.
        """
        self._released_count = 0
        try:
            yield
        finally:
            pass

    def release_gradients_after_step(self) -> int:
        """Post-step mode: iterate all params and set grad=None one by one.

        Call this AFTER optimizer.step() — the optimizer has already consumed
        the gradients, so we can free them.  Returns the number of gradients
        released.
        """
        count = 0
        for p in self._params:
            if p.grad is not None:
                p.grad = None
                count += 1
        return count

    @property
    def needs_external_optimizer_step(self) -> bool:
        """Whether the caller should still call optimizer.step().

        In during_backward mode, per-param optimizers handle updates in hooks,
        so the standard optimizer.step() should be SKIPPED.
        In post_step mode, the standard step is still needed.
        """
        return self.mode == "post_step"

    def sync_learning_rate(self) -> None:
        """Propagate LR from the original optimizer to all per-param instances.

        Call this after lr_scheduler.step() so per-param optimizers pick up
        the updated learning rate.
        """
        if self._original_optimizer is None or self.mode != "during_backward":
            return
        for group in self._original_optimizer.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                per_opt = self._per_param_optimizers.get(id(p))
                if per_opt is not None:
                    for pg in per_opt.param_groups:
                        pg["lr"] = lr

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "param_count": len(self._params),
            "released_count": self._released_count,
            "peak_grad_mb": self._peak_grad_bytes / (1024 * 1024),
        }

    def cleanup(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._params.clear()
        self._per_param_optimizers.clear()
        self._original_optimizer = None

