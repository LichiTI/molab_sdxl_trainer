# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Opt-in ``torch.compile`` wrapping for arbitrary optimizer ``step`` methods.

Most third-party optimizers (the pytorch_optimizer plugin catalog in
particular) update parameters in a per-parameter Python loop, which costs
hundreds of micro kernel launches per step. Wrapping ``optimizer.step`` in
``torch.compile`` lets dynamo horizontally fuse those updates into a handful
of launches without touching the optimizer implementation.

Design points (absorbed launch-hygiene program, 2026-06-11):

* **Safe by construction** — the wrapper compiles lazily on the first step
  call; any exception permanently restores the original eager step and
  records the reason. Training never breaks because of this knob.
* **Tensor-lr recipe** — a compiled step guards on python-float param-group
  hyperparameters, so an LR scheduler that writes a new float every step
  would trigger a recompile storm straight into the cache limit. Converting
  ``group["lr"]`` to a 0-dim tensor up front keeps the guard stable; torch's
  ``LRScheduler`` updates tensor lrs in place via ``fill_``. A disposable
  probe instance first verifies the class actually accepts tensor lr —
  third-party foreach paths (e.g. pytorch_optimizer collecting lr into
  ``_foreach_addcdiv_`` scalar lists) may not, in which case lr stays float.
* **Skip guards** — backends whose step is already a single fused kernel or
  a cpp extension (bitsandbytes) gain nothing from compilation, and the
  mn-lora hijacker's step is host-logic heavy; both are skipped with a note.
"""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    from .dynamo_budget import pin_recompile_limit
except ImportError:  # pragma: no cover - direct script import fallback
    from core.lulynx_trainer.dynamo_budget import pin_recompile_limit


_MIN_RECOMPILE_LIMIT = 32

# Module substrings whose optimizers we refuse to compile (with reason).
_SKIP_MODULE_MARKERS = (
    ("bitsandbytes", "bitsandbytes step is a cpp extension; compile cannot fuse it"),
    ("mn_lora", "mn_lora hijacker step is host-logic heavy; compile would only graph-break"),
)


def _log_via(log: Optional[Callable[[str], None]], message: str) -> None:
    if log is not None:
        try:
            log(message)
        except Exception:
            pass


def _skip_reason_for(optimizer: Any) -> str:
    module = str(type(optimizer).__module__ or "").lower()
    for marker, reason in _SKIP_MODULE_MARKERS:
        if marker in module:
            return reason
    if not torch.cuda.is_available():
        return "CUDA unavailable; compiled step targets GPU launch overhead"
    if not hasattr(optimizer, "param_groups") or not hasattr(optimizer, "step"):
        return "object does not expose param_groups/step"
    return ""


def _probe_tensor_lr_support(optimizer: Any) -> bool:
    """Check on a disposable instance whether this optimizer class accepts a
    0-dim tensor ``lr`` end-to-end (construction + one step).

    The official compiled-optimizer recipe only guarantees tensor lr for
    ``torch.optim`` classes; third-party per-param/foreach implementations may
    collect lr into scalar lists (e.g. ``_foreach_addcdiv_`` scalars) and blow
    up. The probe never touches the real optimizer's state.
    """
    try:
        import inspect

        param = torch.zeros(2, 2, device="cuda", requires_grad=True)
        kwargs = dict(getattr(optimizer, "defaults", {}) or {})
        # defaults may carry base-class extras the ctor does not accept
        # (e.g. AdamW.defaults includes decoupled_weight_decay)
        sig_params = inspect.signature(type(optimizer).__init__).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
        kwargs["lr"] = torch.tensor(float(kwargs.get("lr", 1e-3)), dtype=torch.float32)
        probe = type(optimizer)([param], **kwargs)
        param.grad = torch.zeros_like(param)
        probe.step()
        return True
    except Exception:
        return False


def _tensorize_group_lrs(optimizer: Any) -> int:
    """Convert python-float ``lr`` entries to 0-dim tensors (official recipe).

    Returns how many groups were converted. torch's ``LRScheduler.step``
    updates tensor lrs with ``fill_`` so the tensor identity — and therefore
    the dynamo guard — survives schedule updates.
    """
    converted = 0
    for group in optimizer.param_groups:
        lr = group.get("lr")
        if isinstance(lr, float):
            group["lr"] = torch.tensor(lr, dtype=torch.float32)
            converted += 1
    return converted


def _restore_float_lrs(optimizer: Any) -> None:
    """Undo lr tensorization (keep the *current* scheduled value)."""
    for group in optimizer.param_groups:
        lr = group.get("lr")
        if isinstance(lr, torch.Tensor):
            group["lr"] = float(lr)


def wrap_optimizer_step_compiled(
    optimizer: Any,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Replace ``optimizer.step`` with a lazily-compiled, self-healing wrapper.

    Returns a report dict::

        {"wrapped": bool, "skipped_reason": str, "lr_groups_tensorized": int,
         "notes": [str, ...]}

    The instance attribute is replaced (not the class), so only this optimizer
    is affected. Calls that pass a ``closure`` bypass the compiled path. The
    first compiled call that raises restores the eager step permanently.
    """
    notes: List[str] = []
    skip = _skip_reason_for(optimizer)
    if skip:
        _log_via(log, f"[compiled_step] skipped: {skip}")
        return {"wrapped": False, "skipped_reason": skip, "lr_groups_tensorized": 0, "notes": [skip]}

    converted = 0
    if _probe_tensor_lr_support(optimizer):
        converted = _tensorize_group_lrs(optimizer)
        if converted:
            notes.append(f"converted lr to tensor in {converted} param group(s) (guard-stable under LR schedulers)")
    else:
        notes.append(
            "lr kept as python float: this optimizer class rejects tensor lr; "
            "an LR scheduler may trigger bounded recompiles, then dynamo falls back to eager"
        )

    effective_limit = pin_recompile_limit(_MIN_RECOMPILE_LIMIT, log=log)
    if effective_limit:
        notes.append(f"dynamo recompile limit pinned to {effective_limit}")

    eager_step = optimizer.step
    state: Dict[str, Any] = {"compiled": None, "failed": False}

    def _compiled_or_eager(_self: Any, *args: Any, **kwargs: Any) -> Any:
        # Closure-style optimizers (LBFGS and friends) re-run the model inside
        # step; compiling that would capture far more than the update math.
        if state["failed"] or args or kwargs.get("closure") is not None:
            return eager_step(*args, **kwargs)
        if state["compiled"] is None:
            try:
                state["compiled"] = torch.compile(eager_step)
            except Exception as exc:  # pragma: no cover - backend specific
                state["failed"] = True
                optimizer.step = eager_step
                _log_via(log, f"[compiled_step] compile unavailable, eager restored: {exc}")
                return eager_step(*args, **kwargs)
        try:
            return state["compiled"](**kwargs)
        except Exception as exc:
            state["failed"] = True
            optimizer.step = eager_step
            if converted:
                _restore_float_lrs(optimizer)
            _log_via(log, f"[compiled_step] compiled step failed, eager restored permanently: {exc}")
            return eager_step(*args, **kwargs)

    # Bind as a real method: torch's LRScheduler patches ``opt.step`` and
    # reads ``step_fn.__func__`` — a bare function would AttributeError there.
    optimizer.step = types.MethodType(_compiled_or_eager, optimizer)
    notes.append(f"step wrapped with torch.compile for {type(optimizer).__name__} (lazy compile, eager fallback)")
    _log_via(log, f"[compiled_step] {notes[-1]}")
    return {"wrapped": True, "skipped_reason": "", "lr_groups_tensorized": converted, "notes": notes}


__all__ = ["wrap_optimizer_step_compiled"]
