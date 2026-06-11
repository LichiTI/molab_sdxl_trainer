# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Small optimizer-step contracts used by the native training loop."""

from __future__ import annotations

from typing import Any


_LOSS_VALUE_ATTR = "_lulynx_loss_value_for_step"
_REQUIRES_CREATE_GRAPH_ATTR = "_lulynx_requires_create_graph_backward"
_USES_FUSED_BACKWARD_ATTR = "_lulynx_uses_fused_backward"
_FUSED_BACKWARD_METHOD = "_lulynx_fused_backward"
_REQUIRES_STEP_CLOSURE_ATTR = "_lulynx_requires_step_closure"
_STEP_CLOSURE_REQUIRES_INITIAL_BACKWARD_ATTR = "_lulynx_step_closure_requires_initial_backward"
_STEP_CLOSURE_ATTR = "_lulynx_step_closure"


def bind_loss_value_closure(optimizer: Any, loss_value: Any) -> int:
    """Bind the current loss to optimizers that need loss-only closures.

    Some optimizers require ``optimizer.step(closure)`` but only use the closure
    to read the already-computed loss scalar. The normal Lulynx training loop
    has already run backward by optimizer-step time, so this contract exposes
    that scalar without re-running forward/backward.
    """
    return _bind_loss_value_closure(optimizer, loss_value, seen=set())


def optimizer_requires_create_graph_backward(optimizer: Any) -> bool:
    """Return whether any optimizer wrapper requires create_graph backward."""
    return _optimizer_requires_create_graph_backward(optimizer, seen=set())


def optimizer_uses_fused_backward(optimizer: Any) -> bool:
    """Return whether any optimizer wrapper consumes loss through fused_backward."""
    return _find_fused_backward_optimizer(optimizer, seen=set()) is not None


def run_optimizer_fused_backward(optimizer: Any, loss: Any, lr: float) -> bool:
    """Run a fused-backward optimizer contract if one exists."""
    target = _find_fused_backward_optimizer(optimizer, seen=set())
    if target is None:
        return False
    method = getattr(target, _FUSED_BACKWARD_METHOD, None)
    if not callable(method):
        raise RuntimeError("Fused-backward optimizer is missing its Lulynx contract method.")
    method(loss, lr)
    return True


def bind_step_closure(optimizer: Any, closure: Any) -> int:
    """Bind a recompute closure to optimizers that require step(closure)."""
    return _bind_step_closure(optimizer, closure, seen=set())


def optimizer_requires_step_closure(optimizer: Any) -> bool:
    """Return whether any optimizer wrapper requires optimizer.step(closure)."""
    return _optimizer_requires_step_closure(optimizer, seen=set())


def optimizer_step_closure_requires_initial_backward(optimizer: Any) -> bool:
    """Return whether a closure optimizer also needs the pre-step gradients."""
    return _optimizer_step_closure_requires_initial_backward(optimizer, seen=set())


def _bind_loss_value_closure(optimizer: Any, loss_value: Any, *, seen: set[int]) -> int:
    if optimizer is None:
        return 0
    key = id(optimizer)
    if key in seen:
        return 0
    seen.add(key)

    count = 0
    if hasattr(optimizer, _LOSS_VALUE_ATTR):
        setattr(optimizer, _LOSS_VALUE_ATTR, loss_value)
        count += 1

    for child_name in ("_base", "optimizer", "base_optimizer"):
        child = getattr(optimizer, child_name, None)
        if child is not None:
            count += _bind_loss_value_closure(child, loss_value, seen=seen)

    for child in getattr(optimizer, "optimizers", []) or []:
        count += _bind_loss_value_closure(child, loss_value, seen=seen)
    return count


def _optimizer_requires_create_graph_backward(optimizer: Any, *, seen: set[int]) -> bool:
    if optimizer is None:
        return False
    key = id(optimizer)
    if key in seen:
        return False
    seen.add(key)

    if bool(getattr(optimizer, _REQUIRES_CREATE_GRAPH_ATTR, False)):
        return True

    for child_name in ("_base", "optimizer", "base_optimizer"):
        child = getattr(optimizer, child_name, None)
        if child is not None and _optimizer_requires_create_graph_backward(child, seen=seen):
            return True

    return any(_optimizer_requires_create_graph_backward(child, seen=seen) for child in getattr(optimizer, "optimizers", []) or [])


def _find_fused_backward_optimizer(optimizer: Any, *, seen: set[int]) -> Any:
    if optimizer is None:
        return None
    key = id(optimizer)
    if key in seen:
        return None
    seen.add(key)

    if bool(getattr(optimizer, _USES_FUSED_BACKWARD_ATTR, False)):
        return optimizer

    for child_name in ("_base", "optimizer", "base_optimizer"):
        child = getattr(optimizer, child_name, None)
        found = _find_fused_backward_optimizer(child, seen=seen)
        if found is not None:
            return found

    for child in getattr(optimizer, "optimizers", []) or []:
        found = _find_fused_backward_optimizer(child, seen=seen)
        if found is not None:
            return found
    return None


def _bind_step_closure(optimizer: Any, closure: Any, *, seen: set[int]) -> int:
    if optimizer is None:
        return 0
    key = id(optimizer)
    if key in seen:
        return 0
    seen.add(key)

    count = 0
    if bool(getattr(optimizer, _REQUIRES_STEP_CLOSURE_ATTR, False)):
        setattr(optimizer, _STEP_CLOSURE_ATTR, closure)
        count += 1

    for child_name in ("_base", "optimizer", "base_optimizer"):
        child = getattr(optimizer, child_name, None)
        if child is not None:
            count += _bind_step_closure(child, closure, seen=seen)
    for child in getattr(optimizer, "optimizers", []) or []:
        count += _bind_step_closure(child, closure, seen=seen)
    return count


def _optimizer_requires_step_closure(optimizer: Any, *, seen: set[int]) -> bool:
    if optimizer is None:
        return False
    key = id(optimizer)
    if key in seen:
        return False
    seen.add(key)

    if bool(getattr(optimizer, _REQUIRES_STEP_CLOSURE_ATTR, False)):
        return True
    for child_name in ("_base", "optimizer", "base_optimizer"):
        child = getattr(optimizer, child_name, None)
        if child is not None and _optimizer_requires_step_closure(child, seen=seen):
            return True
    return any(_optimizer_requires_step_closure(child, seen=seen) for child in getattr(optimizer, "optimizers", []) or [])


def _optimizer_step_closure_requires_initial_backward(optimizer: Any, *, seen: set[int]) -> bool:
    if optimizer is None:
        return False
    key = id(optimizer)
    if key in seen:
        return False
    seen.add(key)

    if bool(getattr(optimizer, _STEP_CLOSURE_REQUIRES_INITIAL_BACKWARD_ATTR, False)):
        return True
    for child_name in ("_base", "optimizer", "base_optimizer"):
        child = getattr(optimizer, child_name, None)
        if child is not None and _optimizer_step_closure_requires_initial_backward(child, seen=seen):
            return True
    return any(_optimizer_step_closure_requires_initial_backward(child, seen=seen) for child in getattr(optimizer, "optimizers", []) or [])
