"""Training loop hook integration.

Provides functions that the training loop calls at each step.  Uses a
snapshot JSON file + mtime caching for fast-path optimization: when no
plugins are active, only a single ``os.stat()`` call is made per event
check — zero JSON parsing, zero EventBus dispatch.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Snapshot cache ─────────────────────────────────────────────────────

_snapshot_path: Path | None = None
_cached_mtime: float = -1.0
_cached_events: dict[str, bool] = {}
_cached_active_ids: list[str] = []


def _get_snapshot_path() -> Path:
    global _snapshot_path
    if _snapshot_path is None:
        from .plugin_runtime import get_plugin_runtime
        rt = get_plugin_runtime()
        _snapshot_path = rt._snapshot_path
    return _snapshot_path


def _refresh_cache() -> None:
    """Refresh the cached snapshot if the file has changed."""
    global _cached_mtime, _cached_events, _cached_active_ids

    path = _get_snapshot_path()
    try:
        st = os.stat(path)
        mtime = st.st_mtime
    except OSError:
        # File doesn't exist — no plugins active
        _cached_mtime = -1.0
        _cached_events = {}
        _cached_active_ids = []
        return

    if mtime == _cached_mtime:
        return  # Cache hit

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cached_events = data.get("events", {})
        _cached_active_ids = data.get("active_plugin_ids", [])
        _cached_mtime = mtime
    except (json.JSONDecodeError, OSError):
        _cached_events = {}
        _cached_active_ids = []


def _has_active_handlers(event: str) -> bool:
    """Check if any plugin has registered handlers for *event*.

    Fast-path: returns False with a single ``os.stat()`` call when no
    plugins are active.
    """
    _refresh_cache()
    return _cached_events.get(event, False)


def _get_runtime():
    from .plugin_runtime import get_plugin_runtime
    return get_plugin_runtime()


# ── Training hook functions ────────────────────────────────────────────

def emit_before_forward_event(
    *,
    training_type: str | None,
    global_step: int,
    micro_batch_index: int,
    micro_batch_count: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    sync_gradients: bool,
    source: str = "training_loop",
) -> dict[str, Any] | None:
    """Emit a ``before_forward`` event.  Returns the dispatch report or None."""
    if not _has_active_handlers("before_forward"):
        return None

    from lulynx_plugin_core.protocol import build_before_forward_payload
    payload = build_before_forward_payload(
        training_type=training_type,
        global_step=global_step,
        micro_batch_index=micro_batch_index,
        micro_batch_count=micro_batch_count,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sync_gradients=sync_gradients,
        source=source,
    )
    return _get_runtime().emit_event("before_forward", payload)


def emit_after_loss_event(
    *,
    training_type: str | None,
    global_step: int,
    micro_batch_index: int,
    micro_batch_count: int,
    micro_batch_size: int,
    loss_value: float,
    loss_scale: float,
    weighted_loss: float,
    gradient_accumulation_steps: int,
    sync_gradients: bool,
    source: str = "training_loop",
) -> dict[str, Any] | None:
    """Emit an ``after_loss`` event.  Returns the dispatch report or None."""
    if not _has_active_handlers("after_loss"):
        return None

    from lulynx_plugin_core.protocol import build_after_loss_payload
    payload = build_after_loss_payload(
        training_type=training_type,
        global_step=global_step,
        micro_batch_index=micro_batch_index,
        micro_batch_count=micro_batch_count,
        micro_batch_size=micro_batch_size,
        loss_value=loss_value,
        loss_scale=loss_scale,
        weighted_loss=weighted_loss,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sync_gradients=sync_gradients,
        source=source,
    )
    return _get_runtime().emit_event("after_loss", payload)


def apply_modify_loss_event(
    *,
    training_type: str | None,
    global_step: int,
    micro_batch_index: int,
    micro_batch_count: int,
    micro_batch_size: int,
    loss_value: float,
    loss_scale: float,
    gradient_accumulation_steps: int,
    sync_gradients: bool,
    source: str = "training_loop",
) -> tuple[float, dict[str, Any] | None]:
    """Emit a ``modify_loss`` mutation event.

    Returns ``(modified_loss, report)``.  If no handlers are active,
    returns ``(loss_value, None)`` without dispatching.
    """
    if not _has_active_handlers("modify_loss"):
        return loss_value, None

    from lulynx_plugin_core.protocol import build_modify_loss_payload
    from lulynx_plugin_core.dispatch_protocol import apply_loss_mutation

    payload = build_modify_loss_payload(
        training_type=training_type,
        global_step=global_step,
        micro_batch_index=micro_batch_index,
        micro_batch_count=micro_batch_count,
        micro_batch_size=micro_batch_size,
        loss_value=loss_value,
        loss_scale=loss_scale,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sync_gradients=sync_gradients,
        source=source,
    )

    rt = _get_runtime()
    report = rt.emit_mutation_event("modify_loss", payload)

    # Apply mutation if handler modified the payload
    result_payload = report.get("result_payload", {})
    mutation = result_payload.get("mutation", {})
    scale = float(mutation.get("scale", 1.0))
    bias = float(mutation.get("bias", 0.0))

    outcome = apply_loss_mutation(loss_value, scale=scale, bias=bias)
    return outcome.final_loss, report


def emit_after_backward_event(
    *,
    training_type: str | None,
    global_step: int,
    micro_batch_index: int,
    micro_batch_count: int,
    micro_batch_size: int,
    loss_value: float,
    loss_scale: float,
    backward_loss: float,
    weighted_loss: float,
    gradient_accumulation_steps: int,
    sync_gradients: bool,
    source: str = "training_loop",
) -> dict[str, Any] | None:
    """Emit an ``after_backward`` event.  Returns the dispatch report or None."""
    if not _has_active_handlers("after_backward"):
        return None

    from lulynx_route_contract import classify_route
    contract = classify_route(training_type or "")

    payload = {
        "protocol_version": "tier2.training.v1",
        "event": "after_backward",
        "training_type": contract.training_type,
        "route_kind": contract.kind,
        "route_label": contract.label,
        "route_capabilities": list(contract.capability_tags),
        "source": source,
        "global_step": global_step,
        "gradient_accumulation_steps": max(1, gradient_accumulation_steps),
        "sync_gradients": sync_gradients,
        "micro_batch_index": max(1, micro_batch_index),
        "micro_batch_count": max(1, micro_batch_count),
        "micro_batch_size": max(1, micro_batch_size),
        "loss": loss_value,
        "loss_scale": loss_scale,
        "backward_loss": backward_loss,
        "weighted_loss": weighted_loss,
    }
    return _get_runtime().emit_event("after_backward", payload)


def emit_before_optimizer_step_event(
    *,
    training_type: str | None,
    global_step: int,
    current_loss: float,
    optimizer_type: str,
    scheduler_type: str,
    learning_rates: list[float],
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    sync_gradients: bool,
    source: str = "training_loop",
) -> dict[str, Any] | None:
    """Emit a ``before_optimizer_step`` event.  Returns the dispatch report or None."""
    if not _has_active_handlers("before_optimizer_step"):
        return None

    from lulynx_route_contract import classify_route
    contract = classify_route(training_type or "")

    payload = {
        "protocol_version": "tier2.training.v1",
        "event": "before_optimizer_step",
        "training_type": contract.training_type,
        "route_kind": contract.kind,
        "route_label": contract.label,
        "route_capabilities": list(contract.capability_tags),
        "source": source,
        "global_step": global_step,
        "gradient_accumulation_steps": max(1, gradient_accumulation_steps),
        "sync_gradients": sync_gradients,
        "current_loss": current_loss,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
        "learning_rates": learning_rates,
        "max_grad_norm": max_grad_norm,
    }
    return _get_runtime().emit_event("before_optimizer_step", payload)


def emit_after_optimizer_step_event(
    *,
    training_type: str | None,
    global_step: int,
    current_loss: float,
    optimizer_type: str,
    scheduler_type: str,
    learning_rates: list[float],
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    sync_gradients: bool,
    optimizer_step_executed: bool,
    scheduler_step_executed: bool,
    zero_grad_called: bool,
    source: str = "training_loop",
) -> dict[str, Any] | None:
    """Emit an ``after_optimizer_step`` event.  Returns the dispatch report or None."""
    if not _has_active_handlers("after_optimizer_step"):
        return None

    from lulynx_route_contract import classify_route
    contract = classify_route(training_type or "")

    payload = {
        "protocol_version": "tier2.training.v1",
        "event": "after_optimizer_step",
        "training_type": contract.training_type,
        "route_kind": contract.kind,
        "route_label": contract.label,
        "route_capabilities": list(contract.capability_tags),
        "source": source,
        "global_step": global_step,
        "gradient_accumulation_steps": max(1, gradient_accumulation_steps),
        "sync_gradients": sync_gradients,
        "current_loss": current_loss,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
        "learning_rates": learning_rates,
        "max_grad_norm": max_grad_norm,
        "optimizer_step_executed": optimizer_step_executed,
        "scheduler_step_executed": scheduler_step_executed,
        "zero_grad_called": zero_grad_called,
    }
    return _get_runtime().emit_event("after_optimizer_step", payload)
