"""Runtime profile helpers for CPU/offloaded activation checkpointing."""

from __future__ import annotations

from typing import Any, Mapping


def _stats(ctx: Any) -> dict[str, Any]:
    if ctx is None:
        return {}
    value = getattr(ctx, "stats", {})
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def build_offloaded_checkpoint_runtime_profile(
    *,
    requested: bool,
    mode: str,
    pool_gb: float = 0.0,
    context: Any = None,
) -> dict[str, Any]:
    normalized_mode = str(mode or "standard").strip().lower().replace("-", "_")
    ctx_active = context is not None
    enabled = bool(requested)
    profile: dict[str, Any] = {
        "enabled": enabled,
        "requested": bool(requested),
        "source": "training_loop",
        "mode": normalized_mode,
        "pinned_async_active": bool(ctx_active),
        "pool_gb": float(pool_gb or 0.0),
        "stats": _stats(context),
    }
    if requested and normalized_mode == "pinned_async" and not ctx_active:
        profile["disabled_reason"] = "pinned_async_context_unavailable"
    elif not requested:
        profile["disabled_reason"] = "not_requested"
    return profile


__all__ = ["build_offloaded_checkpoint_runtime_profile"]
