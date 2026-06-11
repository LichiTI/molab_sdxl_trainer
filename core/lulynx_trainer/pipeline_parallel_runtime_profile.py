"""Runtime profile helpers for pipeline-parallel training experiments."""

from __future__ import annotations

from typing import Any, Mapping


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _stats(manager: Any) -> dict[str, Any]:
    if manager is None:
        return {}
    value = getattr(manager, "stats", {})
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def build_pipeline_parallel_runtime_profile(
    *,
    requested: bool,
    chunks: int,
    split_points: str = "",
    available: bool = False,
    block_accessor: str = "",
    manager: Any = None,
    partition_stages: int = 0,
    disabled_reason: str = "",
    error: str = "",
) -> dict[str, Any]:
    stats = _stats(manager)
    active = bool(getattr(manager, "is_active", False)) if manager is not None else False
    if not disabled_reason and requested and not active:
        if not available:
            disabled_reason = "requires_at_least_two_cuda_gpus"
        elif _safe_int(partition_stages, 0) < 2:
            disabled_reason = "not_enough_pipeline_stages"
    profile: dict[str, Any] = {
        "enabled": active,
        "requested": bool(requested),
        "source": "training_loop",
        "chunks": max(_safe_int(chunks, 1), 1),
        "split_points": str(split_points or ""),
        "available": bool(available),
        "block_accessor": str(block_accessor or ""),
        "partition_stages": _safe_int(partition_stages, 0),
        "stats": stats,
    }
    if disabled_reason:
        profile["disabled_reason"] = str(disabled_reason)
    if error:
        profile["error"] = str(error)
        profile["enabled"] = False
    return profile


__all__ = ["build_pipeline_parallel_runtime_profile"]
