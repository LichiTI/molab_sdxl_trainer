"""Route-aware DataLoader policy helpers.

The cached training routes have very different bottlenecks from raw image
loading.  Keep their worker/prefetch policy centralized so compile/static
routes can log and test the exact policy they use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _is_auto(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "auto"}


def _int_or_default(value: Any, default: int) -> int:
    if _is_auto(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _auto_worker_count(*, os_name: str, cpu_count: int) -> int:
    cpu_count = max(int(cpu_count or 1), 1)
    if os_name == "nt":
        # Windows multiprocessing startup is expensive and easier to trip with
        # large cache objects, so stay conservative.
        return 2 if cpu_count >= 8 else 1
    return max(1, min(4, cpu_count // 2))


@dataclass(frozen=True)
class CachedDataLoaderPolicy:
    route: str
    cached: bool
    auto_policy: bool
    num_workers: int
    persistent_workers: bool
    pin_memory: bool
    prefetch_factor: int | None
    reason: str

    def log_lines(self) -> Iterable[str]:
        prefetch = "none" if self.prefetch_factor is None else str(self.prefetch_factor)
        yield (
            "[dataloader-policy] "
            f"route={self.route} cached={'yes' if self.cached else 'no'} "
            f"workers={self.num_workers} prefetch_factor={prefetch} "
            f"pin_memory={'true' if self.pin_memory else 'false'} "
            f"persistent_workers={'true' if self.persistent_workers else 'false'} "
            f"reason=\"{self.reason}\""
        )


def resolve_cached_dataloader_policy(
    config: Any,
    *,
    route: str,
    cached: bool,
    cuda_available: bool,
    os_name: str | None = None,
    cpu_count: int | None = None,
) -> CachedDataLoaderPolicy:
    """Resolve DataLoader knobs for cached training routes.

    Existing explicit knobs remain respected when the auto policy is disabled.
    When enabled, the cached route gets conservative platform-aware defaults
    unless a dedicated cached_* override is supplied.
    """

    route_name = str(route or "unknown").strip().lower()
    auto_policy = _boolish(getattr(config, "cached_dataloader_auto_policy", True), default=True)
    base_workers = int(getattr(config, "dataloader_num_workers", 0) or 0)
    base_prefetch = int(getattr(config, "prefetch_factor", 2) or 2)
    base_pin = _boolish(getattr(config, "pin_memory", True), default=True)
    base_persistent = _boolish(getattr(config, "persistent_data_loader_workers", False), default=False)

    if not cached or not auto_policy:
        workers = max(base_workers, 0)
        return CachedDataLoaderPolicy(
            route=route_name,
            cached=bool(cached),
            auto_policy=False,
            num_workers=workers,
            persistent_workers=base_persistent if workers > 0 else False,
            pin_memory=base_pin,
            prefetch_factor=base_prefetch if workers > 0 else None,
            reason="explicit base DataLoader policy",
        )

    resolved_os = os.name if os_name is None else os_name
    resolved_cpu = os.cpu_count() if cpu_count is None else cpu_count
    auto_workers = _auto_worker_count(os_name=resolved_os, cpu_count=int(resolved_cpu or 1))

    workers = _int_or_default(getattr(config, "cached_dataloader_workers", "auto"), auto_workers)
    workers = max(workers, 0)

    auto_prefetch = 4 if workers > 2 else 2
    prefetch = _int_or_default(
        getattr(config, "cached_dataloader_prefetch_factor", "auto"),
        auto_prefetch,
    )
    if workers <= 0:
        prefetch_factor: int | None = None
    else:
        prefetch_factor = max(prefetch, 1)

    pin_override = getattr(config, "cached_dataloader_pin_memory", "auto")
    pin_memory = bool(cuda_available) if _is_auto(pin_override) else _boolish(pin_override, default=base_pin)
    persistent = workers > 0

    return CachedDataLoaderPolicy(
        route=route_name,
        cached=True,
        auto_policy=True,
        num_workers=workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        reason="auto cached CUDA policy" if cuda_available else "auto cached CPU policy",
    )
