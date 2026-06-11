"""Manual PCIe GPU cache v0 for CPU-pinned frozen Linear weights.

This module is intentionally conservative: it only caches already active
CPU-pinned modules, never trainable adapter weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class PcieCacheV0Report:
    enabled: bool
    mode: str
    reason: str
    budget_mb: float
    selected_count: int = 0
    skipped_count: int = 0
    cache_mb: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    error_count: int = 0
    selected: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "reason": self.reason,
            "budget_mb": round(float(self.budget_mb), 3),
            "selected_count": int(self.selected_count),
            "skipped_count": int(self.skipped_count),
            "cache_mb": round(float(self.cache_mb), 3),
            "hit_count": int(self.hit_count),
            "miss_count": int(self.miss_count),
            "error_count": int(self.error_count),
            "selected": list(self.selected),
        }


def _module_parameter_count(module: Any) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0
    try:
        return sum(int(param.numel()) for param in parameters(recurse=False))
    except Exception:
        return 0


def _transfer_stats(module: Any) -> dict[str, Any]:
    getter = getattr(module, "get_transfer_format_stats", None)
    if not callable(getter):
        return {}
    try:
        return dict(getter())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "gpu_cache_errors": 1}


def _cache_stats(module: Any) -> dict[str, Any]:
    getter = getattr(module, "get_cpu_pinned_gpu_cache_stats", None)
    if not callable(getter):
        return {}
    try:
        return dict(getter())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "errors": 1}


def _clear_module_cache(module: Any) -> None:
    clearer = getattr(module, "clear_cpu_pinned_gpu_cache", None)
    if callable(clearer):
        try:
            clearer()
        except Exception:
            pass


def apply_pcie_cache_v0(
    units: Iterable[tuple[str, Any, int] | tuple[str, Any, int, int]],
    *,
    enabled: bool,
    mode: str,
    budget_mb: float,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> PcieCacheV0Report:
    normalized_mode = str(mode or "observe").strip().lower()
    budget = max(float(budget_mb or 0.0), 0.0)
    rows: list[tuple[str, Any, int, int, float]] = []
    for unit in units:
        if len(unit) >= 4:
            name, module, parameter_count, block_index = unit[:4]
        else:
            name, module, parameter_count = unit[:3]
            block_index = -1
        _clear_module_cache(module)
        if not bool(getattr(module, "lulynx_weight_residency_active", False)):
            continue
        stats = _transfer_stats(module)
        errors = int(stats.get("pack_errors", 0) or 0) + int(stats.get("decode_errors", 0) or 0)
        if errors > 0:
            continue
        transfer_mb = float(stats.get("transfer_mb", 0.0) or 0.0)
        if transfer_mb <= 0:
            continue
        rows.append((str(name), module, int(parameter_count or _module_parameter_count(module)), int(block_index), transfer_mb))
    if not enabled or normalized_mode != "cache_v0":
        return PcieCacheV0Report(False, normalized_mode, "disabled", budget, skipped_count=len(rows))
    if budget <= 0:
        return PcieCacheV0Report(False, normalized_mode, "budget_zero", budget, skipped_count=len(rows))
    if device is None or torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return PcieCacheV0Report(False, normalized_mode, "requires_cuda", budget, skipped_count=len(rows))

    rows.sort(key=lambda item: (-item[4], -item[3], -item[2], item[0]))
    selected: list[dict[str, Any]] = []
    selected_modules: list[Any] = []
    used_mb = 0.0
    errors = 0
    for name, module, parameter_count, block_index, transfer_mb in rows:
        if used_mb + transfer_mb > budget:
            continue
        enabler = getattr(module, "enable_cpu_pinned_gpu_cache", None)
        if not callable(enabler):
            continue
        try:
            ok = bool(enabler(device=device, dtype=dtype))
        except Exception:
            ok = False
        stats = _cache_stats(module)
        if not ok or not bool(stats.get("enabled", False)):
            errors += int(stats.get("errors", 1) or 1)
            continue
        cache_mb = float(stats.get("cache_mb", transfer_mb) or 0.0)
        used_mb += cache_mb
        selected.append(
            {
                "name": name,
                "block_index": int(block_index),
                "parameter_count": int(parameter_count),
                "transfer_mb": round(float(transfer_mb), 3),
                "cache_mb": round(float(cache_mb), 3),
            }
        )
        selected_modules.append(module)
    if errors > 0:
        for module in selected_modules:
            _clear_module_cache(module)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return PcieCacheV0Report(
            False,
            normalized_mode,
            "cache_error_fallback",
            budget,
            skipped_count=len(rows),
            error_count=errors,
        )
    return collect_pcie_cache_v0_report(rows, mode=normalized_mode, budget_mb=budget, reason="active", selected=selected, extra_errors=errors)


def collect_pcie_cache_v0_report(
    units: Iterable[tuple[str, Any, int] | tuple[str, Any, int, int]],
    *,
    mode: str,
    budget_mb: float,
    reason: str = "active",
    selected: list[dict[str, Any]] | None = None,
    extra_errors: int = 0,
) -> PcieCacheV0Report:
    selected_rows = list(selected or [])
    selected_names = {str(row.get("name", "")) for row in selected_rows}
    cache_mb = 0.0
    hits = 0
    misses = 0
    errors = int(extra_errors or 0)
    eligible = 0
    for unit in units:
        if len(unit) >= 4:
            name, module, _parameter_count, _block_index = unit[:4]
        else:
            name, module, _parameter_count = unit[:3]
        if not bool(getattr(module, "lulynx_weight_residency_active", False)):
            continue
        eligible += 1
        stats = _cache_stats(module)
        if stats.get("enabled"):
            cache_mb += float(stats.get("cache_mb", 0.0) or 0.0)
        hits += int(stats.get("hits", 0) or 0)
        misses += int(stats.get("misses", 0) or 0)
        errors += int(stats.get("errors", 0) or 0)
        if stats.get("enabled") and str(name) not in selected_names:
            selected_rows.append({"name": str(name), "cache_mb": float(stats.get("cache_mb", 0.0) or 0.0)})
    return PcieCacheV0Report(
        enabled=len(selected_rows) > 0,
        mode=str(mode or "cache_v0"),
        reason=reason if selected_rows else "no_selected_cache",
        budget_mb=max(float(budget_mb or 0.0), 0.0),
        selected_count=len(selected_rows),
        skipped_count=max(eligible - len(selected_rows), 0),
        cache_mb=cache_mb,
        hit_count=hits,
        miss_count=misses,
        error_count=errors,
        selected=tuple(selected_rows[:24]),
    )
