"""Optional dataset adapter for LXCS cache sidecars.

This is P2 plumbing, kept conservative: callers must opt in, and any missing
or failed sidecar returns ``None`` so the existing cache loader can continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .lossless_cache_sidecar import load_lossless_cache_sidecar_arrays_from_file, sidecar_path_for_cache
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_cache_sidecar import load_lossless_cache_sidecar_arrays_from_file, sidecar_path_for_cache


@dataclass(frozen=True)
class LosslessCacheDatasetAdapterConfig:
    enabled: bool = False
    strict: bool = False
    sidecar_suffix: str = ".lxcs"


def load_lossless_cache_arrays_for_dataset(
    cache_path: str | Path,
    *,
    config: LosslessCacheDatasetAdapterConfig | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cfg = config or LosslessCacheDatasetAdapterConfig()
    path = Path(cache_path)
    report: dict[str, Any] = {
        "provider": "lxcs_dataset_adapter_v1",
        "enabled": bool(cfg.enabled),
        "source": str(path),
        "training_path_enabled": False,
        "fallback_to_raw_cache": True,
    }
    if not cfg.enabled:
        report["reason"] = "disabled"
        return None, report
    if path.suffix.lower() not in {".npz", ".npy"}:
        report["reason"] = "unsupported_cache_suffix"
        return None, report

    sidecar = sidecar_path_for_cache(path, suffix=cfg.sidecar_suffix)
    report["sidecar_path"] = str(sidecar)
    if not sidecar.is_file():
        report["reason"] = "sidecar_missing"
        if cfg.strict:
            raise FileNotFoundError(f"LXCS sidecar missing: {sidecar}")
        return None, report

    try:
        arrays = load_lossless_cache_sidecar_arrays_from_file(sidecar)
    except Exception as exc:
        report["reason"] = "sidecar_decode_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if cfg.strict:
            raise
        return None, report

    report.update(
        {
            "ok": True,
            "reason": "sidecar_loaded",
            "entry_count": len(arrays),
            "fallback_to_raw_cache": False,
            "training_path_enabled": True,
        }
    )
    return arrays, report


__all__ = [
    "LosslessCacheDatasetAdapterConfig",
    "load_lossless_cache_arrays_for_dataset",
]
