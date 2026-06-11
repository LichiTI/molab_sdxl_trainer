"""Shadow parity probes for LXCS cache sidecars.

P1 deliberately runs beside the trainer.  It compares original cache arrays
against LXCS-decoded arrays without changing dataset reads, so failures can be
reported as blockers instead of affecting training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .lossless_cache_sidecar import (
    decode_lossless_cache_sidecar_arrays,
    encode_lossless_cache_sidecar,
    load_numpy_cache_entries,
)
from .lossless_tensor_block import DEFAULT_FAST_CACHE_CODECS


@dataclass(frozen=True)
class CacheShadowParityOptions:
    chunk_size: int = 1 << 20
    codecs: tuple[str, ...] = DEFAULT_FAST_CACHE_CODECS
    min_saving: float = 0.02
    max_entries: int = 0


def _sha256_array(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _array_summary(name: str, array: Any) -> dict[str, Any]:
    return {
        "name": str(name),
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "nbytes": int(array.nbytes),
        "sha256": _sha256_array(array),
    }


def _compare_arrays(name: str, source: Any, decoded: Any) -> dict[str, Any]:
    import numpy as np

    source_summary = _array_summary(name, source)
    decoded_summary = _array_summary(name, decoded)
    dtype_match = str(source.dtype) == str(decoded.dtype)
    shape_match = tuple(source.shape) == tuple(decoded.shape)
    hash_match = source_summary["sha256"] == decoded_summary["sha256"]
    value_match = bool(dtype_match and shape_match and np.array_equal(source, decoded))
    ok = bool(dtype_match and shape_match and hash_match and value_match)
    return {
        "name": str(name),
        "ok": ok,
        "dtype_match": dtype_match,
        "shape_match": shape_match,
        "hash_match": hash_match,
        "value_match": value_match,
        "source": source_summary,
        "decoded": decoded_summary,
    }


def run_lossless_cache_shadow_parity(
    cache_path: str | Path,
    *,
    options: CacheShadowParityOptions | None = None,
) -> dict[str, Any]:
    """Build an in-memory LXCS sidecar and compare it to the source cache."""

    import numpy as np

    opts = options or CacheShadowParityOptions()
    path = Path(cache_path)
    entries = list(load_numpy_cache_entries(path))
    if opts.max_entries > 0:
        entries = entries[: opts.max_entries]
    if not entries:
        return {
            "ok": False,
            "source": str(path),
            "reason": "no_numpy_entries",
            "training_path_enabled": False,
        }

    sidecar = encode_lossless_cache_sidecar(
        entries,
        chunk_size=max(int(opts.chunk_size), 1),
        codecs=opts.codecs,
        min_saving=float(opts.min_saving),
    )
    decoded = decode_lossless_cache_sidecar_arrays(sidecar)
    rows: list[dict[str, Any]] = []
    missing_decoded: list[str] = []
    for entry in entries:
        name = str(entry.name)
        source = np.frombuffer(bytes(entry.data), dtype=np.dtype(str(entry.metadata["dtype"]))).reshape(
            tuple(entry.metadata["shape"])
        )
        if name not in decoded:
            missing_decoded.append(name)
            rows.append({"name": name, "ok": False, "reason": "missing_decoded_array"})
            continue
        rows.append(_compare_arrays(name, source, decoded[name]))

    ok = all(bool(row.get("ok")) for row in rows) and not missing_decoded
    raw_bytes = sum(int(row.get("source", {}).get("nbytes", 0)) for row in rows if isinstance(row.get("source"), dict))
    return {
        "ok": ok,
        "provider": "lxcs_shadow_parity_v1",
        "source": str(path),
        "entry_count": len(entries),
        "matched_entries": sum(1 for row in rows if row.get("ok")),
        "mismatch_count": sum(1 for row in rows if not row.get("ok")),
        "raw_bytes": raw_bytes,
        "sidecar_bytes": len(sidecar),
        "compression_ratio": round(len(sidecar) / max(float(raw_bytes), 1.0), 6),
        "missing_decoded": missing_decoded,
        "entries": rows,
        "training_path_enabled": False,
        "shadow_only": True,
    }


def run_lossless_cache_shadow_parity_matrix(
    paths: Iterable[str | Path],
    *,
    options: CacheShadowParityOptions | None = None,
) -> dict[str, Any]:
    reports = [run_lossless_cache_shadow_parity(path, options=options) for path in paths]
    ok_count = sum(1 for report in reports if report.get("ok"))
    raw_bytes = sum(int(report.get("raw_bytes") or 0) for report in reports)
    sidecar_bytes = sum(int(report.get("sidecar_bytes") or 0) for report in reports)
    return {
        "ok": ok_count == len(reports),
        "provider": "lxcs_shadow_parity_matrix_v1",
        "case_count": len(reports),
        "ok_count": ok_count,
        "mismatch_count": len(reports) - ok_count,
        "raw_bytes": raw_bytes,
        "sidecar_bytes": sidecar_bytes,
        "overall_ratio": round(sidecar_bytes / max(float(raw_bytes), 1.0), 6),
        "reports": reports,
        "training_path_enabled": False,
        "shadow_only": True,
    }


__all__ = [
    "CacheShadowParityOptions",
    "run_lossless_cache_shadow_parity",
    "run_lossless_cache_shadow_parity_matrix",
]
