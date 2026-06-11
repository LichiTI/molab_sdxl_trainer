"""Sampled payload parity checks for image GGUF exports.

This module compares exported GGUF tensor payloads against the source
safetensors files. It is intentionally bounded: by default it checks a small,
deterministic tensor sample and a prefix of each tensor payload. It does not
instantiate models, run inference, or change runtime loadability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_MAX_TENSORS = 8
DEFAULT_MAX_ELEMENTS_PER_TENSOR = 4096


def check_image_gguf_payload_parity(
    source_paths: str | Path | Iterable[str | Path],
    gguf_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    max_elements_per_tensor: int = DEFAULT_MAX_ELEMENTS_PER_TENSOR,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Return a bounded parity report for an image GGUF export."""

    sources = _normalize_sources(source_paths)
    path = Path(gguf_path)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")
    try:
        import gguf
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("image GGUF payload parity requires gguf and safetensors") from exc

    gguf_reader = gguf.GGUFReader(str(path))
    gguf_tensors = {str(tensor.name): tensor for tensor in gguf_reader.tensors}
    sidecar = _read_sidecar(path, sidecar_path)
    selected_names = _select_tensor_names(sources, gguf_tensors, safe_open, max_tensors=max_tensors)
    records: list[dict[str, Any]] = []
    source_tensor_count = 0
    duplicate_source_tensors = 0
    seen: set[str] = set()

    for source in sources:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                source_tensor_count += 1
                if name in seen:
                    duplicate_source_tensors += 1
                    continue
                seen.add(str(name))
                if name not in selected_names:
                    continue
                records.append(
                    _compare_tensor(
                        source,
                        str(name),
                        handle,
                        gguf_tensors.get(str(name)),
                        max_elements=max_elements_per_tensor,
                        atol=atol,
                        rtol=rtol,
                    )
                )

    missing = [name for name in selected_names if name not in {str(item.get("name")) for item in records}]
    for name in missing:
        records.append({"name": name, "ok": False, "issue": "selected tensor missing from source payload"})
    failed = [item for item in records if not item.get("ok")]
    max_abs_error = max((float(item.get("max_abs_error") or 0.0) for item in records), default=0.0)
    max_rel_error = max((float(item.get("max_rel_error") or 0.0) for item in records), default=0.0)
    return {
        "schema_version": 1,
        "checker": "image_gguf_payload_parity_v1",
        "ok": not failed and bool(records),
        "report_only": True,
        "reads_tensor_payloads": True,
        "builds_model_modules": False,
        "runs_forward_pass": False,
        "runtime_loadable_enabled": False,
        "training_path_enabled": False,
        "source_paths": [str(source) for source in sources],
        "gguf_path": str(path),
        "sidecar_path": str(sidecar.get("path") or ""),
        "sidecar_present": bool(sidecar.get("path")),
        "source_tensor_count": source_tensor_count,
        "gguf_tensor_count": len(gguf_tensors),
        "duplicate_source_tensors": duplicate_source_tensors,
        "sampled_tensor_count": len(records),
        "max_tensors": int(max_tensors),
        "max_elements_per_tensor": int(max_elements_per_tensor),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "failed_tensor_count": len(failed),
        "failed_tensors": [str(item.get("name") or item.get("issue") or "") for item in failed[:16]],
        "records": records,
    }


def _normalize_sources(source_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(source_paths, (str, Path)):
        raw = [source_paths]
    else:
        raw = list(source_paths)
    sources = [Path(path) for path in raw]
    if not sources:
        raise ValueError("source_paths must not be empty")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"source file not found: {source}")
        if source.suffix.lower() != ".safetensors":
            raise ValueError("payload parity currently accepts .safetensors source files only")
    return sources


def _read_sidecar(gguf_path: Path, sidecar_path: str | Path | None) -> dict[str, Any]:
    path = Path(sidecar_path) if sidecar_path else gguf_path.with_suffix(gguf_path.suffix + ".manifest.json")
    if not path.is_file():
        return {"path": "", "payload": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return {"path": str(path), "payload": payload if isinstance(payload, dict) else {}}


def _select_tensor_names(sources: list[Path], gguf_tensors: dict[str, Any], safe_open: Any, *, max_tensors: int) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for source in sources:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for name in sorted(str(key) for key in handle.keys()):
                if name in seen or name not in gguf_tensors:
                    continue
                seen.add(name)
                names.append(name)
                if len(names) >= max(int(max_tensors), 1):
                    return names
    return names


def _compare_tensor(
    source_path: Path,
    name: str,
    handle: Any,
    gguf_tensor: Any,
    *,
    max_elements: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if gguf_tensor is None:
        return {"name": name, "source_path": str(source_path), "ok": False, "issue": "tensor missing from GGUF"}
    target_array = np.asarray(getattr(gguf_tensor, "data", None))
    source_tensor = handle.get_tensor(name).detach().cpu().contiguous()
    source_dtype = str(source_tensor.dtype).replace("torch.", "")
    expected_tensor = _convert_source_for_target(source_tensor, target_array.dtype)
    source_array = expected_tensor.numpy()
    source_shape = list(source_tensor.shape)
    target_shape = list(target_array.shape)
    if source_shape != target_shape:
        return {
            "name": name,
            "source_path": str(source_path),
            "ok": False,
            "issue": "shape mismatch",
            "source_shape": source_shape,
            "gguf_shape": target_shape,
            "source_dtype": source_dtype,
            "expected_dtype": str(source_array.dtype),
            "gguf_dtype": str(target_array.dtype),
        }
    source_flat = source_array.reshape(-1)
    target_flat = target_array.reshape(-1)
    sample_count = min(int(max_elements), int(source_flat.size), int(target_flat.size))
    source_sample = source_flat[:sample_count]
    target_sample = target_flat[:sample_count]
    source_compare = _to_float64(source_sample)
    target_compare = _to_float64(target_sample)
    diff = np.abs(source_compare - target_compare)
    source_abs = np.abs(source_compare)
    rel = diff / np.maximum(source_abs, 1e-12)
    finite = np.isfinite(source_compare) & np.isfinite(target_compare)
    within = np.isclose(source_compare, target_compare, rtol=float(rtol), atol=float(atol), equal_nan=True)
    nonzero_source = source_compare != 0
    zero_after_nonzero = int(np.count_nonzero(nonzero_source & (target_compare == 0)))
    inf_count = int(np.count_nonzero(np.isinf(target_compare)))
    nan_count = int(np.count_nonzero(np.isnan(target_compare)))
    mismatch_count = int(np.count_nonzero(~within))
    max_abs_error = float(np.max(diff)) if sample_count else 0.0
    mean_abs_error = float(np.mean(diff)) if sample_count else 0.0
    max_rel_error = float(np.max(rel[finite])) if np.any(finite) else 0.0
    ok = mismatch_count == 0 and zero_after_nonzero == 0 and inf_count == 0 and nan_count == 0
    return {
        "name": name,
        "source_path": str(source_path),
        "ok": bool(ok),
        "source_shape": source_shape,
        "gguf_shape": target_shape,
        "source_dtype": source_dtype,
        "expected_dtype": str(source_array.dtype),
        "gguf_dtype": str(target_array.dtype),
        "source_numel": int(source_flat.size),
        "sampled_elements": int(sample_count),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "mismatch_count": mismatch_count,
        "zero_after_nonzero_count": zero_after_nonzero,
        "target_inf_count": inf_count,
        "target_nan_count": nan_count,
        "max_source_abs": float(np.max(source_abs)) if sample_count else 0.0,
    }


def _to_float64(array: np.ndarray) -> np.ndarray:
    try:
        return array.astype(np.float64, copy=False)
    except TypeError:
        return np.asarray(array, dtype=np.float64)


def _convert_source_for_target(tensor: torch.Tensor, target_dtype: np.dtype[Any]) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor
    target = np.dtype(target_dtype)
    if target == np.dtype("float16"):
        return tensor.to(torch.float16).contiguous()
    if target == np.dtype("float32"):
        return tensor.to(torch.float32).contiguous()
    if target == np.dtype("float64"):
        return tensor.to(torch.float64).contiguous()
    return tensor.to(torch.float32).contiguous()


__all__ = ["check_image_gguf_payload_parity"]
