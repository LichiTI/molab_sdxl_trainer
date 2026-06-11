"""Native cache-reader batch materialization for supported cached datasets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from core.turbocore_cache_reader_training_probe import run_native_cache_reader_training_probe


def run_native_cache_reader_batch_dispatch(
    dataset: Any,
    *,
    sample_indices: Sequence[int],
    max_decode_payload_bytes: int,
    max_batch_cpu_payload_buffer_bytes: int,
) -> dict[str, Any]:
    """Materialize a supported cache batch from native payload bytes."""

    indices = [int(index) for index in sample_indices]
    if not indices:
        return _blocked("sample_indices_missing")
    native = run_native_cache_reader_training_probe(
        dataset,
        indices,
        max_decode_payload_bytes=max(int(max_decode_payload_bytes), 1),
        max_batch_cpu_payload_buffer_bytes=max(int(max_batch_cpu_payload_buffer_bytes), 1),
    )
    payload = _as_dict(native.get("batch_cpu_payload_shadow"))
    latents, reason = _tensor_from_batch_payload(payload)
    if latents is None:
        return _blocked(reason or "native_batch_payload_unavailable", native_probe=native)
    reference = _reference_batch(dataset, indices)
    parity = _tensor_parity(latents, reference.get("latents"))
    if not bool(parity.get("ok", False)):
        return _blocked("native_batch_tensor_parity_failed", native_probe=native, parity=parity)
    batch = dict(reference)
    batch["latents"] = latents
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_batch_dispatch_v0",
        "ok": True,
        "dataset_class": type(dataset).__name__,
        "sample_indices": indices,
        "sample_count": len(indices),
        "batch_size": len(indices),
        "native_runtime": True,
        "training_dispatch": True,
        "training_path_enabled": True,
        "native_dispatch_eligible": True,
        "would_allow_native_dispatch": True,
        "fallback_to_python_batch": False,
        "returns_tensor_payloads": True,
        "cache_reader_path_enabled": True,
        "prefetch_queue_training_path_enabled": False,
        "batch": batch,
        "tensor_payloads": {"latents": latents},
        "parity": parity,
        "native_probe": _compact_native_probe(native),
        "blocked_reasons": [],
    }


def _reference_batch(dataset: Any, sample_indices: Sequence[int]) -> dict[str, Any]:
    items = [dataset[int(index)] for index in sample_indices]
    dataset_class = type(dataset).__name__
    if dataset_class == "NewbieCachedDataset":
        from core.lulynx_trainer.newbie_cached_dataset import newbie_cached_collate

        return newbie_cached_collate(items)
    if dataset_class == "AnimaCachedDataset":
        from core.lulynx_trainer.anima_cached_dataset import anima_cached_collate

        return anima_cached_collate(items, fixed_text_tokens=int(getattr(dataset, "fixed_text_tokens", 0) or 0))
    raise ValueError(f"Unsupported native cache-reader dispatch dataset: {dataset_class}")


def _tensor_from_batch_payload(payload: Mapping[str, Any]) -> tuple[torch.Tensor | None, str]:
    if not bool(payload.get("batch_cpu_payload_ready", False)):
        reasons = payload.get("blocked_reasons") if isinstance(payload.get("blocked_reasons"), list) else []
        return None, str(reasons[0]) if reasons else "batch_cpu_payload_not_ready"
    raw = payload.get("batch_cpu_payload_bytes")
    if isinstance(raw, list):
        data = bytes(int(item) & 0xFF for item in raw)
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        data = bytes(raw)
    else:
        return None, "batch_cpu_payload_bytes_missing"
    dtype = _dtype_from_canonical(str(payload.get("canonical_dtype") or ""))
    shape = [int(dim) for dim in list(payload.get("shape", []) or [])]
    if dtype is None:
        return None, "batch_cpu_payload_dtype_not_supported"
    if not shape:
        return None, "batch_cpu_payload_shape_missing"
    expected = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if len(data) != expected:
        return None, "batch_cpu_payload_byte_length_mismatch"
    array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    return torch.from_numpy(array), ""


def _tensor_parity(native: torch.Tensor, reference: Any) -> dict[str, Any]:
    if not isinstance(reference, torch.Tensor):
        return {"ok": False, "reason": "reference_latents_missing"}
    left = native.detach().cpu().float()
    right = reference.detach().cpu().float()
    if tuple(left.shape) != tuple(right.shape):
        return {"ok": False, "reason": "shape_mismatch", "native_shape": list(left.shape), "reference_shape": list(right.shape)}
    diff = (left - right).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    return {
        "ok": max_abs <= 1e-6,
        "max_abs_diff": max_abs,
        "shape": list(left.shape),
        "dtype": str(native.dtype).replace("torch.", ""),
    }


def _compact_native_probe(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(report.get("batch_cpu_payload_shadow"))
    return {
        "chunk_count": int(report.get("chunk_count", 0) or 0),
        "data_payload_bytes_read": int(report.get("data_payload_bytes_read", 0) or 0),
        "record_count": len(list(report.get("records", []) or [])),
        "batch_cpu_payload_ready": bool(payload.get("batch_cpu_payload_ready", False)),
        "batch_cpu_payload_byte_count": int(payload.get("batch_cpu_payload_byte_count", 0) or 0),
        "shape": list(payload.get("shape", []) or []),
        "canonical_dtype": str(payload.get("canonical_dtype") or ""),
    }


def _dtype_from_canonical(value: str) -> np.dtype[Any] | None:
    mapping = {
        "float64": np.float64,
        "float32": np.float32,
        "float16": np.float16,
        "int64": np.int64,
        "int32": np.int32,
        "int16": np.int16,
        "int8": np.int8,
        "uint64": np.uint64,
        "uint32": np.uint32,
        "uint16": np.uint16,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    dtype = mapping.get(str(value or "").strip().lower())
    return None if dtype is None else np.dtype(dtype)


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "provider": "native_cache_reader_batch_dispatch_v0",
        "ok": False,
        "reason": reason,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_eligible": False,
        "would_allow_native_dispatch": False,
        "fallback_to_python_batch": True,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "blocked_reasons": [reason],
    }
    payload.update(extra)
    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["run_native_cache_reader_batch_dispatch"]
