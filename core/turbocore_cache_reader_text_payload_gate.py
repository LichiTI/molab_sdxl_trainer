"""Bounded text/conditioning payload parity guard for cache reader shadows."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch

from core.turbocore_cache_reader_payload_ownership import TEXT_PAYLOAD_FIELDS
from core.turbocore_cache_reader_shadow_layout import NativeCacheReaderDecodeShadowSession
from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest
from core.turbocore_cache_reader_training_probe import build_selected_cache_reader_dataset_view


_FNV_OFFSET_BASIS = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_MAX_MISMATCHES = 8

_FIELD_TO_NATIVE_KEYS = {
    "encoder_hidden_states": {"encoder_hidden_states", "prompt_embeds", "gemma_hidden_states"},
    "attention_mask": {"attention_mask", "attn_mask", "gemma_attention_mask"},
    "pooled_prompt_embeds": {"pooled_prompt_embeds", "clip_pooled_features", "text_embeds"},
    "t5_input_ids": {"t5_input_ids"},
    "t5_attention_mask": {"t5_attention_mask"},
    "qwen3_hidden_states": {"qwen3_hidden_states"},
    "qwen3_attention_mask": {"qwen3_attention_mask"},
}


def _fnv1a_64(payload: bytes) -> int:
    checksum = _FNV_OFFSET_BASIS
    for value in payload:
        checksum ^= int(value)
        checksum = (checksum * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return checksum


def _collate_reference_batch(dataset: Any, sample_indices: Sequence[int]) -> Dict[str, Any]:
    items = [dataset[int(index)] for index in sample_indices]
    dataset_class = type(dataset).__name__
    if dataset_class == "NewbieCachedDataset":
        from core.lulynx_trainer.newbie_cached_dataset import newbie_cached_collate

        return newbie_cached_collate(items)
    if dataset_class == "AnimaCachedDataset":
        from core.lulynx_trainer.anima_cached_dataset import anima_cached_collate

        return anima_cached_collate(
            items,
            fixed_text_tokens=int(getattr(dataset, "fixed_text_tokens", 0) or 0),
        )
    raise ValueError(f"Unsupported cached dataset class for text payload guard: {dataset_class}")


def _tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    flat = tensor.reshape(-1).to(dtype=torch.float64)
    element_count = int(flat.numel())
    payload = tensor.numpy().tobytes(order="C") if element_count else b""
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "canonical_dtype": str(tensor.dtype).replace("torch.", ""),
        "element_count": element_count,
        "payload_byte_count": int(tensor.element_size() * element_count),
        "payload_checksum": _fnv1a_64(payload),
        "decoded_sum": float(flat.sum().item()) if element_count else 0.0,
        "decoded_min": float(flat.min().item()) if element_count else 0.0,
        "decoded_max": float(flat.max().item()) if element_count else 0.0,
        "sample_values": [float(item) for item in flat[:4].tolist()],
    }


def _canonical_dtype(dtype: np.dtype[Any]) -> str:
    normalized = np.dtype(dtype)
    if normalized == np.dtype(np.float64):
        return "float64"
    if normalized == np.dtype(np.float32):
        return "float32"
    if normalized == np.dtype(np.float16):
        return "float16"
    if normalized == np.dtype(np.int64):
        return "int64"
    if normalized == np.dtype(np.int32):
        return "int32"
    if normalized == np.dtype(np.int16):
        return "int16"
    if normalized == np.dtype(np.int8):
        return "int8"
    if normalized == np.dtype(np.uint64):
        return "uint64"
    if normalized == np.dtype(np.uint32):
        return "uint32"
    if normalized == np.dtype(np.uint16):
        return "uint16"
    if normalized == np.dtype(np.uint8):
        return "uint8"
    if normalized == np.dtype(np.bool_):
        return "bool"
    return str(normalized)


def _numpy_summary(array: np.ndarray) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    flat = contiguous.reshape(-1)
    flat64 = flat.astype(np.float64, copy=False)
    element_count = int(flat.size)
    payload = contiguous.tobytes(order="C") if element_count else b""
    return {
        "shape": [int(dim) for dim in contiguous.shape],
        "canonical_dtype": _canonical_dtype(contiguous.dtype),
        "element_count": element_count,
        "payload_byte_count": int(contiguous.nbytes),
        "payload_checksum": _fnv1a_64(payload),
        "decoded_sum": float(flat64.sum()) if element_count else 0.0,
        "decoded_min": float(flat64.min()) if element_count else 0.0,
        "decoded_max": float(flat64.max()) if element_count else 0.0,
        "sample_values": [float(item) for item in flat64[:4].tolist()],
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


def _record_array(record: dict[str, Any]) -> np.ndarray | None:
    dtype = _dtype_from_canonical(str(record.get("canonical_dtype") or ""))
    if dtype is None:
        return None
    payload = _payload_bytes(record)
    shape = [int(dim) for dim in list(record.get("shape", []) or [])]
    expected = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize) if shape else 0
    if not payload or len(payload) != expected:
        return None
    try:
        return np.frombuffer(payload, dtype=dtype).reshape(shape)
    except Exception:
        return None


def _normalize_native_array(field: str, array: np.ndarray) -> np.ndarray:
    normalized = np.asarray(array)
    if field in {"encoder_hidden_states", "attention_mask"} and normalized.ndim >= 2 and normalized.shape[0] == 1:
        normalized = normalized[0]
    if field in {"attention_mask", "t5_attention_mask", "qwen3_attention_mask"}:
        normalized = normalized.astype(np.bool_, copy=False)
    if field == "encoder_hidden_states":
        normalized = normalized.astype(np.float32, copy=False)
    if field == "pooled_prompt_embeds" and normalized.ndim == 2 and normalized.shape[0] == 1:
        normalized = normalized[0]
    return np.ascontiguousarray(normalized)


def _native_field_summary(records: list[dict[str, Any]], field: str, keys: set[str], batch_size: int) -> Dict[str, Any] | None:
    matched = [record for record in records if str(record.get("tensor_key") or "") in keys]
    if len(matched) != batch_size or not matched:
        return None
    arrays: list[np.ndarray] = []
    for record in matched:
        array = _record_array(record)
        if array is None:
            return None
        arrays.append(_normalize_native_array(field, array))
    if any(array.shape != arrays[0].shape for array in arrays):
        return None
    summary = _numpy_summary(np.stack(arrays, axis=0))
    summary["source_tensor_count"] = len(matched)
    summary["source_item_shapes"] = [[int(dim) for dim in array.shape] for array in arrays]
    summary["tensor_keys"] = sorted(str(record.get("tensor_key") or "") for record in matched)
    return summary


def _payload_bytes(record: dict[str, Any]) -> bytes:
    values = record.get("cpu_payload_buffer_bytes")
    if isinstance(values, (bytes, bytearray)):
        return bytes(values)
    if isinstance(values, list):
        return bytes(int(value) & 0xFF for value in values)
    return b""


def _float_close(left: float, right: float, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= max(atol, rtol * max(abs(float(left)), abs(float(right)), 1.0))


def _compare(native: Dict[str, Any], python: Dict[str, Any]) -> Dict[str, Any]:
    fields = ("shape", "canonical_dtype", "element_count", "payload_byte_count", "payload_checksum")
    mismatches: list[dict[str, Any]] = []
    matches = 0
    for field in fields:
        if native.get(field) == python.get(field):
            matches += 1
        elif len(mismatches) < _MAX_MISMATCHES:
            mismatches.append({"field": field, "native": native.get(field), "python": python.get(field)})
    for field in ("decoded_sum", "decoded_min", "decoded_max"):
        if _float_close(float(native.get(field, 0.0) or 0.0), float(python.get(field, 0.0) or 0.0)):
            matches += 1
        elif len(mismatches) < _MAX_MISMATCHES:
            mismatches.append({"field": field, "native": native.get(field), "python": python.get(field)})
    return {
        "field_count": 8,
        "field_matches": matches,
        "passed": matches == 8 and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def run_cache_reader_text_payload_parity_shadow(
    dataset: Any,
    *,
    sample_indices: Sequence[int],
    max_decode_payload_bytes: int,
    max_text_payload_buffer_bytes: int,
) -> Dict[str, Any]:
    indices = [int(index) for index in sample_indices]
    if not indices:
        return {"ok": False, "reason": "no_sample_indices", "training_path_enabled": False}
    try:
        batch = _collate_reference_batch(dataset, indices)
        selected_view = build_selected_cache_reader_dataset_view(dataset, indices)
        manifest = build_cache_reader_shadow_manifest(selected_view, max_files=max(len(indices), 1) * 2)
        with NativeCacheReaderDecodeShadowSession(
            manifest,
            max_files=max(len(indices), 1) * 2,
            max_tensors_per_file=32,
            max_decode_payload_bytes=max(int(max_decode_payload_bytes), 1),
            selected_only=False,
        ) as session:
            chunk = session.run_cpu_payload_chunk(
                cursor=0,
                max_tensors=32 * max(len(indices), 1),
                max_cpu_payload_buffer_bytes=max(int(max_text_payload_buffer_bytes), 1),
            )
    except Exception as exc:
        return {
            "schema_version": 1,
            "provider": "native_cache_reader_text_payload_parity_shadow",
            "ok": False,
            "reason": "text_payload_parity_shadow_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
        }
    native_records = [dict(record) for record in list(chunk.get("records", []) or [])]
    fields: dict[str, Any] = {}
    passed = True
    for field in sorted(TEXT_PAYLOAD_FIELDS):
        value = batch.get(field)
        if not isinstance(value, torch.Tensor):
            continue
        keys = _FIELD_TO_NATIVE_KEYS.get(field, {field})
        native_summary = _native_field_summary(native_records, field, keys, len(indices))
        python_summary = _tensor_summary(value)
        if native_summary is None:
            fields[field] = {
                "ok": False,
                "reason": "native_text_tensor_summary_missing",
                "python": python_summary,
                "native_keys": sorted(keys),
            }
            passed = False
            continue
        comparison = _compare(native_summary, python_summary)
        fields[field] = {
            "ok": bool(comparison["passed"]),
            "native": native_summary,
            "python": python_summary,
            "comparison": comparison,
        }
        passed = passed and bool(comparison["passed"])
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_text_payload_parity_shadow",
        "ok": passed,
        "debug_only": True,
        "shadow_run": True,
        "sample_indices": indices,
        "sample_count": len(indices),
        "text_payload_parity_guard_ran": True,
        "text_payload_parity_guard_passed": passed,
        "text_payload_field_count": len(fields),
        "text_payload_fields": sorted(fields.keys()),
        "fields": fields,
        "native_tensor_decode_count": int(chunk.get("tensor_decode_count", 0) or 0),
        "native_data_payload_bytes_read": int(chunk.get("data_payload_bytes_read", 0) or 0),
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


__all__ = ["run_cache_reader_text_payload_parity_shadow"]
