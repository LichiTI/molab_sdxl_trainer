"""Batch-level parity guard for the native cache reader training gate."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch

from core.turbocore_cache_reader_tensor_handoff_gate import run_torch_tensor_handoff_guards
from core.turbocore_cache_reader_payload_ownership import build_cache_reader_payload_ownership_shadow


_MAX_MISMATCHES = 8


def _float_close(left: float, right: float, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= max(atol, rtol * max(abs(float(left)), abs(float(right)), 1.0))


def _tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    tensor = value.detach().cpu()
    flat = tensor.reshape(-1).to(dtype=torch.float64)
    element_count = int(flat.numel())
    finite_count = int(torch.isfinite(flat).sum().item()) if element_count else 0
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "canonical_dtype": str(tensor.dtype).replace("torch.", ""),
        "element_count": element_count,
        "decoded_element_count": element_count,
        "decoded_finite_count": finite_count,
        "decoded_sum": float(flat.sum().item()) if element_count else 0.0,
        "decoded_min": float(flat.min().item()) if element_count else 0.0,
        "decoded_max": float(flat.max().item()) if element_count else 0.0,
        "sample_values": [float(item) for item in flat[:4].tolist()],
    }


def _numpy_summary(value: np.ndarray) -> Dict[str, Any]:
    array = np.ascontiguousarray(value)
    flat = array.reshape(-1)
    flat64 = flat.astype(np.float64, copy=False)
    element_count = int(flat.size)
    finite_count = int(np.isfinite(flat64).sum()) if element_count else 0
    return {
        "shape": [int(dim) for dim in array.shape],
        "canonical_dtype": str(array.dtype),
        "element_count": element_count,
        "decoded_element_count": element_count,
        "decoded_finite_count": finite_count,
        "decoded_sum": float(flat64.sum()) if element_count else 0.0,
        "decoded_min": float(flat64.min()) if element_count else 0.0,
        "decoded_max": float(flat64.max()) if element_count else 0.0,
        "sample_values": [float(item) for item in flat64[:4].tolist()],
        "payload_checksum": _fnv1a_64(array.tobytes(order="C")),
        "payload_byte_count": int(array.nbytes),
    }


def _fnv1a_64(payload: bytes) -> int:
    checksum = 14_695_981_039_346_656_037
    for value in payload:
        checksum ^= int(value)
        checksum = (checksum * 1_099_511_628_211) & 0xFFFFFFFFFFFFFFFF
    return checksum


def _normalized_item_shape(dataset_class: str, record: Dict[str, Any]) -> list[int]:
    shape = [int(dim) for dim in list(record.get("shape", []) or [])]
    if dataset_class == "NewbieCachedDataset" and len(shape) == 4 and shape[0] == 1:
        return shape[1:]
    return shape


def _native_latent_batch_summary(
    dataset_class: str,
    native_records: Sequence[Dict[str, Any]],
    native_batch_summary: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any] | None, list[str]]:
    if native_batch_summary and bool(native_batch_summary.get("batch_summary_ready", False)):
        return {
            "shape": [int(dim) for dim in list(native_batch_summary.get("shape", []) or [])],
            "canonical_dtype": str(native_batch_summary.get("canonical_dtype") or ""),
            "element_count": int(native_batch_summary.get("element_count", 0) or 0),
            "decoded_element_count": int(native_batch_summary.get("decoded_element_count", 0) or 0),
            "decoded_finite_count": int(native_batch_summary.get("decoded_finite_count", 0) or 0),
            "decoded_sum": float(native_batch_summary.get("decoded_sum", 0.0) or 0.0),
            "decoded_min": float(native_batch_summary.get("decoded_min", 0.0) or 0.0),
            "decoded_max": float(native_batch_summary.get("decoded_max", 0.0) or 0.0),
            "sample_values": list(native_batch_summary.get("sample_values", []) or []),
            "source_item_shapes": list(native_batch_summary.get("source_item_shapes", []) or []),
            "source_tensor_count": int(native_batch_summary.get("source_tensor_count", 0) or 0),
            "native_batch_summary_provider": str(native_batch_summary.get("provider") or "native_cache_reader_decode_session_batch_summary"),
        }, []
    if native_batch_summary and native_batch_summary.get("blocked_reasons"):
        return None, [str(reason) for reason in list(native_batch_summary.get("blocked_reasons", []) or [])]
    if not native_records:
        return None, ["native_latent_records_missing"]
    item_shapes = [_normalized_item_shape(dataset_class, dict(record)) for record in native_records]
    first_shape = item_shapes[0]
    if any(shape != first_shape for shape in item_shapes):
        return None, ["variable_latent_shape_batch_parity_not_ready"]
    finite_records = [record for record in native_records if int(record.get("decoded_element_count", 0) or 0) > 0]
    if len(finite_records) != len(native_records):
        return None, ["native_latent_decode_summary_incomplete"]
    mins = [float(record.get("decoded_min", 0.0) or 0.0) for record in native_records]
    maxs = [float(record.get("decoded_max", 0.0) or 0.0) for record in native_records]
    return {
        "shape": [len(native_records), *first_shape],
        "canonical_dtype": str(native_records[0].get("canonical_dtype") or ""),
        "element_count": sum(int(record.get("element_count", 0) or 0) for record in native_records),
        "decoded_element_count": sum(int(record.get("decoded_element_count", 0) or 0) for record in native_records),
        "decoded_finite_count": sum(int(record.get("decoded_finite_count", 0) or 0) for record in native_records),
        "decoded_sum": sum(float(record.get("decoded_sum", 0.0) or 0.0) for record in native_records),
        "decoded_min": min(mins) if mins else 0.0,
        "decoded_max": max(maxs) if maxs else 0.0,
        "sample_values": list(native_records[0].get("sample_values", []) or []),
        "source_item_shapes": item_shapes,
        "source_tensor_count": len(native_records),
    }, []


def _batch_transform_blockers(dataset: Any) -> list[str]:
    dataset_class = type(dataset).__name__
    blockers: list[str] = []
    if int(getattr(dataset, "latent_crop_size", 0) or 0) > 0:
        blockers.append("latent_crop_batch_parity_not_ready")
    if dataset_class == "AnimaCachedDataset" and int(getattr(dataset, "fixed_visual_tokens", 0) or 0) > 0:
        blockers.append("fixed_visual_token_padding_batch_parity_not_ready")
    return blockers


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
    raise ValueError(f"Unsupported cached dataset class for batch parity: {dataset_class}")


def _compare_batch_summaries(native_summary: Dict[str, Any], python_summary: Dict[str, Any]) -> Dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    match_count = 0
    for field in ("shape", "canonical_dtype", "element_count", "decoded_element_count", "decoded_finite_count"):
        if native_summary.get(field) == python_summary.get(field):
            match_count += 1
            continue
        if len(mismatches) < _MAX_MISMATCHES:
            mismatches.append({"field": field, "native": native_summary.get(field), "python": python_summary.get(field)})
    for field in ("decoded_sum", "decoded_min", "decoded_max"):
        if _float_close(float(native_summary.get(field, 0.0) or 0.0), float(python_summary.get(field, 0.0) or 0.0)):
            match_count += 1
            continue
        if len(mismatches) < _MAX_MISMATCHES:
            mismatches.append({"field": field, "native": native_summary.get(field), "python": python_summary.get(field)})
    native_values = list(native_summary.get("sample_values", []) or [])
    python_values = list(python_summary.get("sample_values", []) or [])
    if len(native_values) == len(python_values) and all(
        _float_close(float(native), float(python)) for native, python in zip(native_values, python_values)
    ):
        match_count += 1
    elif len(mismatches) < _MAX_MISMATCHES:
        mismatches.append({"field": "sample_values", "native": native_values, "python": python_values})
    return {
        "batch_parity_field_count": 9,
        "batch_parity_field_matches": match_count,
        "batch_parity_guard_passed": match_count == 9 and not mismatches,
        "batch_mismatch_count": len(mismatches),
        "batch_mismatches": mismatches,
    }


def _dtype_from_canonical(value: str) -> np.dtype[Any] | None:
    normalized = str(value or "").strip().lower()
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
    dtype = mapping.get(normalized)
    return None if dtype is None else np.dtype(dtype)


def _native_batch_payload_summary(native_batch_payload_shadow: Dict[str, Any]) -> tuple[Dict[str, Any] | None, list[str]]:
    if not native_batch_payload_shadow:
        return None, []
    if not bool(native_batch_payload_shadow.get("batch_cpu_payload_ready", False)):
        blocked = [str(reason) for reason in list(native_batch_payload_shadow.get("blocked_reasons", []) or [])]
        return None, blocked or ["batch_cpu_payload_shadow_not_ready"]
    payload = native_batch_payload_shadow.get("batch_cpu_payload_bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None, ["batch_cpu_payload_bytes_missing"]
    dtype = _dtype_from_canonical(str(native_batch_payload_shadow.get("canonical_dtype") or ""))
    if dtype is None:
        return None, ["batch_cpu_payload_dtype_not_supported"]
    shape = [int(dim) for dim in list(native_batch_payload_shadow.get("shape", []) or [])]
    if not shape:
        return None, ["batch_cpu_payload_shape_missing"]
    expected = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    byte_count = int(native_batch_payload_shadow.get("batch_cpu_payload_byte_count", 0) or 0)
    payload_view = memoryview(payload)
    if payload_view.nbytes != expected or byte_count != expected:
        return None, ["batch_cpu_payload_byte_length_mismatch"]
    try:
        array = np.frombuffer(payload_view, dtype=dtype).reshape(shape)
    except Exception:
        return None, ["batch_cpu_payload_numpy_view_failed"]
    summary = _numpy_summary(array)
    summary["provider"] = str(native_batch_payload_shadow.get("provider") or "")
    summary["native_batch_payload_provider"] = str(native_batch_payload_shadow.get("provider") or "")
    summary["batch_cpu_payload_byte_count"] = byte_count
    summary["batch_cpu_payload_ready"] = True
    return summary, []


def _compare_batch_payload(native_payload_summary: Dict[str, Any], python_latents: torch.Tensor) -> Dict[str, Any]:
    python_array = np.ascontiguousarray(python_latents.detach().cpu().numpy())
    python_summary = _numpy_summary(python_array)
    comparison = _compare_batch_summaries(native_payload_summary, python_summary)
    checksum_match = native_payload_summary.get("payload_checksum") == python_summary.get("payload_checksum")
    byte_count_match = native_payload_summary.get("payload_byte_count") == python_summary.get("payload_byte_count")
    field_matches = int(comparison.get("batch_parity_field_matches", 0) or 0)
    field_count = int(comparison.get("batch_parity_field_count", 0) or 0) + 2
    mismatches = list(comparison.get("batch_mismatches", []) or [])
    if checksum_match:
        field_matches += 1
    elif len(mismatches) < _MAX_MISMATCHES:
        mismatches.append({"field": "payload_checksum", "native": native_payload_summary.get("payload_checksum"), "python": python_summary.get("payload_checksum")})
    if byte_count_match:
        field_matches += 1
    elif len(mismatches) < _MAX_MISMATCHES:
        mismatches.append({"field": "payload_byte_count", "native": native_payload_summary.get("payload_byte_count"), "python": python_summary.get("payload_byte_count")})
    return {
        "batch_payload_parity_guard_ran": True,
        "batch_payload_parity_guard_passed": field_matches == field_count and not mismatches,
        "batch_payload_parity_field_count": field_count,
        "batch_payload_parity_field_matches": field_matches,
        "batch_payload_mismatch_count": len(mismatches),
        "batch_payload_mismatches": mismatches,
        "native_batch_payload_reference": native_payload_summary,
        "python_batch_payload_reference": python_summary,
    }


def run_cache_reader_batch_parity_guard(
    dataset: Any,
    *,
    sample_indices: Sequence[int],
    native_records: Sequence[Dict[str, Any]],
    native_batch_summary: Dict[str, Any] | None = None,
    native_batch_payload_shadow: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "schema_version": 1,
        "probe": "turbocore_cache_reader_batch_parity_guard",
        "provider": "native_cache_reader_training_batch_parity_guard",
        "ok": True,
        "experimental_gate": True,
        "dataset_class": type(dataset).__name__,
        "sample_indices": [int(index) for index in sample_indices],
        "sample_count": len(sample_indices),
        "batch_parity_guard_ran": False,
        "batch_parity_guard_passed": False,
        "batch_payload_parity_guard_ran": False,
        "batch_payload_parity_guard_passed": False,
        "torch_tensor_handoff_guard_ran": False,
        "torch_tensor_handoff_guard_passed": False,
        "torch_owned_tensor_handoff_guard_ran": False,
        "torch_owned_tensor_handoff_guard_passed": False,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }
    blockers = _batch_transform_blockers(dataset)
    if blockers:
        return {**base, "blocked_reasons": blockers}
    native_summary, native_blockers = _native_latent_batch_summary(
        type(dataset).__name__,
        native_records,
        native_batch_summary,
    )
    if native_blockers or native_summary is None:
        return {**base, "blocked_reasons": native_blockers}
    try:
        batch = _collate_reference_batch(dataset, sample_indices)
        latents = batch.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise ValueError("collated_latents_missing")
        python_summary = _tensor_summary(latents)
        comparison = _compare_batch_summaries(native_summary, python_summary)
        payload_comparison: Dict[str, Any] = {
            "batch_payload_parity_guard_ran": False,
            "batch_payload_parity_guard_passed": False,
        }
        torch_handoff_comparison: Dict[str, Any] = {
            "torch_tensor_handoff_guard_ran": False,
            "torch_tensor_handoff_guard_passed": False,
        }
        torch_owned_handoff_comparison: Dict[str, Any] = {
            "torch_owned_tensor_handoff_guard_ran": False,
            "torch_owned_tensor_handoff_guard_passed": False,
        }
        native_payload_summary, payload_blockers = _native_batch_payload_summary(dict(native_batch_payload_shadow or {}))
        if payload_blockers:
            payload_comparison = {
                "batch_payload_parity_guard_ran": False,
                "batch_payload_parity_guard_passed": False,
                "batch_payload_blocked_reasons": payload_blockers,
            }
        elif native_payload_summary is not None:
            payload_comparison = _compare_batch_payload(native_payload_summary, latents)
            handoff_report = run_torch_tensor_handoff_guards(
                dict(native_batch_payload_shadow or {}),
                latents,
                max_mismatches=_MAX_MISMATCHES,
            )
            torch_handoff_comparison = handoff_report
            torch_owned_handoff_comparison = handoff_report
        payload_ownership = build_cache_reader_payload_ownership_shadow(
            batch,
            sample_indices=sample_indices,
            batch_payload_parity_passed=bool(payload_comparison.get("batch_payload_parity_guard_passed", False)),
        )
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "reason": "batch_parity_reference_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **base,
        "batch_parity_guard_ran": True,
        "batch_parity_guard_passed": bool(comparison.get("batch_parity_guard_passed", False)),
        "batch_parity_field_count": int(comparison.get("batch_parity_field_count", 0) or 0),
        "batch_parity_field_matches": int(comparison.get("batch_parity_field_matches", 0) or 0),
        "batch_mismatch_count": int(comparison.get("batch_mismatch_count", 0) or 0),
        "batch_mismatches": list(comparison.get("batch_mismatches", []) or []),
        "batch_payload_parity_guard_ran": bool(payload_comparison.get("batch_payload_parity_guard_ran", False)),
        "batch_payload_parity_guard_passed": bool(payload_comparison.get("batch_payload_parity_guard_passed", False)),
        "batch_payload_parity_field_count": int(payload_comparison.get("batch_payload_parity_field_count", 0) or 0),
        "batch_payload_parity_field_matches": int(payload_comparison.get("batch_payload_parity_field_matches", 0) or 0),
        "batch_payload_mismatch_count": int(payload_comparison.get("batch_payload_mismatch_count", 0) or 0),
        "batch_payload_mismatches": list(payload_comparison.get("batch_payload_mismatches", []) or []),
        "torch_tensor_handoff_guard_ran": bool(torch_handoff_comparison.get("torch_tensor_handoff_guard_ran", False)),
        "torch_tensor_handoff_guard_passed": bool(torch_handoff_comparison.get("torch_tensor_handoff_guard_passed", False)),
        "torch_tensor_handoff_field_count": int(torch_handoff_comparison.get("torch_tensor_handoff_field_count", 0) or 0),
        "torch_tensor_handoff_field_matches": int(torch_handoff_comparison.get("torch_tensor_handoff_field_matches", 0) or 0),
        "torch_tensor_handoff_mismatch_count": int(torch_handoff_comparison.get("torch_tensor_handoff_mismatch_count", 0) or 0),
        "torch_tensor_handoff_mismatches": list(torch_handoff_comparison.get("torch_tensor_handoff_mismatches", []) or []),
        "torch_owned_tensor_handoff_guard_ran": bool(torch_owned_handoff_comparison.get("torch_owned_tensor_handoff_guard_ran", False)),
        "torch_owned_tensor_handoff_guard_passed": bool(torch_owned_handoff_comparison.get("torch_owned_tensor_handoff_guard_passed", False)),
        "torch_owned_tensor_handoff_field_count": int(torch_owned_handoff_comparison.get("torch_owned_tensor_handoff_field_count", 0) or 0),
        "torch_owned_tensor_handoff_field_matches": int(torch_owned_handoff_comparison.get("torch_owned_tensor_handoff_field_matches", 0) or 0),
        "torch_owned_tensor_handoff_mismatch_count": int(torch_owned_handoff_comparison.get("torch_owned_tensor_handoff_mismatch_count", 0) or 0),
        "torch_owned_tensor_handoff_mismatches": list(torch_owned_handoff_comparison.get("torch_owned_tensor_handoff_mismatches", []) or []),
        "native_latent_batch_reference": native_summary,
        "native_batch_payload_reference": dict(payload_comparison.get("native_batch_payload_reference", {}) or {}),
        "native_torch_tensor_handoff_reference": dict(torch_handoff_comparison.get("native_torch_tensor_handoff_reference", {}) or {}),
        "native_torch_owned_tensor_handoff_reference": dict(torch_owned_handoff_comparison.get("native_torch_owned_tensor_handoff_reference", {}) or {}),
        "payload_ownership_shadow": payload_ownership,
        "python_batch_payload_reference": dict(payload_comparison.get("python_batch_payload_reference", {}) or {}),
        "python_batch_reference": {
            "latents": python_summary,
            "keys": sorted(str(key) for key in batch.keys()),
        },
    }


__all__ = ["run_cache_reader_batch_parity_guard"]
