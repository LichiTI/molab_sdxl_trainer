"""Experimental training gate for native cache reader parity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from core.turbocore_cache_reader_batch_gate import run_cache_reader_batch_parity_guard
from core.turbocore_cache_reader_dispatch_eligibility import build_cache_reader_dispatch_eligibility_report
from core.turbocore_cache_reader_dispatch_contract import build_cache_reader_batch_dispatch_contract_shadow
from core.turbocore_cache_reader_handoff_session import run_cache_reader_batch_handoff_shadow_session_probe
from core.turbocore_cache_reader_text_payload_gate import run_cache_reader_text_payload_parity_shadow
from core.turbocore_cache_reader_training_probe import (
    compact_batch_cpu_payload_shadow,
    run_native_cache_reader_training_probe,
)
from core.turbocore_cached_dataset_prefetch_native import optional_env_int, truthy_env


ENABLE_EXPERIMENTAL_ENV = "LULYNX_ENABLE_NATIVE_CACHE_READER_TRAINING_EXPERIMENTAL"
DISABLE_EXPERIMENTAL_ENV = "LULYNX_DISABLE_NATIVE_CACHE_READER_TRAINING_EXPERIMENTAL"
PARITY_BATCHES_ENV = "LULYNX_NATIVE_CACHE_READER_TRAINING_PARITY_BATCHES"
PARITY_MAX_BYTES_ENV = "LULYNX_NATIVE_CACHE_READER_TRAINING_PARITY_MAX_BYTES"
CPU_PAYLOAD_BUFFER_BYTES_ENV = "LULYNX_NATIVE_CACHE_READER_CPU_PAYLOAD_BUFFER_BYTES"
BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV = "LULYNX_NATIVE_CACHE_READER_BATCH_CPU_PAYLOAD_BUFFER_BYTES"
BATCH_HANDOFF_SESSION_ENV = "LULYNX_ENABLE_NATIVE_CACHE_READER_BATCH_HANDOFF_SESSION_SHADOW"
BATCH_DISPATCH_CONTRACT_ENV = "LULYNX_ENABLE_NATIVE_CACHE_READER_BATCH_DISPATCH_CONTRACT_SHADOW"
DISPATCH_STRICT_FALLBACK_ENV = "LULYNX_ENABLE_NATIVE_CACHE_READER_DISPATCH_STRICT_FALLBACK"
TEXT_PAYLOAD_PARITY_ENV = "LULYNX_ENABLE_NATIVE_CACHE_READER_TEXT_PAYLOAD_PARITY_SHADOW"
TEXT_PAYLOAD_BUFFER_BYTES_ENV = "LULYNX_NATIVE_CACHE_READER_TEXT_PAYLOAD_BUFFER_BYTES"

_FNV_OFFSET_BASIS = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_MAX_MISMATCHES = 8


def _fnv1a_64(payload: bytes) -> int:
    checksum = _FNV_OFFSET_BASIS
    for value in payload:
        checksum ^= int(value)
        checksum = (checksum * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return checksum


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _byte_order(dtype: np.dtype[Any]) -> str:
    order = np.dtype(dtype).byteorder
    if order == ">":
        return "big"
    if order == "=":
        return "little" if np.little_endian else "big"
    if order == "|":
        return "none"
    return "little"


def _normalized_byte_order(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "native":
        return "little" if np.little_endian else "big"
    return normalized


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


def _float_close(left: float, right: float, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= max(atol, rtol * max(abs(float(left)), abs(float(right)), 1.0))


def _sample_values(values: np.ndarray) -> list[float]:
    flat = np.asarray(values).reshape(-1)
    return [float(value) for value in flat[:4].astype(np.float64, copy=False).tolist()]


def _python_decode_summary(path: Path, tensor_key: str, array: np.ndarray, *, role: str) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    flat = contiguous.reshape(-1)
    finite_count = int(np.isfinite(flat).sum()) if np.issubdtype(flat.dtype, np.floating) else int(flat.size)
    decoded_sum = float(flat.astype(np.float64, copy=False).sum()) if flat.size else 0.0
    decoded_min = float(flat.min()) if flat.size else 0.0
    decoded_max = float(flat.max()) if flat.size else 0.0
    payload = contiguous.tobytes(order="C")
    dtype = np.dtype(contiguous.dtype)
    declared = int(contiguous.nbytes)
    return {
        "path": str(path),
        "role": role,
        "suffix": path.suffix.lower(),
        "format": path.suffix.lower().lstrip("."),
        "tensor_key": tensor_key,
        "canonical_dtype": _canonical_dtype(dtype),
        "byte_order": _byte_order(dtype),
        "shape": [int(value) for value in contiguous.shape],
        "element_count": int(contiguous.size),
        "declared_payload_bytes": declared,
        "expected_payload_bytes": int(contiguous.size * dtype.itemsize),
        "data_payload_bytes_read": declared,
        "payload_checksum": _fnv1a_64(payload),
        "decoded_element_count": int(contiguous.size),
        "decoded_finite_count": finite_count,
        "decoded_sum": decoded_sum,
        "decoded_min": decoded_min,
        "decoded_max": decoded_max,
        "sample_values": _sample_values(flat),
    }


def _build_newbie_python_records(dataset: Any, samples: Sequence[Any]) -> List[Dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        arrays = dataset._load_newbie_cache_arrays(Path(sample.cache_path))
        records.append(
            _python_decode_summary(
                Path(sample.cache_path),
                "latents",
                np.asarray(arrays.latents),
                role="cache",
            )
        )
    return records


def _build_anima_python_records(dataset: Any, samples: Sequence[Any]) -> List[Dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        path = Path(sample.latent_path)
        data = dataset._load_cache_file(path)
        keys = dataset._cache_keys(data)
        latent_keys = sorted(
            (key for key in keys if key.startswith("latents_")),
            key=lambda key: int(np.prod(dataset._array_shape(data, key)[-2:])),
        )
        if not latent_keys:
            raise ValueError(f"No latents_* arrays found in {path}")
        tensor_key = str(latent_keys[0])
        array = _as_numpy(dataset._tensor_from_cache(data, tensor_key))
        records.append(_python_decode_summary(path, tensor_key, array, role="latent"))
    return records


def _compare_records(native_records: Sequence[Dict[str, Any]], python_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    tensor_count = min(len(native_records), len(python_records))
    match_count = 0
    if len(native_records) != len(python_records):
        mismatches.append(
            {
                "field": "tensor_count",
                "native": len(native_records),
                "python": len(python_records),
                "reason": "tensor_count_mismatch",
            }
        )
    for native_record, python_record in zip(native_records, python_records):
        record_mismatches = 0
        for field in (
            "path",
            "tensor_key",
            "canonical_dtype",
            "shape",
            "element_count",
            "declared_payload_bytes",
            "expected_payload_bytes",
            "data_payload_bytes_read",
            "payload_checksum",
            "decoded_element_count",
            "decoded_finite_count",
        ):
            if native_record.get(field) != python_record.get(field):
                record_mismatches += 1
                if len(mismatches) < _MAX_MISMATCHES:
                    mismatches.append(
                        {
                            "path": str(native_record.get("path") or python_record.get("path") or ""),
                            "tensor_key": str(native_record.get("tensor_key") or python_record.get("tensor_key") or ""),
                            "field": field,
                            "native": native_record.get(field),
                            "python": python_record.get(field),
                        }
                    )
        if _normalized_byte_order(native_record.get("byte_order")) != _normalized_byte_order(python_record.get("byte_order")):
            record_mismatches += 1
            if len(mismatches) < _MAX_MISMATCHES:
                mismatches.append(
                    {
                        "path": str(native_record.get("path") or python_record.get("path") or ""),
                        "tensor_key": str(native_record.get("tensor_key") or python_record.get("tensor_key") or ""),
                        "field": "byte_order",
                        "native": native_record.get("byte_order"),
                        "python": python_record.get("byte_order"),
                    }
                )
        for field in ("decoded_sum", "decoded_min", "decoded_max"):
            if not _float_close(float(native_record.get(field, 0.0) or 0.0), float(python_record.get(field, 0.0) or 0.0)):
                record_mismatches += 1
                if len(mismatches) < _MAX_MISMATCHES:
                    mismatches.append(
                        {
                            "path": str(native_record.get("path") or python_record.get("path") or ""),
                            "tensor_key": str(native_record.get("tensor_key") or python_record.get("tensor_key") or ""),
                            "field": field,
                            "native": native_record.get(field),
                            "python": python_record.get(field),
                        }
                    )
        native_values = list(native_record.get("sample_values", []) or [])
        python_values = list(python_record.get("sample_values", []) or [])
        if len(native_values) != len(python_values) or any(
            not _float_close(float(native_value), float(python_value))
            for native_value, python_value in zip(native_values, python_values)
        ):
            record_mismatches += 1
            if len(mismatches) < _MAX_MISMATCHES:
                mismatches.append(
                    {
                        "path": str(native_record.get("path") or python_record.get("path") or ""),
                        "tensor_key": str(native_record.get("tensor_key") or python_record.get("tensor_key") or ""),
                        "field": "sample_values",
                        "native": native_values,
                        "python": python_values,
                    }
                )
        if record_mismatches == 0:
            match_count += 1
    return {
        "tensor_parity_count": tensor_count,
        "tensor_parity_matches": match_count,
        "parity_guard_passed": match_count == tensor_count and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def run_cache_reader_training_experimental_gate(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
    max_parity_batches: int | None = None,
    max_decode_payload_bytes: int | None = None,
    max_cpu_payload_buffer_bytes: int | None = None,
    max_batch_cpu_payload_buffer_bytes: int | None = None,
    max_text_payload_buffer_bytes: int | None = None,
    enable_batch_handoff_session_shadow: bool | None = None,
    enable_batch_dispatch_contract_shadow: bool | None = None,
    enable_text_payload_parity_shadow: bool | None = None,
) -> Dict[str, Any]:
    if truthy_env(DISABLE_EXPERIMENTAL_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_training_gate",
            "provider": "native_cache_reader_training_gate_policy",
            "ok": True,
            "skipped": True,
            "reason": "training_experimental_gate_disabled_by_env",
            "training_experimental_allowed": False,
            "training_path_enabled": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "returns_tensor_payloads": False,
        }
    if not truthy_env(ENABLE_EXPERIMENTAL_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_training_gate",
            "provider": "native_cache_reader_training_gate_policy",
            "ok": True,
            "skipped": True,
            "reason": "training_experimental_gate_disabled",
            "training_experimental_allowed": False,
            "training_path_enabled": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "returns_tensor_payloads": False,
        }

    resolved_batches = max(int(max_parity_batches or optional_env_int(PARITY_BATCHES_ENV) or 1), 1)
    resolved_batch_size = max(int(batch_size or 1), 1)
    resolved_max_bytes = max(int(max_decode_payload_bytes or optional_env_int(PARITY_MAX_BYTES_ENV) or 16 * 1024 * 1024), 1)
    resolved_cpu_payload_bytes = max(int(max_cpu_payload_buffer_bytes or optional_env_int(CPU_PAYLOAD_BUFFER_BYTES_ENV) or 0), 0)
    resolved_batch_cpu_payload_bytes = max(
        int(max_batch_cpu_payload_buffer_bytes or optional_env_int(BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV) or 0),
        0,
    )
    resolved_text_payload_bytes = max(int(max_text_payload_buffer_bytes or optional_env_int(TEXT_PAYLOAD_BUFFER_BYTES_ENV) or 0), 0)
    handoff_session_enabled = bool(enable_batch_handoff_session_shadow) if enable_batch_handoff_session_shadow is not None else truthy_env(BATCH_HANDOFF_SESSION_ENV)
    dispatch_contract_enabled = bool(enable_batch_dispatch_contract_shadow) if enable_batch_dispatch_contract_shadow is not None else truthy_env(BATCH_DISPATCH_CONTRACT_ENV)
    text_payload_parity_enabled = bool(enable_text_payload_parity_shadow) if enable_text_payload_parity_shadow is not None else truthy_env(TEXT_PAYLOAD_PARITY_ENV)
    handoff_session_enabled = handoff_session_enabled or dispatch_contract_enabled
    sample_count = min(len(list(getattr(dataset, "samples", []) or [])), resolved_batches * resolved_batch_size)
    sample_indices = list(range(sample_count))
    dispatch_eligibility = build_cache_reader_dispatch_eligibility_report(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        strict_fallback=truthy_env(DISPATCH_STRICT_FALLBACK_ENV),
    )
    blockers = [str(item) for item in list(dispatch_eligibility.get("shadow_gate_blockers", []) or [])]
    base_report: Dict[str, Any] = {
        "schema_version": 1,
        "probe": "turbocore_cache_reader_training_gate",
        "provider": "native_cache_reader_training_gate",
        "ok": True,
        "skipped": False,
        "experimental_gate": True,
        "dataset_class": type(dataset).__name__,
        "sample_count": sample_count,
        "batch_size": resolved_batch_size,
        "planned_parity_batches": resolved_batches,
        "max_cpu_payload_buffer_bytes": resolved_cpu_payload_bytes,
        "cpu_payload_buffer_shadow": resolved_cpu_payload_bytes > 0,
        "max_batch_cpu_payload_buffer_bytes": resolved_batch_cpu_payload_bytes,
        "batch_cpu_payload_buffer_shadow": resolved_batch_cpu_payload_bytes > 0,
        "batch_handoff_session_shadow": handoff_session_enabled,
        "batch_dispatch_contract_shadow": dispatch_contract_enabled,
        "text_payload_parity_shadow": text_payload_parity_enabled,
        "max_text_payload_buffer_bytes": resolved_text_payload_bytes,
        "dispatch_strict_fallback": bool(dispatch_eligibility.get("strict_fallback", False)),
        "dispatch_eligibility_shadow_gate_ready": bool(dispatch_eligibility.get("shadow_gate_ready", False)),
        "native_dispatch_eligible": False,
        "native_dispatch_blockers": list(dispatch_eligibility.get("native_dispatch_blockers", []) or []),
        "dispatch_eligibility": dispatch_eligibility,
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "worker_count": max(int(num_workers or 0), 0),
        "prefetch_factor": None if prefetch_factor is None else max(int(prefetch_factor), 1),
        "training_experimental_allowed": False,
        "parity_guard_ran": False,
        "parity_guard_passed": False,
        "batch_parity_guard_ran": False,
        "batch_parity_guard_passed": False,
        "batch_payload_parity_guard_ran": False,
        "batch_payload_parity_guard_passed": False,
        "torch_tensor_handoff_guard_ran": False,
        "torch_tensor_handoff_guard_passed": False,
        "torch_owned_tensor_handoff_guard_ran": False,
        "torch_owned_tensor_handoff_guard_passed": False,
        "text_payload_parity_guard_ran": False,
        "text_payload_parity_guard_passed": False,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }
    if blockers:
        base_report["blocked_reasons"] = blockers
        return base_report
    if sample_count <= 0:
        base_report["blocked_reasons"] = ["no_cached_samples_available"]
        return base_report

    samples = [list(getattr(dataset, "samples", []) or [])[index] for index in sample_indices]
    try:
        if type(dataset).__name__ == "NewbieCachedDataset":
            python_records = _build_newbie_python_records(dataset, samples)
        else:
            python_records = _build_anima_python_records(dataset, samples)
        native_report = run_native_cache_reader_training_probe(
            dataset,
            sample_indices,
            max_decode_payload_bytes=resolved_max_bytes,
            max_cpu_payload_buffer_bytes=resolved_cpu_payload_bytes,
            max_batch_cpu_payload_buffer_bytes=resolved_batch_cpu_payload_bytes,
        )
        native_records = [dict(record) for record in list(native_report.get("records", []) or [])]
        comparison = _compare_records(native_records, python_records)
    except Exception as exc:
        return {
            **base_report,
            "ok": False,
            "reason": "training_experimental_parity_probe_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
        }

    batch_report: Dict[str, Any] = {
        "batch_parity_guard_ran": False,
        "batch_parity_guard_passed": False,
        "blocked_reasons": ["raw_latent_parity_guard_failed"],
    }
    if bool(comparison.get("parity_guard_passed", False)):
        batch_report = run_cache_reader_batch_parity_guard(
            dataset,
            sample_indices=sample_indices,
            native_records=native_records,
            native_batch_summary=dict(native_report.get("native_latent_batch_summary", {}) or {}),
            native_batch_payload_shadow=dict(native_report.get("batch_cpu_payload_shadow", {}) or {}),
        )

    batch_passed = bool(batch_report.get("batch_parity_guard_passed", False))
    payload_required = resolved_batch_cpu_payload_bytes > 0
    payload_passed = bool(batch_report.get("batch_payload_parity_guard_passed", False)) if payload_required else True
    handoff_passed = bool(batch_report.get("torch_tensor_handoff_guard_passed", False)) if payload_required else True
    owned_handoff_passed = bool(batch_report.get("torch_owned_tensor_handoff_guard_passed", False)) if payload_required else True
    raw_passed = bool(comparison.get("parity_guard_passed", False))
    native_bytes = sum(int(record.get("data_payload_bytes_read", 0) or 0) for record in native_records)
    python_bytes = sum(int(record.get("data_payload_bytes_read", 0) or 0) for record in python_records)
    handoff_session_report: Dict[str, Any] = {}
    handoff_session_passed = True
    dispatch_contract_report: Dict[str, Any] = {}
    text_payload_report: Dict[str, Any] = {}
    if handoff_session_enabled and resolved_batch_cpu_payload_bytes > 0 and raw_passed:
        try:
            handoff_session_report = run_cache_reader_batch_handoff_shadow_session_probe(
                dataset,
                sample_indices=sample_indices,
                batch_size=resolved_batch_size,
                max_decode_payload_bytes=resolved_max_bytes,
                max_batch_cpu_payload_buffer_bytes=resolved_batch_cpu_payload_bytes,
            )
            handoff_session_passed = bool(handoff_session_report.get("ok", False)) and bool(
                handoff_session_report.get("torch_owned_tensor_handoff_guard_passed", False)
            )
        except Exception as exc:
            handoff_session_passed = False
            handoff_session_report = {
                "schema_version": 1,
                "provider": "native_cache_reader_batch_handoff_shadow_session",
                "ok": False,
                "reason": "batch_handoff_session_shadow_failed",
                "native_error": f"{type(exc).__name__}: {exc}",
                "training_path_enabled": False,
            }
    if dispatch_contract_enabled:
        dispatch_contract_report = build_cache_reader_batch_dispatch_contract_shadow(
            handoff_session_report,
            dispatch_eligibility=dispatch_eligibility,
        )
    if text_payload_parity_enabled and resolved_text_payload_bytes > 0 and raw_passed:
        text_payload_report = run_cache_reader_text_payload_parity_shadow(
            dataset,
            sample_indices=sample_indices,
            max_decode_payload_bytes=resolved_max_bytes,
            max_text_payload_buffer_bytes=resolved_text_payload_bytes,
        )
    return {
        **base_report,
        "native_runtime": True,
        "parity_guard_ran": True,
        "parity_guard_passed": raw_passed,
        "batch_parity_guard_ran": bool(batch_report.get("batch_parity_guard_ran", False)),
        "batch_parity_guard_passed": batch_passed,
        "batch_payload_parity_guard_ran": bool(batch_report.get("batch_payload_parity_guard_ran", False)),
        "batch_payload_parity_guard_passed": bool(batch_report.get("batch_payload_parity_guard_passed", False)),
        "torch_tensor_handoff_guard_ran": bool(batch_report.get("torch_tensor_handoff_guard_ran", False)),
        "torch_tensor_handoff_guard_passed": bool(batch_report.get("torch_tensor_handoff_guard_passed", False)),
        "torch_owned_tensor_handoff_guard_ran": bool(batch_report.get("torch_owned_tensor_handoff_guard_ran", False)),
        "torch_owned_tensor_handoff_guard_passed": bool(batch_report.get("torch_owned_tensor_handoff_guard_passed", False)),
        "text_payload_parity_guard_ran": bool(text_payload_report.get("text_payload_parity_guard_ran", False)),
        "text_payload_parity_guard_passed": bool(text_payload_report.get("text_payload_parity_guard_passed", False)),
        "batch_handoff_session_shadow_ran": bool(handoff_session_report),
        "batch_handoff_session_shadow_passed": handoff_session_passed if handoff_session_enabled else False,
        "batch_dispatch_contract_shadow_ran": bool(dispatch_contract_report),
        "batch_dispatch_contract_ready": bool(dispatch_contract_report.get("dispatch_contract_ready", False)) if dispatch_contract_report else False,
        "batch_dispatch_contract_would_allow_native_dispatch": False,
        "training_experimental_allowed": raw_passed and batch_passed and payload_passed and handoff_passed and owned_handoff_passed and handoff_session_passed,
        "tensor_parity_count": int(comparison.get("tensor_parity_count", 0) or 0),
        "tensor_parity_matches": int(comparison.get("tensor_parity_matches", 0) or 0),
        "mismatch_count": int(comparison.get("mismatch_count", 0) or 0),
        "mismatches": list(comparison.get("mismatches", []) or []),
        "batch_parity": batch_report,
        "text_payload_parity": text_payload_report,
        "sample_indices": sample_indices,
        "native_data_payload_bytes_read": native_bytes,
        "python_data_payload_bytes_read": python_bytes,
        "native_probe": {
            "chunk_count": int(native_report.get("chunk_count", 0) or 0),
            "data_payload_bytes_read": int(native_report.get("data_payload_bytes_read", 0) or 0),
            "tensor_decode_count": len(native_records),
            "session_create": dict(native_report.get("session_create", {}) or {}),
            "session_stats": dict(native_report.get("session_stats", {}) or {}),
            "native_latent_batch_summary": dict(native_report.get("native_latent_batch_summary", {}) or {}),
            "native_latent_batch_summaries": list(native_report.get("native_latent_batch_summaries", []) or []),
            "batch_cpu_payload_shadow": compact_batch_cpu_payload_shadow(native_report.get("batch_cpu_payload_shadow")),
        },
        "batch_handoff_session": handoff_session_report,
        "batch_dispatch_contract": dispatch_contract_report,
        "python_reference": {
            "tensor_decode_count": len(python_records),
            "data_payload_bytes_read": python_bytes,
        },
    }


def maybe_attach_cache_reader_training_experimental_gate(
    dataloader: Any,
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    prefetch_factor: int | None = None,
) -> Any:
    if truthy_env(DISABLE_EXPERIMENTAL_ENV) or not truthy_env(ENABLE_EXPERIMENTAL_ENV):
        return dataloader
    report = run_cache_reader_training_experimental_gate(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    for target in (dataloader, dataset):
        try:
            setattr(target, "native_cache_reader_training_gate", report)
        except Exception:
            pass
    return dataloader


__all__ = [
    "DISABLE_EXPERIMENTAL_ENV",
    "ENABLE_EXPERIMENTAL_ENV",
    "BATCH_HANDOFF_SESSION_ENV",
    "BATCH_DISPATCH_CONTRACT_ENV",
    "DISPATCH_STRICT_FALLBACK_ENV",
    "BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV",
    "CPU_PAYLOAD_BUFFER_BYTES_ENV",
    "TEXT_PAYLOAD_BUFFER_BYTES_ENV",
    "TEXT_PAYLOAD_PARITY_ENV",
    "PARITY_BATCHES_ENV",
    "PARITY_MAX_BYTES_ENV",
    "maybe_attach_cache_reader_training_experimental_gate",
    "run_cache_reader_training_experimental_gate",
]
