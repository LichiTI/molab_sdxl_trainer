# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Probe-only dataset adapter contract for .lynx tensor-cache manifests.

This module deliberately does not wire .lynx manifests into the training hot
path. It gives devtools and future runtime/request adapters a small, auditable
contract for ordered batch reads, fallback reporting, and manifest reader
lifetime before any product gate is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .lossless_cache_tensor_reader import TensorGroupManifestReader
except ImportError:  # pragma: no cover - direct script loading
    from lossless_cache_tensor_reader import TensorGroupManifestReader


@dataclass(frozen=True)
class LosslessTensorManifestAdapterConfig:
    enabled: bool = False
    strict: bool = False
    verify_crc32: bool = True
    eager_readers: bool = False
    copy_arrays: bool = True
    copy_tensor_bytes: bool = True
    use_tensor_records: bool = True


def _array_record(name: str, array: Any) -> dict[str, Any]:
    raw = array.tobytes(order="C")
    return {
        "name": str(name),
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "raw_size": len(raw),
        "data": raw,
    }


def _tensor_record(name: str, tensor: dict[str, Any]) -> dict[str, Any]:
    precision = dict(tensor.get("precision") or {})
    data = tensor.get("data") or b""
    copied = isinstance(data, bytes)
    return {
        "name": str(name),
        "dtype": str(precision.get("storage_dtype") or ""),
        "shape": [int(item) for item in precision.get("storage_shape") or []],
        "raw_size": int(tensor.get("raw_size") or 0),
        "data": data if not copied else bytes(data),
        "zero_copy_view": bool(tensor.get("zero_copy_view")),
        "data_view_kind": str(tensor.get("data_view_kind") or ("bytes" if copied else "memoryview")),
    }


class LosslessTensorManifestDatasetAdapter:
    """Ordered batch loader for a .lynx manifest, gated behind explicit config."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        config: LosslessTensorManifestAdapterConfig | None = None,
    ) -> None:
        self.path = Path(manifest_path)
        self.config = config or LosslessTensorManifestAdapterConfig()
        self._reader: TensorGroupManifestReader | None = None

    def __enter__(self) -> "LosslessTensorManifestDatasetAdapter":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._reader is None and self.config.enabled:
            self._reader = TensorGroupManifestReader(self.path, eager=bool(self.config.eager_readers))

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def load_batch(self, sample_ids: Iterable[str]) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        wanted = [str(item) for item in sample_ids]
        report: dict[str, Any] = {
            "provider": "lynx_manifest_dataset_adapter_v1",
            "enabled": bool(self.config.enabled),
            "manifest_path": str(self.path),
            "requested_sample_count": len(wanted),
            "training_path_enabled": False,
            "resource_center_allowed": False,
            "fallback_to_raw_npz": True,
            "same_process_fallback_to_raw_npz": True,
        }
        if not self.config.enabled:
            report["reason"] = "disabled"
            return None, report
        if not self.path.is_file():
            report["reason"] = "manifest_missing"
            if self.config.strict:
                raise FileNotFoundError(f".lynx manifest missing: {self.path}")
            return None, report

        try:
            self.open()
            if self._reader is None:
                raise RuntimeError("manifest reader did not open")
            batch_plan = self._reader.plan_batch(wanted)
            if self.config.use_tensor_records:
                loaded = self._reader.load_tensors(
                    wanted,
                    verify_crc32=bool(self.config.verify_crc32),
                    copy_bytes=bool(self.config.copy_tensor_bytes),
                )
            else:
                loaded = self._reader.load_samples(
                    wanted,
                    verify_crc32=bool(self.config.verify_crc32),
                    copy_arrays=bool(self.config.copy_arrays),
                )
        except Exception as exc:
            report["reason"] = "manifest_decode_failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            if self.config.strict:
                raise
            return None, report

        rows: list[dict[str, Any]] = []
        missing_samples: list[str] = []
        for sample_id in wanted:
            tensors = loaded.get(sample_id)
            if tensors is None:
                missing_samples.append(sample_id)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "tensor_names": [],
                        "tensor_count": 0,
                        "tensors": {},
                    }
                )
                continue
            if self.config.use_tensor_records:
                records = {name: _tensor_record(name, tensor) for name, tensor in tensors.items()}
            else:
                records = {name: _array_record(name, array) for name, array in tensors.items()}
            rows.append(
                {
                    "sample_id": sample_id,
                    "tensor_names": list(records),
                    "tensor_count": len(records),
                    "tensors": records,
                }
            )

        report.update(
            {
                "ok": not missing_samples,
                "reason": "manifest_loaded" if not missing_samples else "sample_missing",
                "loaded_sample_count": len(wanted) - len(missing_samples),
                "missing_sample_count": len(missing_samples),
                "missing_sample_ids": missing_samples,
                "tensor_count": sum(int(row.get("tensor_count") or 0) for row in rows),
                "fallback_to_raw_npz": bool(missing_samples),
                "same_process_fallback_to_raw_npz": bool(missing_samples),
                "record_mode": "raw_tensor_records" if self.config.use_tensor_records else "numpy_arrays",
                "uses_raw_tensor_records": bool(self.config.use_tensor_records),
                "copy_tensor_bytes": bool(self.config.copy_tensor_bytes),
                "zero_copy_tensor_view_count": sum(
                    1
                    for row in rows
                    for tensor in (row.get("tensors") or {}).values()
                    if isinstance(tensor, dict) and tensor.get("zero_copy_view")
                ),
                "tensor_view_record_count": sum(
                    1
                    for row in rows
                    for tensor in (row.get("tensors") or {}).values()
                    if isinstance(tensor, dict) and tensor.get("data_view_kind") == "memoryview"
                ),
                "batch_plan": batch_plan,
                "batch_plan_shard_count": int(batch_plan.get("shard_count") or 0),
                "batch_plan_read_round_count": int(batch_plan.get("read_round_count") or 0),
                "batch_plan_missing_sample_count": int(batch_plan.get("missing_sample_count") or 0),
                "batch_plan_request_shard_transition_count": int(
                    batch_plan.get("request_shard_transition_count") or 0
                ),
                "batch_plan_groups": batch_plan.get("groups") or [],
            }
        )
        if missing_samples and self.config.strict:
            raise KeyError(f".lynx manifest missing sample ids: {missing_samples}")
        return rows, report


def load_lossless_tensor_manifest_batch_for_dataset(
    manifest_path: str | Path,
    sample_ids: Iterable[str],
    *,
    config: LosslessTensorManifestAdapterConfig | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    with LosslessTensorManifestDatasetAdapter(manifest_path, config=config) as adapter:
        return adapter.load_batch(sample_ids)


__all__ = [
    "LosslessTensorManifestAdapterConfig",
    "LosslessTensorManifestDatasetAdapter",
    "load_lossless_tensor_manifest_batch_for_dataset",
]
