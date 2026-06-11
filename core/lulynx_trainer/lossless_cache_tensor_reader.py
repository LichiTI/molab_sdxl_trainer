# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Persistent mmap reader for research .lynx tensor cache shards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mmap
from pathlib import Path
from typing import Any, Iterable

from .lossless_cache_tensor_container import (
    MANIFEST_FORMAT_NAME,
    _crc32,
    _decode_payload,
    _parse_container,
    inspect_tensor_group_container,
)
from .lossless_cache_tensor_manifest_validation import validate_manifest_index
from .lossless_cache_sidecar import load_numpy_cache_entries


@dataclass(frozen=True)
class _TensorSpec:
    sample_id: str
    name: str
    offset: int
    end: int
    encoded_size: int
    raw_size: int
    codec: str
    crc32: int
    dtype: Any
    shape: tuple[int, ...]
    precision: dict[str, Any]
    metadata: dict[str, Any]
    numpy_array_compatible: bool


class TensorGroupContainerReader:
    """Keep a .lynx shard mmap and parsed tensor specs open for repeated reads."""

    def __init__(self, path: str | Path) -> None:
        import numpy as np

        self.path = Path(path)
        self._closed = False
        self._mmap_access = "read_only"
        self._handle = self.path.open("rb")
        self._mapped = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.header, self.body_offset, self._payload = _parse_container(self._mapped)
        specs: list[_TensorSpec] = []
        sample_specs: dict[str, list[_TensorSpec]] = {}
        array_specs: dict[str, list[_TensorSpec]] = {}
        for tensor in self.header.get("tensors") or []:
            precision = dict(tensor.get("precision") or {})
            numpy_array_compatible = bool(precision.get("numpy_array_compatible"))
            dtype = precision.get("storage_dtype")
            shape = precision.get("storage_shape")
            if numpy_array_compatible and dtype and isinstance(shape, list):
                storage_dtype = np.dtype(str(dtype))
                storage_shape = tuple(int(item) for item in shape)
            else:
                storage_dtype = None
                storage_shape = ()
            encoded_size = int(tensor.get("encoded_size") or 0)
            offset = self.body_offset + int(tensor.get("offset") or 0)
            spec = _TensorSpec(
                sample_id=str(tensor.get("sample_id") or ""),
                name=str(tensor.get("name") or ""),
                offset=offset,
                end=offset + encoded_size,
                encoded_size=encoded_size,
                raw_size=int(tensor.get("raw_size") or 0),
                codec=str(tensor.get("codec") or "raw"),
                crc32=int(tensor.get("crc32") or 0),
                dtype=storage_dtype,
                shape=storage_shape,
                precision=precision,
                metadata=dict(tensor.get("metadata") or {}),
                numpy_array_compatible=numpy_array_compatible and storage_dtype is not None,
            )
            specs.append(spec)
            sample_specs.setdefault(spec.sample_id, []).append(spec)
            if spec.numpy_array_compatible:
                array_specs.setdefault(spec.sample_id, []).append(spec)
        self.specs = tuple(specs)
        self._specs_by_sample = {sample: tuple(items) for sample, items in sample_specs.items()}
        self._array_specs_by_sample = {sample: tuple(items) for sample, items in array_specs.items()}
        sample_order = [
            str(sample.get("sample_id") or "")
            for sample in self.header.get("samples") or []
            if sample.get("sample_id")
        ]
        self._sample_order = tuple(sample_order or self._specs_by_sample)
        self._sample_order_index = {sample_id: index for index, sample_id in enumerate(self._sample_order)}

    def __enter__(self) -> "TensorGroupContainerReader":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return bool(getattr(self, "_closed", False))

    @property
    def mmap_access(self) -> str:
        return str(getattr(self, "_mmap_access", ""))

    def _ensure_open(self) -> None:
        if self.closed or getattr(self, "_payload", None) is None:
            raise RuntimeError("LYNX_TENSOR_CACHE persistent mmap reader is closed")

    def lifecycle_report(self) -> dict[str, Any]:
        mapped = getattr(self, "_mapped", None)
        handle = getattr(self, "_handle", None)
        closed = self.closed
        return {
            "provider": "lynx_tensor_group_container_persistent_mmap_reader_v1",
            "persistent_mmap_reader_ready": bool(
                not closed and mapped is not None and self.mmap_access == "read_only"
            ),
            "path": str(self.path),
            "mmap_access": self.mmap_access,
            "read_only": self.mmap_access == "read_only",
            "closed": closed,
            "mmap_open": mapped is not None and not closed,
            "file_handle_open": bool(handle is not None and not handle.closed and not closed),
            "sample_count": int(self.header.get("sample_count") or 0),
            "tensor_count": int(self.header.get("tensor_count") or 0),
            "training_path_enabled": False,
            "resource_center_allowed": False,
        }

    def close(self) -> None:
        payload = getattr(self, "_payload", None)
        if payload is not None:
            payload.release()
            self._payload = None
        mapped = getattr(self, "_mapped", None)
        if mapped is not None:
            mapped.close()
            self._mapped = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None
        self._closed = True

    def load_arrays(
        self,
        *,
        sample_ids: set[str] | None = None,
        verify_crc32: bool = True,
        copy_arrays: bool = True,
    ) -> dict[str, dict[str, Any]]:
        import numpy as np

        self._ensure_open()
        output: dict[str, dict[str, Any]] = {}
        for sample_id, specs in self._iter_sample_specs(sample_ids, arrays_only=True):
            sample_output: dict[str, Any] = {}
            for spec in specs:
                raw = self._payload[spec.offset : spec.end]
                raw = _decode_payload(raw, spec.codec, spec.raw_size)
                if len(raw) != spec.raw_size:
                    raise ValueError(f"LYNX_TENSOR_CACHE tensor {spec.sample_id}/{spec.name} size mismatch")
                if verify_crc32 and _crc32(raw) != spec.crc32:
                    raise ValueError(f"LYNX_TENSOR_CACHE tensor {spec.sample_id}/{spec.name} checksum mismatch")
                array = np.frombuffer(raw, dtype=spec.dtype).reshape(spec.shape)
                sample_output[spec.name] = array.copy() if copy_arrays else array
            if sample_output:
                output[sample_id] = sample_output
        return output

    def _iter_sample_specs(
        self,
        sample_ids: set[str] | None,
        *,
        arrays_only: bool,
    ) -> Iterable[tuple[str, tuple[_TensorSpec, ...]]]:
        specs_by_sample = self._array_specs_by_sample if arrays_only else self._specs_by_sample
        if sample_ids is None:
            for sample_id in self._sample_order:
                specs = specs_by_sample.get(sample_id)
                if specs:
                    yield sample_id, specs
            return
        wanted = {str(item) for item in sample_ids}
        ordered_wanted = sorted(
            (sample_id for sample_id in wanted if sample_id in specs_by_sample),
            key=lambda sample_id: self._sample_order_index.get(sample_id, len(self._sample_order)),
        )
        for sample_id in ordered_wanted:
            specs = specs_by_sample.get(sample_id)
            if specs:
                yield sample_id, specs

    def load_tensors(
        self,
        *,
        sample_ids: set[str] | None = None,
        verify_crc32: bool = True,
        copy_bytes: bool = True,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Load raw tensor payloads, including opaque packed precision formats."""

        self._ensure_open()
        output: dict[str, dict[str, dict[str, Any]]] = {}
        for sample_id, specs in self._iter_sample_specs(sample_ids, arrays_only=False):
            sample_output: dict[str, dict[str, Any]] = {}
            for spec in specs:
                raw = self._payload[spec.offset : spec.end]
                raw = _decode_payload(raw, spec.codec, spec.raw_size)
                if len(raw) != spec.raw_size:
                    raise ValueError(f"LYNX_TENSOR_CACHE tensor {spec.sample_id}/{spec.name} size mismatch")
                if verify_crc32 and _crc32(raw) != spec.crc32:
                    raise ValueError(f"LYNX_TENSOR_CACHE tensor {spec.sample_id}/{spec.name} checksum mismatch")
                sample_output[spec.name] = {
                    "data": bytes(raw) if copy_bytes else raw,
                    "raw_size": spec.raw_size,
                    "codec": spec.codec,
                    "crc32": spec.crc32,
                    "precision": dict(spec.precision),
                    "metadata": dict(spec.metadata),
                    "numpy_array_compatible": spec.numpy_array_compatible,
                    "zero_copy_view": bool(not copy_bytes and spec.codec == "raw"),
                    "data_view_kind": "memoryview" if not copy_bytes and spec.codec == "raw" else "bytes",
                }
            if sample_output:
                output[sample_id] = sample_output
        return output


def validate_tensor_group_manifest_index(
    manifest_path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate manifest sample-index checksum and shard/header parity."""

    return validate_manifest_index(
        manifest_path,
        manifest_format_name=MANIFEST_FORMAT_NAME,
        inspect_container=inspect_tensor_group_container,
        strict=strict,
    )


def trusted_materialized_decode_report(
    path: str | Path,
    *,
    verify_crc32: bool = True,
) -> dict[str, Any]:
    """Compare decoded .lynx tensor bytes with their source .npz entries."""

    rows: list[dict[str, Any]] = []
    missing_source_count = 0
    mismatch_count = 0
    with TensorGroupContainerReader(path) as reader:
        decoded = reader.load_tensors(verify_crc32=verify_crc32, copy_bytes=True)
        source_entries: dict[str, dict[str, bytes]] = {}
        for sample in reader.header.get("samples") or []:
            sample_id = str(sample.get("sample_id") or "")
            source_path = Path(str(sample.get("source") or ""))
            if not source_path.is_file():
                source_entries[sample_id] = {}
                missing_source_count += 1
                continue
            source_entries[sample_id] = {
                entry.name: bytes(entry.data)
                for entry in load_numpy_cache_entries(source_path)
            }
        for tensor in reader.header.get("tensors") or []:
            sample_id = str(tensor.get("sample_id") or "")
            name = str(tensor.get("name") or "")
            actual = bytes((decoded.get(sample_id) or {}).get(name, {}).get("data") or b"")
            expected = (source_entries.get(sample_id) or {}).get(name)
            matched = expected is not None and actual == expected
            if not matched:
                mismatch_count += 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "name": name,
                    "source_present": expected is not None,
                    "byte_for_byte_match": matched,
                    "raw_size": len(actual),
                    "source_raw_size": len(expected or b""),
                    "decoded_sha256": hashlib.sha256(actual).hexdigest(),
                    "source_sha256": hashlib.sha256(expected or b"").hexdigest(),
                }
            )
    ready = bool(rows) and missing_source_count == 0 and mismatch_count == 0
    return {
        "provider": "lynx_tensor_group_container_trusted_materialized_decode_v1",
        "ok": ready,
        "trusted_materialized_decode_ready": ready,
        "path": str(path),
        "tensor_count": len(rows),
        "byte_for_byte_match_count": sum(1 for row in rows if row["byte_for_byte_match"]),
        "missing_source_count": missing_source_count,
        "mismatch_count": mismatch_count,
        "rows": rows,
        "verify_crc32": bool(verify_crc32),
        "training_path_enabled": False,
        "resource_center_allowed": False,
    }


class TensorGroupManifestReader:
    """Keep per-shard .lynx readers behind a manifest sample index."""

    def __init__(self, manifest_path: str | Path, *, eager: bool = False) -> None:
        self.path = Path(manifest_path)
        self._closed = False
        self._eager_readers = bool(eager)
        self.index_validation_report = validate_tensor_group_manifest_index(self.path, strict=True)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != MANIFEST_FORMAT_NAME or int(self.manifest.get("version") or 0) != 1:
            raise ValueError("unsupported LYNX_TENSOR_CACHE_MANIFEST version")
        self._shard_paths: list[Path] = []
        self._readers: dict[int, TensorGroupContainerReader] = {}
        self._sample_to_shard: dict[str, int] = {}
        self._sample_order: list[str] = []
        for shard_index, shard in enumerate(self.manifest.get("shards") or []):
            shard_path = self._resolve_shard_path(str(shard.get("path") or ""))
            self._shard_paths.append(shard_path)
            for sample_id in shard.get("sample_ids") or []:
                sample_text = str(sample_id)
                if sample_text in self._sample_to_shard:
                    raise ValueError(f"duplicate LYNX_TENSOR_CACHE manifest sample_id: {sample_text}")
                self._sample_to_shard[sample_text] = shard_index
                self._sample_order.append(sample_text)
        if eager:
            for shard_index in range(len(self._shard_paths)):
                self._reader_for(shard_index)

    @property
    def shard_count(self) -> int:
        return len(self._shard_paths)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self._sample_order)

    @property
    def closed(self) -> bool:
        return bool(getattr(self, "_closed", False))

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("LYNX_TENSOR_CACHE manifest reader is closed")

    def lifecycle_report(self) -> dict[str, Any]:
        reader_reports = [
            reader.lifecycle_report()
            for reader in getattr(self, "_readers", {}).values()
        ]
        all_open_readers_read_only = all(
            bool(report.get("read_only")) and bool(report.get("mmap_open"))
            for report in reader_reports
        )
        open_reader_count = len(reader_reports)
        shard_count = len(getattr(self, "_shard_paths", []))
        return {
            "provider": "lynx_tensor_group_manifest_persistent_mmap_reader_v1",
            "persistent_mmap_reader_ready": bool(
                not self.closed
                and (open_reader_count == 0 or all_open_readers_read_only)
                and bool(self.index_validation_report.get("ok"))
            ),
            "manifest_path": str(self.path),
            "closed": self.closed,
            "eager_readers": bool(getattr(self, "_eager_readers", False)),
            "shard_count": shard_count,
            "sample_count": len(getattr(self, "_sample_order", [])),
            "open_reader_count": open_reader_count,
            "all_open_readers_read_only": all_open_readers_read_only,
            "reader_reports": reader_reports,
            "index_validation_ok": bool(self.index_validation_report.get("ok")),
            "training_path_enabled": False,
            "resource_center_allowed": False,
        }

    def plan_batch(self, sample_ids: Iterable[str] | None = None) -> dict[str, Any]:
        """Return the shard-local read plan while preserving request order."""

        self._ensure_open()
        wanted = [str(item) for item in sample_ids] if sample_ids is not None else list(self._sample_order)
        shard_batches: dict[int, list[str]] = {}
        missing: list[str] = []
        for sample_id in wanted:
            shard_index = self._sample_to_shard.get(sample_id)
            if shard_index is None:
                missing.append(sample_id)
                continue
            shard_batches.setdefault(shard_index, []).append(sample_id)
        transitions = 0
        previous: int | None = None
        for sample_id in wanted:
            shard_index = self._sample_to_shard.get(sample_id)
            if shard_index is None:
                continue
            if previous is not None and previous != shard_index:
                transitions += 1
            previous = shard_index
        return {
            "requested_sample_count": len(wanted),
            "known_sample_count": len(wanted) - len(missing),
            "missing_sample_count": len(missing),
            "missing_sample_ids": missing,
            "shard_count": len(shard_batches),
            "read_round_count": len(shard_batches),
            "request_shard_transition_count": transitions,
            "groups": [
                {
                    "shard_index": int(shard_index),
                    "sample_count": len(batch),
                    "sample_ids": list(batch),
                }
                for shard_index, batch in sorted(shard_batches.items())
            ],
        }

    def __enter__(self) -> "TensorGroupManifestReader":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def _resolve_shard_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidate = self.path.parent / path
        if candidate.exists():
            return candidate
        return path

    def _reader_for(self, shard_index: int) -> TensorGroupContainerReader:
        self._ensure_open()
        reader = self._readers.get(shard_index)
        if reader is None:
            reader = TensorGroupContainerReader(self._shard_paths[shard_index])
            self._readers[shard_index] = reader
        return reader

    def close(self) -> None:
        readers = list(getattr(self, "_readers", {}).values())
        self._readers.clear()
        for reader in readers:
            reader.close()
        self._closed = True

    def load_samples(
        self,
        sample_ids: Iterable[str] | None = None,
        *,
        verify_crc32: bool = True,
        copy_arrays: bool = True,
    ) -> dict[str, dict[str, Any]]:
        self._ensure_open()
        wanted = [str(item) for item in sample_ids] if sample_ids is not None else list(self._sample_order)
        by_shard: dict[int, set[str]] = {}
        for sample_id in wanted:
            shard_index = self._sample_to_shard.get(sample_id)
            if shard_index is None:
                continue
            by_shard.setdefault(shard_index, set()).add(sample_id)
        output: dict[str, dict[str, Any]] = {}
        for shard_index in sorted(by_shard):
            output.update(
                self._reader_for(shard_index).load_arrays(
                    sample_ids=by_shard[shard_index],
                    verify_crc32=verify_crc32,
                    copy_arrays=copy_arrays,
                )
            )
        return {sample_id: output[sample_id] for sample_id in wanted if sample_id in output}

    def load_tensors(
        self,
        sample_ids: Iterable[str] | None = None,
        *,
        verify_crc32: bool = True,
        copy_bytes: bool = True,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        self._ensure_open()
        wanted = [str(item) for item in sample_ids] if sample_ids is not None else list(self._sample_order)
        by_shard: dict[int, set[str]] = {}
        for sample_id in wanted:
            shard_index = self._sample_to_shard.get(sample_id)
            if shard_index is None:
                continue
            by_shard.setdefault(shard_index, set()).add(sample_id)
        output: dict[str, dict[str, dict[str, Any]]] = {}
        for shard_index in sorted(by_shard):
            output.update(
                self._reader_for(shard_index).load_tensors(
                    sample_ids=by_shard[shard_index],
                    verify_crc32=verify_crc32,
                    copy_bytes=copy_bytes,
                )
            )
        return {sample_id: output[sample_id] for sample_id in wanted if sample_id in output}


__all__ = [
    "TensorGroupContainerReader",
    "TensorGroupManifestReader",
    "trusted_materialized_decode_report",
    "validate_tensor_group_manifest_index",
]
