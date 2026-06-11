# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Group/shard tensor container prototype for cache fragmentation research.

This module is deliberately not wired into training. It packs tensors from
multiple numpy cache files into one versioned shard with a JSON index and a
contiguous raw payload area, so devtools can measure whether grouped reads
beat per-image ``.npz`` loading before any runtime integration is considered.
"""

from __future__ import annotations

import json
import mmap
import os
from pathlib import Path
import struct
from typing import Any, Iterable
import zlib

try:
    from .lossless_cache_sidecar import LosslessCacheEntry, load_numpy_cache_entries
    from .lossless_cache_tensor_manifest_validation import manifest_sample_index_crc32
except ImportError:  # pragma: no cover - direct script loading
    from lossless_cache_sidecar import LosslessCacheEntry, load_numpy_cache_entries
    from lossless_cache_tensor_manifest_validation import manifest_sample_index_crc32


MAGIC = b"LYNXTC1\n"
FORMAT_NAME = "LYNX_TENSOR_CACHE"
MANIFEST_FORMAT_NAME = "LYNX_TENSOR_CACHE_MANIFEST"
GROUP_SUFFIX = ".lynx"
MANIFEST_SUFFIX = ".lynx.json"
CODEC_PRESETS: dict[str, tuple[str, ...]] = {
    "raw": ("raw",),
    "zlib": ("zlib",),
    "zstd1": ("zstd1",),
    "lz4fast": ("lz4fast",),
    "fast-cache": ("zstd1", "lz4fast", "zlib"),
}
PRECISION_FORMATS: dict[str, dict[str, Any]] = {
    "bool": {"logical_bits": 1, "storage_dtype": "bool", "storage_bits": 8, "packing": "numpy_bool"},
    "uint8": {"logical_bits": 8, "storage_dtype": "uint8", "storage_bits": 8, "packing": "none"},
    "int8": {"logical_bits": 8, "storage_dtype": "int8", "storage_bits": 8, "packing": "none"},
    "uint16": {"logical_bits": 16, "storage_dtype": "uint16", "storage_bits": 16, "packing": "none"},
    "int16": {"logical_bits": 16, "storage_dtype": "int16", "storage_bits": 16, "packing": "none"},
    "uint32": {"logical_bits": 32, "storage_dtype": "uint32", "storage_bits": 32, "packing": "none"},
    "int32": {"logical_bits": 32, "storage_dtype": "int32", "storage_bits": 32, "packing": "none"},
    "uint64": {"logical_bits": 64, "storage_dtype": "uint64", "storage_bits": 64, "packing": "none"},
    "int64": {"logical_bits": 64, "storage_dtype": "int64", "storage_bits": 64, "packing": "none"},
    "float16": {"logical_bits": 16, "storage_dtype": "float16", "storage_bits": 16, "packing": "none"},
    "bfloat16": {"logical_bits": 16, "storage_dtype": "uint16", "storage_bits": 16, "packing": "bf16_bits"},
    "float32": {"logical_bits": 32, "storage_dtype": "float32", "storage_bits": 32, "packing": "none"},
    "float64": {"logical_bits": 64, "storage_dtype": "float64", "storage_bits": 64, "packing": "none"},
    "fp8_e4m3": {"logical_bits": 8, "storage_dtype": "uint8", "storage_bits": 8, "packing": "fp8_e4m3_byte"},
    "fp8_e5m2": {"logical_bits": 8, "storage_dtype": "uint8", "storage_bits": 8, "packing": "fp8_e5m2_byte"},
    "fp4_e2m1": {"logical_bits": 4, "storage_dtype": "uint8", "storage_bits": 8, "packing": "two_fp4_per_byte"},
    "nvfp4": {"logical_bits": 4, "storage_dtype": "uint8", "storage_bits": 8, "packing": "two_nvfp4_per_byte"},
}
DTYPE_ALIASES = {
    "fp16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "float": "float32",
    "fp32": "float32",
    "double": "float64",
    "fp64": "float64",
    "fp8": "fp8_e4m3",
    "float8": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
    "fp4": "fp4_e2m1",
}
NUMPY_STORAGE_DTYPES = {
    "bool",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "float16",
    "float32",
    "float64",
}


def _optional_module(name: str) -> Any:
    try:
        import importlib

        return importlib.import_module(name)
    except Exception:
        return None


def _crc32(data: bytes | bytearray | memoryview) -> int:
    return int(zlib.crc32(data) & 0xFFFFFFFF)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _repoish(path: str | Path) -> str:
    return str(Path(path))


def _compact_metadata(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(json.dumps(dict(value), default=str))


def _sample_id(path: Path, index: int) -> str:
    return f"{index:06d}:{path.name}"


def _normalize_dtype(value: Any, default: str = "opaque_bytes") -> str:
    token = str(value or default).replace("torch.", "").replace("np.", "").strip().lower()
    return DTYPE_ALIASES.get(token, token)


def _shape_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    shape: list[int] = []
    for item in value:
        try:
            shape.append(int(item))
        except Exception:
            return []
    return shape


def _precision_descriptor(entry: LosslessCacheEntry, raw_size: int) -> dict[str, Any]:
    metadata = dict(entry.metadata or {})
    logical_dtype = _normalize_dtype(
        metadata.get("logical_dtype") or metadata.get("precision") or metadata.get("dtype")
    )
    spec = PRECISION_FORMATS.get(logical_dtype, {})
    vendor_format = str(metadata.get("vendor_format") or metadata.get("quant_format") or "").strip()
    storage_dtype = _normalize_dtype(
        metadata.get("storage_dtype") or metadata.get("dtype") or spec.get("storage_dtype")
    )
    if storage_dtype == "opaque_bytes":
        storage_dtype = "uint8"
    logical_shape = _shape_list(metadata.get("logical_shape") or metadata.get("shape"))
    storage_shape = _shape_list(metadata.get("storage_shape") or metadata.get("shape"))
    element_size = max(int(entry.element_size or metadata.get("element_size") or 1), 1)
    storage_bits = int(metadata.get("storage_bits") or spec.get("storage_bits") or element_size * 8)
    logical_bits = int(metadata.get("logical_bits") or spec.get("logical_bits") or storage_bits)
    packing = str(metadata.get("packing") or spec.get("packing") or "none")
    byte_order = str(metadata.get("byte_order") or ("<" if element_size > 1 else "|"))
    precision_params = metadata.get("precision_params") or metadata.get("quant_params") or {}
    if not isinstance(precision_params, dict):
        precision_params = {"value": str(precision_params)}
    numpy_array_compatible = bool(
        storage_dtype in NUMPY_STORAGE_DTYPES
        and storage_shape
        and packing in {"none", "numpy_bool"}
        and logical_dtype in NUMPY_STORAGE_DTYPES.union({"bool"})
    )
    return {
        "logical_dtype": logical_dtype,
        "storage_dtype": storage_dtype,
        "vendor_format": vendor_format,
        "logical_bits": logical_bits,
        "storage_bits": storage_bits,
        "element_size": element_size,
        "logical_shape": logical_shape,
        "storage_shape": storage_shape,
        "packing": packing,
        "precision_params": _compact_metadata(precision_params),
        "byte_order": byte_order,
        "raw_size": int(raw_size),
        "known_precision": bool(spec),
        "numpy_array_compatible": numpy_array_compatible,
    }


def _codec_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in CODEC_PRESETS:
            return CODEC_PRESETS[key]
        return tuple(item.strip().lower() for item in key.split("+") if item.strip())
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _encode_payload(raw: bytes, codec: str) -> bytes | None:
    if codec == "raw":
        return raw
    if codec == "zlib":
        return zlib.compress(raw, level=1)
    if codec == "zstd1":
        zstd = _optional_module("zstandard")
        if zstd is None:
            return None
        return zstd.ZstdCompressor(level=1).compress(raw)
    if codec == "lz4fast":
        lz4_frame = _optional_module("lz4.frame")
        if lz4_frame is None:
            return None
        return lz4_frame.compress(raw, compression_level=0)
    return None


def _decode_payload(payload: bytes | memoryview, codec: str, raw_size: int) -> bytes | memoryview:
    if codec == "raw":
        return payload
    if codec == "zlib":
        return zlib.decompress(payload)
    if codec == "zstd1":
        zstd = _optional_module("zstandard")
        if zstd is None:
            raise ValueError("LYNX_TENSOR_CACHE zstd1 codec requires optional dependency: zstandard")
        return zstd.ZstdDecompressor().decompress(payload, max_output_size=raw_size)
    if codec == "lz4fast":
        lz4_frame = _optional_module("lz4.frame")
        if lz4_frame is None:
            raise ValueError("LYNX_TENSOR_CACHE lz4fast codec requires optional dependency: lz4")
        return lz4_frame.decompress(payload)
    raise ValueError(f"unsupported LYNX_TENSOR_CACHE codec: {codec}")


def _select_payload(raw: bytes, codecs: tuple[str, ...], min_saving: float) -> tuple[str, bytes]:
    candidates: list[tuple[str, bytes]] = [("raw", raw)]
    for codec in codecs:
        if codec == "raw":
            continue
        payload = _encode_payload(raw, codec)
        if payload is not None:
            candidates.append((codec, payload))
    codec, payload = min(candidates, key=lambda item: len(item[1]))
    if codec != "raw" and len(payload) <= len(raw) * (1.0 - float(min_saving)):
        return codec, payload
    return "raw", raw


def _parse_container(container: bytes | bytearray | memoryview) -> tuple[dict[str, Any], int, memoryview]:
    payload = memoryview(container)
    if not payload[: len(MAGIC)].tobytes() == MAGIC:
        raise ValueError("not a LYNX_TENSOR_CACHE v1 tensor container")
    header_offset = len(MAGIC)
    header_size = struct.unpack_from("<Q", payload, header_offset)[0]
    body_offset = header_offset + 8 + int(header_size)
    header = json.loads(payload[header_offset + 8 : body_offset].tobytes().decode("utf-8"))
    if header.get("format") != FORMAT_NAME or int(header.get("version") or 0) != 1:
        raise ValueError("unsupported LYNX_TENSOR_CACHE tensor container version")
    return header, body_offset, payload


def encode_tensor_group_container(
    groups: Iterable[tuple[str, str | Path, Iterable[LosslessCacheEntry]]],
    *,
    codecs: str | Iterable[str] = "raw",
    min_saving: float = 0.0,
) -> bytes:
    """Encode grouped tensor entries into one contiguous shard.

    ``groups`` contains ``(sample_id, source_path, entries)`` tuples. Payloads
    are stored contiguous in group order. The default codec is raw so existing
    probes isolate layout effects; codec matrices can opt into compression.
    """

    codec_tuple = _codec_tuple(codecs)
    body = bytearray()
    samples: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    codec_counts: dict[str, int] = {}
    for group_index, (sample, source, entries) in enumerate(groups):
        source_text = _repoish(source)
        sample_text = str(sample or "").strip() or f"{group_index:06d}:{Path(source_text).name}"
        tensor_start = len(tensors)
        raw_bytes = 0
        for entry in entries:
            name = str(entry.name or "").strip()
            if not name:
                raise ValueError("LYNX_TENSOR_CACHE tensor name is required")
            key = (sample_text, name)
            if key in seen:
                raise ValueError(f"duplicate LYNX_TENSOR_CACHE tensor key: {sample_text}/{name}")
            seen.add(key)
            raw = bytes(entry.data)
            precision = _precision_descriptor(entry, len(raw))
            codec, payload = _select_payload(raw, codec_tuple, float(min_saving))
            codec_counts[codec] = codec_counts.get(codec, 0) + 1
            offset = len(body)
            body.extend(payload)
            tensors.append(
                {
                    "sample_id": sample_text,
                    "source": source_text,
                    "name": name,
                    "offset": offset,
                    "raw_size": len(raw),
                    "encoded_size": len(payload),
                    "codec": codec,
                    "crc32": _crc32(raw),
                    "metadata": _compact_metadata(entry.metadata),
                    "precision": precision,
                    "logical_dtype": precision["logical_dtype"],
                    "storage_dtype": precision["storage_dtype"],
                    "logical_bits": precision["logical_bits"],
                    "storage_bits": precision["storage_bits"],
                    "logical_shape": precision["logical_shape"],
                    "storage_shape": precision["storage_shape"],
                    "packing": precision["packing"],
                    "vendor_format": precision["vendor_format"],
                    "precision_params": precision["precision_params"],
                    "known_precision": precision["known_precision"],
                    "numpy_array_compatible": precision["numpy_array_compatible"],
                }
            )
            raw_bytes += len(raw)
        samples.append(
            {
                "sample_id": sample_text,
                "source": source_text,
                "tensor_start": tensor_start,
                "tensor_count": len(tensors) - tensor_start,
                "raw_bytes": raw_bytes,
            }
        )
    header = {
        "format": FORMAT_NAME,
        "version": 1,
        "layout": "group_shard_contiguous",
        "sample_count": len(samples),
        "tensor_count": len(tensors),
        "total_raw_size": sum(int(sample["raw_bytes"]) for sample in samples),
        "total_encoded_size": len(body),
        "codecs": list(codec_tuple),
        "codec_counts": codec_counts,
        "precision_contract_version": 1,
        "supported_precision_formats": sorted(PRECISION_FORMATS),
        "samples": samples,
        "tensors": tensors,
    }
    header_bytes = _json_bytes(header)
    return MAGIC + struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(body)


def inspect_tensor_group_container(container: bytes | bytearray | memoryview) -> dict[str, Any]:
    header, _body_offset, _payload = _parse_container(container)
    return header


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload)
        inspect_tensor_group_container(tmp.read_bytes())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload)
        parsed = json.loads(tmp.read_text(encoding="utf-8"))
        if parsed.get("format") != value.get("format") or int(parsed.get("version") or 0) != int(
            value.get("version") or 0
        ):
            raise ValueError("manifest validation failed before atomic publish")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def decode_tensor_group_container_arrays(
    container: bytes | bytearray | memoryview,
    *,
    sample_ids: set[str] | None = None,
    verify_crc32: bool = True,
    copy_arrays: bool = True,
) -> dict[str, dict[str, Any]]:
    """Decode arrays grouped by sample id."""

    import numpy as np

    header, body_offset, payload = _parse_container(container)
    wanted = set(sample_ids or [])
    output: dict[str, dict[str, Any]] = {}
    for tensor in header.get("tensors") or []:
        sample_id = str(tensor.get("sample_id") or "")
        if wanted and sample_id not in wanted:
            continue
        offset = body_offset + int(tensor.get("offset") or 0)
        encoded_size = int(tensor.get("encoded_size") or 0)
        raw_size = int(tensor.get("raw_size") or 0)
        raw = payload[offset : offset + encoded_size]
        codec = str(tensor.get("codec") or "raw")
        raw = _decode_payload(raw, codec, raw_size)
        if len(raw) != raw_size:
            raise ValueError(f"LYNX_TENSOR_CACHE tensor {sample_id}/{tensor.get('name')} size mismatch")
        if verify_crc32 and _crc32(raw) != int(tensor.get("crc32") or 0):
            raise ValueError(f"LYNX_TENSOR_CACHE tensor {sample_id}/{tensor.get('name')} checksum mismatch")
        precision = dict(tensor.get("precision") or {})
        if not bool(precision.get("numpy_array_compatible")):
            continue
        dtype = precision.get("storage_dtype")
        shape = precision.get("storage_shape")
        if not dtype or not isinstance(shape, list):
            continue
        array = np.frombuffer(raw, dtype=np.dtype(str(dtype))).reshape(tuple(int(item) for item in shape))
        output.setdefault(sample_id, {})[str(tensor.get("name") or "")] = (
            array.copy() if copy_arrays else array
        )
    return output


def _write_tensor_group_blob(
    groups: list[tuple[str, str | Path, tuple[LosslessCacheEntry, ...]]],
    *,
    container_path: str | Path,
    codecs: str | Iterable[str],
    min_saving: float,
    input_mode: str,
) -> dict[str, Any]:
    blob = encode_tensor_group_container(groups, codecs=codecs, min_saving=min_saving)
    target = Path(container_path)
    _atomic_write_bytes(target, blob)
    header = inspect_tensor_group_container(blob)
    raw_bytes = int(header.get("total_source_raw_size") or header.get("total_raw_size") or 0)
    return {
        "ok": bool(groups),
        "container_path": str(target),
        "input_mode": input_mode,
        "npz_intermediate_required": input_mode != "native_entries",
        "atomic_publish": True,
        "immutable_shard": True,
        "sample_count": len(groups),
        "tensor_count": int(header.get("tensor_count") or 0),
        "raw_bytes": raw_bytes,
        "container_bytes": len(blob),
        "compression_ratio": round(len(blob) / max(float(raw_bytes), 1.0), 6),
        "codec_counts": header.get("codec_counts") or {},
        "codecs": header.get("codecs") or [],
    }


def write_tensor_group_container_entries_file(
    groups: Iterable[tuple[str, str | Path, Iterable[LosslessCacheEntry]]],
    *,
    container_path: str | Path,
    codecs: str | Iterable[str] = "raw",
    min_saving: float = 0.0,
) -> dict[str, Any]:
    """Write generated tensor entries directly to a .lynx shard.

    This is the native cache-writer contract for future dataset/cache builders:
    callers pass already-generated tensor payloads, so no .npz intermediate is
    required. The shard remains immutable and is atomically published.
    """

    materialized: list[tuple[str, str | Path, tuple[LosslessCacheEntry, ...]]] = []
    for index, (sample, source, entries) in enumerate(groups):
        entry_tuple = tuple(entries)
        if not entry_tuple:
            continue
        sample_text = str(sample or "").strip() or f"{index:06d}:{Path(str(source)).name}"
        materialized.append((sample_text, source, entry_tuple))
    return _write_tensor_group_blob(
        materialized,
        container_path=container_path,
        codecs=codecs,
        min_saving=min_saving,
        input_mode="native_entries",
    )


def write_tensor_group_container_file(
    cache_paths: Iterable[str | Path],
    *,
    container_path: str | Path,
    codecs: str | Iterable[str] = "raw",
    min_saving: float = 0.0,
) -> dict[str, Any]:
    groups: list[tuple[str, Path, tuple[LosslessCacheEntry, ...]]] = []
    for index, pathish in enumerate(cache_paths):
        path = Path(pathish)
        entries = load_numpy_cache_entries(path)
        if not entries:
            continue
        groups.append((_sample_id(path, index), path, entries))
    return _write_tensor_group_blob(
        groups,
        container_path=container_path,
        codecs=codecs,
        min_saving=min_saving,
        input_mode="npz_paths",
    )


def write_tensor_group_manifest_file(
    shard_paths: Iterable[str | Path],
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    codec_counts: dict[str, int] = {}
    total_raw = 0
    total_encoded = 0
    sample_count = 0
    tensor_count = 0
    for index, pathish in enumerate(shard_paths):
        path = Path(pathish)
        header = inspect_tensor_group_container(path.read_bytes())
        shard_codec_counts = dict(header.get("codec_counts") or {})
        for codec, count in shard_codec_counts.items():
            codec_counts[str(codec)] = codec_counts.get(str(codec), 0) + int(count)
        shard_sample_count = int(header.get("sample_count") or 0)
        shard_tensor_count = int(header.get("tensor_count") or 0)
        shard_raw = int(header.get("total_raw_size") or 0)
        shard_encoded = int(header.get("total_encoded_size") or 0)
        shards.append(
            {
                "shard_index": index,
                "path": _repoish(path),
                "sample_count": shard_sample_count,
                "tensor_count": shard_tensor_count,
                "total_raw_size": shard_raw,
                "total_encoded_size": shard_encoded,
                "codec_counts": shard_codec_counts,
                "sample_ids": [
                    str(sample.get("sample_id") or "")
                    for sample in header.get("samples") or []
                    if sample.get("sample_id")
                ],
            }
        )
        sample_count += shard_sample_count
        tensor_count += shard_tensor_count
        total_raw += shard_raw
        total_encoded += shard_encoded
    manifest = {
        "format": MANIFEST_FORMAT_NAME,
        "version": 1,
        "shard_policy": "immutable_shards_manifest_swap",
        "append_policy": "new_shard_then_atomic_manifest_publish",
        "shard_count": len(shards),
        "sample_count": sample_count,
        "tensor_count": tensor_count,
        "total_raw_size": total_raw,
        "total_encoded_size": total_encoded,
        "codec_counts": codec_counts,
        "shards": shards,
    }
    manifest["sample_index_crc32"] = manifest_sample_index_crc32(manifest)
    target = Path(manifest_path)
    _atomic_write_json(target, manifest)
    return {
        "ok": True,
        "manifest_path": str(target),
        "atomic_publish": True,
        "append_policy": manifest["append_policy"],
        "shard_count": len(shards),
        "sample_count": sample_count,
        "tensor_count": tensor_count,
        "total_raw_size": total_raw,
        "total_encoded_size": total_encoded,
    }


def load_tensor_group_container_arrays_from_file(
    path: str | Path,
    *,
    sample_ids: set[str] | None = None,
    verify_crc32: bool = True,
    copy_arrays: bool = True,
) -> dict[str, dict[str, Any]]:
    return decode_tensor_group_container_arrays(
        Path(path).read_bytes(),
        sample_ids=sample_ids,
        verify_crc32=verify_crc32,
        copy_arrays=copy_arrays,
    )


def load_tensor_group_container_arrays_mmap_from_file(
    path: str | Path,
    *,
    sample_ids: set[str] | None = None,
    verify_crc32: bool = True,
) -> dict[str, dict[str, Any]]:
    """Decode a .lynx shard through mmap and return copied numpy arrays."""

    with Path(path).open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            return decode_tensor_group_container_arrays(
                mapped,
                sample_ids=sample_ids,
                verify_crc32=verify_crc32,
                copy_arrays=True,
            )


__all__ = [
    "GROUP_SUFFIX",
    "FORMAT_NAME",
    "MANIFEST_FORMAT_NAME",
    "MANIFEST_SUFFIX",
    "decode_tensor_group_container_arrays",
    "encode_tensor_group_container",
    "inspect_tensor_group_container",
    "load_tensor_group_container_arrays_from_file",
    "load_tensor_group_container_arrays_mmap_from_file",
    "write_tensor_group_manifest_file",
    "write_tensor_group_container_file",
    "write_tensor_group_container_entries_file",
]
