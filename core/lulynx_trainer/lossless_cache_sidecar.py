"""Lossless cache sidecar prototype for Windows cache/native decode research.

This module deliberately stays outside the trainer hot path.  It lets us
encode cached tensor payloads into a bytewise-lossless sidecar, decode them
back, and measure whether fast native codecs are worth wiring into prefetch
and H2D later.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping

try:
    from .lossless_tensor_block import (
        DEFAULT_FAST_CACHE_CODECS,
        decode_lossless_tensor_block,
        encode_lossless_tensor_block,
    )
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_tensor_block import (
        DEFAULT_FAST_CACHE_CODECS,
        decode_lossless_tensor_block,
        encode_lossless_tensor_block,
    )


MAGIC = b"LXCS1\n"
LXTB_MAGIC = b"LXTB1\n"
SIDECAR_SUFFIX = ".lxcs"


@dataclass(frozen=True)
class LosslessCacheEntry:
    name: str
    data: bytes | bytearray | memoryview
    element_size: int = 1
    metadata: Mapping[str, Any] | None = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _parse_sidecar(container: bytes | bytearray | memoryview) -> tuple[dict[str, Any], int, bytes]:
    payload = bytes(container)
    if not payload.startswith(MAGIC):
        raise ValueError("not an LXCS v1 cache sidecar")
    header_offset = len(MAGIC)
    header_size = struct.unpack_from("<I", payload, header_offset)[0]
    body_offset = header_offset + 4 + int(header_size)
    header = json.loads(payload[header_offset + 4 : body_offset].decode("utf-8"))
    if header.get("format") != "LXCS" or int(header.get("version") or 0) != 1:
        raise ValueError("unsupported LXCS sidecar version")
    return header, body_offset, payload


def _read_lxtb_header(container: bytes) -> dict[str, Any]:
    if not container.startswith(LXTB_MAGIC):
        return {}
    header_offset = len(LXTB_MAGIC)
    header_size = struct.unpack_from("<I", container, header_offset)[0]
    start = header_offset + 4
    return json.loads(container[start : start + int(header_size)].decode("utf-8"))


def _lxtb_codec_counts(container: bytes) -> dict[str, int]:
    header = _read_lxtb_header(container)
    counts: dict[str, int] = {}
    for chunk in header.get("chunks") or []:
        codec = str(chunk.get("codec") or "")
        if not codec:
            continue
        counts[codec] = counts.get(codec, 0) + 1
    return counts


def _compact_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(json.dumps(dict(value), default=str))


def encode_lossless_cache_sidecar(
    entries: Iterable[LosslessCacheEntry],
    *,
    chunk_size: int = 1 << 20,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
    min_saving: float = 0.02,
) -> bytes:
    """Encode named cache payloads into an LXCS sidecar.

    The sidecar contains one LXTB blob per named payload.  It is safe to keep
    beside the original cache because decoding verifies each entry's SHA256
    and does not mutate the source cache file.
    """

    seen: set[str] = set()
    body = bytearray()
    header_entries: list[dict[str, Any]] = []
    raw_total = 0
    for entry in entries:
        name = str(entry.name or "").strip()
        if not name:
            raise ValueError("LXCS entry name is required")
        if name in seen:
            raise ValueError(f"duplicate LXCS entry name: {name}")
        seen.add(name)

        raw = bytes(entry.data)
        element_size = max(int(entry.element_size or 1), 1)
        encoded = encode_lossless_tensor_block(
            raw,
            element_size=element_size,
            chunk_size=chunk_size,
            codecs=codecs,
            min_saving=min_saving,
        )
        header_entries.append(
            {
                "name": name,
                "raw_size": len(raw),
                "encoded_size": len(encoded),
                "sha256": _sha256(raw),
                "element_size": element_size,
                "metadata": _compact_metadata(entry.metadata),
                "codec_counts": _lxtb_codec_counts(encoded),
            }
        )
        body.extend(encoded)
        raw_total += len(raw)

    header = {
        "format": "LXCS",
        "version": 1,
        "entry_count": len(header_entries),
        "total_raw_size": raw_total,
        "chunk_size": max(int(chunk_size), 1),
        "codecs": [str(codec) for codec in codecs],
        "entries": header_entries,
    }
    header_bytes = _json_bytes(header)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + bytes(body)


def inspect_lossless_cache_sidecar(container: bytes | bytearray | memoryview) -> dict[str, Any]:
    header, _body_offset, _payload = _parse_sidecar(container)
    return header


def decode_lossless_cache_sidecar(
    container: bytes | bytearray | memoryview,
    *,
    verify_sha256: bool = True,
) -> dict[str, bytes]:
    header, cursor, payload = _parse_sidecar(container)
    decoded: dict[str, bytes] = {}
    for entry in header.get("entries") or []:
        encoded_size = int(entry["encoded_size"])
        blob = payload[cursor : cursor + encoded_size]
        cursor += encoded_size
        raw = decode_lossless_tensor_block(blob)
        expected_size = int(entry.get("raw_size") or 0)
        if len(raw) != expected_size:
            raise ValueError(f"LXCS entry {entry.get('name')} decoded size mismatch")
        if verify_sha256 and _sha256(raw) != str(entry.get("sha256") or ""):
            raise ValueError(f"LXCS entry {entry.get('name')} checksum mismatch")
        decoded[str(entry["name"])] = raw
    return decoded


def analyze_lossless_cache_sidecar(
    entries: Iterable[LosslessCacheEntry],
    *,
    chunk_size: int = 1 << 20,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
    min_saving: float = 0.02,
) -> dict[str, Any]:
    entry_list = list(entries)
    container = encode_lossless_cache_sidecar(
        entry_list,
        chunk_size=chunk_size,
        codecs=codecs,
        min_saving=min_saving,
    )
    decoded = decode_lossless_cache_sidecar(container)
    raw_total = sum(len(bytes(entry.data)) for entry in entry_list)
    codec_counts: dict[str, int] = {}
    header = inspect_lossless_cache_sidecar(container)
    for entry in header.get("entries") or []:
        for codec, count in dict(entry.get("codec_counts") or {}).items():
            codec_counts[str(codec)] = codec_counts.get(str(codec), 0) + int(count)
    return {
        "format": "LXCS",
        "version": 1,
        "entry_count": len(entry_list),
        "raw_bytes": raw_total,
        "sidecar_bytes": len(container),
        "compression_ratio": round(len(container) / max(float(raw_total), 1.0), 6),
        "saved_bytes": raw_total - len(container),
        "codec_counts": codec_counts,
        "roundtrip_ok": all(decoded.get(entry.name) == bytes(entry.data) for entry in entry_list),
    }


def load_numpy_cache_entries(path: str | Path) -> tuple[LosslessCacheEntry, ...]:
    """Load `.npz` or `.npy` arrays as cache sidecar entries."""

    import numpy as np

    cache_path = Path(path)
    suffix = cache_path.suffix.lower()
    entries: list[LosslessCacheEntry] = []
    if suffix == ".npz":
        with np.load(cache_path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.ascontiguousarray(archive[name])
                if array.dtype.hasobject:
                    continue
                entries.append(
                    LosslessCacheEntry(
                        name=str(name),
                        data=array.tobytes(order="C"),
                        element_size=max(int(array.dtype.itemsize), 1),
                        metadata={
                            "source": str(cache_path),
                            "kind": "npz_array",
                            "dtype": str(array.dtype),
                            "shape": [int(item) for item in array.shape],
                        },
                    )
                )
        return tuple(entries)
    if suffix == ".npy":
        array = np.ascontiguousarray(np.load(cache_path, allow_pickle=False))
        if array.dtype.hasobject:
            raise ValueError(f"object dtype is not supported for LXCS: {cache_path}")
        return (
            LosslessCacheEntry(
                name=cache_path.stem,
                data=array.tobytes(order="C"),
                element_size=max(int(array.dtype.itemsize), 1),
                metadata={
                    "source": str(cache_path),
                    "kind": "npy_array",
                    "dtype": str(array.dtype),
                    "shape": [int(item) for item in array.shape],
                },
            ),
        )
    raise ValueError(f"LXCS numpy cache loader supports .npz/.npy, got: {cache_path}")


def decode_lossless_cache_sidecar_arrays(
    container: bytes | bytearray | memoryview,
    *,
    verify_sha256: bool = True,
    copy_arrays: bool = True,
) -> dict[str, Any]:
    """Decode entries with numpy dtype/shape metadata back into arrays."""

    import numpy as np

    header, cursor, payload = _parse_sidecar(container)
    arrays: dict[str, Any] = {}
    for entry in header.get("entries") or []:
        encoded_size = int(entry["encoded_size"])
        blob = payload[cursor : cursor + encoded_size]
        cursor += encoded_size
        raw = decode_lossless_tensor_block(blob)
        expected_size = int(entry.get("raw_size") or 0)
        if len(raw) != expected_size:
            raise ValueError(f"LXCS entry {entry.get('name')} decoded size mismatch")
        if verify_sha256 and _sha256(raw) != str(entry.get("sha256") or ""):
            raise ValueError(f"LXCS entry {entry.get('name')} checksum mismatch")

        name = str(entry["name"])
        metadata = dict(entry.get("metadata") or {})
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        if not dtype or not isinstance(shape, list):
            continue
        array = np.frombuffer(raw, dtype=np.dtype(str(dtype))).reshape(tuple(shape))
        arrays[name] = array.copy() if copy_arrays else array
    return arrays


def sidecar_path_for_cache(path: str | Path, *, suffix: str = SIDECAR_SUFFIX) -> Path:
    cache_path = Path(path)
    return cache_path.with_name(f"{cache_path.name}{suffix}")


def write_lossless_cache_sidecar_file(
    cache_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    chunk_size: int = 1 << 20,
    codecs: Iterable[str] = DEFAULT_FAST_CACHE_CODECS,
    min_saving: float = 0.02,
) -> dict[str, Any]:
    cache = Path(cache_path)
    target = Path(sidecar_path) if sidecar_path is not None else sidecar_path_for_cache(cache)
    entries = load_numpy_cache_entries(cache)
    blob = encode_lossless_cache_sidecar(
        entries,
        chunk_size=chunk_size,
        codecs=codecs,
        min_saving=min_saving,
    )
    target.write_bytes(blob)
    raw_bytes = sum(len(bytes(entry.data)) for entry in entries)
    return {
        "ok": True,
        "source": str(cache),
        "sidecar_path": str(target),
        "entry_count": len(entries),
        "raw_bytes": raw_bytes,
        "sidecar_bytes": len(blob),
        "compression_ratio": round(len(blob) / max(float(raw_bytes), 1.0), 6),
    }


def load_lossless_cache_sidecar_arrays_from_file(
    path: str | Path,
    *,
    verify_sha256: bool = True,
    copy_arrays: bool = True,
) -> dict[str, Any]:
    return decode_lossless_cache_sidecar_arrays(
        Path(path).read_bytes(),
        verify_sha256=verify_sha256,
        copy_arrays=copy_arrays,
    )


__all__ = [
    "LosslessCacheEntry",
    "analyze_lossless_cache_sidecar",
    "decode_lossless_cache_sidecar",
    "decode_lossless_cache_sidecar_arrays",
    "encode_lossless_cache_sidecar",
    "inspect_lossless_cache_sidecar",
    "load_numpy_cache_entries",
    "load_lossless_cache_sidecar_arrays_from_file",
    "sidecar_path_for_cache",
    "write_lossless_cache_sidecar_file",
]
