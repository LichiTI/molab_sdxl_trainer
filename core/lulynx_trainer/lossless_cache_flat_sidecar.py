# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Flat LXCS sidecar prototype for cache decode overhead research.

This module is deliberately not wired into training. It removes the nested
LXTB-per-entry container used by ``lossless_cache_sidecar`` so we can measure
whether a single header plus contiguous entry payloads can get close to raw
``.npz`` load cost before considering native decode work.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable
import zlib

try:
    from .lossless_cache_sidecar import LosslessCacheEntry, load_numpy_cache_entries
except ImportError:  # pragma: no cover - direct script smoke loading
    from lossless_cache_sidecar import LosslessCacheEntry, load_numpy_cache_entries


MAGIC = b"LXFS1\n"
FLAT_SUFFIX = ".lxfs"
CODEC_PRESETS: dict[str, tuple[str, ...]] = {
    "raw": ("raw",),
    "zstd1": ("zstd1",),
    "lz4fast": ("lz4fast",),
    "fast-cache": ("zstd1", "lz4fast", "zlib"),
}


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _crc32(data: bytes | bytearray | memoryview) -> int:
    return int(zlib.crc32(data) & 0xFFFFFFFF)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _codec_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in CODEC_PRESETS:
            return CODEC_PRESETS[key]
        return tuple(item.strip().lower() for item in key.split("+") if item.strip())
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _compact_metadata(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(json.dumps(dict(value), default=str))


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
            raise ValueError("LXFS zstd1 codec requires optional dependency: zstandard")
        return zstd.ZstdDecompressor().decompress(payload, max_output_size=raw_size)
    if codec == "lz4fast":
        lz4_frame = _optional_module("lz4.frame")
        if lz4_frame is None:
            raise ValueError("LXFS lz4fast codec requires optional dependency: lz4")
        return lz4_frame.decompress(payload)
    raise ValueError(f"unsupported LXFS codec: {codec}")


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


def encode_flat_lossless_cache_sidecar(
    entries: Iterable[LosslessCacheEntry],
    *,
    codecs: str | Iterable[str] = "fast-cache",
    min_saving: float = 0.0,
) -> bytes:
    codec_tuple = _codec_tuple(codecs)
    body = bytearray()
    rows: list[dict[str, Any]] = []
    raw_total = 0
    seen: set[str] = set()
    for entry in entries:
        name = str(entry.name or "").strip()
        if not name:
            raise ValueError("LXFS entry name is required")
        if name in seen:
            raise ValueError(f"duplicate LXFS entry name: {name}")
        seen.add(name)
        raw = bytes(entry.data)
        codec, payload = _select_payload(raw, codec_tuple, float(min_saving))
        rows.append(
            {
                "name": name,
                "codec": codec,
                "raw_size": len(raw),
                "encoded_size": len(payload),
                "crc32": _crc32(raw),
                "metadata": _compact_metadata(entry.metadata),
            }
        )
        body.extend(payload)
        raw_total += len(raw)
    header = {
        "format": "LXFS",
        "version": 1,
        "entry_count": len(rows),
        "total_raw_size": raw_total,
        "codecs": list(codec_tuple),
        "entries": rows,
    }
    header_bytes = _json_bytes(header)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + bytes(body)


def inspect_flat_lossless_cache_sidecar(container: bytes | bytearray | memoryview) -> dict[str, Any]:
    payload = bytes(container)
    if not payload.startswith(MAGIC):
        raise ValueError("not an LXFS v1 cache sidecar")
    header_offset = len(MAGIC)
    header_size = struct.unpack_from("<I", payload, header_offset)[0]
    start = header_offset + 4
    header = json.loads(payload[start : start + int(header_size)].decode("utf-8"))
    if header.get("format") != "LXFS" or int(header.get("version") or 0) != 1:
        raise ValueError("unsupported LXFS sidecar version")
    return header


def decode_flat_lossless_cache_sidecar_arrays(
    container: bytes | bytearray | memoryview,
    *,
    verify_crc32: bool = True,
    copy_arrays: bool = True,
) -> dict[str, Any]:
    import numpy as np

    payload_bytes = bytes(container)
    if not payload_bytes.startswith(MAGIC):
        raise ValueError("not an LXFS v1 cache sidecar")
    header_offset = len(MAGIC)
    header_size = struct.unpack_from("<I", payload_bytes, header_offset)[0]
    body_offset = header_offset + 4 + int(header_size)
    header = json.loads(payload_bytes[header_offset + 4 : body_offset].decode("utf-8"))
    payload_view = memoryview(payload_bytes)
    cursor = body_offset
    arrays: dict[str, Any] = {}
    for entry in header.get("entries") or []:
        encoded_size = int(entry["encoded_size"])
        encoded = payload_view[cursor : cursor + encoded_size]
        cursor += encoded_size
        raw_size = int(entry.get("raw_size") or 0)
        raw = _decode_payload(encoded, str(entry.get("codec") or "raw"), raw_size)
        if len(raw) != raw_size:
            raise ValueError(f"LXFS entry {entry.get('name')} decoded size mismatch")
        if verify_crc32 and _crc32(raw) != int(entry.get("crc32") or 0):
            raise ValueError(f"LXFS entry {entry.get('name')} checksum mismatch")
        metadata = dict(entry.get("metadata") or {})
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        if not dtype or not isinstance(shape, list):
            continue
        array = np.frombuffer(raw, dtype=np.dtype(str(dtype))).reshape(tuple(shape))
        arrays[str(entry["name"])] = array.copy() if copy_arrays else array
    return arrays


def write_flat_lossless_cache_sidecar_file(
    cache_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    codecs: str | Iterable[str] = "fast-cache",
    min_saving: float = 0.0,
) -> dict[str, Any]:
    cache = Path(cache_path)
    target = Path(sidecar_path) if sidecar_path is not None else cache.with_name(f"{cache.name}{FLAT_SUFFIX}")
    entries = load_numpy_cache_entries(cache)
    blob = encode_flat_lossless_cache_sidecar(entries, codecs=codecs, min_saving=min_saving)
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


def load_flat_lossless_cache_sidecar_arrays_from_file(
    path: str | Path,
    *,
    verify_crc32: bool = True,
    copy_arrays: bool = True,
) -> dict[str, Any]:
    return decode_flat_lossless_cache_sidecar_arrays(
        Path(path).read_bytes(),
        verify_crc32=verify_crc32,
        copy_arrays=copy_arrays,
    )


__all__ = [
    "decode_flat_lossless_cache_sidecar_arrays",
    "encode_flat_lossless_cache_sidecar",
    "inspect_flat_lossless_cache_sidecar",
    "load_flat_lossless_cache_sidecar_arrays_from_file",
    "write_flat_lossless_cache_sidecar_file",
]
