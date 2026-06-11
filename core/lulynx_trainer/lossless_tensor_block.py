"""Lossless tensor block container for PCIe/cache transfer research.

LXTB v1 is intentionally CPU-only and opt-in.  It provides an exact
round-trip container so we can measure compression candidates before wiring
anything into the trainer hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import math
import struct
import zlib
from typing import Any, Iterable


MAGIC = b"LXTB1\n"
DEFAULT_CODECS = ("bitpack_bool", "sparse_zero_bitmap", "rle_byte", "zlib")
FAST_CACHE_CODECS = ("zstd1", "zstd3", "zstd6", "lz4fast", "lz4hc")
DEFAULT_FAST_CACHE_CODECS = ("zstd1", "lz4fast", "sparse_zero_bitmap", "rle_byte", "zlib")


@dataclass(frozen=True)
class EncodedChunk:
    codec: str
    payload: bytes
    raw_size: int
    metadata: dict[str, Any]


def _crc32(data: bytes) -> int:
    return int(zlib.crc32(data) & 0xFFFFFFFF)


@lru_cache(maxsize=None)
def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def available_lossless_tensor_codecs() -> tuple[str, ...]:
    """Return codecs available in the current Python environment."""
    codecs = ["raw", "bitpack_bool", "sparse_zero_bitmap", "rle_byte", "zlib"]
    if _optional_module("zstandard") is not None:
        codecs.extend(["zstd1", "zstd3", "zstd6"])
    if _optional_module("lz4.frame") is not None:
        codecs.extend(["lz4fast", "lz4hc"])
    return tuple(codecs)


def _pack_bits(values: Iterable[int]) -> bytes:
    out = bytearray()
    current = 0
    bit = 0
    for value in values:
        if int(value):
            current |= 1 << bit
        bit += 1
        if bit == 8:
            out.append(current)
            current = 0
            bit = 0
    if bit:
        out.append(current)
    return bytes(out)


def _unpack_bits(data: bytes, count: int) -> bytes:
    out = bytearray(count)
    written = 0
    for item in data:
        value = int(item)
        for bit in range(8):
            if written >= count:
                return bytes(out)
            out[written] = 1 if (value >> bit) & 1 else 0
            written += 1
    return bytes(out)


def _encode_bitpack_bool(data: bytes) -> EncodedChunk | None:
    if not data or any(item not in (0, 1) for item in data):
        return None
    return EncodedChunk(
        "bitpack_bool",
        _pack_bits(data),
        len(data),
        {"logical_values": len(data)},
    )


def _decode_bitpack_bool(payload: bytes, raw_size: int, metadata: dict[str, Any]) -> bytes:
    count = int(metadata.get("logical_values") or raw_size)
    return _unpack_bits(payload, count)[:raw_size]


def _encode_sparse_zero_bitmap(data: bytes, *, element_size: int) -> EncodedChunk | None:
    if element_size <= 0 or not data or len(data) % element_size != 0:
        return None
    n_elements = len(data) // element_size
    bitmap_values: list[int] = []
    values = bytearray()
    zero = b"\0" * element_size
    nonzero = 0
    for index in range(n_elements):
        start = index * element_size
        item = data[start : start + element_size]
        if item == zero:
            bitmap_values.append(0)
            continue
        bitmap_values.append(1)
        values.extend(item)
        nonzero += 1
    if nonzero == n_elements:
        return None
    bitmap = _pack_bits(bitmap_values)
    return EncodedChunk(
        "sparse_zero_bitmap",
        bitmap + bytes(values),
        len(data),
        {
            "element_size": int(element_size),
            "n_elements": int(n_elements),
            "bitmap_bytes": len(bitmap),
            "nonzero_elements": int(nonzero),
        },
    )


def _decode_sparse_zero_bitmap(payload: bytes, raw_size: int, metadata: dict[str, Any]) -> bytes:
    element_size = int(metadata["element_size"])
    n_elements = int(metadata["n_elements"])
    bitmap_bytes = int(metadata["bitmap_bytes"])
    bitmap = payload[:bitmap_bytes]
    values = memoryview(payload[bitmap_bytes:])
    out = bytearray(raw_size)
    value_offset = 0
    for index, flag in enumerate(_unpack_bits(bitmap, n_elements)):
        if not flag:
            continue
        start = index * element_size
        end = start + element_size
        next_offset = value_offset + element_size
        out[start:end] = values[value_offset:next_offset]
        value_offset = next_offset
    return bytes(out)


def _encode_rle_byte(data: bytes) -> EncodedChunk | None:
    if not data:
        return None
    out = bytearray()
    index = 0
    size = len(data)
    while index < size:
        value = data[index]
        run = 1
        while index + run < size and data[index + run] == value and run < 65535:
            run += 1
        out.extend(struct.pack("<HB", run, value))
        index += run
    return EncodedChunk("rle_byte", bytes(out), len(data), {"unit": "byte", "run_width": 16})


def _decode_rle_byte(payload: bytes, raw_size: int, metadata: dict[str, Any]) -> bytes:
    out = bytearray()
    if len(payload) % 3:
        raise ValueError("invalid rle_byte payload length")
    for offset in range(0, len(payload), 3):
        run, value = struct.unpack_from("<HB", payload, offset)
        out.extend(bytes([value]) * int(run))
    if len(out) != raw_size:
        raise ValueError(f"rle_byte decoded {len(out)} bytes, expected {raw_size}")
    return bytes(out)


def _encode_zlib(data: bytes) -> EncodedChunk | None:
    if not data:
        return None
    return EncodedChunk("zlib", zlib.compress(data, level=1), len(data), {"level": 1})


def _encode_zstd(data: bytes, *, level: int) -> EncodedChunk | None:
    zstd = _optional_module("zstandard")
    if zstd is None or not data:
        return None
    payload = zstd.ZstdCompressor(level=level).compress(data)
    return EncodedChunk(f"zstd{level}", payload, len(data), {"level": int(level)})


def _decode_zstd(payload: bytes, raw_size: int) -> bytes:
    zstd = _optional_module("zstandard")
    if zstd is None:
        raise ValueError("LXTB zstd codec requires optional dependency: zstandard")
    return zstd.ZstdDecompressor().decompress(payload, max_output_size=raw_size)


def _encode_lz4(data: bytes, *, codec: str) -> EncodedChunk | None:
    lz4_frame = _optional_module("lz4.frame")
    if lz4_frame is None or not data:
        return None
    if codec == "lz4hc":
        payload = lz4_frame.compress(data, compression_level=9)
        return EncodedChunk(codec, payload, len(data), {"compression_level": 9})
    payload = lz4_frame.compress(data, compression_level=0)
    return EncodedChunk("lz4fast", payload, len(data), {"compression_level": 0})


def _decode_lz4(payload: bytes) -> bytes:
    lz4_frame = _optional_module("lz4.frame")
    if lz4_frame is None:
        raise ValueError("LXTB lz4 codec requires optional dependency: lz4")
    return lz4_frame.decompress(payload)


def _decode_chunk(codec: str, payload: bytes, raw_size: int, metadata: dict[str, Any]) -> bytes:
    if codec == "raw":
        return payload
    if codec == "bitpack_bool":
        return _decode_bitpack_bool(payload, raw_size, metadata)
    if codec == "sparse_zero_bitmap":
        return _decode_sparse_zero_bitmap(payload, raw_size, metadata)
    if codec == "rle_byte":
        return _decode_rle_byte(payload, raw_size, metadata)
    if codec == "zlib":
        return zlib.decompress(payload)
    if codec in {"zstd1", "zstd3", "zstd6"}:
        return _decode_zstd(payload, raw_size)
    if codec in {"lz4fast", "lz4hc"}:
        return _decode_lz4(payload)
    raise ValueError(f"unsupported LXTB codec: {codec}")


def _encode_chunk(
    data: bytes,
    *,
    element_size: int,
    codecs: Iterable[str],
    min_saving: float,
) -> EncodedChunk:
    enabled = {str(codec).strip().lower() for codec in codecs}
    candidates = [EncodedChunk("raw", data, len(data), {})]
    if "bitpack_bool" in enabled:
        item = _encode_bitpack_bool(data)
        if item is not None:
            candidates.append(item)
    if "sparse_zero_bitmap" in enabled:
        item = _encode_sparse_zero_bitmap(data, element_size=element_size)
        if item is not None:
            candidates.append(item)
    if "rle_byte" in enabled:
        item = _encode_rle_byte(data)
        if item is not None:
            candidates.append(item)
    if "zlib" in enabled:
        item = _encode_zlib(data)
        if item is not None:
            candidates.append(item)
    for level in (1, 3, 6):
        if f"zstd{level}" in enabled:
            item = _encode_zstd(data, level=level)
            if item is not None:
                candidates.append(item)
    for codec in ("lz4fast", "lz4hc"):
        if codec in enabled:
            item = _encode_lz4(data, codec=codec)
            if item is not None:
                candidates.append(item)

    best = min(candidates, key=lambda item: len(item.payload))
    raw_size = max(len(data), 1)
    if best.codec != "raw" and len(best.payload) <= raw_size * (1.0 - float(min_saving)):
        return best
    return candidates[0]


def encode_lossless_tensor_block(
    data: bytes | bytearray | memoryview,
    *,
    element_size: int = 1,
    chunk_size: int = 1 << 20,
    codecs: Iterable[str] = DEFAULT_CODECS,
    min_saving: float = 0.02,
) -> bytes:
    """Encode bytes into an exact LXTB v1 container.

    The encoder uses per-chunk best-of selection.  If no codec clears
    ``min_saving``, that chunk is stored as raw bytes.
    """

    raw = bytes(data)
    chunk_size = max(int(chunk_size), 1)
    element_size = max(int(element_size), 1)
    chunks: list[dict[str, Any]] = []
    body = bytearray()
    for index in range(0, len(raw), chunk_size):
        chunk = raw[index : index + chunk_size]
        encoded = _encode_chunk(chunk, element_size=element_size, codecs=codecs, min_saving=min_saving)
        chunks.append(
            {
                "codec": encoded.codec,
                "raw_size": int(encoded.raw_size),
                "encoded_size": len(encoded.payload),
                "crc32": _crc32(chunk),
                "metadata": encoded.metadata,
            }
        )
        body.extend(encoded.payload)
    header = {
        "format": "LXTB",
        "version": 1,
        "total_raw_size": len(raw),
        "element_size": element_size,
        "chunk_size": chunk_size,
        "chunks": chunks,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + bytes(body)


def decode_lossless_tensor_block(container: bytes | bytearray | memoryview) -> bytes:
    payload = bytes(container)
    if not payload.startswith(MAGIC):
        raise ValueError("not an LXTB v1 container")
    header_offset = len(MAGIC)
    header_size = struct.unpack_from("<I", payload, header_offset)[0]
    body_offset = header_offset + 4 + int(header_size)
    header = json.loads(payload[header_offset + 4 : body_offset].decode("utf-8"))
    out = bytearray()
    cursor = body_offset
    for chunk in header.get("chunks") or []:
        encoded_size = int(chunk["encoded_size"])
        raw_size = int(chunk["raw_size"])
        codec = str(chunk["codec"])
        encoded = payload[cursor : cursor + encoded_size]
        cursor += encoded_size
        decoded = _decode_chunk(codec, encoded, raw_size, dict(chunk.get("metadata") or {}))
        if len(decoded) != raw_size:
            raise ValueError(f"LXTB chunk decoded {len(decoded)} bytes, expected {raw_size}")
        if _crc32(decoded) != int(chunk["crc32"]):
            raise ValueError("LXTB chunk checksum mismatch")
        out.extend(decoded)
    if len(out) != int(header.get("total_raw_size") or 0):
        raise ValueError("LXTB decoded size mismatch")
    return bytes(out)


def analyze_lossless_tensor_block(
    data: bytes | bytearray | memoryview,
    *,
    element_size: int = 1,
    chunk_size: int = 1 << 20,
    codecs: Iterable[str] = DEFAULT_CODECS,
    min_saving: float = 0.02,
) -> dict[str, Any]:
    raw = bytes(data)
    container = encode_lossless_tensor_block(
        raw,
        element_size=element_size,
        chunk_size=chunk_size,
        codecs=codecs,
        min_saving=min_saving,
    )
    decoded = decode_lossless_tensor_block(container)
    if decoded != raw:
        raise AssertionError("LXTB round-trip mismatch")
    header_size = struct.unpack_from("<I", container, len(MAGIC))[0]
    header_start = len(MAGIC) + 4
    header = json.loads(container[header_start : header_start + header_size].decode("utf-8"))
    codec_counts: dict[str, int] = {}
    encoded_payload_bytes = 0
    for chunk in header["chunks"]:
        codec = str(chunk["codec"])
        codec_counts[codec] = codec_counts.get(codec, 0) + 1
        encoded_payload_bytes += int(chunk["encoded_size"])
    ratio = float(len(container)) / max(float(len(raw)), 1.0)
    return {
        "format": "LXTB",
        "version": 1,
        "raw_bytes": len(raw),
        "container_bytes": len(container),
        "encoded_payload_bytes": encoded_payload_bytes,
        "compression_ratio": round(ratio, 6),
        "saved_bytes": len(raw) - len(container),
        "chunk_count": len(header["chunks"]),
        "codec_counts": codec_counts,
        "element_size": int(element_size),
        "chunk_size": int(chunk_size),
        "roundtrip_ok": True,
    }


__all__ = [
    "DEFAULT_CODECS",
    "DEFAULT_FAST_CACHE_CODECS",
    "FAST_CACHE_CODECS",
    "available_lossless_tensor_codecs",
    "decode_lossless_tensor_block",
    "encode_lossless_tensor_block",
    "analyze_lossless_tensor_block",
]
