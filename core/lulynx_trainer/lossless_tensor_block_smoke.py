"""Smoke checks for the LXTB lossless tensor block prototype."""

from __future__ import annotations

import os
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_tensor_block import (  # type: ignore[no-redef]
        analyze_lossless_tensor_block,
        available_lossless_tensor_codecs,
        decode_lossless_tensor_block,
        encode_lossless_tensor_block,
    )
else:
    from .lossless_tensor_block import (
        analyze_lossless_tensor_block,
        available_lossless_tensor_codecs,
        decode_lossless_tensor_block,
        encode_lossless_tensor_block,
    )


def _roundtrip(
    data: bytes,
    *,
    element_size: int = 1,
    chunk_size: int = 64,
    codecs: tuple[str, ...] = ("bitpack_bool", "sparse_zero_bitmap", "rle_byte", "zlib"),
) -> dict[str, object]:
    blob = encode_lossless_tensor_block(data, element_size=element_size, chunk_size=chunk_size, codecs=codecs)
    assert decode_lossless_tensor_block(blob) == data
    return analyze_lossless_tensor_block(data, element_size=element_size, chunk_size=chunk_size, codecs=codecs)


def test_random_falls_back_to_raw() -> None:
    data = os.urandom(512)
    stats = _roundtrip(data)
    assert stats["roundtrip_ok"] is True
    assert stats["codec_counts"] == {"raw": 8}


def test_bool_bitpack() -> None:
    data = bytes([0, 1, 0, 0, 1, 1, 0, 1]) * 64
    stats = _roundtrip(data, codecs=("bitpack_bool",))
    assert stats["roundtrip_ok"] is True
    assert "bitpack_bool" in stats["codec_counts"]


def test_sparse_zero_bitmap_elements() -> None:
    element = (123).to_bytes(2, "little", signed=False)
    data = bytearray(2 * 256)
    for index in range(0, 256, 32):
        start = index * 2
        data[start : start + 2] = element
    stats = _roundtrip(bytes(data), element_size=2, codecs=("sparse_zero_bitmap",))
    assert stats["roundtrip_ok"] is True
    assert "sparse_zero_bitmap" in stats["codec_counts"]


def test_byte_rle() -> None:
    data = (b"a" * 96) + (b"b" * 80) + (b"c" * 48)
    stats = _roundtrip(data)
    assert stats["roundtrip_ok"] is True
    assert "rle_byte" in stats["codec_counts"]


def test_optional_fast_cache_codecs() -> None:
    available = set(available_lossless_tensor_codecs())
    codecs = tuple(codec for codec in ("zstd1", "lz4fast") if codec in available)
    if not codecs:
        return
    data = (b"latent-cache-block-" * 8192) + bytes(range(64)) * 128
    stats = _roundtrip(data, chunk_size=4096, codecs=codecs)
    assert stats["roundtrip_ok"] is True
    assert set(stats["codec_counts"]).issubset(set(codecs) | {"raw"})
    assert stats["compression_ratio"] < 0.8


def main() -> int:
    test_random_falls_back_to_raw()
    test_bool_bitpack()
    test_sparse_zero_bitmap_elements()
    test_byte_rle()
    test_optional_fast_cache_codecs()
    print("lossless_tensor_block_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
