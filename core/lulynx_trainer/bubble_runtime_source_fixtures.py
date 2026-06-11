"""Generated source-data fixtures for bubble runtime benchmarks."""

from __future__ import annotations

import json
import random
import struct
import zlib
from pathlib import Path
from typing import Any


def _write_png_chunk(handle: Any, chunk_type: bytes, data: bytes) -> None:
    handle.write(struct.pack(">I", len(data)))
    handle.write(chunk_type)
    handle.write(data)
    handle.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(chunk_type)) & 0xFFFFFFFF))


def _write_entropy_png(path: Path, *, width: int, height: int, seed: int) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    rng = random.Random(seed)
    with tmp_path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        _write_png_chunk(handle, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        compressor = zlib.compressobj(level=1)
        for _row_index in range(height):
            chunk = compressor.compress(b"\x00" + rng.randbytes(width * 3))
            if chunk:
                _write_png_chunk(handle, b"IDAT", chunk)
        chunk = compressor.flush()
        if chunk:
            _write_png_chunk(handle, b"IDAT", chunk)
        _write_png_chunk(handle, b"IEND", b"")
    tmp_path.replace(path)


def _valid_png_fixture_file(path: Path, *, min_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _caption_payload(index: int) -> dict[str, Any]:
    return {
        "concept": f"lulynx_sample_{index:02d}",
        "trigger": ["lulynx", "natural_data_wait"],
        "tags": ["high_entropy_png", "raw_decode", f"sample_{index:02d}"],
        "categories": {
            "quality": ["sharp", "detailed"],
            "style": ["benchmark_fixture", "non_injected"],
        },
        "nl": (
            "A deterministic high entropy training image used to measure "
            "natural data supply wait without benchmark-only sleep injection."
        ),
    }


def _write_caption_sidecars(target: Path, stem: str, *, index: int, image_suffix: str = ".png") -> dict[str, str]:
    style = index % 4
    if style == 0:
        path = target / f"{stem}.json"
        if not path.is_file():
            path.write_text(json.dumps(_caption_payload(index), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"style": "json_caption", "path": str(path)}
    if style == 1:
        path = target / f"{stem}.caption"
        if not path.is_file():
            path.write_text(
                f"lulynx, natural data wait, caption sidecar, sample {index}",
                encoding="utf-8",
            )
        return {"style": "caption_sidecar", "path": str(path)}
    if style == 2:
        path = target / f"{stem}{image_suffix}.txt"
        if not path.is_file():
            path.write_text(
                f"lulynx natural data wait image-suffix caption sample {index}",
                encoding="utf-8",
            )
        return {"style": "image_suffix_txt_caption", "path": str(path)}
    path = target / f"{stem}.txt"
    if not path.is_file():
        path.write_text(
            f"lulynx natural data wait heavy raw decode sample {index}, high entropy png",
            encoding="utf-8",
        )
    return {"style": "txt_caption", "path": str(path)}


def _cache_patterns_for_stem(stem: str) -> tuple[str, ...]:
    return (
        f"{stem}_*_anima.npz",
        f"{stem}_anima_te.npz",
        f"{stem}_*_newbie.npz",
        f"{stem}_newbie.npz",
        f"{stem}_newbie.safetensors",
        f"{stem}_newbie.pt",
        f"{stem}_te_outputs.npz",
        f"{stem}.jpg.safetensors",
        f"{stem}.png.safetensors",
        f"{stem}.txt.safetensors",
    )


def _remove_cache_sidecars(target: Path, stems: list[str]) -> list[str]:
    removed: list[str] = []
    for stem in stems:
        for pattern in _cache_patterns_for_stem(stem):
            for cache_file in target.glob(pattern):
                if not cache_file.is_file():
                    continue
                removed.append(str(cache_file))
                cache_file.unlink()
    for metadata_name in ("lulynx_cache_manifest_newbie.json", "lulynx_cache_metadata_newbie.json"):
        metadata = target / metadata_name
        if metadata.is_file():
            removed.append(str(metadata))
            metadata.unlink()
    return removed


def prepare_heavy_raw_decode_source(
    target: Path,
    *,
    samples: int = 4,
    size: int = 4096,
    seed: int = 1337,
    mixed_sidecars: bool = False,
    cache_state: str = "",
) -> dict[str, Any]:
    """Create a deterministic raw decode fixture without committing large images."""

    target.mkdir(parents=True, exist_ok=True)
    sample_count = max(int(samples), 1)
    width = height = max(int(size), 64)
    min_bytes = max(width * height // 2, 1024)
    images: list[str] = []
    sidecars: list[dict[str, str]] = []
    stems: list[str] = []
    for index in range(sample_count):
        stem = f"heavy_raw_{index:02d}"
        stems.append(stem)
        image_path = target / f"{stem}.png"
        if not _valid_png_fixture_file(image_path, min_bytes=min_bytes):
            _write_entropy_png(image_path, width=width, height=height, seed=seed + index)
        if mixed_sidecars:
            sidecars.append(_write_caption_sidecars(target, stem, index=index))
        else:
            caption_path = target / f"{stem}.txt"
            if not caption_path.is_file():
                caption_path.write_text(
                    f"lulynx natural data wait heavy raw decode sample {index}, high entropy png",
                    encoding="utf-8",
                )
            sidecars.append({"style": "txt_caption", "path": str(caption_path)})
        images.append(str(image_path))
    removed_cache_files = _remove_cache_sidecars(target, stems) if cache_state == "missing_at_start" else []
    sentinel = {
        "schema_version": 1,
        "fixture": (
            "heavy_raw_decode_cache_miss_mixed_sidecars_v0"
            if cache_state == "missing_at_start"
            else ("heavy_raw_decode_mixed_sidecars_v0" if mixed_sidecars else "heavy_raw_decode_png_v0")
        ),
        "samples": sample_count,
        "width": width,
        "height": height,
        "seed": int(seed),
        "cache_state": str(cache_state or "unspecified"),
        "cache_present_before": False if cache_state == "missing_at_start" else None,
        "removed_cache_files": removed_cache_files,
        "variants": (
            ["high_entropy_png", "json_caption", "caption_sidecar", "image_suffix_txt_caption", "txt_caption"]
            if mixed_sidecars
            else ["high_entropy_png", "txt_caption"]
        ),
        "images": images,
        "sidecars": sidecars,
    }
    (target / "fixture_manifest.json").write_text(
        json.dumps(sentinel, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sentinel


def prepare_heavy_raw_decode_mixed_sidecar_source(
    target: Path,
    *,
    samples: int = 4,
    size: int = 4096,
    seed: int = 1337,
) -> dict[str, Any]:
    return prepare_heavy_raw_decode_source(
        target,
        samples=samples,
        size=size,
        seed=seed,
        mixed_sidecars=True,
    )


def prepare_heavy_raw_decode_cache_miss_mixed_sidecar_source(
    target: Path,
    *,
    samples: int = 4,
    size: int = 4096,
    seed: int = 1337,
) -> dict[str, Any]:
    return prepare_heavy_raw_decode_source(
        target,
        samples=samples,
        size=size,
        seed=seed,
        mixed_sidecars=True,
        cache_state="missing_at_start",
    )


__all__ = [
    "prepare_heavy_raw_decode_cache_miss_mixed_sidecar_source",
    "prepare_heavy_raw_decode_mixed_sidecar_source",
    "prepare_heavy_raw_decode_source",
]
