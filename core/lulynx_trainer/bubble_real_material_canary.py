"""Real-material source fixtures for bubble natural-load canaries."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .dataset_discovery import caption_candidates_for_stem
from .newbie_cache_contract import is_newbie_cache_file, newbie_cache_contract_for_files


REAL_MATERIAL_CANARY_FIXTURE = "real_material_canary_v0"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _iter_images(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            yield child


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_file(source: Path, target: Path) -> Path | None:
    if not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


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


def _cache_inventory(files: Iterable[Path]) -> dict[str, Any]:
    patterns = {
        "anima_latent": "*_anima.npz",
        "anima_text": "*_anima_te.npz",
        "newbie_npz": "*_newbie.npz",
        "newbie_safetensors": "*_newbie.safetensors",
        "newbie_pt": "*_newbie.pt",
        "text_outputs": "*_te_outputs.npz",
        "sdxl_cache": "*.safetensors",
    }
    paths = [path for path in files if path.is_file()]
    names = [path.name for path in paths]
    counts = {
        key: sum(1 for name in names if fnmatch.fnmatch(name, pattern))
        for key, pattern in patterns.items()
    }
    newbie_files = [path for path in paths if is_newbie_cache_file(path)]
    newbie_contract = newbie_cache_contract_for_files(newbie_files)
    return {
        "schema_version": 1,
        "cache_inventory": "real_material_canary_cache_inventory_v0",
        "counts": counts,
        "has_anima_cache": bool(counts["anima_latent"] and counts["anima_text"]),
        "has_newbie_cache": bool(newbie_contract.get("ok")),
        "newbie_contract_ok": bool(newbie_contract.get("ok")),
        "newbie_contract_reasons": list(newbie_contract.get("reasons") or []),
        "newbie_pooled_shapes": list(newbie_contract.get("pooled_shapes") or []),
        "has_sdxl_cache": bool(counts["sdxl_cache"]),
    }


def _has_family_cache(family: str, inventory: dict[str, Any]) -> bool:
    if family == "anima":
        return bool(inventory.get("has_anima_cache"))
    if family == "newbie":
        return bool(inventory.get("has_newbie_cache"))
    if family == "sdxl":
        return bool(inventory.get("has_sdxl_cache"))
    return False


def _copied_file_record(source_root: Path, target_root: Path, path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(target_root).as_posix(),
        "source_relative_path": path.relative_to(source_root).as_posix()
        if path.is_relative_to(source_root)
        else "",
        "bytes": path.stat().st_size,
        "sha1": _sha1(path),
    }


def prepare_real_material_canary_source(
    source: Path,
    target: Path,
    *,
    family: str,
    samples: int = 8,
    sample_offset: int = 0,
    native_cache_mode: str = "cache_first",
    label: str = "",
) -> dict[str, Any]:
    """Copy a small real-material slice and write an auditable fixture manifest."""

    source = source.resolve()
    target.mkdir(parents=True, exist_ok=True)
    all_images = list(_iter_images(source))
    offset = max(int(sample_offset), 0)
    images = all_images[offset: offset + max(int(samples), 1)]
    if not images:
        raise FileNotFoundError(f"No images found in {source}")

    copied: list[Path] = []
    for image in images:
        copied_image = _copy_if_file(image, target / image.name)
        if copied_image is not None:
            copied.append(copied_image)
        for caption in caption_candidates_for_stem(image.parent, image.stem, ".txt"):
            copied_caption = _copy_if_file(caption, target / caption.name)
            if copied_caption is not None:
                copied.append(copied_caption)
        for pattern in _cache_patterns_for_stem(image.stem):
            for cache_file in sorted(image.parent.glob(pattern)):
                copied_cache = _copy_if_file(cache_file, target / cache_file.name)
                if copied_cache is not None:
                    copied.append(copied_cache)

    normalized_family = str(family or "").strip().lower().replace("-", "_")
    if normalized_family == "dit":
        normalized_family = "newbie"
    unique_files = sorted({path for path in copied if path.is_file()})
    inventory = _cache_inventory(unique_files)
    has_family_cache = _has_family_cache(normalized_family, inventory)
    cache_state = "warm_cache" if has_family_cache else "cache_inventory_missing"
    counts = dict(inventory.get("counts") or {})
    newbie_file_count = sum(
        int(counts.get(key) or 0)
        for key in ("newbie_npz", "newbie_safetensors", "newbie_pt")
    )
    if normalized_family == "newbie" and not has_family_cache and newbie_file_count:
        cache_state = "invalid_family_cache"
    records = [_copied_file_record(target, target, path) for path in unique_files]
    manifest_seed = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    source_manifest_sha1 = hashlib.sha1(manifest_seed).hexdigest()
    manifest = {
        "schema_version": 1,
        "fixture": REAL_MATERIAL_CANARY_FIXTURE,
        "family": normalized_family,
        "label": str(label or "real_material_canary"),
        "source_root": str(source),
        "source_image_total": len(all_images),
        "sample_offset": offset,
        "samples": len(images),
        "source_image_count": len(images),
        "source_file_count": len(unique_files),
        "source_manifest_sha1": source_manifest_sha1,
        "native_cache_mode": str(native_cache_mode or "cache_first"),
        "cache_state": cache_state,
        "cache_present_before": bool(has_family_cache),
        "cache_has_family_cache": bool(has_family_cache),
        "cache_inventory": inventory,
        "variants": ["real_material", "caption_sidecars", "cache_sidecars"],
        "files": records,
    }
    (target / "fixture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "REAL_MATERIAL_CANARY_FIXTURE",
    "prepare_real_material_canary_source",
]
