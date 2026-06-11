"""Source/cache readiness scan for bubble real-material canaries."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .bubble_real_material_canary import IMAGE_SUFFIXES, REAL_MATERIAL_CANARY_FIXTURE
from .dataset_discovery import caption_candidates_for_stem
from .newbie_cache_contract import find_newbie_cache_files, newbie_cache_contract_for_files


REAL_MATERIAL_SOURCE_SCAN_REPORT = "bubble_real_material_source_scan_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
DEFAULT_SCAN_FAMILIES = ("sdxl", "anima", "newbie")


def _safe_stat_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1_or_error(path: Path) -> tuple[str, str]:
    try:
        return _sha1(path), ""
    except OSError as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _glob(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _normalize_family(value: str) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if family in {"dit", "newbie_dit"} else family


def _family_list(families: Sequence[str] | None) -> list[str]:
    selected = [_normalize_family(item) for item in families or DEFAULT_SCAN_FAMILIES]
    return [item for item in selected if item in DEFAULT_SCAN_FAMILIES] or list(DEFAULT_SCAN_FAMILIES)


def _sample_cache_files(root: Path, stem: str, family: str) -> list[Path]:
    if family == "sdxl":
        return [
            *_glob(root, f"{stem}.jpg.safetensors"),
            *_glob(root, f"{stem}.png.safetensors"),
            *_glob(root, f"{stem}.jpeg.safetensors"),
            *_glob(root, f"{stem}.webp.safetensors"),
            *_glob(root, f"{stem}.txt.safetensors"),
        ]
    if family == "anima":
        return [
            *_glob(root, f"{stem}_*_anima.npz"),
            *_glob(root, f"{stem}_anima_te.npz"),
        ]
    if family == "newbie":
        return [
            *_glob(root, f"{stem}_*_newbie.npz"),
            *_glob(root, f"{stem}_newbie.npz"),
            *_glob(root, f"{stem}_newbie.safetensors"),
            *_glob(root, f"{stem}_newbie.pt"),
        ]
    return []


def _fixture_cache_patterns_for_stem(stem: str) -> tuple[str, ...]:
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


def _fixture_files_for_images(root: Path, images: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for image in images:
        files.append(image)
        files.extend(path for path in caption_candidates_for_stem(root, image.stem, ".txt") if path.is_file())
        for pattern in _fixture_cache_patterns_for_stem(image.stem):
            files.extend(_glob(root, pattern))
    return sorted({path for path in files if path.is_file()})


def _sample_has_family_cache(root: Path, stem: str, family: str) -> bool:
    if family == "anima":
        return bool(_glob(root, f"{stem}_*_anima.npz")) and bool(_glob(root, f"{stem}_anima_te.npz"))
    if family == "newbie":
        files = _sample_cache_files(root, stem, family)
        return bool(files) and bool(newbie_cache_contract_for_files(files).get("ok"))
    return bool(_sample_cache_files(root, stem, family))


def _cache_inventory(root: Path) -> dict[str, Any]:
    counts = {
        "anima_latent": len(_glob(root, "*_anima.npz")),
        "anima_text": len(_glob(root, "*_anima_te.npz")),
        "newbie_npz": len(_glob(root, "*_newbie.npz")),
        "newbie_safetensors": len(_glob(root, "*_newbie.safetensors")),
        "newbie_pt": len(_glob(root, "*_newbie.pt")),
        "text_outputs": len(_glob(root, "*_te_outputs.npz")),
        "sdxl_cache": len(_glob(root, "*.safetensors")),
    }
    bytes_by_family = {
        "sdxl": sum(_safe_stat_size(path) for path in _glob(root, "*.safetensors")),
        "anima": sum(
            _safe_stat_size(path)
            for path in [*_glob(root, "*_anima.npz"), *_glob(root, "*_anima_te.npz")]
        ),
        "newbie": sum(
            _safe_stat_size(path)
            for path in [
                *_glob(root, "*_newbie.npz"),
                *_glob(root, "*_newbie.safetensors"),
                *_glob(root, "*_newbie.pt"),
            ]
        ),
    }
    newbie_contract = newbie_cache_contract_for_files(find_newbie_cache_files(root, recursive=False))
    return {
        "schema_version": 1,
        "cache_inventory": "real_material_source_scan_cache_inventory_v0",
        "counts": counts,
        "bytes": bytes_by_family,
        "has_sdxl_cache": bool(counts["sdxl_cache"]),
        "has_anima_cache": bool(counts["anima_latent"] and counts["anima_text"]),
        "has_newbie_cache": bool(newbie_contract.get("ok")),
        "newbie_contract_ok": bool(newbie_contract.get("ok")),
        "newbie_contract_reasons": list(newbie_contract.get("reasons") or []),
        "newbie_pooled_shapes": list(newbie_contract.get("pooled_shapes") or []),
    }


def _metadata_inventory(root: Path) -> dict[str, Any]:
    names = [
        "fixture_manifest.json",
        "lulynx_cache_manifest_newbie.json",
        "lulynx_cache_metadata_newbie.json",
        "lulynx_cache_metadata_anima.json",
        "_metadata.json",
    ]
    found = [name for name in names if (root / name).is_file()]
    return {
        "schema_version": 1,
        "metadata_inventory": "real_material_source_scan_metadata_inventory_v0",
        "found": found,
        "metadata_found": bool(found),
        "has_newbie_manifest": "lulynx_cache_manifest_newbie.json" in found,
        "has_newbie_metadata": "lulynx_cache_metadata_newbie.json" in found,
        "has_anima_metadata": "lulynx_cache_metadata_anima.json" in found,
    }


def _fixture_file_record(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    sha1, error = _sha1_or_error(path)
    record = {
        "relative_path": relative,
        "source_relative_path": relative,
        "bytes": _safe_stat_size(path),
        "sha1": sha1,
    }
    if error:
        record["sha1_error"] = error
    return record


def _source_manifest_sha1(root: Path, images: Sequence[Path]) -> str:
    records = [_fixture_file_record(root, path) for path in _fixture_files_for_images(root, images)]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _source_manifest_hash_warnings(root: Path, images: Sequence[Path]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for path in _fixture_files_for_images(root, images):
        sha1, error = _sha1_or_error(path)
        if error:
            warnings.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "reason": "sha1_unavailable",
                    "error": error,
                }
            )
    return warnings


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _image_stats(images: Sequence[Path]) -> dict[str, Any]:
    bytes_values = [_safe_stat_size(path) for path in images]
    sizes = [size for path in images if (size := _image_size(path)) is not None]
    pixels = [width * height for width, height in sizes]
    return {
        "image_count": len(images),
        "total_image_bytes": sum(bytes_values),
        "mean_image_bytes": round(mean(bytes_values), 4) if bytes_values else 0.0,
        "max_image_bytes": max(bytes_values, default=0),
        "mean_pixels": round(mean(pixels), 4) if pixels else 0.0,
        "max_pixels": max(pixels, default=0),
        "max_width": max((width for width, _ in sizes), default=0),
        "max_height": max((height for _, height in sizes), default=0),
        "dimension_probe_available": bool(sizes),
    }


def _sample_records(
    root: Path,
    images: Sequence[Path],
    families: Sequence[str],
    *,
    sample_offset: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        captions = [path for path in caption_candidates_for_stem(root, image.stem, ".txt") if path.is_file()]
        family_cache: dict[str, dict[str, Any]] = {}
        for family in families:
            files = _sample_cache_files(root, image.stem, family)
            contract = newbie_cache_contract_for_files(files) if family == "newbie" else {"ok": bool(files), "reasons": []}
            has_cache = (
                bool(_glob(root, f"{image.stem}_*_anima.npz"))
                and bool(_glob(root, f"{image.stem}_anima_te.npz"))
                if family == "anima"
                else bool(files) and bool(contract.get("ok"))
            )
            family_cache[family] = {
                "has_cache": has_cache,
                "cache_file_count": len(files),
                "cache_bytes": sum(_safe_stat_size(path) for path in files),
                "latent_present": bool(_glob(root, f"{image.stem}_*_anima.npz")) if family == "anima" else None,
                "text_present": bool(_glob(root, f"{image.stem}_anima_te.npz")) if family == "anima" else None,
                "contract_ok": bool(contract.get("ok")),
                "contract_reasons": list(contract.get("reasons") or []),
                "pooled_shapes": list(contract.get("pooled_shapes") or []),
                "invalid_cache": bool(files) and not bool(contract.get("ok")),
            }
        records.append(
            {
                "image": image.name,
                "stem": image.stem,
                "sample_index": int(sample_offset) + index,
                "bytes": _safe_stat_size(image),
                "caption_count": len(captions),
                "captions": [path.name for path in captions[:5]],
                "family_cache": family_cache,
            }
        )
    return records


def _family_readiness(
    root: Path,
    sample_records: Sequence[dict[str, Any]],
    family: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    sample_count = len(sample_records)
    cached = [
        record
        for record in sample_records
        if bool(record.get("family_cache", {}).get(family, {}).get("has_cache"))
    ]
    missing = [
        str(record.get("stem") or "")
        for record in sample_records
        if not bool(record.get("family_cache", {}).get(family, {}).get("has_cache"))
    ]
    invalid = [
        str(record.get("stem") or "")
        for record in sample_records
        if bool(record.get("family_cache", {}).get(family, {}).get("invalid_cache"))
    ]
    contract_reasons = sorted(
        {
            str(reason)
            for record in sample_records
            for reason in record.get("family_cache", {}).get(family, {}).get("contract_reasons", [])
            if reason
        }
    )
    cache_bytes = sum(
        int(record.get("family_cache", {}).get(family, {}).get("cache_bytes") or 0)
        for record in sample_records
    )
    coverage = len(cached) / max(sample_count, 1)
    cache_ready = sample_count > 0 and coverage >= 1.0
    counts = dict(inventory.get("counts") or {})
    if family == "sdxl":
        family_inventory_ready = bool(inventory.get("has_sdxl_cache"))
    elif family == "anima":
        family_inventory_ready = bool(inventory.get("has_anima_cache"))
    else:
        family_inventory_ready = bool(inventory.get("has_newbie_cache"))
    reasons: list[str] = []
    if not sample_count:
        reasons.append("no_sample_images")
    if not family_inventory_ready:
        reasons.append(f"{family}_family_cache_inventory_missing")
    if not cache_ready:
        reasons.append(f"{family}_sample_cache_coverage_incomplete")
    if invalid:
        reasons.append(f"{family}_sample_cache_contract_invalid")
        reasons.extend(contract_reasons)
    status = "ready" if cache_ready and family_inventory_ready else "blocked_missing_family_cache"
    if invalid:
        status = "blocked_invalid_family_cache"
    return {
        "family": family,
        "status": status,
        "cache_ready": bool(cache_ready and family_inventory_ready),
        "sample_cache_coverage": round(coverage, 6),
        "sample_cache_count": len(cached),
        "sample_count": sample_count,
        "sample_cache_bytes": cache_bytes,
        "mean_sample_cache_bytes": round(cache_bytes / max(sample_count, 1), 4),
        "missing_stems": missing,
        "invalid_stems": invalid,
        "inventory_counts": counts,
        "blocked_reasons": reasons,
        "recommended_action": "run_real_material_canary" if status == "ready" else f"prepare_{family}_warm_cache",
        "source_root": str(root),
    }


def _cache_pair_coverage(sample_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(sample_records)
    latent = [
        record for record in sample_records
        if bool(record.get("family_cache", {}).get("anima", {}).get("latent_present"))
    ]
    text = [
        record for record in sample_records
        if bool(record.get("family_cache", {}).get("anima", {}).get("text_present"))
    ]
    pair = [
        record for record in sample_records
        if bool(record.get("family_cache", {}).get("anima", {}).get("latent_present"))
        and bool(record.get("family_cache", {}).get("anima", {}).get("text_present"))
    ]
    return {
        "anima_latent_coverage": round(len(latent) / max(count, 1), 6),
        "anima_text_coverage": round(len(text) / max(count, 1), 6),
        "anima_pair_coverage": round(len(pair) / max(count, 1), 6),
    }


def _pressure_score(image_stats: dict[str, Any], family_readiness: Sequence[dict[str, Any]]) -> float:
    mean_image_mb = float(image_stats.get("mean_image_bytes") or 0.0) / (1024.0 * 1024.0)
    mean_pixels_mp = float(image_stats.get("mean_pixels") or 0.0) / 1_000_000.0
    max_cache_mb = max(
        (float(item.get("mean_sample_cache_bytes") or 0.0) / (1024.0 * 1024.0) for item in family_readiness),
        default=0.0,
    )
    return round(mean_image_mb + mean_pixels_mp * 0.15 + max_cache_mb * 0.5, 6)


def _next_canary_cases(root: Path, family_readiness: Sequence[dict[str, Any]], sample_offset: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in family_readiness:
        family = str(item.get("family") or "")
        if not bool(item.get("cache_ready")):
            continue
        for suffix, workers, prefetch in (
            ("workers0", 0, 2),
            ("workers2_prefetch4", 2, 4),
        ):
            cases.append(
                {
                    "case_id": f"{family}_real_material_cache_first_{suffix}_canary",
                    "family": family,
                    "source_data": str(root),
                    "sample_offset": int(sample_offset),
                    "dataloader_workers": workers,
                    "dataloader_prefetch_factor": prefetch,
                    "native_cache_mode": "cache_first",
                }
            )
    return cases


def scan_real_material_source(
    root: Path,
    *,
    samples: int = 8,
    sample_offset: int = 0,
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Scan one directory using the same direct-child source assumptions as P60."""

    root = Path(root).resolve()
    selected_families = _family_list(families)
    images = _iter_images(root)
    offset = max(int(sample_offset), 0)
    sample_images = images[offset: offset + max(int(samples), 1)]
    inventory = _cache_inventory(root)
    records = _sample_records(root, sample_images, selected_families, sample_offset=offset)
    readiness = [
        _family_readiness(root, records, family, inventory)
        for family in selected_families
    ]
    captions_ready = sum(1 for record in records if int(record.get("caption_count") or 0) > 0)
    stats = _image_stats(sample_images)
    score = _pressure_score(stats, readiness)
    ready_families = [item["family"] for item in readiness if item.get("cache_ready")]
    source_sha1 = _source_manifest_sha1(root, sample_images)
    hash_warnings = _source_manifest_hash_warnings(root, sample_images)
    return {
        "schema_version": 1,
        "report": REAL_MATERIAL_SOURCE_SCAN_REPORT,
        "roadmap": ROADMAP,
        "source_fixture": REAL_MATERIAL_CANARY_FIXTURE,
        "root": str(root),
        "samples_requested": int(samples),
        "sample_offset": offset,
        "source_image_count": len(images),
        "sample_image_count": len(sample_images),
        "caption_sample_coverage": round(captions_ready / max(len(records), 1), 6),
        "source_manifest_sha1": source_sha1,
        "source_manifest_hash_warnings": hash_warnings,
        "cache_inventory": inventory,
        "metadata_inventory": _metadata_inventory(root),
        "cache_pair_coverage": _cache_pair_coverage(records),
        "image_stats": stats,
        "family_readiness": readiness,
        "missing_stems_by_family": {
            item["family"]: list(item.get("missing_stems") or [])
            for item in readiness
            if item.get("missing_stems")
        },
        "ready_families": ready_families,
        "blocked_families": [item["family"] for item in readiness if not item.get("cache_ready")],
        "pressure_score": score,
        "candidate_rank_score": round(score * max(len(ready_families), 1), 6),
        "next_canary_cases": _next_canary_cases(root, readiness, offset),
        "sample_records": records,
    }


def _candidate_dirs(root: Path, max_depth: int) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    candidates: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()
    while stack:
        current, depth = stack.pop()
        resolved = current.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if current.is_dir():
            if _iter_images(current):
                candidates.append(current)
            if depth < max(int(max_depth), 0):
                stack.extend((child, depth + 1) for child in sorted(current.iterdir(), reverse=True) if child.is_dir())
    return sorted(candidates)


def scan_real_material_sources(
    roots: Iterable[Path],
    *,
    samples: int = 8,
    sample_offset: int = 0,
    families: Sequence[str] | None = None,
    max_depth: int = 1,
    min_images: int = 1,
) -> dict[str, Any]:
    """Scan candidate source directories and sort them by readiness and pressure."""

    scans = [
        scan_real_material_source(path, samples=samples, sample_offset=sample_offset, families=families)
        for root in roots
        for path in _candidate_dirs(Path(root), max_depth=max_depth)
    ]
    scans = [item for item in scans if int(item.get("source_image_count") or 0) >= int(min_images)]
    scans.sort(
        key=lambda item: (
            len(item.get("ready_families") or []),
            float(item.get("candidate_rank_score") or 0.0),
            int(item.get("source_image_count") or 0),
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "report": REAL_MATERIAL_SOURCE_SCAN_REPORT,
        "roadmap": ROADMAP,
        "roots": [str(Path(root)) for root in roots],
        "samples_requested": int(samples),
        "sample_offset": max(int(sample_offset), 0),
        "max_depth": int(max_depth),
        "min_images": int(min_images),
        "family_count": len(_family_list(families)),
        "candidate_count": len(scans),
        "candidates": scans,
    }


def scan_real_material_windows(
    root: Path,
    *,
    samples: int = 8,
    stride: int = 8,
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Scan sample windows within one source directory."""

    root = Path(root).resolve()
    image_count = len(_iter_images(root))
    window_size = max(int(samples), 1)
    step = max(int(stride), 1)
    offsets = list(range(0, max(image_count, 1), step))
    scans = [
        scan_real_material_source(root, samples=window_size, sample_offset=offset, families=families)
        for offset in offsets
        if offset < image_count
    ]
    scans = [item for item in scans if int(item.get("sample_image_count") or 0) > 0]
    scans.sort(
        key=lambda item: (
            len(item.get("ready_families") or []),
            float(item.get("candidate_rank_score") or 0.0),
            float(item.get("image_stats", {}).get("mean_image_bytes") or 0.0),
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "report": REAL_MATERIAL_SOURCE_SCAN_REPORT,
        "roadmap": ROADMAP,
        "root": str(root),
        "mode": "sample_windows",
        "samples_requested": window_size,
        "stride": step,
        "source_image_count": image_count,
        "window_count": len(scans),
        "windows": scans,
    }


def dumps_scan_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_SCAN_FAMILIES",
    "REAL_MATERIAL_SOURCE_SCAN_REPORT",
    "ROADMAP",
    "dumps_scan_report",
    "scan_real_material_source",
    "scan_real_material_sources",
    "scan_real_material_windows",
]
