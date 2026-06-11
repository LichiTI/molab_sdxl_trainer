"""Dataset, Anima cache, and Concept Geometry preflight reports."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

try:
    from .dataset_discovery import DatasetSampleRecord, discover_dataset_samples
except ImportError:  # pragma: no cover - direct script loading
    from dataset_discovery import DatasetSampleRecord, discover_dataset_samples


_TAG_SPLIT_RE = re.compile(r"[\n,]+")
_GENERIC_TAGS = {
    "1girl",
    "1boy",
    "solo",
    "best quality",
    "masterpiece",
    "high quality",
    "furry",
    "portrait",
    "reference sheet",
}


@dataclass(frozen=True)
class InspectorOptions:
    data_dir: Path
    caption_extension: str = ".txt"
    geometry_path: Path | None = None
    concept_geometry_sampler_mode: str = ""
    h_lora_sampler_mode: str = ""
    batch_size: int = 1
    top_tags: int = 40


def inspect_dataset(options: InspectorOptions) -> dict[str, Any]:
    discovery = discover_dataset_samples(options.data_dir, options.caption_extension)
    samples = list(discovery.samples)
    dataset_report = _dataset_report(discovery.data_dir, discovery.scan_mode, samples, options.top_tags)
    cache_report = _cache_report(samples)
    geometry_report = _geometry_report(
        samples,
        options.geometry_path,
        sampler_mode=options.concept_geometry_sampler_mode or options.h_lora_sampler_mode,
        batch_size=options.batch_size,
    )
    warnings = list(discovery.warnings)
    warnings.extend(dataset_report.get("warnings", []))
    warnings.extend(cache_report.get("warnings", []))
    warnings.extend(geometry_report.get("warnings", []))
    metrics = _stable_metrics(dataset_report, cache_report, geometry_report)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(discovery.data_dir),
        "scan_mode": discovery.scan_mode,
        "subset_count": len(discovery.subsets),
        "subsets": [
            {
                "root": str(subset.root),
                "name": subset.root.name,
                "concept": subset.concept,
                "repeats": subset.repeats,
                "source": subset.source,
            }
            for subset in discovery.subsets
        ],
        "dataset": dataset_report,
        "cache": cache_report,
        "concept_geometry": geometry_report,
        "h_lora": geometry_report,
        "metrics": metrics,
        "visualization": _visualization_payload(dataset_report, geometry_report),
        "warnings": _dedupe(warnings),
        "sample_preview": [_sample_payload(sample) for sample in samples[:12]],
    }


def _dataset_report(root: Path, scan_mode: str, samples: list[DatasetSampleRecord], top_tags: int) -> dict[str, Any]:
    image_samples = [sample for sample in samples if sample.image_path is not None]
    caption_samples = [sample for sample in samples if sample.caption_path is not None]
    warnings: list[str] = []
    extension_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    empty_caption_count = 0
    broken_image_count = 0
    alpha_capable_count = 0
    total_tags = 0

    for sample in samples:
        concept_counts[sample.concept or "ungrouped"] += 1
        if sample.image_path is not None:
            extension_counts[sample.image_path.suffix.lower()] += 1
            if sample.image_path.suffix.lower() in {".png", ".webp"}:
                alpha_capable_count += 1
            size = _read_image_size(sample.image_path)
            if size is None:
                broken_image_count += 1
            else:
                w, h = size
                resolution_counts[f"{w}x{h}"] += 1
                orientation_counts[_orientation(w, h)] += 1
        if sample.caption_path is not None:
            tags = _read_caption_tags(sample.caption_path)
            if not tags:
                empty_caption_count += 1
            else:
                tag_counts.update(tags)
                total_tags += len(tags)

    image_count = len(image_samples)
    caption_count = len(caption_samples)
    missing_caption_count = sum(1 for sample in image_samples if sample.caption_path is None)
    effective_image_count = sum(max(sample.repeats, 1) for sample in image_samples)
    caption_coverage = caption_count / max(image_count, 1)
    orphan_caption_count = _orphan_caption_count(root, samples)
    generic_tag_hits = sum(count for tag, count in tag_counts.items() if tag in _GENERIC_TAGS)
    generic_tag_ratio = generic_tag_hits / max(sum(tag_counts.values()), 1)

    if image_count == 0 and any(sample.text_cache_path or sample.latent_path for sample in samples):
        warnings.append("No raw images were discovered; this appears to be cache-first data.")
    if missing_caption_count:
        warnings.append(f"{missing_caption_count} images have no matching caption.")
    if empty_caption_count:
        warnings.append(f"{empty_caption_count} captions are empty after tag parsing.")
    if orphan_caption_count:
        warnings.append(f"{orphan_caption_count} caption files do not match a discovered sample.")
    if broken_image_count:
        warnings.append(f"{broken_image_count} images could not be opened for size probing.")
    if scan_mode == "native_recursive":
        warnings.append("No N_concept subset layout was detected; using native recursive discovery.")
    if generic_tag_ratio > 0.6 and tag_counts:
        warnings.append("Most caption tags are generic; concept separation may rely on folders or cache features.")

    return {
        "sample_count": len(samples),
        "image_count": image_count,
        "effective_image_count": effective_image_count,
        "caption_count": caption_count,
        "caption_coverage": round(caption_coverage, 4),
        "images_without_caption_count": missing_caption_count,
        "orphan_caption_count": orphan_caption_count,
        "empty_caption_count": empty_caption_count,
        "broken_image_count": broken_image_count,
        "alpha_capable_image_count": alpha_capable_count,
        "unique_tag_count": len(tag_counts),
        "average_tags_per_caption": round(total_tags / max(caption_count, 1), 4),
        "generic_tag_ratio": round(generic_tag_ratio, 4),
        "concept_group_count": len(concept_counts),
        "concept_groups": _counter_payload(concept_counts),
        "top_tags": _counter_payload(tag_counts, top_tags),
        "image_extensions": _counter_payload(extension_counts),
        "orientations": _counter_payload(orientation_counts),
        "resolutions": _counter_payload(resolution_counts, 20),
        "warnings": warnings,
    }


def _stable_metrics(dataset: dict[str, Any], cache: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_sample_count": int(dataset.get("sample_count", 0) or 0),
        "image_count": int(dataset.get("image_count", 0) or 0),
        "effective_image_count": int(dataset.get("effective_image_count", 0) or 0),
        "caption_coverage": float(dataset.get("caption_coverage", 0.0) or 0.0),
        "unique_tag_count": int(dataset.get("unique_tag_count", 0) or 0),
        "average_tags_per_caption": float(dataset.get("average_tags_per_caption", 0.0) or 0.0),
        "tag_generic_ratio": float(dataset.get("generic_tag_ratio", 0.0) or 0.0),
        "dataset_concept_group_count": int(dataset.get("concept_group_count", 0) or 0),
        "cache_pair_count": int(cache.get("cache_pair_count", 0) or 0),
        "cache_attach_rate": float(cache.get("cache_attach_rate", 0.0) or 0.0),
        "missing_latent_count": int(cache.get("missing_latent_count", 0) or 0),
        "missing_text_count": int(cache.get("missing_text_count", 0) or 0),
        "geometry_enabled": bool(geometry.get("enabled", False)),
        "geometry_attach_rate": float(geometry.get("geometry_attach_rate", 0.0) or 0.0),
        "geometry_concept_group_count": int(geometry.get("concept_group_count", 0) or 0),
        "neighbor_same_ratio": float(geometry.get("neighbor_same_ratio", 0.0) or 0.0),
        "sibling_same_ratio": float(geometry.get("sibling_same_ratio", 0.0) or 0.0),
        "conflict_mean": float(geometry.get("conflict_mean", 0.0) or 0.0),
        "risk": str(geometry.get("risk", "not_enabled") or "not_enabled"),
    }


def _visualization_payload(dataset: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "concept_distribution": list(dataset.get("concept_groups", [])),
        "top_tags": list(dataset.get("top_tags", [])),
        "resolution_distribution": list(dataset.get("resolutions", [])),
        "orientation_distribution": list(dataset.get("orientations", [])),
        "geometry_scatter": list(geometry.get("scatter_2d", [])),
        "geometry_edges": list(geometry.get("graph_edges", [])),
        "conflict_histogram": list(geometry.get("conflict_histogram", [])),
    }


def _cache_report(samples: list[DatasetSampleRecord]) -> dict[str, Any]:
    warnings: list[str] = []
    cache_samples = [sample for sample in samples if sample.latent_path is not None or sample.text_cache_path is not None]
    paired = [sample for sample in cache_samples if sample.latent_path is not None and sample.text_cache_path is not None]
    missing_latent = [sample.sample_id for sample in cache_samples if sample.text_cache_path is not None and sample.latent_path is None]
    missing_text = [sample.sample_id for sample in cache_samples if sample.latent_path is not None and sample.text_cache_path is None]
    latent_shapes: Counter[str] = Counter()
    text_tokens: Counter[str] = Counter()
    unreadable: list[str] = []

    for sample in paired[:256]:
        if sample.latent_path is not None:
            shape = _read_npz_shape(sample.latent_path, prefix="latents")
            if shape:
                latent_shapes[shape] += 1
            elif sample.latent_path.suffix.lower() == ".npz":
                unreadable.append(str(sample.latent_path))
        if sample.text_cache_path is not None:
            tokens = _read_prompt_token_count(sample.text_cache_path)
            if tokens:
                text_tokens[str(tokens)] += 1

    if missing_latent:
        warnings.append(f"{len(missing_latent)} text cache files have no matching latent cache.")
    if missing_text:
        warnings.append(f"{len(missing_text)} latent cache files have no matching text cache.")
    if unreadable:
        warnings.append(f"{len(unreadable)} npz cache files could not be inspected.")

    return {
        "cache_sample_count": len(cache_samples),
        "cache_pair_count": len(paired),
        "cache_attach_rate": round(len(paired) / max(len(cache_samples), 1), 4),
        "missing_latent_count": len(missing_latent),
        "missing_text_count": len(missing_text),
        "missing_latent_samples": missing_latent[:12],
        "missing_text_samples": missing_text[:12],
        "latent_shapes": _counter_payload(latent_shapes, 20),
        "text_token_counts": _counter_payload(text_tokens, 20),
        "warnings": warnings,
    }


def _geometry_report(
    samples: list[DatasetSampleRecord],
    geometry_path: Path | None,
    *,
    sampler_mode: str,
    batch_size: int,
) -> dict[str, Any]:
    if geometry_path is None or not str(geometry_path).strip():
        return {"enabled": False, "warnings": []}
    path = Path(geometry_path)
    if not path.is_file():
        return {"enabled": True, "geometry_path": str(path), "warnings": [f"Concept Geometry file not found: {path}"]}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "enabled": True,
            "geometry_path": str(path),
            "warnings": [f"Failed to parse Concept Geometry file: {type(exc).__name__}: {exc}"],
        }

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    raw_samples = payload.get("samples", payload) if isinstance(payload, dict) else {}
    geometry_samples = raw_samples if isinstance(raw_samples, dict) else {}
    dataset_ids = {sample.sample_id for sample in samples}
    dataset_stems = {sample.stem for sample in samples}
    attached_ids = [sid for sid in geometry_samples.keys() if sid in dataset_ids or sid in dataset_stems]
    concept_counts: Counter[str] = Counter()
    conflict_values: list[float] = []
    neighbor_same = 0
    neighbor_total = 0
    sibling_same = 0
    sibling_total = 0
    tag_source_count = 0
    scatter: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    geometry_group = {
        str(sample_id): str(item.get("concept_group", "") or "")
        for sample_id, item in geometry_samples.items()
        if isinstance(item, dict)
    }
    group_order = {name: idx for idx, name in enumerate(sorted(set(geometry_group.values()) or {"ungrouped"}))}
    total_groups = max(len(group_order), 1)
    for sample_index, (sample_id, item) in enumerate(sorted(geometry_samples.items(), key=lambda pair: str(pair[0]))):
        if not isinstance(item, dict):
            continue
        group = str(item.get("concept_group", "") or "ungrouped")
        concept_counts[group] += 1
        conflict = float(item.get("conflict_score", 0.0) or 0.0)
        density = float(item.get("density", 0.0) or 0.0)
        conflict_values.append(conflict)
        angle = (2.0 * np.pi * group_order.get(group, 0) / total_groups) + 0.37 * (sample_index % 7)
        radius = 0.35 + 0.65 * (1.0 - min(max(density, 0.0), 1.0))
        scatter.append(
            {
                "id": str(sample_id),
                "concept_group": group,
                "stage": str(item.get("stage", "mid") or "mid"),
                "density": round(density, 4),
                "conflict_score": round(conflict, 4),
                "x": round(float(np.cos(angle) * radius), 4),
                "y": round(float(np.sin(angle) * radius), 4),
            }
        )
        sources = item.get("feature_sources", {})
        if isinstance(sources, dict) and "tags" in sources:
            tag_source_count += 1
        elif isinstance(sources, list) and "tags" in sources:
            tag_source_count += 1
        for neighbor in _string_list(item.get("neighbor_ids", [])):
            if neighbor in geometry_group:
                edges.append({"source": str(sample_id), "target": neighbor, "kind": "neighbor"})
                neighbor_total += 1
                if geometry_group.get(neighbor) == group:
                    neighbor_same += 1
        for sibling in _string_list(item.get("sibling_ids", [])):
            if sibling in geometry_group:
                edges.append({"source": str(sample_id), "target": sibling, "kind": "sibling"})
                sibling_total += 1
                if geometry_group.get(sibling) == group:
                    sibling_same += 1

    attach_rate = len(attached_ids) / max(len(samples), 1)
    neighbor_ratio = neighbor_same / max(neighbor_total, 1)
    sibling_ratio = sibling_same / max(sibling_total, 1)
    conflict_mean = sum(conflict_values) / max(len(conflict_values), 1)
    warnings: list[str] = []
    risk = "ok"
    if attach_rate < 0.98:
        warnings.append(f"Geometry attach rate is {attach_rate:.2%}; some geometry samples do not match dataset samples.")
        risk = "cache_mismatch"
    if len(concept_counts) <= 1 and len(samples) > 1:
        warnings.append("Geometry has a single concept group; Concept Geometry Sampling can weight density but cannot prove multi-concept separation.")
        risk = "collapsed" if risk == "ok" else risk
    if neighbor_total and neighbor_ratio < 0.65:
        warnings.append(f"Neighbor same-concept ratio is low ({neighbor_ratio:.2f}); geometry may be mixed.")
        risk = "weak_multiconcept" if risk == "ok" else risk
    if str(sampler_mode or "").lower().replace("-", "_") == "concept_batch" and int(batch_size or 1) <= 1:
        warnings.append("concept_batch is enabled with batch_size <= 1; sampler will fall back to weighted curriculum.")
    if tag_source_count == 0 and geometry_samples:
        warnings.append("Geometry entries do not show tag features; semantic projection may be feature-only.")

    return {
        "enabled": True,
        "geometry_path": str(path),
        "geometry_version": int(meta.get("geometry_version", 1) or 1) if isinstance(meta, dict) else 1,
        "backend_requested": str(meta.get("backend_requested", "") or "") if isinstance(meta, dict) else "",
        "backend_resolved": str(meta.get("backend_resolved", "") or "") if isinstance(meta, dict) else "",
        "feature_sources": list(meta.get("feature_sources", [])) if isinstance(meta.get("feature_sources", []), list) else [],
        "fallback_reasons": list(meta.get("fallback_reasons", [])) if isinstance(meta.get("fallback_reasons", []), list) else [],
        "dataset_sample_count": len(samples),
        "geometry_sample_count": len(geometry_samples),
        "attached_count": len(attached_ids),
        "geometry_attach_rate": round(attach_rate, 4),
        "concept_group_count": len(concept_counts),
        "concept_groups": _counter_payload(concept_counts),
        "neighbor_same_ratio": round(neighbor_ratio, 4),
        "sibling_same_ratio": round(sibling_ratio, 4),
        "conflict_mean": round(conflict_mean, 4),
        "conflict_histogram": _histogram(conflict_values, bins=5),
        "scatter_2d": scatter[:1000],
        "graph_edges": edges[:3000],
        "risk": risk,
        "warnings": warnings,
    }


def _sample_payload(sample: DatasetSampleRecord) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "stem": sample.stem,
        "subset": sample.subset_name,
        "concept": sample.concept,
        "repeats": sample.repeats,
        "image": str(sample.image_path) if sample.image_path else "",
        "caption": str(sample.caption_path) if sample.caption_path else "",
        "latent": str(sample.latent_path) if sample.latent_path else "",
        "text_cache": str(sample.text_cache_path) if sample.text_cache_path else "",
    }


def _read_caption_tags(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return []
    tags = []
    for raw in _TAG_SPLIT_RE.split(text):
        tag = raw.strip().lower().replace("_", " ")
        if tag:
            tags.append(re.sub(r"\s+", " ", tag))
    return tags


def _read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:
        return None


def _read_npz_shape(path: Path, *, prefix: str) -> str:
    try:
        with np.load(str(path), mmap_mode="r") as data:
            for key in data.files:
                if key.startswith(prefix):
                    return "x".join(str(part) for part in data[key].shape)
    except Exception:
        return ""
    return ""


def _read_prompt_token_count(path: Path) -> int:
    try:
        if path.suffix.lower() == ".npz":
            with np.load(str(path), mmap_mode="r") as data:
                if "prompt_embeds" in data.files:
                    return int(data["prompt_embeds"].shape[0])
    except Exception:
        return 0
    return 0


def _orphan_caption_count(root: Path, samples: list[DatasetSampleRecord]) -> int:
    sample_caption_paths = {sample.caption_path.resolve() for sample in samples if sample.caption_path is not None}
    count = 0
    for suffix in (".txt", ".caption"):
        for path in root.rglob(f"*{suffix}"):
            if path.is_file() and path.resolve() not in sample_caption_paths:
                count += 1
    return count


def _orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _counter_payload(counter: Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        rows = rows[:limit]
    return [{"name": key, "count": int(value)} for key, value in rows]


def _histogram(values: list[float], bins: int = 5) -> list[dict[str, Any]]:
    if not values:
        return []
    bins = max(int(bins or 5), 1)
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(max(int(float(value) * bins), 0), bins - 1)
        counts[idx] += 1
    return [
        {
            "range": f"{idx / bins:.2f}-{(idx + 1) / bins:.2f}",
            "count": count,
        }
        for idx, count in enumerate(counts)
    ]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _print_summary(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    cache = report["cache"]
    geometry = report["concept_geometry"]
    print(
        "[dataset-inspector] "
        f"scan_mode={report['scan_mode']} subsets={report['subset_count']} "
        f"samples={dataset['sample_count']} images={dataset['image_count']} "
        f"captions={dataset['caption_count']} coverage={dataset['caption_coverage']:.2f} "
        f"concepts={dataset['concept_group_count']}"
    )
    print(
        "[dataset-cache] "
        f"pairs={cache['cache_pair_count']} missing_latent={cache['missing_latent_count']} "
        f"missing_text={cache['missing_text_count']} attach={cache['cache_attach_rate']:.2f}"
    )
    if geometry.get("enabled"):
        attach = float(geometry.get("geometry_attach_rate", 0.0) or 0.0)
        neighbor = float(geometry.get("neighbor_same_ratio", 0.0) or 0.0)
        print(
            "[concept-geometry-preflight] "
            f"version={geometry.get('geometry_version')} backend={geometry.get('backend_resolved')} "
            f"attach={attach:.2f} concepts={geometry.get('concept_group_count', 0)} "
            f"neighbor_same={neighbor:.2f} risk={geometry.get('risk', 'unknown')}"
        )
    for warning in report.get("warnings", []):
        print(f"[dataset-warning] {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect trainer dataset/cache/Concept Geometry alignment.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--caption-extension", default=".txt")
    parser.add_argument("--geometry", default="")
    parser.add_argument("--sampler-mode", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = inspect_dataset(
        InspectorOptions(
            data_dir=Path(args.data_dir),
            caption_extension=args.caption_extension,
            geometry_path=Path(args.geometry) if str(args.geometry or "").strip() else None,
            concept_geometry_sampler_mode=args.sampler_mode,
            batch_size=args.batch_size,
        )
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
