"""Report generated samples for experimental SDXL Turbo/LCM LoRA outputs.

This is intentionally a file/statistics report, not an image-quality
evaluation. It records whether sample images exist and basic facts about them
so the launcher can distinguish "smoke passed" from "samples were reviewed".
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SHA256_INLINE_LIMIT = 16 * 1024 * 1024
_MAX_SAMPLE_DETAILS = 64


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def _load_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"sidecar root must be an object: {path}")
    return data


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image_size(path: Path) -> tuple[list[int] | None, str | None]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional Pillow
        return None, f"Pillow unavailable: {exc.__class__.__name__}"

    try:
        with Image.open(path) as image:
            width, height = image.size
            return [int(width), int(height)], None
    except Exception as exc:
        return None, f"{path.name}: {exc.__class__.__name__}: {exc}"


def _collect_sample_files(samples_dir: Path | None) -> tuple[list[Path], list[str]]:
    if samples_dir is None:
        return [], ["No samples directory was provided."]
    if not samples_dir.exists():
        return [], [f"Samples path does not exist: {samples_dir}"]
    if samples_dir.is_file():
        if samples_dir.suffix.lower() in _IMAGE_SUFFIXES:
            return [samples_dir], []
        return [], [f"Samples path is not a supported image file: {samples_dir}"]
    if not samples_dir.is_dir():
        return [], [f"Samples path is not a directory: {samples_dir}"]

    files = sorted(
        path
        for path in samples_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not files:
        return [], [f"No supported sample images found in: {samples_dir}"]
    return files, []


def _relative_path(path: Path, root: Path | None) -> str:
    if root is not None and root.is_dir():
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def _dimension_summary(dimensions: list[list[int]]) -> list[dict[str, int]]:
    counter = Counter((int(width), int(height)) for width, height in dimensions)
    return [
        {"width": width, "height": height, "count": count}
        for (width, height), count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def report_turbo_lora_samples(
    output_path: Path,
    *,
    samples_dir: Path | None = None,
    write_sidecar: bool = False,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    samples_root = samples_dir.resolve() if samples_dir is not None else None
    sidecar_path = _sidecar_path(output_path)
    sidecar = _load_sidecar(sidecar_path)
    checked_at = _utc_now()
    issues: list[str] = []

    if not output_path.is_file():
        issues.append(f"Output file does not exist: {output_path}")

    sample_files, sample_issues = _collect_sample_files(samples_root)
    issues.extend(sample_issues)

    if not output_path.is_file():
        status = "failed"
        level = "none"
        note = "Output file is missing; sample reporting was not completed."
    elif not sample_files:
        status = "no_samples"
        level = "none"
        note = "No generated samples were provided; quality is not evaluated."
    else:
        status = "basic_report"
        level = "file_stats"
        note = "Basic sample file report only; visual quality is not evaluated."

    details: list[dict[str, Any]] = []
    extension_counts: Counter[str] = Counter()
    dimensions: list[list[int]] = []
    total_size = 0

    for sample in sample_files:
        stat = sample.stat()
        size = int(stat.st_size)
        total_size += size
        extension = sample.suffix.lower()
        extension_counts[extension] += 1
        image_size, image_issue = _read_image_size(sample)
        if image_size:
            dimensions.append(image_size)
        if image_issue:
            issues.append(image_issue)

        if len(details) < _MAX_SAMPLE_DETAILS:
            entry: dict[str, Any] = {
                "path": str(sample),
                "relative_path": _relative_path(sample, samples_root),
                "extension": extension,
                "size_bytes": size,
                "modified_at": _file_time(sample),
            }
            if image_size:
                entry["width"] = image_size[0]
                entry["height"] = image_size[1]
            if size <= _SHA256_INLINE_LIMIT:
                entry["sha256"] = _sha256_file(sample)
            else:
                entry["hash_status"] = "deferred_large_file"
            details.append(entry)

    sample_evaluation = {
        "status": status,
        "level": level,
        "checked_at": checked_at,
        "output_path": str(output_path),
        "metadata_sidecar": str(sidecar_path),
        "samples_dir": str(samples_root) if samples_root is not None else "",
        "sample_count": len(sample_files),
        "total_size_bytes": total_size,
        "extension_counts": dict(sorted(extension_counts.items())),
        "dimensions": _dimension_summary(dimensions),
        "files": details,
        "truncated": len(sample_files) > len(details),
        "issues": issues[:32],
        "note": note,
        "quality_boundary": "not_final_quality_validation",
    }

    if write_sidecar:
        updated = dict(sidecar)
        if "lulynx.quality_status" not in updated:
            updated["lulynx.quality_status"] = "not_quality_validated"
        if "lulynx.quality_note" not in updated:
            updated["lulynx.quality_note"] = "Sample report only; final quality validation has not been completed."
        updated.update({
            "lulynx.sample_eval_status": status,
            "lulynx.sample_eval_level": level,
            "lulynx.sample_eval_sample_count": len(sample_files),
            "lulynx.sample_eval_checked_at": checked_at,
            "lulynx.sample_eval_note": note,
            "lulynx.sample_eval_samples_dir": str(samples_root) if samples_root is not None else "",
            "lulynx.sample_eval_extension_counts": json.dumps(dict(sorted(extension_counts.items())), sort_keys=True),
            "lulynx.sample_eval_dimension_count": len(sample_evaluation["dimensions"]),
            "sample_evaluation": sample_evaluation,
        })
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": status,
        "output_path": str(output_path),
        "metadata_sidecar": str(sidecar_path),
        "sample_evaluation": sample_evaluation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report generated samples for a Lulynx SDXL Turbo/LCM LoRA output")
    parser.add_argument("output", help="Path to the output .safetensors file")
    parser.add_argument("--samples-dir", default="", help="Directory containing generated sample images")
    parser.add_argument("--write-sidecar", action="store_true", help="Write sample report fields back to the metadata sidecar")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir) if str(args.samples_dir).strip() else None
    result = report_turbo_lora_samples(
        Path(args.output),
        samples_dir=samples_dir,
        write_sidecar=args.write_sidecar,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"no_samples", "basic_report"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
