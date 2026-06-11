# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Manifest and fingerprint helpers for cache-first training datasets.

The manifest is deliberately lightweight and pure-Python: it records which raw
files and cache artifacts existed when a cache build completed, plus cheap file
fingerprints.  Training code can later decide whether to trust, warn, rebuild,
or fail based on this report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CACHE_MANIFEST_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class CacheManifestValidationReport:
    ok: bool
    manifest_path: Path
    missing_files: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "manifest_path": str(self.manifest_path),
            "missing_files": list(self.missing_files),
            "changed_files": list(self.changed_files),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class CacheTrustReport:
    ok: bool
    mode: str
    manifest_path: Path
    strict_sha256_required: bool
    hash_validation_skipped: bool = False
    missing_files: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "mode": self.mode,
            "manifest_path": str(self.manifest_path),
            "strict_sha256_required": bool(self.strict_sha256_required),
            "hash_validation_skipped": bool(self.hash_validation_skipped),
            "missing_files": list(self.missing_files),
            "changed_files": list(self.changed_files),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CacheManifestWriteResult:
    manifest_path: Path
    sample_count: int
    cache_file_count: int


@dataclass(frozen=True)
class CacheSampleRecord:
    stem: str
    source_image: str = ""
    caption: str = ""
    cache_files: tuple[str, ...] = ()
    fingerprints: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def manifest_path_for(root: str | Path, family: str) -> Path:
    normalized = _normalize_family(family)
    return Path(root) / f"lulynx_cache_manifest_{normalized}.json"


def write_cache_manifest(
    root: str | Path,
    *,
    family: str,
    builder: str,
    config: Optional[Dict[str, Any]] = None,
    include_sha256: bool = False,
) -> CacheManifestWriteResult:
    """Discover cache artifacts under *root* and write a manifest JSON file."""

    root_path = Path(root)
    samples = discover_cache_samples(root_path, family=family, include_sha256=include_sha256)
    manifest_path = manifest_path_for(root_path, family)
    payload = {
        "manifest_version": CACHE_MANIFEST_VERSION,
        "family": _normalize_family(family),
        "builder": builder,
        "root": str(root_path),
        "config": dict(config or {}),
        "samples": [_sample_to_json(sample) for sample in samples],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cache_file_count = sum(len(sample.cache_files) for sample in samples)
    return CacheManifestWriteResult(
        manifest_path=manifest_path,
        sample_count=len(samples),
        cache_file_count=cache_file_count,
    )


def load_cache_manifest(root: str | Path, *, family: str) -> Dict[str, Any]:
    path = manifest_path_for(root, family)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cache_manifest(
    root: str | Path,
    *,
    family: str,
    include_sha256: bool = False,
) -> CacheManifestValidationReport:
    """Validate recorded file fingerprints against the current filesystem."""

    root_path = Path(root)
    path = manifest_path_for(root_path, family)
    if not path.is_file():
        return CacheManifestValidationReport(
            ok=False,
            manifest_path=path,
            errors=(f"cache manifest not found: {path}",),
        )

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CacheManifestValidationReport(
            ok=False,
            manifest_path=path,
            errors=(f"cache manifest cannot be read: {type(exc).__name__}: {exc}",),
        )

    missing: List[str] = []
    changed: List[str] = []
    errors: List[str] = []

    if int(manifest.get("manifest_version", 0) or 0) != CACHE_MANIFEST_VERSION:
        errors.append(
            f"cache manifest version mismatch: expected {CACHE_MANIFEST_VERSION}, "
            f"got {manifest.get('manifest_version')!r}"
        )
    if str(manifest.get("family", "")).lower() != _normalize_family(family):
        errors.append(
            f"cache manifest family mismatch: expected {_normalize_family(family)}, "
            f"got {manifest.get('family')!r}"
        )

    for sample in manifest.get("samples", []):
        fingerprints = sample.get("fingerprints", {})
        if not isinstance(fingerprints, dict):
            errors.append(f"sample {sample.get('stem', '<unknown>')} has invalid fingerprints")
            continue
        for rel_path, expected in fingerprints.items():
            file_path = root_path / rel_path
            if not file_path.is_file():
                missing.append(rel_path)
                continue
            actual = fingerprint_file(file_path, include_sha256=include_sha256)
            if _fingerprint_changed(expected, actual, include_sha256=include_sha256):
                changed.append(rel_path)

    return CacheManifestValidationReport(
        ok=not missing and not changed and not errors,
        manifest_path=path,
        missing_files=tuple(sorted(set(missing))),
        changed_files=tuple(sorted(set(changed))),
        errors=tuple(errors),
    )


def build_cache_trust_report(
    root: str | Path,
    *,
    family: str,
    mode: str = "strict",
) -> CacheTrustReport:
    """Build an auditable cache-trust report for cache-first trainer manifests.

    ``strict`` is the default production posture and validates sha256 hashes.
    ``trusted`` is an explicit fast-local-iteration mode: it skips hash
    validation, still checks the cheaper manifest fingerprints, and emits a
    warning that must be visible in run metadata.
    """

    normalized_mode = str(mode or "strict").strip().lower().replace("-", "_")
    if normalized_mode not in {"strict", "trusted"}:
        normalized_mode = "strict"
    include_sha256 = normalized_mode == "strict"
    validation = validate_cache_manifest(root, family=family, include_sha256=include_sha256)
    warnings: list[str] = []
    if normalized_mode == "trusted":
        warnings.append("hash_validation_skipped_trusted_cache")
    return CacheTrustReport(
        ok=bool(validation.ok),
        mode=normalized_mode,
        manifest_path=validation.manifest_path,
        strict_sha256_required=include_sha256,
        hash_validation_skipped=not include_sha256,
        missing_files=validation.missing_files,
        changed_files=validation.changed_files,
        errors=validation.errors,
        warnings=tuple(warnings),
    )


def discover_cache_samples(
    root: str | Path,
    *,
    family: str,
    include_sha256: bool = False,
) -> List[CacheSampleRecord]:
    family = _normalize_family(family)
    root_path = Path(root)
    if family == "anima":
        return _discover_anima_samples(root_path, include_sha256=include_sha256)
    if family == "newbie":
        return _discover_newbie_samples(root_path, include_sha256=include_sha256)
    raise ValueError(f"Unsupported cache manifest family: {family}")


def fingerprint_file(path: str | Path, *, include_sha256: bool = False) -> Dict[str, Any]:
    file_path = Path(path)
    stat = file_path.stat()
    result: Dict[str, Any] = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        result["sha256"] = h.hexdigest()
    return result


def _discover_anima_samples(root: Path, *, include_sha256: bool) -> List[CacheSampleRecord]:
    text_by_stem: Dict[str, Path] = {}
    for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
        for path in root.rglob(f"*{suffix}"):
            stem = path.name[: -len(suffix)]
            text_by_stem.setdefault(stem, path)

    records: List[CacheSampleRecord] = []
    for stem, text_path in sorted(text_by_stem.items()):
        latent_candidates: List[Path] = []
        for ext in (".npz", ".safetensors", ".pt"):
            latent_candidates.extend(root.rglob(f"{stem}_*_anima{ext}"))
        latent_candidates = sorted(set(latent_candidates), key=lambda p: str(p.relative_to(root)))
        if not latent_candidates:
            continue
        cache_files = tuple(_rel(root, path) for path in [text_path, *latent_candidates])
        source_image = _find_source_image(root, stem)
        caption = _find_caption(root, stem)
        records.append(
            CacheSampleRecord(
                stem=stem,
                source_image=_rel(root, source_image) if source_image else "",
                caption=_rel(root, caption) if caption else "",
                cache_files=cache_files,
                fingerprints=_fingerprints_for(
                    root,
                    [p for p in [source_image, caption, text_path, *latent_candidates] if p is not None],
                    include_sha256=include_sha256,
                ),
            )
        )
    return records


def _discover_newbie_samples(root: Path, *, include_sha256: bool) -> List[CacheSampleRecord]:
    cache_by_stem: Dict[str, Path] = {}
    for suffix in ("_newbie.npz", "_newbie.safetensors", "_newbie.pt"):
        for path in root.rglob(f"*{suffix}"):
            stem = path.name[: -len(suffix)]
            cache_by_stem.setdefault(stem, path)

    records: List[CacheSampleRecord] = []
    for stem, cache_path in sorted(cache_by_stem.items()):
        source_image = _find_source_image(root, stem)
        caption = _find_caption(root, stem)
        records.append(
            CacheSampleRecord(
                stem=stem,
                source_image=_rel(root, source_image) if source_image else "",
                caption=_rel(root, caption) if caption else "",
                cache_files=(_rel(root, cache_path),),
                fingerprints=_fingerprints_for(
                    root,
                    [p for p in [source_image, caption, cache_path] if p is not None],
                    include_sha256=include_sha256,
                ),
            )
        )
    return records


def _find_source_image(root: Path, stem: str) -> Optional[Path]:
    for suffix in sorted(IMAGE_SUFFIXES):
        direct = root / f"{stem}{suffix}"
        if direct.is_file():
            return direct
    for path in sorted(root.rglob(f"{stem}.*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def _find_caption(root: Path, stem: str) -> Optional[Path]:
    for suffix in (".txt", ".caption"):
        direct = root / f"{stem}{suffix}"
        if direct.is_file():
            return direct
    for suffix in (".txt", ".caption"):
        matches = sorted(root.rglob(f"{stem}{suffix}"))
        if matches:
            return matches[0]
    return None


def _fingerprints_for(
    root: Path,
    paths: Iterable[Path],
    *,
    include_sha256: bool,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if path.is_file():
            result[_rel(root, path)] = fingerprint_file(path, include_sha256=include_sha256)
    return result


def _fingerprint_changed(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    *,
    include_sha256: bool,
) -> bool:
    for key in ("size", "mtime_ns"):
        if int(expected.get(key, -1)) != int(actual.get(key, -2)):
            return True
    if include_sha256 and expected.get("sha256") != actual.get("sha256"):
        return True
    return False


def _sample_to_json(sample: CacheSampleRecord) -> Dict[str, Any]:
    return {
        "stem": sample.stem,
        "source_image": sample.source_image,
        "caption": sample.caption,
        "cache_files": list(sample.cache_files),
        "fingerprints": sample.fingerprints,
    }


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _normalize_family(family: str) -> str:
    return str(family or "").strip().lower().replace("-", "_")
