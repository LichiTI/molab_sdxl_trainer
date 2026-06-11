# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Newbie cache manifest parity validation.

The generic cache manifest proves that source/cache files have not changed.
This module adds Newbie-specific semantic validation: cache schema version,
latent/text/pooled/mask fields, finite tensors, loss-mask shape, and builder
configuration hints such as prompt prefix/template.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    from .cache_manifest import load_cache_manifest, manifest_path_for, validate_cache_manifest
    from .newbie_cached_dataset import NewbieCacheSchema, load_newbie_cache_arrays
except ImportError:  # pragma: no cover - standalone smoke loading
    from cache_manifest import load_cache_manifest, manifest_path_for, validate_cache_manifest
    from newbie_cached_dataset import NewbieCacheSchema, load_newbie_cache_arrays


@dataclass(frozen=True)
class NewbieCacheParitySample:
    cache_file: str
    latent_shape: Tuple[int, ...]
    hidden_shape: Tuple[int, ...]
    pooled_shape: Tuple[int, ...] = ()
    attention_mask_shape: Tuple[int, ...] = ()
    loss_mask_shape: Tuple[int, ...] = ()


@dataclass(frozen=True)
class NewbieCacheParityReport:
    ok: bool
    manifest_path: Path
    sample_count: int
    cache_file_count: int
    samples: Tuple[NewbieCacheParitySample, ...] = ()
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


def validate_newbie_cache_parity(
    root: str | Path,
    *,
    schema: Optional[NewbieCacheSchema] = None,
    expected_gemma3_prompt: str = "",
    require_manifest: bool = True,
    include_fingerprint_validation: bool = True,
) -> NewbieCacheParityReport:
    """Validate Newbie cache files against the manifest and cache schema."""

    root_path = Path(root)
    manifest_path = manifest_path_for(root_path, "newbie")
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    samples: List[NewbieCacheParitySample] = []

    if include_fingerprint_validation:
        fp_report = validate_cache_manifest(root_path, family="newbie")
        if not fp_report.ok:
            details = "; ".join(
                list(fp_report.errors)
                + [f"missing={list(fp_report.missing_files)}" if fp_report.missing_files else ""]
                + [f"changed={list(fp_report.changed_files)}" if fp_report.changed_files else ""]
            ).strip("; ")
            if require_manifest:
                errors.append(f"Newbie cache fingerprint validation failed: {details}")
            else:
                warnings.append(f"Newbie cache fingerprint validation warning: {details}")

    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = load_cache_manifest(root_path, family="newbie")
        except Exception as exc:
            errors.append(f"Newbie cache manifest cannot be read: {type(exc).__name__}: {exc}")
    elif require_manifest:
        errors.append(f"Newbie cache manifest missing: {manifest_path}")
    else:
        warnings.append(f"Newbie cache manifest missing: {manifest_path}")

    if manifest:
        family = str(manifest.get("family", "") or "").lower()
        if family != "newbie":
            errors.append(f"Newbie cache manifest family mismatch: {family!r}")
        config = manifest.get("config", {})
        if not isinstance(config, Mapping):
            config = {}
            warnings.append("Newbie cache manifest config section is invalid")
        schema_version = int(config.get("schema_version", 0) or 0)
        if schema_version and schema_version != NewbieCacheSchema().version:
            errors.append(
                f"Newbie cache manifest schema_version mismatch: expected {NewbieCacheSchema().version}, got {schema_version}"
            )
        configured_prompt = str(config.get("gemma3_prompt", "") or "")
        if expected_gemma3_prompt and configured_prompt != expected_gemma3_prompt:
            errors.append(
                "Newbie cache Gemma3 prompt template mismatch: "
                f"expected={expected_gemma3_prompt!r}, manifest={configured_prompt!r}"
            )
        elif configured_prompt:
            notes.append("Newbie cache manifest records Gemma3 prompt template")
        else:
            warnings.append("Newbie cache manifest has no Gemma3 prompt template record")

    raw_samples = manifest.get("samples", []) if manifest else []
    if not isinstance(raw_samples, list):
        raw_samples = []
        errors.append("Newbie cache manifest samples section is invalid")

    resolved_schema = schema or NewbieCacheSchema()
    cache_file_count = 0
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            errors.append("Newbie cache manifest contains a non-object sample")
            continue
        cache_files = sample.get("cache_files", [])
        if not isinstance(cache_files, list):
            errors.append(f"Newbie sample {sample.get('stem', '<unknown>')} has invalid cache_files")
            continue
        newbie_cache_files = [
            str(path)
            for path in cache_files
            if str(path).endswith(("_newbie.npz", "_newbie.safetensors", "_newbie.pt"))
        ]
        cache_file_count += len(newbie_cache_files)
        if len(newbie_cache_files) != 1:
            errors.append(
                f"Newbie sample {sample.get('stem', '<unknown>')} expected 1 cache file, got {len(newbie_cache_files)}"
            )
            continue
        rel_cache = newbie_cache_files[0]
        cache_path = root_path / rel_cache
        try:
            arrays = load_newbie_cache_arrays(cache_path, resolved_schema)
            samples.append(
                NewbieCacheParitySample(
                    cache_file=rel_cache,
                    latent_shape=tuple(int(v) for v in arrays.latents.shape),
                    hidden_shape=tuple(int(v) for v in arrays.encoder_hidden_states.shape),
                    pooled_shape=tuple(int(v) for v in arrays.pooled_prompt_embeds.shape)
                    if arrays.pooled_prompt_embeds is not None
                    else (),
                    attention_mask_shape=tuple(int(v) for v in arrays.attention_mask.shape)
                    if arrays.attention_mask is not None
                    else (),
                    loss_mask_shape=tuple(int(v) for v in arrays.loss_mask.shape)
                    if arrays.loss_mask is not None
                    else (),
                )
            )
        except Exception as exc:
            errors.append(f"Newbie cache semantic validation failed for {rel_cache}: {type(exc).__name__}: {exc}")

    if manifest and not raw_samples:
        errors.append("Newbie cache manifest contains no samples")
    if manifest and len(samples) != len(raw_samples):
        warnings.append(f"Newbie cache parity validated {len(samples)}/{len(raw_samples)} manifest samples")

    return NewbieCacheParityReport(
        ok=not errors,
        manifest_path=manifest_path,
        sample_count=len(raw_samples),
        cache_file_count=cache_file_count,
        samples=tuple(samples),
        errors=tuple(errors),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def report_to_json(report: NewbieCacheParityReport) -> str:
    """Return a compact JSON representation for logs/tests."""

    payload = {
        "ok": report.ok,
        "manifest_path": str(report.manifest_path),
        "sample_count": report.sample_count,
        "cache_file_count": report.cache_file_count,
        "samples": [sample.__dict__ for sample in report.samples],
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "notes": list(report.notes),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
