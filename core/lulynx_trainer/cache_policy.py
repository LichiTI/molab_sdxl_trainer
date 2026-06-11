# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Cache policy resolution for manifest-driven cache-first routes.

This module keeps top-level imports pure stdlib so it can run during
config/preflight without importing torch, diffusers, or native runtime
dependencies.
Newbie cache readiness performs a lazy contract check only when Newbie cache
files are discovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple

try:
    from .cache_manifest import (
        discover_cache_samples,
        manifest_path_for,
        validate_cache_manifest,
    )
except ImportError:  # pragma: no cover - standalone smoke loading
    from cache_manifest import (
        discover_cache_samples,
        manifest_path_for,
        validate_cache_manifest,
    )


@dataclass(frozen=True)
class CachePolicyReport:
    family: str
    mode: str
    data_dir: Path
    manifest_path: Path
    manifest_present: bool
    manifest_ok: bool
    cache_sample_count: int
    cache_file_count: int
    can_use_cache: bool
    should_rebuild: bool
    cache_ready: bool = False
    ready_cache_sample_count: int = 0
    ready_cache_file_count: int = 0
    cache_contract_ok: bool = True
    cache_contract_reasons: Tuple[str, ...] = ()
    cache_contract_pooled_shapes: Tuple[Tuple[int, ...], ...] = ()
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


def resolve_cache_policy(
    data_dir: str | Path,
    *,
    family: str,
    mode: str = "",
    trust_cache: bool = False,
    rebuild_requested: bool = False,
    force_cache_only: bool = False,
    strict_manifest: bool = False,
) -> CachePolicyReport:
    """Resolve cache usability from manifest state and route cache flags.

    ``strict_manifest`` is deliberately opt-in. Existing users may already have
    valid legacy cache files without a manifest, so the default behavior warns
    but keeps legacy cache-first training usable.
    """

    normalized_family = _normalize_family(family)
    normalized_mode = _normalize_mode(mode)
    root = Path(data_dir)
    manifest_path = manifest_path_for(root, normalized_family)
    manifest_present = manifest_path.is_file()

    samples = []
    warnings = []
    errors = []
    notes = []
    manifest_ok = False

    if root.is_dir():
        try:
            samples = discover_cache_samples(root, family=normalized_family)
        except Exception as exc:
            errors.append(f"cache discovery failed: {type(exc).__name__}: {exc}")
    else:
        errors.append(f"cache data_dir does not exist: {root}")

    cache_file_count = sum(len(sample.cache_files) for sample in samples)
    has_cache = bool(samples)
    cache_ready = has_cache
    ready_cache_sample_count = len(samples) if has_cache else 0
    ready_cache_file_count = cache_file_count if has_cache else 0
    cache_contract_ok = True
    cache_contract_reasons: Tuple[str, ...] = ()
    cache_contract_pooled_shapes: Tuple[Tuple[int, ...], ...] = ()

    if manifest_present:
        validation = validate_cache_manifest(root, family=normalized_family)
        manifest_ok = validation.ok
        if validation.ok:
            notes.append(f"cache manifest validated: {manifest_path.name}")
        else:
            details = _join_nonempty(
                validation.errors,
                _prefix("missing", validation.missing_files),
                _prefix("changed", validation.changed_files),
            )
            if trust_cache:
                warnings.append(
                    "cache manifest validation failed but trust_cache=True; "
                    f"using cache anyway ({details})"
                )
            elif strict_manifest:
                errors.append(f"cache manifest validation failed: {details}")
            else:
                warnings.append(
                    "cache manifest validation failed; using legacy-compatible cache behavior. "
                    f"Set trust_cache=True to silence this warning or rebuild cache. ({details})"
                )
    elif has_cache:
        if strict_manifest:
            errors.append(f"cache manifest missing: {manifest_path}")
        else:
            warnings.append(
                f"cache manifest missing; using legacy cache files without fingerprint validation: {manifest_path.name}"
            )
    else:
        notes.append("no cache samples discovered")

    if normalized_family == "newbie" and has_cache:
        contract = _resolve_newbie_cache_contract(root, samples)
        cache_contract_ok = bool(contract.get("ok"))
        cache_contract_reasons = tuple(str(reason) for reason in contract.get("reasons", ()))
        cache_contract_pooled_shapes = _shape_tuples(contract.get("pooled_shapes", ()))
        ready_cache_file_count = int(contract.get("cache_file_count") or 0) if cache_contract_ok else 0
        ready_cache_sample_count = len(samples) if cache_contract_ok else 0
        cache_ready = cache_contract_ok
        if cache_contract_ok:
            notes.append("newbie cache contract validated")
        else:
            errors.append(
                "newbie cache contract invalid: "
                f"{_join_nonempty(cache_contract_reasons)}"
            )

    should_rebuild = bool(rebuild_requested or normalized_mode == "rebuild_cache")
    if should_rebuild:
        notes.append("cache rebuild requested")

    if force_cache_only and not has_cache and not should_rebuild:
        errors.append("force_cache_only requested but no cache samples were discovered")

    can_use_cache = has_cache and not errors
    if strict_manifest and manifest_present and not manifest_ok and not trust_cache:
        can_use_cache = False

    return CachePolicyReport(
        family=normalized_family,
        mode=normalized_mode,
        data_dir=root,
        manifest_path=manifest_path,
        manifest_present=manifest_present,
        manifest_ok=manifest_ok,
        cache_sample_count=len(samples),
        cache_file_count=cache_file_count,
        can_use_cache=can_use_cache,
        should_rebuild=should_rebuild,
        cache_ready=cache_ready,
        ready_cache_sample_count=ready_cache_sample_count,
        ready_cache_file_count=ready_cache_file_count,
        cache_contract_ok=cache_contract_ok,
        cache_contract_reasons=cache_contract_reasons,
        cache_contract_pooled_shapes=cache_contract_pooled_shapes,
        errors=tuple(errors),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def _normalize_family(family: str) -> str:
    return str(family or "").strip().lower().replace("-", "_")


def _normalize_mode(mode: str) -> str:
    return str(mode or "").strip().lower().replace("-", "_")


def _prefix(label: str, values: Iterable[str]) -> Tuple[str, ...]:
    values = tuple(values)
    if not values:
        return ()
    preview = ", ".join(values[:5])
    if len(values) > 5:
        preview += f", ... (+{len(values) - 5})"
    return (f"{label}: {preview}",)


def _join_nonempty(*parts: Iterable[str]) -> str:
    flattened = [str(part) for group in parts for part in group if str(part)]
    return "; ".join(flattened) if flattened else "unknown validation error"


def _resolve_newbie_cache_contract(root: Path, samples: Iterable[Any]) -> dict[str, Any]:
    try:
        from .newbie_cache_contract import newbie_cache_contract_for_files
    except ImportError:  # pragma: no cover - standalone smoke loading
        from newbie_cache_contract import newbie_cache_contract_for_files

    return newbie_cache_contract_for_files(_cache_file_paths(root, samples))


def _cache_file_paths(root: Path, samples: Iterable[Any]) -> Tuple[Path, ...]:
    paths = []
    for sample in samples:
        for rel_path in getattr(sample, "cache_files", ()):
            path = Path(rel_path)
            paths.append(path if path.is_absolute() else root / path)
    return tuple(paths)


def _shape_tuples(shapes: Iterable[Any]) -> Tuple[Tuple[int, ...], ...]:
    result = []
    for shape in shapes:
        try:
            result.append(tuple(int(dim) for dim in shape))
        except TypeError:
            continue
    return tuple(result)
