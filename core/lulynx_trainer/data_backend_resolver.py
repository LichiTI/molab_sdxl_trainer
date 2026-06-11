# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Dataset backend strategy resolver.

This module is intentionally stdlib-only.  It does not construct training
datasets and is safe to import during config/preflight probes.  The default
route keeps CaptionDataset unchanged; explicit WebDataset uses the materialized
CaptionDataset bridge when compatible shards are present.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATA_BACKEND_CHOICES = {"auto", "caption", "raw", "webdataset", "dali"}
WEBDATASET_SHARD_SUFFIXES = (".tar", ".tar.gz", ".tgz")


PackageAvailability = Callable[[str], bool]


@dataclass(frozen=True)
class WebDatasetShardPlan:
    data_dir: str
    shard_count: int
    shards: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "shard_count": self.shard_count,
            "shards": list(self.shards),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DataBackendDecision:
    requested_backend: str
    resolved_backend: str
    webdataset_available: bool
    dali_available: bool
    webdataset_shards: WebDatasetShardPlan
    fallback_reason: str = ""
    warnings: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    training_integration: str = "not_wired"
    experimental: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "webdataset_available": self.webdataset_available,
            "dali_available": self.dali_available,
            "webdataset_shards": self.webdataset_shards.as_dict(),
            "fallback_reason": self.fallback_reason,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "training_integration": self.training_integration,
            "experimental": self.experimental,
        }


def normalize_data_backend(value: Any) -> str:
    backend = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "pil": "caption",
        "imagefolder": "caption",
        "image_folder": "caption",
        "caption_dataset": "caption",
        "captiondataset": "caption",
        "folder": "caption",
        "raw_caption": "caption",
        "raw": "raw",
        "tar": "webdataset",
        "tars": "webdataset",
        "wds": "webdataset",
        "web_dataset": "webdataset",
        "nvidia_dali": "dali",
    }
    backend = aliases.get(backend.replace(" ", ""), backend)
    return backend if backend in DATA_BACKEND_CHOICES else "auto"


def package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def discover_webdataset_shards(
    data_dir: str | Path | None,
    *,
    max_preview: int = 8,
    recursive: bool = False,
) -> WebDatasetShardPlan:
    if data_dir is None:
        return WebDatasetShardPlan(data_dir="", shard_count=0, warnings=("data_dir not supplied",))

    root = Path(data_dir)
    warnings: List[str] = []
    if not root.exists():
        return WebDatasetShardPlan(data_dir=str(root), shard_count=0, warnings=(f"data_dir does not exist: {root}",))
    if root.is_file():
        candidates = [root] if _looks_like_webdataset_shard(root) else []
    else:
        iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
        candidates = [path for path in iterator if path.is_file() and _looks_like_webdataset_shard(path)]

    candidates = sorted(candidates, key=lambda path: str(path).lower())
    if len(candidates) > max_preview:
        warnings.append(f"showing first {max_preview} of {len(candidates)} detected WebDataset shards")

    preview = tuple(str(path) for path in candidates[:max_preview])
    return WebDatasetShardPlan(data_dir=str(root), shard_count=len(candidates), shards=preview, warnings=tuple(warnings))


def resolve_data_backend(
    requested: Any = "auto",
    *,
    data_dir: str | Path | None = None,
    package_availability: Optional[PackageAvailability | Mapping[str, bool]] = None,
    recursive_shard_scan: bool = False,
) -> DataBackendDecision:
    requested_backend = normalize_data_backend(requested)
    availability = _availability_lookup(package_availability)
    webdataset_package_ok = availability("webdataset")
    dali_ok = availability("nvidia.dali") or availability("nvidia_dali")
    shard_plan = discover_webdataset_shards(data_dir, recursive=recursive_shard_scan)
    materialized_webdataset_ok = shard_plan.shard_count > 0
    webdataset_ok = webdataset_package_ok or materialized_webdataset_ok
    warnings: List[str] = list(shard_plan.warnings)
    notes: List[str] = []
    fallback_reason = ""
    experimental = requested_backend in {"webdataset", "dali"}

    if requested_backend == "auto":
        resolved = "caption"
        notes.append("auto keeps the existing CaptionDataset path")
        if shard_plan.shard_count:
            notes.append("WebDataset shards are available for explicit materialized CaptionDataset training")
        return DataBackendDecision(
            requested_backend=requested_backend,
            resolved_backend=resolved,
            webdataset_available=webdataset_ok,
            dali_available=dali_ok,
            webdataset_shards=shard_plan,
            warnings=tuple(warnings),
            notes=tuple(notes),
            experimental=False,
        )

    if requested_backend in {"caption", "raw"}:
        return DataBackendDecision(
            requested_backend=requested_backend,
            resolved_backend="caption",
            webdataset_available=webdataset_ok,
            dali_available=dali_ok,
            webdataset_shards=shard_plan,
            warnings=tuple(warnings),
            notes=("existing CaptionDataset-compatible path",),
        )

    if requested_backend == "webdataset":
        if shard_plan.shard_count <= 0:
            fallback_reason = "no .tar/.tar.gz WebDataset shards were detected"
        if fallback_reason:
            warnings.append(f"data_backend=webdataset resolved to caption: {fallback_reason}")
            return DataBackendDecision(
                requested_backend=requested_backend,
                resolved_backend="caption",
                webdataset_available=webdataset_ok,
                dali_available=dali_ok,
                webdataset_shards=shard_plan,
                fallback_reason=fallback_reason,
                warnings=tuple(warnings),
                notes=("explicit WebDataset needs materialized .tar/.tar.gz/.tgz shards",),
                experimental=True,
            )
        if not webdataset_package_ok:
            notes.append("webdataset package is unavailable; using the built-in materialized CaptionDataset bridge")
        return DataBackendDecision(
            requested_backend=requested_backend,
            resolved_backend="webdataset",
            webdataset_available=webdataset_ok,
            dali_available=dali_ok,
            webdataset_shards=shard_plan,
            warnings=tuple(warnings),
            notes=tuple(notes + ["WebDataset shards will use the materialized CaptionDataset bridge"]),
            training_integration="materialized_captiondataset_bridge",
            experimental=True,
        )

    if requested_backend == "dali":
        fallback_reason = "DALI backend is reserved for a future high-dependency data pipeline"
        if not dali_ok:
            fallback_reason += "; nvidia.dali is not installed"
        warnings.append(f"data_backend=dali resolved to caption: {fallback_reason}")
        return DataBackendDecision(
            requested_backend=requested_backend,
            resolved_backend="caption",
            webdataset_available=webdataset_ok,
            dali_available=dali_ok,
            webdataset_shards=shard_plan,
            fallback_reason=fallback_reason,
            warnings=tuple(warnings),
            experimental=True,
        )

    return DataBackendDecision(
        requested_backend=requested_backend,
        resolved_backend="caption",
        webdataset_available=webdataset_ok,
        dali_available=dali_ok,
        webdataset_shards=shard_plan,
        fallback_reason="unknown backend resolved through default CaptionDataset path",
        warnings=tuple(warnings),
    )


def _availability_lookup(
    package_availability: Optional[PackageAvailability | Mapping[str, bool]],
) -> PackageAvailability:
    if package_availability is None:
        return package_available
    if isinstance(package_availability, Mapping):
        lower = {str(key).lower(): bool(value) for key, value in package_availability.items()}

        def lookup(name: str) -> bool:
            return bool(lower.get(str(name).lower(), False))

        return lookup
    return package_availability


def _looks_like_webdataset_shard(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in WEBDATASET_SHARD_SUFFIXES)
