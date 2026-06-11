"""Contracts for image-model GGUF compatibility probes.

Phase 1 is report-only: it describes component coverage and does not write
GGUF files or claim runtime loadability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class ImageGGUFComponent(str, Enum):
    VAE = "vae"
    CLIP = "clip"
    T5 = "t5"
    SD15_UNET = "sd15_unet"
    SDXL_UNET = "sdxl_unet"
    ANIMA_DIT = "anima_dit"
    NEWBIE_DIT = "newbie_dit"
    GENERIC_TENSOR_BUNDLE = "generic_tensor_bundle"
    UNKNOWN = "unknown"


class ImageGGUFCompatibility(str, Enum):
    UNKNOWN = "unknown"
    PROBE_ONLY = "probe_only"
    CONTAINER_CANDIDATE = "container_candidate"
    CONTAINER_COMPATIBLE = "container_compatible"
    RUNTIME_LOADABLE = "runtime_loadable"


@dataclass(frozen=True)
class TensorInfo:
    key: str
    shape: list[int]
    dtype: str

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rank"] = self.rank
        payload["numel"] = self.numel
        return payload


@dataclass(frozen=True)
class ImageGGUFManifest:
    schema_version: int
    adapter_id: str
    component: str
    family: str
    source_path: str
    source_format: str
    compatibility: str
    ok: bool
    tensor_count: int
    matched_tensors: int
    missing_required_tensors: list[str]
    missing_required_prefixes: list[str]
    unexpected_tensors_sample: list[str]
    required_tensors: list[str]
    required_prefixes: list[str]
    dtype_counts: dict[str, int]
    rank_counts: dict[str, int]
    shape_summary: dict[str, Any]
    tensor_samples: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageGGUFExportResult:
    schema_version: int
    ok: bool
    output_path: str
    sidecar_path: str
    component: str
    family: str
    compatibility: str
    source_paths: list[str]
    tensor_count: int
    converted_tensors: int
    skipped_tensors: int
    output_size_bytes: int
    gguf_arch: str
    gguf_file_type: str
    dtype_counts: dict[str, int]
    rank_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageGGUFExportPlan:
    schema_version: int
    ok: bool
    component: str
    family: str
    compatibility: str
    source_paths: list[str]
    tensor_count: int
    unique_tensor_count: int
    duplicate_tensor_count: int
    converted_tensors: int
    skipped_tensors: int
    estimated_tensor_bytes: int
    estimated_container_overhead_bytes: int
    estimated_output_size_bytes: int
    gguf_arch: str
    gguf_file_type: str
    dtype_counts: dict[str, int]
    rank_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageGGUFAdapter(Protocol):
    adapter_id: str
    component: ImageGGUFComponent
    family: str

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        ...

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        ...


__all__ = [
    "ImageGGUFAdapter",
    "ImageGGUFCompatibility",
    "ImageGGUFComponent",
    "ImageGGUFExportPlan",
    "ImageGGUFExportResult",
    "ImageGGUFManifest",
    "TensorInfo",
]
