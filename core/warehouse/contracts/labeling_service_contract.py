"""
Y-4 - Aesthetic Labeling Service - Interface Contract

Defines the service protocol, annotation store contract, and source provider
contract for a multi-source image annotation workflow.

This module contains NO behavioral implementation.  It specifies:
- What the labeling service lifecycle looks like (config, fetch, annotate).
- What the annotation store must support (CRUD, filtering, stats).
- What image sources must provide (candidates, health, indexing).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SampleStatus(enum.Enum):
    """Review status of an annotation sample."""
    UNREVIEWED = "unreviewed"
    LABELED = "labeled"
    SKIPPED = "skipped"


class ContentCategory(enum.Enum):
    """Content type classification for a sample."""
    ANIME_ILLUST = "anime_illust"
    MANGA = "manga"
    AI_GEN = "ai_gen"
    PHOTO_REAL = "photo_real"
    GARBAGE = "garbage"
    OTHER = "other"


class ScoreDimension(enum.Enum):
    """Scoring dimensions for annotation."""
    AESTHETIC = "aesthetic"
    COMPOSITION = "composition"
    COLOR = "color"
    SEXUAL = "sexual"


# ---------------------------------------------------------------------------
# Data Models (frozen, no behavior)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnotationScores:
    """Per-dimension annotation scores (1-5 scale)."""
    aesthetic: int | None = None
    composition: int | None = None
    color: int | None = None
    sexual: int | None = None


@dataclass(frozen=True)
class AnnotationRecord:
    """Full annotation state for a single sample."""
    sample_id: int
    status: SampleStatus
    scores: AnnotationScores
    in_domain: bool
    content_type: ContentCategory
    exclude_from_score_train: bool
    exclude_from_cls_train: bool
    exclude_reason: str | None
    note: str | None
    updated_at: str | None


@dataclass(frozen=True)
class SampleRecord:
    """A fetched image sample with its metadata."""
    sample_id: int
    source: str
    source_post_id: str | None
    source_page_url: str | None
    original_url: str | None
    created_at: str | None
    local_path: str
    width: int
    height: int
    sha256: str
    annotation: AnnotationRecord | None


@dataclass(frozen=True)
class SourceCandidate:
    """A raw candidate fetched from an image source before dedup/storage."""
    source: str
    source_post_id: str
    source_page_url: str
    original_url: str


@dataclass(frozen=True)
class SourceHealthStatus:
    """Health check result for a single image source."""
    source_name: str
    enabled: bool
    ok: bool
    message: str


@dataclass(frozen=True)
class LabelingStats:
    """Aggregate statistics about the labeling database."""
    total_samples: int
    labeled_count: int
    skipped_count: int
    unreviewed_count: int
    local_indexed_files: int


@dataclass(frozen=True)
class PaginatedResult:
    """Paginated list of samples."""
    items: tuple[SampleRecord, ...]
    page: int
    size: int
    total: int
    pages: int


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class AnnotationStoreProtocol(Protocol):
    """Persistent store for sample metadata and annotation records.

    Implementations MUST:
    - Support atomic upsert of annotation records.
    - Support filtering by status, source, content type, score range.
    - Support SHA-256 dedup and source+post_id dedup.
    - Track sample position for sequential review workflows.
    """

    def get_sample(self, sample_id: int) -> SampleRecord | None: ...

    def get_sample_by_sha(self, sha256: str) -> SampleRecord | None: ...

    def get_sample_by_source(self, source: str, post_id: str) -> SampleRecord | None: ...

    def list_samples(
        self,
        *,
        page: int = 1,
        size: int = 30,
        status: SampleStatus | None = None,
        source: str | None = None,
        content_type: ContentCategory | None = None,
        score_dim: ScoreDimension | None = None,
        score_value: int | None = None,
        order: str = "desc",
    ) -> PaginatedResult: ...

    def upsert_annotation(
        self,
        sample_id: int,
        annotation: AnnotationRecord,
    ) -> None: ...

    def delete_sample(self, sample_id: int) -> bool: ...

    def is_reviewed(self, sample_id: int) -> bool: ...

    def get_stats(self) -> LabelingStats: ...


@runtime_checkable
class SourceProviderProtocol(Protocol):
    """Abstraction for an image source (Danbooru, e621, local filesystem, etc.).

    Implementations MUST:
    - Provide weighted random candidate fetching.
    - Support health checks.
    - Support local index building for filesystem sources.
    """

    @property
    def source_name(self) -> str: ...

    def fetch_candidate(self) -> SourceCandidate | None:
        """Fetch the next random candidate from this source. Returns None on failure."""
        ...

    def load_image(self, candidate: SourceCandidate, *, timeout_sec: float = 8.0) -> bytes:
        """Load the image bytes for a candidate. Raises on failure."""
        ...

    def check_health(self) -> SourceHealthStatus:
        """Check if this source is reachable and functional."""
        ...

    def has_local_files(self) -> bool:
        """Return True if this source has indexed local files."""
        ...

    def reindex_local(self) -> int:
        """Rebuild the local file index. Returns file count."""
        ...


@runtime_checkable
class SourceSamplerProtocol(Protocol):
    """Weighted source selection across multiple providers."""

    def pick_source(self, available: set[str]) -> str:
        """Select a source name based on configured weights."""
        ...

    def enabled_sources(self) -> set[str]:
        """Return the set of currently enabled source names."""
        ...


@runtime_checkable
class LabelingServiceProtocol(Protocol):
    """High-level labeling service combining store, sources, and config.

    Implementations MUST:
    - Be safe for concurrent access from multiple threads.
    - Support config hot-reload without data loss.
    - Deduplicate by SHA-256 across all sources.
    - Apply source cooldown on failure.
    """

    def get_config(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        """Return current service configuration."""
        ...

    def save_and_apply_config(self, new_cfg: dict[str, Any]) -> dict[str, Any]:
        """Merge, validate, and apply new configuration. Thread-safe."""
        ...

    def get_sample(self, sample_id: int) -> SampleRecord:
        """Retrieve a single sample by ID. Raises if not found."""
        ...

    def next_sample(
        self,
        *,
        override_weights: dict[str, float] | None = None,
        avoid_ids: set[int] | None = None,
    ) -> SampleRecord:
        """Fetch the next unreviewed sample. Thread-safe. Raises on exhaustion."""
        ...

    def annotate(
        self,
        *,
        sample_id: int,
        scores: AnnotationScores,
        in_domain: bool = True,
        content_type: ContentCategory = ContentCategory.ANIME_ILLUST,
        exclude_from_score_train: bool = False,
        exclude_from_cls_train: bool = False,
        exclude_reason: str | None = None,
        note: str | None = None,
    ) -> None:
        """Submit a full annotation for a sample. Thread-safe."""
        ...

    def skip(
        self,
        *,
        sample_id: int,
        in_domain: bool = True,
        content_type: ContentCategory = ContentCategory.ANIME_ILLUST,
        exclude_reason: str | None = None,
        note: str | None = None,
    ) -> None:
        """Mark a sample as skipped. Thread-safe."""
        ...

    def list_samples(self, **kwargs: Any) -> PaginatedResult:
        """List samples with filtering and pagination."""
        ...

    def delete_sample(self, sample_id: int, *, delete_image: bool = True) -> dict[str, Any]:
        """Delete a sample and optionally its image file."""
        ...

    def stats(self) -> LabelingStats:
        """Return aggregate labeling statistics."""
        ...

    def check_source_health(self, *, refresh: bool = False) -> list[SourceHealthStatus]:
        """Check health of all configured sources."""
        ...

    def image_path(self, filename: str) -> Path:
        """Resolve the local path for an image filename. Raises if not found."""
        ...
