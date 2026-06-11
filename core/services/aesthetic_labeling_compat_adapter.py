"""Compatibility adapter for the legacy aesthetic labeling page.

The adapter keeps the old HTTP response shape while delegating all state,
queue, history, and file operations to the Lulynx-native labeling service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.aesthetic_labeling import AestheticLabelStore
from backend.core.services.aesthetic_labeling_service import DEFAULT_SETTINGS, AestheticLabelingService


class AestheticLabelingCompatAdapter:
    def __init__(self, store: AestheticLabelStore | None = None) -> None:
        self.service = AestheticLabelingService(store)
        self.store = self.service.store.legacy_store

    def settings(self) -> dict[str, Any]:
        return self.service.settings()

    def save_settings(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.service.save_settings(config)

    def source_names(self) -> list[str]:
        return self.service.source_names()

    def source_records(self) -> list[dict[str, Any]]:
        return self.service.source_records()

    def add_source(self, path: str, name: str = "") -> dict[str, Any]:
        return self.service.add_source(path, name)

    def list_samples(
        self,
        *,
        page: int = 1,
        size: int = 24,
        status: str = "all",
        source: str = "",
        order: str = "desc",
        after_id: int = 0,
    ) -> dict[str, Any]:
        return self.service.list_samples(page=page, size=size, status=status, source=source, order=order, after_id=after_id)

    def next_sample(self, avoid_sample_ids: list[int] | None = None, after_sample_id: int = 0) -> dict[str, Any] | None:
        return self.service.next_sample(avoid_sample_ids=avoid_sample_ids, after_sample_id=after_sample_id)

    def sample(self, sample_id: int | str) -> dict[str, Any] | None:
        return self.service.sample(sample_id)

    def image_path(self, sample_id: int | str, *, thumb: bool = False, thumb_size: int = 480) -> Path | None:
        return self.service.image_path(sample_id, thumb=thumb, thumb_size=thumb_size)

    def annotate(self, body: dict[str, Any]) -> dict[str, Any] | None:
        return self.service.annotate(body)

    def set_label(self, image_path: str, label: str = "", score: Any = None, notes: str = "") -> dict[str, Any]:
        return self.service.set_label(image_path, label, score, notes)

    def annotate_dim(self, body: dict[str, Any]) -> dict[str, Any] | None:
        return self.service.annotate_dim(body)

    def skip(self, body: dict[str, Any]) -> dict[str, Any] | None:
        return self.service.skip(body)

    def last_reviewed(self, status: str = "labeled") -> dict[str, Any]:
        return self.service.last_reviewed(status)

    def stats(self) -> dict[str, Any]:
        return self.service.stats()

    def source_health(self) -> dict[str, Any]:
        return self.service.source_health()

    def reindex_local(self, *, hash_local: bool = False, progress_callback=None) -> dict[str, Any]:
        return self.service.reindex_local(hash_local=hash_local, progress_callback=progress_callback)

    def delete_sample(self, sample_id: int | str, delete_image: bool = False) -> dict[str, Any]:
        return self.service.delete_sample(sample_id, delete_image=delete_image)
