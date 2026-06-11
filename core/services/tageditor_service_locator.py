"""Lazy service locator for tag-editor compatibility routes."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


TagEditorFactory = Callable[[], Any]
TagAnalysisFactory = Callable[[Any], Any]
TagSuggestionFactory = Callable[[Any], Any]
TagJobStoreFactory = Callable[[], Any]
DatasetManifestFactory = Callable[[Any, Any], Any]


def _default_tag_editor_factory() -> Any:
    from core.services.tag_editor_service import TagEditorService

    return TagEditorService()


def _default_tag_analysis_factory(tag_editor: Any) -> Any:
    from core.services.tag_analysis_service import TagAnalysisService

    return TagAnalysisService(tag_editor)


def _default_tag_suggestion_factory(analysis_service: Any) -> Any:
    from core.services.tag_suggestion_service import TagSuggestionService

    return TagSuggestionService(analysis_service)


def _default_tag_job_store_factory() -> Any:
    from core.services.tag_job_store import TagJobStore

    return TagJobStore()


def _default_dataset_manifest_factory(tag_editor: Any, analysis_service: Any) -> Any:
    from core.services.dataset_manifest_service import DatasetManifestService

    return DatasetManifestService(tag_editor, analysis_service)


class TageditorServiceLocator:
    """Own the lazy service graph used by tag-editor route adapters."""

    def __init__(
        self,
        *,
        tag_editor_factory: TagEditorFactory = _default_tag_editor_factory,
        tag_analysis_factory: TagAnalysisFactory = _default_tag_analysis_factory,
        tag_suggestion_factory: TagSuggestionFactory = _default_tag_suggestion_factory,
        tag_job_store_factory: TagJobStoreFactory = _default_tag_job_store_factory,
        dataset_manifest_factory: DatasetManifestFactory = _default_dataset_manifest_factory,
    ) -> None:
        self._tag_editor_factory = tag_editor_factory
        self._tag_analysis_factory = tag_analysis_factory
        self._tag_suggestion_factory = tag_suggestion_factory
        self._tag_job_store_factory = tag_job_store_factory
        self._dataset_manifest_factory = dataset_manifest_factory
        self._tag_editor_service: Any = None
        self._tag_analysis_service: Any = None
        self._tag_suggestion_service: Any = None
        self._tag_job_store: Any = None
        self._dataset_manifest_service: Any = None

    def tag_editor_service(self) -> Any:
        if self._tag_editor_service is None:
            self._tag_editor_service = self._tag_editor_factory()
        return self._tag_editor_service

    def tag_analysis_service(self) -> Any:
        if self._tag_analysis_service is None:
            self._tag_analysis_service = self._tag_analysis_factory(self.tag_editor_service())
        return self._tag_analysis_service

    def tag_suggestion_service(self) -> Any:
        if self._tag_suggestion_service is None:
            self._tag_suggestion_service = self._tag_suggestion_factory(self.tag_analysis_service())
        return self._tag_suggestion_service

    def tag_job_store(self) -> Any:
        if self._tag_job_store is None:
            self._tag_job_store = self._tag_job_store_factory()
        return self._tag_job_store

    def dataset_manifest_service(self) -> Any:
        if self._dataset_manifest_service is None:
            self._dataset_manifest_service = self._dataset_manifest_factory(
                self.tag_editor_service(),
                self.tag_analysis_service(),
            )
        return self._dataset_manifest_service

    def invalidate_tag_cache(self, dataset_path: str) -> None:
        if str(dataset_path or "").strip():
            self.tag_job_store().invalidate_dataset(str(dataset_path))

    def is_cached_result_current(
        self,
        envelope: Optional[Dict[str, Any]],
        *,
        dataset_path: str,
        caption_extension: str = "",
        recursive: bool = True,
    ) -> bool:
        if not envelope:
            return False
        try:
            payload = dict(envelope.get("payload", {}) or {})
            expected = str(payload.get("dataset_signature", "") or "")
            if not expected:
                return False
            current = self.tag_analysis_service().compute_dataset_signature(
                dataset_path,
                recursive=recursive,
                caption_extension=caption_extension,
            )
            return current == expected
        except Exception:
            return False


_DEFAULT_LOCATOR = TageditorServiceLocator()


def default_tageditor_service_locator() -> TageditorServiceLocator:
    return _DEFAULT_LOCATOR


def tag_editor_service() -> Any:
    return _DEFAULT_LOCATOR.tag_editor_service()


def tag_analysis_service() -> Any:
    return _DEFAULT_LOCATOR.tag_analysis_service()


def tag_suggestion_service() -> Any:
    return _DEFAULT_LOCATOR.tag_suggestion_service()


def tag_job_store() -> Any:
    return _DEFAULT_LOCATOR.tag_job_store()


def dataset_manifest_service() -> Any:
    return _DEFAULT_LOCATOR.dataset_manifest_service()


def invalidate_tag_cache(dataset_path: str) -> None:
    _DEFAULT_LOCATOR.invalidate_tag_cache(dataset_path)


def is_cached_result_current(
    envelope: Optional[Dict[str, Any]],
    *,
    dataset_path: str,
    caption_extension: str = "",
    recursive: bool = True,
) -> bool:
    return _DEFAULT_LOCATOR.is_cached_result_current(
        envelope,
        dataset_path=dataset_path,
        caption_extension=caption_extension,
        recursive=recursive,
    )
