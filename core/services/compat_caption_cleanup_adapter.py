"""Compatibility helpers for caption-cleanup and tag-manager routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import PreprocessRequest, RequestSource

from backend.core.services.caption_cleanup_preview import (
    apply_cleanup_to_dataset,
    collect_cleanup_preview,
    normalize_cleanup_params,
    submit_caption_cleanup_job,
)
from backend.core.services.preprocess_artifacts import attach_preprocess_artifacts
from backend.core.services.tageditor_service_locator import invalidate_tag_cache, tag_job_store


JobManagerFactory = Callable[[], Any | None]


def normalize_caption_cleanup_request(params: dict[str, Any], *, action: str = "clean-captions") -> PreprocessRequest:
    """Build request-native data for legacy caption cleanup/tag-manager routes."""

    data = dict(params or {})
    dataset_path = data.get("dataset_path") or data.get("path") or data.get("dir") or ""
    options = dict(data.get("options") or {}) if isinstance(data.get("options"), dict) else {}
    for key, value in data.items():
        if key not in {
            "action",
            "input_path",
            "output_path",
            "dataset_path",
            "path",
            "dir",
            "caption_extension",
            "recursive",
            "dry_run",
            "options",
            "metadata",
            "schema_id",
            "schema_version",
            "compat_mode",
        }:
            options.setdefault(key, value)
    request_payload = {
        **data,
        "schema_id": data.get("schema_id") or "preprocess.caption-cleanup",
        "action": data.get("action") or action,
        "dataset_path": str(dataset_path or ""),
        "input_path": str(data.get("input_path") or dataset_path or ""),
        "caption_extension": data.get("caption_extension") or ".txt",
        "recursive": data.get("recursive", True),
        "dry_run": data.get("dry_run", True),
        "options": options,
    }
    return PreprocessRequest.from_legacy_payload(request_payload, source=RequestSource.WEBUI)


def _resolve_cleanup_directory(params: dict[str, Any], *, action: str = "clean-captions") -> Path:
    request = normalize_caption_cleanup_request(params, action=action)
    dir_path = request.primary_input()
    if not dir_path:
        raise ValueError("Missing path parameter")
    dataset_dir = Path(dir_path)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return dataset_dir


def _cleanup_mode_params(params: dict[str, Any], *, for_tag_manager: bool) -> dict[str, Any]:
    return normalize_cleanup_params(params, for_tag_manager=for_tag_manager)


def preview_caption_cleanup_payload(
    params: dict[str, Any],
    *,
    for_tag_manager: bool = False,
    include_stats: bool = False,
    attach_artifacts: Callable[[PreprocessRequest, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build preview payload for caption cleanup or tag-manager lite routes."""

    cleanup_request = normalize_caption_cleanup_request(params, action="clean-captions")
    dataset_dir = _resolve_cleanup_directory(params, action="clean-captions")
    cleanup = _cleanup_mode_params(params, for_tag_manager=for_tag_manager)
    preview = collect_cleanup_preview(dataset_dir, cleanup, include_stats=include_stats)
    if attach_artifacts is not None:
        return attach_artifacts(cleanup_request, preview)
    return preview


def apply_caption_cleanup_payload(
    params: dict[str, Any],
    *,
    for_tag_manager: bool = False,
    invalidate_cache: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Apply cleanup synchronously for caption cleanup or tag-manager lite routes."""

    dataset_dir = _resolve_cleanup_directory(params, action="clean-captions")
    cleanup = _cleanup_mode_params(params, for_tag_manager=for_tag_manager)
    return apply_cleanup_to_dataset(
        dataset_dir,
        cleanup,
        invalidate_cache=invalidate_cache,
    )


def _default_job_manager_factory() -> JobManagerFactory:
    from backend.core.locator import Locator

    return Locator.get_jobs


def start_caption_cleanup_job_payload(
    params: dict[str, Any],
    *,
    for_tag_manager: bool = False,
    include_stats: bool = False,
    job_kind: str,
    job_name_prefix: str,
    job_store: Any,
    route_family_default: str,
    invalidate_cache: Callable[[str], None] | None = None,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
) -> dict[str, Any]:
    """Submit async cleanup job and return legacy-compatible payload."""

    dataset_dir = _resolve_cleanup_directory(params, action="clean-captions")
    cleanup = _cleanup_mode_params(params, for_tag_manager=for_tag_manager)
    preview = collect_cleanup_preview(dataset_dir, cleanup, include_stats=include_stats)
    manager = job_manager
    if manager is None:
        manager = (job_manager_factory or _default_job_manager_factory())()
    if manager is None:
        raise RuntimeError("Job manager unavailable")

    job_id = submit_caption_cleanup_job(
        params=params,
        dataset_dir=dataset_dir,
        cleanup=cleanup,
        preview=preview,
        job_manager=manager,
        job_store=job_store,
        kind=job_kind,
        job_name=f"{job_name_prefix}: {dataset_dir.name}",
        route_family_default=route_family_default,
        invalidate_cache=invalidate_cache,
    )
    return {"job_id": job_id, "preview": preview}


def preview_caption_cleanup_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return preview_caption_cleanup_payload(
        params,
        for_tag_manager=False,
        include_stats=False,
        attach_artifacts=attach_preprocess_artifacts,
    )


def apply_caption_cleanup_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return apply_caption_cleanup_payload(
        params,
        for_tag_manager=False,
        invalidate_cache=invalidate_tag_cache,
    )


def start_caption_cleanup_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return start_caption_cleanup_job_payload(
        params,
        for_tag_manager=False,
        include_stats=False,
        job_kind="caption_cleanup",
        job_name_prefix="Caption Cleanup",
        job_store=tag_job_store(),
        route_family_default="generic",
        invalidate_cache=invalidate_tag_cache,
    )


def preview_caption_tag_manager_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return preview_caption_cleanup_payload(
        params,
        for_tag_manager=True,
        include_stats=True,
        attach_artifacts=attach_preprocess_artifacts,
    )


def apply_caption_tag_manager_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return apply_caption_cleanup_payload(
        params,
        for_tag_manager=True,
        invalidate_cache=invalidate_tag_cache,
    )


def start_caption_tag_manager_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return start_caption_cleanup_job_payload(
        params,
        for_tag_manager=True,
        include_stats=True,
        job_kind="tag_manager_lite",
        job_name_prefix="Tag Manager Lite",
        job_store=tag_job_store(),
        route_family_default="generic",
        invalidate_cache=invalidate_tag_cache,
    )
