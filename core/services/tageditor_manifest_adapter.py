"""Route adapter helpers for tag-editor dataset manifests."""

from __future__ import annotations

import time
from typing import Any, Callable


def build_tageditor_manifest_payload(
    params: dict[str, Any],
    *,
    manifest_service: Any,
    job_store: Any | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Build or persist a dataset manifest using the legacy WebUI payload shape."""

    directory = str(params.get("dir", "") or params.get("path", "") or "")
    if not directory:
        raise ValueError("Missing dir parameter")

    manifest = manifest_service.build_manifest(
        directory,
        recursive=bool(params.get("recursive", True)),
        caption_extension=str(params.get("caption_extension", "") or ""),
        route_family=str(params.get("route_family", "") or ""),
        trigger_words=list(params.get("trigger_words", []) or []),
        max_token_count=int(params.get("max_token_count", 75) or 75),
    )
    if bool(params.get("persist", False)):
        if job_store is None:
            raise ValueError("Job store is required to persist manifest results")
        return job_store.save_result(
            kind="manifest",
            job_id=f"manifest_{int(now())}",
            dataset_path=directory,
            route_family=str(params.get("route_family", "") or manifest.get("route_family", "generic")),
            submitted_config=dict(params),
            payload=manifest,
        )
    return manifest


def build_tageditor_manifest_diff_payload(
    params: dict[str, Any],
    *,
    manifest_service: Any,
    job_store: Any | None = None,
) -> dict[str, Any]:
    """Resolve old/new manifests from payload or job ids, then diff them."""

    old_manifest = dict(params.get("old_manifest", {}) or {})
    new_manifest = dict(params.get("new_manifest", {}) or {})
    if not old_manifest or not new_manifest:
        old_job_id = str(params.get("old_job_id", "") or "")
        new_job_id = str(params.get("new_job_id", "") or "")
        old_envelope = job_store.load_job_result(old_job_id) if job_store is not None and old_job_id else None
        new_envelope = job_store.load_job_result(new_job_id) if job_store is not None and new_job_id else None
        old_manifest = dict((old_envelope or {}).get("payload", {}) or old_manifest)
        new_manifest = dict((new_envelope or {}).get("payload", {}) or new_manifest)
    if not old_manifest or not new_manifest:
        raise ValueError("Missing old_manifest/new_manifest or old_job_id/new_job_id")
    return manifest_service.diff_manifests(old_manifest, new_manifest)
