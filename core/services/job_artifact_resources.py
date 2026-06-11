"""Expose job artifact manifests as resource-center compatible items."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def list_job_artifact_resources(job_manager: Any, *, limit: int = 120) -> list[dict[str, Any]]:
    """Return recent job artifacts as lightweight resource items.

    Resource Center primarily discovers files by scanning local roots. Plugin SDK
    runners and request-native jobs can already return `ArtifactManifest`
    objects before a later scanner refresh sees them, so this adapter exposes
    those manifests through the same item shape without inventing a second UI
    path.
    """

    if job_manager is None or not hasattr(job_manager, "get_all_jobs"):
        return []
    try:
        jobs = list(job_manager.get_all_jobs())
    except Exception:
        return []
    jobs.sort(key=lambda job: getattr(job, "finished_at", None) or getattr(job, "started_at", None) or getattr(job, "created_at", None) or datetime.min, reverse=True)

    items: list[dict[str, Any]] = []
    for job in jobs:
        metadata = getattr(job, "metadata", {}) or {}
        for index, artifact in enumerate(_artifact_manifests_from_metadata(metadata)):
            item = _artifact_to_resource_item(job, artifact, index)
            if item:
                items.append(item)
                if len(items) >= limit:
                    return items
    return items


def _artifact_manifests_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, list):
        return [dict(item) for item in artifacts if isinstance(item, dict)]
    run_result = metadata.get("run_result")
    if isinstance(run_result, dict) and isinstance(run_result.get("artifacts"), list):
        return [dict(item) for item in run_result.get("artifacts") if isinstance(item, dict)]
    return []


def _artifact_to_resource_item(job: Any, artifact: dict[str, Any], index: int) -> dict[str, Any] | None:
    files = [dict(item) for item in artifact.get("files") or [] if isinstance(item, dict)]
    primary = _primary_file(files)
    path = str(primary.get("path") or artifact.get("artifact_id") or "").strip()
    if not path:
        return None

    artifact_kind = str(artifact.get("artifact_kind") or "artifact").strip() or "artifact"
    name = Path(path).name if primary.get("path") else str(artifact.get("artifact_id") or f"artifact-{index + 1}")
    size = _coerce_int(primary.get("size_bytes"))
    created_at = str(artifact.get("created_at") or _job_time(job) or "")
    manifest = _compact_artifact_manifest(artifact, files)
    tags = sorted(
        {
            item
            for item in [
                "job-artifact",
                artifact_kind,
                str(artifact.get("schema_id") or ""),
                str(artifact.get("producer") or ""),
                str(primary.get("media_type") or ""),
            ]
            if item
        }
    )
    return {
        "name": name,
        "path": path,
        "relative_path": name,
        "root": "job-artifacts",
        "category": _artifact_category(artifact_kind),
        "kind": _artifact_kind_to_file_kind(artifact_kind, primary),
        "model_type": str((artifact.get("metadata") or {}).get("model_type") or ""),
        "artifact_kind": artifact_kind,
        "model_family": str((artifact.get("metadata") or {}).get("model_family") or ""),
        "detection_source": "job_artifact_manifest",
        "sha256": str(primary.get("sha256") or ""),
        "hash_status": "complete" if primary.get("sha256") else "not_applicable",
        "manifest": manifest,
        "tags": tags,
        "size": size,
        "modified_at": created_at,
        "job_id": str(getattr(job, "id", "") or ""),
        "source": "job_artifact",
    }


def _primary_file(files: list[dict[str, Any]]) -> dict[str, Any]:
    for item in files:
        if str(item.get("role") or "").lower() == "output":
            return item
    return files[0] if files else {}


def _compact_artifact_manifest(artifact: dict[str, Any], files: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "artifact_id": artifact.get("artifact_id"),
        "artifact_kind": artifact.get("artifact_kind"),
        "schema_id": artifact.get("schema_id"),
        "producer": artifact.get("producer"),
        "producer_version": artifact.get("producer_version"),
        "request_id": artifact.get("request_id"),
        "created_at": artifact.get("created_at"),
        "file_count": len(files),
    }
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    validation = artifact.get("validation") if isinstance(artifact.get("validation"), dict) else {}
    if metadata:
        manifest.update(metadata)
    if validation:
        manifest["validation"] = validation
    return {str(key): value for key, value in manifest.items() if value not in (None, "", [])}


def _artifact_category(artifact_kind: str) -> str:
    text = artifact_kind.lower()
    if "dataset" in text or "training" in text or "report" in text:
        return "training"
    if "translation" in text:
        return "translation"
    if "llm" in text:
        return "llm"
    return "models"


def _artifact_kind_to_file_kind(artifact_kind: str, primary_file: dict[str, Any]) -> str:
    path = str(primary_file.get("path") or "")
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix.lstrip(".")
    if "report" in artifact_kind.lower():
        return "report"
    return "artifact"


def _job_time(job: Any) -> str:
    value = getattr(job, "finished_at", None) or getattr(job, "started_at", None) or getattr(job, "created_at", None)
    if hasattr(value, "isoformat"):
        return value.isoformat(timespec="seconds")
    return str(value or "")


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0
