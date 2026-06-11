"""Thin adapter helpers for tag-editor batch mutation routes."""

from __future__ import annotations

from typing import Any, Callable


def build_tageditor_save_batch_request(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(params.get("dir", "") or params.get("path", "") or ""),
        "updates": list(params.get("updates", []) or []),
        "caption_extension": str(params.get("caption_extension", "") or ""),
        "create_backup": bool(params.get("create_backup", False)),
        "backup_label": str(params.get("backup_label", "") or "save_batch"),
    }


def build_tageditor_batch_action_request(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(params.get("dir", "") or params.get("path", "") or ""),
        "action": str(params.get("action", "") or ""),
        "params": dict(params.get("params", {}) or {}),
        "image_paths": list(params.get("image_paths", []) or []),
        "recursive": bool(params.get("recursive", True)),
        "caption_extension": str(params.get("caption_extension", "") or ""),
        "create_backup": bool(params.get("create_backup", False)),
        "filter_payload": dict(params.get("filters", {}) or {}),
        "route_family": str(params.get("route_family", "") or "generic"),
        "submitted_config": dict(params),
    }


def save_tageditor_batch_captions(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    request = build_tageditor_save_batch_request(params)
    if not request["updates"]:
        return {"success": 0, "failed": 0, "errors": []}
    result = tag_editor.save_captions_batch(
        request["updates"],
        caption_extension=request["caption_extension"],
        dataset_dir=request["dataset_path"],
        create_backup=request["create_backup"],
        backup_label=request["backup_label"],
    )
    if invalidate_cache and request["dataset_path"]:
        invalidate_cache(request["dataset_path"])
    return result


def preview_tageditor_batch_action(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    request = build_tageditor_batch_action_request(params)
    if not request["dataset_path"] or not request["action"]:
        raise ValueError("Missing dir or action")
    return tag_editor.preview_batch_action(
        request["dataset_path"],
        action=request["action"],
        params=request["params"],
        image_paths=request["image_paths"],
        recursive=request["recursive"],
        caption_extension=request["caption_extension"],
        filter_payload=request["filter_payload"],
    )


def apply_tageditor_batch_action(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    request = build_tageditor_batch_action_request(params)
    if not request["dataset_path"] or not request["action"]:
        raise ValueError("Missing dir or action")
    result = tag_editor.apply_batch_action(
        request["dataset_path"],
        action=request["action"],
        params=request["params"],
        image_paths=request["image_paths"],
        recursive=request["recursive"],
        caption_extension=request["caption_extension"],
        create_backup=request["create_backup"],
        filter_payload=request["filter_payload"],
    )
    if invalidate_cache:
        invalidate_cache(request["dataset_path"])
    return result


def submit_tageditor_batch_action_job(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    job_store: Any,
    job_manager: Any,
    invalidate_cache: Callable[[str], None] | None = None,
) -> str:
    from backend.core.job_manager import Job, JobType

    request = build_tageditor_batch_action_request(params)
    if not request["dataset_path"] or not request["action"]:
        raise ValueError("Missing dir or action")
    preview = preview_tageditor_batch_action(request["submitted_config"], tag_editor=tag_editor)
    job = Job(
        type=JobType.TAG_BATCH_EDIT,
        name=f"Tag Batch Edit: {request['action']}",
        metadata={
            "dataset_path": request["dataset_path"],
            "action": request["action"],
            "config": request["submitted_config"],
            "preview": preview,
        },
    )

    def _worker(progress_callback=None, cancel_check=None):
        progress_callback = progress_callback or (lambda *_args, **_kwargs: None)
        progress_callback(1, 2)
        if cancel_check and cancel_check():
            return {"cancelled": True}
        result = apply_tageditor_batch_action(
            request["submitted_config"],
            tag_editor=tag_editor,
            invalidate_cache=invalidate_cache,
        )
        envelope = job_store.save_result(
            kind="batch_action",
            job_id=job.id,
            dataset_path=request["dataset_path"],
            route_family=request["route_family"],
            submitted_config=request["submitted_config"],
            payload=result,
            started_at=job.started_at.isoformat() if job.started_at else None,
        )
        job.metadata["result_kind"] = "batch_action"
        job.metadata["result_path"] = envelope.get("dataset_key", "")
        progress_callback(2, 2)
        return envelope

    return job_manager.submit(job, _worker)
