"""Thin adapter helpers for tag-editor interrogate and retag routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.services.caption_cleanup_preview import join_tags, split_tags


def build_tageditor_interrogate_request(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": str(params.get("image_path", "") or params.get("path", "") or ""),
        "method": str(params.get("method", "wd14") or "wd14"),
        "config": dict(params.get("config", {}) or {}),
        "submitted_config": dict(params),
    }


def build_tageditor_retag_request(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(params.get("dir", "") or params.get("path", "") or ""),
        "method": str(params.get("method", "wd14") or "wd14"),
        "config": dict(params.get("config", {}) or {}),
        "recursive": bool(params.get("recursive", True)),
        "caption_extension": str(params.get("caption_extension", "") or ""),
        "conflict": _normalize_retag_conflict(params.get("conflict", "ignore")),
        "route_family": str(params.get("route_family", "") or "generic"),
        "submitted_config": dict(params),
    }


def interrogate_tageditor_image(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    request = build_tageditor_interrogate_request(params)
    if not request["image_path"]:
        raise ValueError("Missing image_path")
    return tag_editor.interrogate_image(
        image_path=request["image_path"],
        method=request["method"],
        config=request["config"],
    )


def _merge_retag_caption(old_caption: str, new_caption: str, *, conflict: str) -> str:
    old_caption = str(old_caption or "").strip()
    new_caption = str(new_caption or "").strip()
    conflict = _normalize_retag_conflict(conflict)
    if old_caption and conflict == "ignore":
        return old_caption
    if old_caption and conflict == "append":
        return join_tags(split_tags(old_caption) + split_tags(new_caption))
    if old_caption and conflict == "prepend":
        return join_tags(split_tags(new_caption) + split_tags(old_caption))
    return new_caption


def _normalize_retag_conflict(value: Any) -> str:
    conflict = str(value or "ignore").strip().lower()
    if conflict in {"copy", "replace", "overwrite"}:
        return "replace"
    if conflict in {"ignore", "append", "prepend"}:
        return conflict
    return "ignore"


def submit_tageditor_retag_job(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    job_store: Any,
    job_manager: Any,
    invalidate_cache: Callable[[str], None] | None = None,
) -> str:
    from backend.core.job_manager import Job, JobType

    request = build_tageditor_retag_request(params)
    if not request["dataset_path"]:
        raise ValueError("Missing dir parameter")
    dataset_path = request["dataset_path"]
    job = Job(
        type=JobType.TAG_RETAG,
        name=f"Tag Retag: {Path(dataset_path).name}",
        metadata={"dataset_path": dataset_path, "config": request["submitted_config"]},
    )

    def _worker(progress_callback=None, cancel_check=None):
        progress_callback = progress_callback or (lambda *_args, **_kwargs: None)
        dataset_dir = Path(dataset_path).resolve()
        items = tag_editor._scan_dataset(
            dataset_dir,
            recursive=request["recursive"],
            caption_extension=request["caption_extension"],
            load_caption_from_filename=False,
        )
        total = max(1, len(items))
        results = []
        for idx, item in enumerate(items, start=1):
            if cancel_check and cancel_check():
                break
            interrogated = tag_editor.interrogate_image(
                image_path=str(item.image_path),
                method=request["method"],
                config=request["config"],
            )
            final_caption = _merge_retag_caption(
                item.caption_text,
                str(interrogated.get("caption", "") or ""),
                conflict=request["conflict"],
            )
            item.caption_path.parent.mkdir(parents=True, exist_ok=True)
            item.caption_path.write_text(final_caption, encoding="utf-8")
            results.append({"image_path": str(item.image_path), "caption": final_caption})
            progress_callback(idx, total)
        if invalidate_cache:
            invalidate_cache(dataset_path)
        envelope = job_store.save_result(
            kind="retag",
            job_id=job.id,
            dataset_path=dataset_path,
            route_family=request["route_family"],
            submitted_config=request["submitted_config"],
            payload={"updated_count": len(results), "samples": results[:20]},
            started_at=job.started_at.isoformat() if job.started_at else None,
        )
        job.metadata["result_kind"] = "retag"
        job.metadata["result_path"] = envelope.get("dataset_key", "")
        return envelope

    return job_manager.submit(job, _worker)
