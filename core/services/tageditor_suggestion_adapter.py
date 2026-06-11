"""Thin adapter helpers for tag-editor suggestion routes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def build_tageditor_suggestion_request(params: dict[str, Any]) -> dict[str, Any]:
    directory = str(params.get("dir", "") or params.get("path", "") or "")
    if not directory:
        raise ValueError("Missing dir parameter")
    return {
        "dataset_path": directory,
        "route_family": str(params.get("route_family", "") or ""),
        "selected_image_paths": list(params.get("image_paths", []) or []),
        "recursive": bool(params.get("recursive", True)),
        "caption_extension": str(params.get("caption_extension", "") or ""),
        "submitted_config": dict(params),
    }


def preview_tageditor_suggestions(
    params: dict[str, Any],
    *,
    suggestion_service: Any,
    job_store: Any,
    is_current: Any,
) -> dict[str, Any]:
    request = build_tageditor_suggestion_request(params)
    latest_analysis = job_store.load_latest(kind="analysis", dataset_path=request["dataset_path"])
    if not latest_analysis or not is_current(
        latest_analysis,
        dataset_path=request["dataset_path"],
        caption_extension=request["caption_extension"],
    ):
        return {"status": "needs_refresh", "latest_analysis": latest_analysis}
    suggestion_report = suggestion_service.build_suggestions(
        request["dataset_path"],
        route_family=request["route_family"],
        selected_image_paths=request["selected_image_paths"],
        analysis_report=dict(latest_analysis.get("payload", {}) or {}),
        recursive=request["recursive"],
        caption_extension=request["caption_extension"],
    )
    job_store.save_result(
        kind="suggestions",
        job_id=f"sync_{int(time.time())}",
        dataset_path=request["dataset_path"],
        route_family=str(request["route_family"] or suggestion_report.get("route_family", "generic")),
        submitted_config=request["submitted_config"],
        payload=suggestion_report,
    )
    return suggestion_report


async def refine_tageditor_suggestions_with_llm(
    params: dict[str, Any],
    *,
    suggestion_service: Any,
    job_store: Any,
) -> dict[str, Any]:
    request = build_tageditor_suggestion_request(params)
    latest_analysis = job_store.load_latest(kind="analysis", dataset_path=request["dataset_path"])
    analysis_report = dict(latest_analysis.get("payload", {}) or {}) if latest_analysis else None
    return await suggestion_service.llm_refine(
        directory=request["dataset_path"],
        route_family=request["route_family"],
        selected_image_paths=request["selected_image_paths"],
        analysis_report=analysis_report,
        provider=str(params.get("provider", "openai") or "openai"),
        api_key=str(params.get("api_key", "") or ""),
        model=str(params.get("model", "") or ""),
        base_url=str(params.get("base_url", "") or ""),
        prompt=str(params.get("prompt", "") or ""),
    )


def submit_tageditor_suggestions_refresh_job(
    params: dict[str, Any],
    *,
    analysis_service: Any,
    suggestion_service: Any,
    job_store: Any,
    job_manager: Any,
) -> str:
    from backend.core.job_manager import Job, JobType

    request = build_tageditor_suggestion_request(params)
    directory = request["dataset_path"]
    config = dict(request["submitted_config"])
    job = Job(
        type=JobType.TAG_SUGGESTIONS_REFRESH,
        name=f"Tag Suggestions Refresh: {Path(directory).name}",
        metadata={"dataset_path": directory, "config": config},
    )

    def _worker(progress_callback=None, cancel_check=None):
        progress_callback = progress_callback or (lambda *_args, **_kwargs: None)
        progress_callback(1, 3)
        analysis_report = analysis_service.analyze_dataset(
            directory,
            route_family=request["route_family"],
            recursive=request["recursive"],
            caption_extension=request["caption_extension"],
        )
        job_store.save_result(
            kind="analysis",
            job_id=f"{job.id}_analysis",
            dataset_path=directory,
            route_family=str(request["route_family"] or analysis_report.get("route_family", "generic")),
            submitted_config=config,
            payload=analysis_report,
            started_at=job.started_at.isoformat() if job.started_at else None,
        )
        progress_callback(2, 3)
        suggestion_report = suggestion_service.build_suggestions(
            directory,
            route_family=request["route_family"],
            selected_image_paths=request["selected_image_paths"],
            analysis_report=analysis_report,
            recursive=request["recursive"],
            caption_extension=request["caption_extension"],
        )
        envelope = job_store.save_result(
            kind="suggestions",
            job_id=job.id,
            dataset_path=directory,
            route_family=str(request["route_family"] or suggestion_report.get("route_family", "generic")),
            submitted_config=config,
            payload=suggestion_report,
            started_at=job.started_at.isoformat() if job.started_at else None,
        )
        progress_callback(3, 3)
        return envelope

    return job_manager.submit(job, _worker)
