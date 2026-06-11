"""Thin adapter helpers for tag-editor analysis routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_tageditor_analysis_request(params: dict[str, Any]) -> dict[str, Any]:
    directory = str(params.get("dir", "") or params.get("path", "") or "")
    if not directory:
        raise ValueError("Missing dir parameter")
    return {
        "dataset_path": directory,
        "route_family": str(params.get("route_family", "") or ""),
        "recursive": bool(params.get("recursive", True)),
        "caption_extension": str(params.get("caption_extension", "") or ""),
        "load_caption_from_filename": bool(params.get("load_caption_from_filename", False)),
        "filename_regex": str(params.get("filename_regex", "") or ""),
        "filename_joiner": str(params.get("filename_joiner", ", ") or ", "),
        "max_token_count": int(params.get("max_token_count", 75) or 75),
        "trigger_words": list(params.get("trigger_words", []) or []),
        "submitted_config": dict(params),
    }


def preview_tageditor_analysis(params: dict[str, Any], *, analysis_service: Any) -> dict[str, Any]:
    request = build_tageditor_analysis_request(params)
    return analysis_service.analyze_dataset(
        request["dataset_path"],
        route_family=request["route_family"],
        recursive=request["recursive"],
        caption_extension=request["caption_extension"],
        load_caption_from_filename=request["load_caption_from_filename"],
        filename_regex=request["filename_regex"],
        filename_joiner=request["filename_joiner"],
        max_token_count=request["max_token_count"],
        trigger_words=request["trigger_words"],
    )


def submit_tageditor_analysis_job(
    params: dict[str, Any],
    *,
    analysis_service: Any,
    job_store: Any,
    job_manager: Any,
) -> str:
    from backend.core.job_manager import Job, JobType

    request = build_tageditor_analysis_request(params)
    dataset_path = request["dataset_path"]
    route_family = request["route_family"]
    config = dict(request["submitted_config"])
    job = Job(
        type=JobType.TAG_ANALYSIS,
        name=f"Tag Analysis: {Path(dataset_path).name}",
        metadata={
            "dataset_path": dataset_path,
            "route_family": route_family,
            "config": config,
        },
    )

    def _worker(progress_callback=None, cancel_check=None):
        progress_callback = progress_callback or (lambda *_args, **_kwargs: None)
        progress_callback(1, 4)
        if cancel_check and cancel_check():
            return {"cancelled": True}
        report = preview_tageditor_analysis(config, analysis_service=analysis_service)
        progress_callback(3, 4)
        envelope = job_store.save_result(
            kind="analysis",
            job_id=job.id,
            dataset_path=dataset_path,
            route_family=route_family or report.get("route_family", "generic"),
            submitted_config=config,
            payload=report,
            started_at=job.started_at.isoformat() if job.started_at else None,
        )
        job.metadata["result_kind"] = "analysis"
        job.metadata["result_path"] = envelope.get("dataset_key", "")
        progress_callback(4, 4)
        return envelope

    return job_manager.submit(job, _worker)
