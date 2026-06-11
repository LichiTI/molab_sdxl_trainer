"""Route adapter helpers for tag-editor persisted result envelopes."""

from __future__ import annotations

from typing import Any, Callable


def load_tageditor_analysis_result(
    params: dict[str, Any],
    *,
    job_store: Any,
    is_current: Callable[..., bool],
) -> dict[str, Any]:
    """Load an analysis result by job id or latest dataset cache."""

    dataset_path = str(params.get("dir", "") or params.get("path", "") or "")
    job_id = str(params.get("job_id", "") or "")
    caption_extension = str(params.get("caption_extension", "") or "")
    if job_id:
        envelope = job_store.load_job_result(job_id)
        if not envelope:
            raise LookupError("Result not found")
        return envelope
    if dataset_path:
        envelope = job_store.load_latest(kind="analysis", dataset_path=dataset_path)
        if not envelope:
            return {"status": "missing"}
        status = "ready" if is_current(envelope, dataset_path=dataset_path, caption_extension=caption_extension) else "stale"
        return {"status": status, **envelope}
    raise ValueError("Missing job_id or dir parameter")


def list_tageditor_results(params: dict[str, Any], *, job_store: Any) -> dict[str, Any]:
    dataset_path = str(params.get("dir", "") or params.get("path", "") or "")
    if not dataset_path:
        raise ValueError("Missing dir parameter")
    results = job_store.list_dataset_results(
        dataset_path=dataset_path,
        kind=str(params.get("kind", "") or ""),
    )
    return {"results": results}


def load_tageditor_job_result(params: dict[str, Any], *, job_store: Any) -> dict[str, Any]:
    job_id = str(params.get("job_id", "") or "")
    if not job_id:
        raise ValueError("Missing job_id")
    envelope = job_store.load_job_result(job_id)
    if not envelope:
        raise LookupError("Result not found")
    return envelope
