"""Lightweight async job manager for tag tooling."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Callable

from core.services.tag_job_store import TagJobStore
from core.services.tag_analysis_service import TagAnalysisService


# Global in-memory job registry
_JOBS: Dict[str, Dict[str, Any]] = {}
_TAG_JOB_TYPES = {"tag_lint", "tag_normalize", "tag_batch_edit", "tag_retag"}


class TagAsyncJobManager:
    """Lightweight async job manager for heavy tag-editor operations."""

    @classmethod
    def submit(cls, *, kind: str, dataset_path: str, route_family: str = "", **params) -> str:
        """Submit an async job and return its job_id."""
        if kind not in _TAG_JOB_TYPES:
            raise ValueError(f"Unsupported job type: {kind}. Supported: {_TAG_JOB_TYPES}")
        job_id = str(uuid.uuid4())
        record: Dict[str, Any] = {
            "job_id": job_id,
            "kind": kind,
            "dataset_path": dataset_path,
            "route_family": route_family or "generic",
            "params": {k: v for k, v in params.items()},
            "status": "pending",
            "result": None,
            "error": None,
        }
        _JOBS[job_id] = record
        return job_id

    @classmethod
    def get_status(cls, job_id: str) -> Dict[str, Any]:
        """Get the status of a submitted job."""
        if job_id not in _JOBS:
            return {"status": "unknown"}
        return dict(_JOBS[job_id])

    @classmethod
    def list_jobs(cls, *, dataset_path: str = "", kind: str = ""):
        """List jobs, optionally filtered by dataset_path and/or kind."""
        results: list[Dict[str, Any]] = []
        for job_id, record in _JOBS.items():
            if dataset_path and record.get("dataset_path") != dataset_path:
                continue
            if kind and record.get("kind") != kind:
                continue
            results.append({
                "job_id": job_id,
                "kind": record.get("kind"),
                "dataset_path": record.get("dataset_path"),
                "status": record.get("status"),
                "route_family": record.get("route_family"),
            })
        return results

    @classmethod
    def run_job_synchronously(cls, *, job_id: str, task_fn: Callable[[], Any]):
        """Run the job synchronously."""
        if job_id not in _JOBS:
            raise ValueError(f"Job {job_id} not found")
        _JOBS[job_id]["status"] = "running"
        try:
            result = task_fn()
            _JOBS[job_id]["status"] = "completed"
            _JOBS[job_id]["result"] = result
            return result
        except Exception as e:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = str(e)
            raise

    @classmethod
    def persist_job_result(cls, *, job_id: str, store: Optional[TagJobStore] = None) -> str:
        """Persist the result of a completed job to the TagJobStore."""
        if job_id not in _JOBS:
            raise ValueError(f"Job {job_id} not found")
        record = _JOBS[job_id]
        if record.get("status") != "completed":
            raise ValueError(f"Job {job_id} is not completed, status: {record.get('status')}")
        if store is None:
            store = TagJobStore()
        store.save_result(
            kind=record["kind"],
            job_id=job_id,
            dataset_path=record["dataset_path"],
            route_family=record["route_family"],
            payload=record["result"],
            submitted_config=record.get("params", {}),
        )
        return job_id

    @classmethod
    def cancel_job(cls, job_id: str):
        """Cancel a pending job."""
        if job_id not in _JOBS:
            raise ValueError(f"Job {job_id} not found")
        if _JOBS[job_id]["status"] == "pending":
            _JOBS[job_id]["status"] = "cancelled"
        return _JOBS[job_id]


def create_tag_lint_job(
    dataset_path: str,
    *,
    route_family: str = "",
    recursive: bool = True,
    max_token_count: int = 75,
    **kwargs,
) -> str:
    """Convenience: create and submit a tag_lint job, return job_id."""
    job_id = TagAsyncJobManager.submit(
        kind="tag_lint",
        dataset_path=dataset_path,
        route_family=route_family,
        recursive=recursive,
        max_token_count=max_token_count,
        **kwargs,
    )
    return job_id


def create_tag_normalize_job(
    dataset_path: str,
    *,
    route_family: str = "",
    **kwargs,
) -> str:
    """Convenience: create and submit a tag_normalize job, return job_id."""
    return TagAsyncJobManager.submit(
        kind="tag_normalize",
        dataset_path=dataset_path,
        route_family=route_family,
        **kwargs,
    )


def execute_tag_lint_job(
    job_id: str,
    *,
    max_token_count: int = 75,
    store_result: bool = True,
) -> Dict[str, Any]:
    """Run a tag_lint job synchronously, persist to store if requested, return the report."""
    record = TagAsyncJobManager.get_status(job_id)
    if record.get("status") == "unknown":
        raise ValueError(f"Job {job_id} not found")

    dataset_path = record["dataset_path"]
    route_family = record["route_family"]
    params = record.get("params", {})

    def _task():
        svc = TagAnalysisService()
        return svc.lint_dataset_lightweight(
            dataset_path,
            route_family=route_family,
            max_token_count=max_token_count,
            **params,
        )

    result = TagAsyncJobManager.run_job_synchronously(job_id=job_id, task_fn=_task)
    if store_result:
        TagAsyncJobManager.persist_job_result(job_id=job_id)
    return result


def execute_tag_normalize_job(
    job_id: str,
    *,
    store_result: bool = True,
) -> Dict[str, Any]:
    """Run a tag_normalize job synchronously, persist to store if requested."""
    record = Taghetti_status(job_id)
    if record.get("status") == "unknown":
        raise ValueError(f"Job {job_id} not found")

    dataset_path = record["dataset_path"]
    params = record.get("params", {})

    def _task():
        # TODO: implement actual normalize logic
        return {"status": "ok", "dataset_path": dataset_path, "params": params}

    result = TagAsyncJobManager.run_job_synchronously(job_id=job_id, task_fn=_task)
    if store_result:
        TagAsyncJobManager.persist_job_result(job_id=job_id)
    return result
