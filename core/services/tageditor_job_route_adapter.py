"""Thin job-route wrappers for tag-editor async compat endpoints."""

from __future__ import annotations

from typing import Any, Callable


JobManagerFactory = Callable[[], Any | None]
Submitter = Callable[..., str]


def _default_job_manager_factory() -> JobManagerFactory:
    from backend.core.locator import Locator

    return Locator.get_jobs


def submit_tageditor_job_route_payload(
    params: dict[str, Any],
    *,
    submitter: Submitter,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    **submit_kwargs: Any,
) -> dict[str, str]:
    """Resolve JobManager for a tag-editor async route and return legacy payload."""

    manager = job_manager
    if manager is None:
        manager = (job_manager_factory or _default_job_manager_factory())()
    if manager is None:
        raise RuntimeError("Job manager unavailable")
    job_id = submitter(
        params,
        job_manager=manager,
        **submit_kwargs,
    )
    return {"job_id": str(job_id)}
