"""Adapter helpers for legacy local task history routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.core.job_manager import Job, JobStatus, JobType


_FILE_PHASE_TO_OFFICIAL_STATUS = {
    "queued": "QUEUED",
    "running": "RUNNING",
    "succeeded": "COMPLETED",
    "failed": "FAILED",
    "cancelled": "CANCELLED",
}


def serialize_task_history(job_manager: Any | None) -> dict[str, Any]:
    if not job_manager:
        return {"tasks": []}
    tasks = []
    for job in job_manager.get_all_jobs():
        tasks.append(
            {
                "id": job.id,
                "name": job.name,
                "type": job.type.value if hasattr(job.type, "value") else str(job.type),
                "status": job.status.value if hasattr(job.status, "value") else str(job.status),
                "progress": job.progress,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "error": job.error,
            }
        )
    return {"tasks": tasks}


def serialize_official_tasks(job_manager: Any | None, telemetry_reader: Any | None = None) -> dict[str, Any]:
    tasks = []
    seen_ids: set[str] = set()
    if job_manager:
        for job in job_manager.get_all_jobs():
            seen_ids.add(str(job.id))
            tasks.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "status": job.status.value.upper(),
                    "progress": job.progress,
                    "type": job.type.value,
                    "error": job.error,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                    "metadata": _metadata_with_closed_multi_batch_claims(job.metadata or {}),
                }
            )
    if telemetry_reader:
        for task in _serialize_file_training_tasks(telemetry_reader):
            task_id = str(task.get("id") or "")
            if not task_id or task_id in seen_ids:
                continue
            seen_ids.add(task_id)
            tasks.append(task)
    return {"tasks": tasks}


def _serialize_file_training_tasks(telemetry_reader: Any) -> list[dict[str, Any]]:
    try:
        from backend.core.services.file_training_task_service import list_file_training_tasks

        file_tasks = list_file_training_tasks(telemetry_reader, limit=80)
        return [_serialize_file_training_task(task) for task in file_tasks]
    except Exception:
        return _serialize_active_file_runs_fallback(telemetry_reader)


def _serialize_file_training_task(task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("task_id") or task.get("id") or "")
    metadata = _metadata_with_closed_multi_batch_claims(task.get("metadata") or {})
    stages = list(task.get("stages") or [])
    status = _FILE_PHASE_TO_OFFICIAL_STATUS.get(str(task.get("phase") or "").lower(), "FAILED")
    payload = {
        "id": task_id,
        "task_id": task_id,
        "name": metadata.get("config_name") or task_id,
        "status": status,
        "progress": _progress_from_file_task_stages(stages),
        "type": "training",
        "error": task.get("error") or None,
        "created_at": metadata.get("queued_at") or task.get("started_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at") or None,
        "metadata": metadata,
        "stages": stages,
    }
    if "queue_position" in metadata:
        payload["queue_position"] = metadata["queue_position"]
    if metadata.get("queue_message"):
        payload["queue_message"] = metadata.get("queue_message")
    return payload


def _serialize_active_file_runs_fallback(telemetry_reader: Any) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for run in telemetry_reader.get_active_runs():
        run_id = run.get("run_id", "")
        if not run_id:
            continue
        tasks.append(
            {
                "id": run_id,
                "task_id": run_id,
                "name": run.get("run_id", ""),
                "status": run.get("status", "UNKNOWN").upper(),
                "progress": 0,
                "type": "training",
                "error": run.get("error"),
                "created_at": run.get("started_at"),
                "started_at": run.get("started_at"),
                "finished_at": None,
                "metadata": {},
            }
        )
    return tasks


def _progress_from_file_task_stages(stages: list[dict[str, Any]]) -> float:
    for stage in reversed(stages):
        detail = stage.get("detail") if isinstance(stage, dict) else None
        if not isinstance(detail, dict):
            continue
        current = _number_or_none(detail.get("current_step"))
        total = _number_or_none(detail.get("total_steps"))
        if current is not None and total and total > 0:
            return max(0.0, min(1.0, current / total))
    return 0.0


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _metadata_with_closed_multi_batch_claims(metadata: Any) -> dict[str, Any]:
    result = dict(metadata or {}) if isinstance(metadata, dict) else {}
    for key in (
        "multi_batch_promotion_gate",
        "multi_batch_dataloader",
        "multi_batch_stability_candidate_evidence",
    ):
        value = result.get(key)
        if isinstance(value, dict):
            value = dict(value)
            value["release_claim_allowed"] = False
            if isinstance(value.get("execution_strategy"), dict):
                value["execution_strategy"] = dict(value["execution_strategy"])
                value["execution_strategy"]["release_claim_allowed"] = False
            result[key] = value
    return result


def delete_official_task(job_manager: Any | None, task_id: str) -> bool:
    if not job_manager:
        return False
    with job_manager._lock:
        if task_id not in job_manager._jobs:
            return False
        del job_manager._jobs[task_id]
        job_manager._save_history()
        return True


def clear_completed_official_tasks(job_manager: Any | None) -> int:
    if not job_manager:
        return 0
    before = len(job_manager.get_all_jobs())
    job_manager.clear_completed()
    return max(0, before - len(job_manager.get_all_jobs()))


def terminate_official_task(job_manager: Any | None, task_id: str) -> bool:
    return bool(job_manager and job_manager.cancel(task_id))


def clear_completed_task_history(job_manager: Any | None) -> None:
    if job_manager:
        job_manager.clear_completed()


def delete_task_history_entry(job_manager: Any | None, task_id: str) -> None:
    if not job_manager:
        return
    job = job_manager.get_job(task_id)
    if not job:
        return
    if job.status == JobStatus.RUNNING:
        job_manager.cancel(task_id)
    with job_manager._lock:
        job_manager._jobs.pop(task_id, None)
    job_manager._save_history()


def _parse_datetime(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except Exception:
        return None


def parse_task_status(value: Any, *, default: JobStatus = JobStatus.COMPLETED) -> JobStatus:
    if not value:
        return default
    try:
        return JobStatus(str(value).lower())
    except ValueError:
        return default


def save_task_history_payload(job_manager: Any | None, body: dict[str, Any]) -> None:
    if not job_manager:
        return
    payload = dict(body or {})
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            _upsert_task(job_manager, dict(task or {}), create_if_missing=True)
        job_manager._save_history()
        return
    task_id = str(payload.get("id", "") or "")
    if not task_id:
        return
    _upsert_task(job_manager, payload, create_if_missing=False)


def _upsert_task(job_manager: Any, task: dict[str, Any], *, create_if_missing: bool) -> None:
    task_id = str(task.get("id", "") or "")
    if not task_id:
        return
    with job_manager._lock:
        existing = job_manager._jobs.get(task_id)
        if existing:
            _update_existing_task(existing, task)
            return
        if not create_if_missing:
            return
        job_manager._jobs[task_id] = Job(
            type=JobType.GENERIC,
            name=task.get("name", task_id),
            status=parse_task_status(task.get("status")),
            id=task_id,
            progress=task.get("progress", 0.0),
            error=task.get("error"),
            created_at=_parse_datetime(task.get("created_at")) or datetime.now(),
            started_at=_parse_datetime(task.get("started_at")),
            finished_at=_parse_datetime(task.get("finished_at")),
            metadata=_metadata_with_closed_multi_batch_claims(task.get("metadata", {}) or {}),
        )


def _update_existing_task(job: Job, task: dict[str, Any]) -> None:
    if "status" in task:
        job.status = parse_task_status(task["status"], default=job.status)
    if "name" in task:
        job.name = task["name"]
    if "metadata" in task:
        job.metadata.update(_metadata_with_closed_multi_batch_claims(task["metadata"] or {}))
    if "progress" in task:
        job.progress = task["progress"]
    if "error" in task:
        job.error = task["error"]
