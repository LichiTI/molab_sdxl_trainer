"""Thin route helpers for task/job-manager backed compatibility endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable


JobManagerFactory = Callable[[], Any | None]
TelemetryReaderFactory = Callable[[], Any | None]
ProcessManagerFactory = Callable[[], Any | None]
GpuManagerFactory = Callable[[], Any | None]
QueueServiceFactory = Callable[[], Any | None]

logger = logging.getLogger("lulynx.task_route_adapter")


def _default_job_manager_factory() -> JobManagerFactory:
    from backend.core.locator import Locator

    return Locator.get_jobs


def _default_telemetry_reader_factory() -> TelemetryReaderFactory:
    from backend.core.telemetry_store import get_file_telemetry_reader

    return get_file_telemetry_reader


def _default_training_process_manager_factory() -> ProcessManagerFactory:
    from backend.core.managers.training_manager import training_process_manager

    return lambda: training_process_manager


def _default_gpu_resource_manager_factory() -> GpuManagerFactory:
    from backend.core.gpu_resource_manager import gpu_resource_manager

    return lambda: gpu_resource_manager


def _default_training_queue_service_factory() -> QueueServiceFactory:
    from backend.core.services.training_queue_service import get_training_queue_service

    return get_training_queue_service


def resolve_route_job_manager(
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
) -> Any | None:
    if job_manager is not None:
        return job_manager
    return (job_manager_factory or _default_job_manager_factory())()


def resolve_route_telemetry_reader(
    *,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
) -> Any | None:
    if telemetry_reader is not None:
        return telemetry_reader
    return (telemetry_reader_factory or _default_telemetry_reader_factory())()


def resolve_route_training_queue_service(
    *,
    queue_service: Any | None = None,
    queue_service_factory: QueueServiceFactory | None = None,
) -> Any | None:
    if queue_service is not None:
        return queue_service
    return (queue_service_factory or _default_training_queue_service_factory())()


def serialize_official_tasks_route_payload(
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    serializer: Callable[[Any | None, Any | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if serializer is None:
        from backend.core.services.task_history_adapter import serialize_official_tasks

        serializer = serialize_official_tasks
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    reader = resolve_route_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    return serializer(manager, reader)


def serialize_local_task_history_route_payload(
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    serializer: Callable[[Any | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if serializer is None:
        from backend.core.services.task_history_adapter import serialize_task_history

        serializer = serialize_task_history
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    return serializer(manager)


def task_output_route_payload(
    task_id: str,
    *,
    tail: int = 100,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    output_builder: Callable[[Any, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if output_builder is None:
        from backend.core.services.telemetry_compat_adapter import get_task_output_payload

        def output_builder(reader: Any, target_task_id: str) -> dict[str, Any]:
            return get_task_output_payload(reader, target_task_id, tail=tail)

    reader = resolve_route_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    return output_builder(reader, task_id)


def delete_official_task_route_result(
    task_id: str,
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    deleter: Callable[[Any | None, str], bool] | None = None,
    file_run_deleter: Callable[[Any, str], bool] | None = None,
) -> bool:
    if deleter is None:
        from backend.core.services.task_history_adapter import delete_official_task

        deleter = delete_official_task
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    if deleter(manager, task_id):
        return True
    if file_run_deleter is None:
        from backend.core.services.file_training_task_service import delete_file_training_task

        file_run_deleter = delete_file_training_task
    reader = resolve_route_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    return bool(reader is not None and file_run_deleter(reader, task_id))


def clear_official_tasks_route_payload(
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    clearer: Callable[[Any | None], int] | None = None,
    file_run_clearer: Callable[[Any], int] | None = None,
) -> dict[str, int]:
    if clearer is None:
        from backend.core.services.task_history_adapter import clear_completed_official_tasks

        clearer = clear_completed_official_tasks
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    deleted = int(clearer(manager) or 0)
    if file_run_clearer is None:
        from backend.core.services.file_training_task_service import clear_finished_file_training_tasks

        file_run_clearer = clear_finished_file_training_tasks
    reader = resolve_route_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    if reader is not None:
        deleted += int(file_run_clearer(reader) or 0)
    return {"deleted": deleted}


def terminate_official_task_route_result(
    task_id: str,
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    queue_service: Any | None = None,
    queue_service_factory: QueueServiceFactory | None = None,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    training_process_manager: Any | None = None,
    training_process_manager_factory: ProcessManagerFactory | None = None,
    gpu_resource_manager: Any | None = None,
    gpu_resource_manager_factory: GpuManagerFactory | None = None,
    terminator: Callable[[Any | None, str], bool] | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    if terminator is None:
        from backend.core.services.task_history_adapter import terminate_official_task

        terminator = terminate_official_task
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    if terminator(manager, task_id):
        return True, None
    resolved_queue_service = resolve_route_training_queue_service(
        queue_service=queue_service,
        queue_service_factory=queue_service_factory,
    )
    if resolved_queue_service is not None:
        try:
            payload = resolved_queue_service.cancel_run(task_id)
        except Exception:
            logger.warning("Failed to cancel queued training run %s via queue service", task_id, exc_info=True)
            payload = None
        if payload is not None:
            return True, payload
    if stop_file_based_training_run(
        task_id,
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
        training_process_manager=training_process_manager,
        training_process_manager_factory=training_process_manager_factory,
        gpu_resource_manager=gpu_resource_manager,
        gpu_resource_manager_factory=gpu_resource_manager_factory,
    ):
        return True, {"stopped": True, "task_id": task_id, "kind": "file_training_run"}
    return False, None


def stop_file_based_training_run(
    task_id: str,
    *,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    training_process_manager: Any | None = None,
    training_process_manager_factory: ProcessManagerFactory | None = None,
    gpu_resource_manager: Any | None = None,
    gpu_resource_manager_factory: GpuManagerFactory | None = None,
    now: Callable[[], datetime] | None = None,
) -> bool:
    """Stop a detached training run that is exposed as an official UI task."""

    reader = resolve_route_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    if reader is None:
        return False
    active_runs = reader.get_active_runs()
    if task_id not in {str(run.get("run_id") or "") for run in active_runs}:
        return False

    process_manager = training_process_manager
    if process_manager is None:
        process_manager = (training_process_manager_factory or _default_training_process_manager_factory())()
    result = process_manager.stop_worker() if process_manager is not None else {"status": "error"}
    if result.get("status") == "error":
        logger.warning("Failed to stop file-based training run %s: %s", task_id, result)
        return False

    _mark_file_run_stopped(reader, active_runs, task_id, now=now)
    _release_training_gpu_lock(
        gpu_resource_manager=gpu_resource_manager,
        gpu_resource_manager_factory=gpu_resource_manager_factory,
    )
    return True


def _mark_file_run_stopped(
    reader: Any,
    active_runs: list[dict[str, Any]],
    task_id: str,
    *,
    now: Callable[[], datetime] | None = None,
) -> None:
    runs_dir = getattr(reader, "_runs_dir", None)
    if runs_dir is None:
        return
    timestamp = (now or datetime.now)().isoformat()
    for run in active_runs:
        run_id = str(run.get("run_id") or "")
        if run_id != task_id:
            continue
        state_file = runs_dir / run_id / "state.json"
        if not state_file.is_file():
            continue
        try:
            state = json.loads(state_file.read_text(encoding="utf-8-sig"))
            if isinstance(state, dict):
                state["status"] = "stopped"
                state["updated_at"] = timestamp
                state["stop_reason"] = "Stopped by user"
                state_file.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            logger.warning("Failed to mark file-based run %s stopped", run_id, exc_info=True)


def _release_training_gpu_lock(
    *,
    gpu_resource_manager: Any | None = None,
    gpu_resource_manager_factory: GpuManagerFactory | None = None,
) -> None:
    manager = gpu_resource_manager
    if manager is None:
        manager = (gpu_resource_manager_factory or _default_gpu_resource_manager_factory())()
    if manager is None:
        return
    lock_info = manager.get_lock_info()
    holder_id = getattr(lock_info, "holder_id", "") if lock_info else ""
    if holder_id.startswith("training_"):
        manager.release_gpu(holder_id)


def clear_local_task_history_route_payload(
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    clearer: Callable[[Any | None], None] | None = None,
) -> None:
    if clearer is None:
        from backend.core.services.task_history_adapter import clear_completed_task_history

        clearer = clear_completed_task_history
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    clearer(manager)


def delete_local_task_history_route_payload(
    task_id: str,
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    deleter: Callable[[Any | None, str], None] | None = None,
) -> None:
    if deleter is None:
        from backend.core.services.task_history_adapter import delete_task_history_entry

        deleter = delete_task_history_entry
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    deleter(manager, task_id)


def save_local_task_history_route_payload(
    body: dict[str, Any],
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    saver: Callable[[Any | None, dict[str, Any]], None] | None = None,
) -> None:
    if saver is None:
        from backend.core.services.task_history_adapter import save_task_history_payload

        saver = save_task_history_payload
    manager = resolve_route_job_manager(job_manager=job_manager, job_manager_factory=job_manager_factory)
    saver(manager, body)


def list_local_task_history_route_payload() -> dict[str, Any]:
    return serialize_local_task_history_route_payload()


def clear_local_task_history_default_route_payload() -> None:
    clear_local_task_history_route_payload()


def delete_local_task_history_default_route_payload(task_id: str) -> None:
    delete_local_task_history_route_payload(task_id)


def save_local_task_history_default_route_payload(body: dict[str, Any]) -> None:
    save_local_task_history_route_payload(body)
