from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from core.services.training_runtime_task_metadata import build_training_runtime_task_metadata


_ACTIVE_TASK_PHASES = {"queued", "running"}


def normalize_training_task_phase(status: Any) -> str:
    value = str(status or "").strip().lower()
    if value == "queued":
        return "queued"
    if value in {"starting", "running", "paused"}:
        return "running"
    if value == "completed":
        return "succeeded"
    if value in {"failed", "orphaned"}:
        return "failed"
    if value in {"stopped", "cancelled", "canceled"}:
        return "cancelled"
    return "failed"


def is_active_training_task_phase(phase: Any) -> bool:
    return str(phase or "").strip().lower() in _ACTIVE_TASK_PHASES


def list_file_training_tasks(reader: Any, *, limit: int | None = 80) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for run in reader.list_runs():
        task = get_file_training_task(
            reader,
            str(run.get("run_id") or ""),
            state=run,
        )
        if task is not None:
            tasks.append(task)
    tasks.sort(key=_task_sort_key, reverse=True)
    if limit is not None:
        return tasks[: max(int(limit), 0)]
    return tasks


def get_file_training_task(
    reader: Any,
    task_id: str,
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    run_id = str(task_id or "").strip()
    if not run_id:
        return None

    payload = dict(state or reader.get_state(run_id) or {})
    if not payload:
        return None

    launch = reader.get_launch_request(run_id) or {}
    raw_status = str(payload.get("status") or "").strip().lower()
    phase = normalize_training_task_phase(raw_status)
    runtime_id = str(
        payload.get("execution_profile_id")
        or launch.get("execution_profile_id")
        or ""
    )
    started_at = str(payload.get("started_at") or payload.get("queued_at") or "")
    finished_at = ""
    if not is_active_training_task_phase(phase):
        finished_at = str(payload.get("finished_at") or payload.get("updated_at") or "")

    run_dir = _run_dir(reader, run_id)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "config_name": payload.get("config_name") or launch.get("config_name") or run_id,
        "model_type": payload.get("model_type") or launch.get("model_type") or "",
        "training_type": payload.get("training_type") or launch.get("training_type") or "",
        "schema_id": payload.get("schema_id") or launch.get("schema_id") or "",
        "execution_profile_id": runtime_id,
        "requested_attention_backend": payload.get("requested_attention_backend") or launch.get("requested_attention_backend") or "",
        "resolved_attention_backend": payload.get("resolved_attention_backend") or launch.get("resolved_attention_backend") or "",
        "queued_at": payload.get("queued_at") or "",
        "queue_message": payload.get("queue_message") or "",
        "raw_status": raw_status,
        "result_path": str(run_dir),
    }
    queue_position = _int_or_none(payload.get("queue_position"))
    if queue_position is not None:
        metadata["queue_position"] = queue_position
    metadata.update(
        build_training_runtime_task_metadata(
            reader=reader,
            run_id=run_id,
            run_dir=run_dir,
            state=payload,
            launch=launch,
        )
    )

    return {
        "task_id": run_id,
        "kind": "training",
        "phase": phase,
        "runtime_id": runtime_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": _task_error(raw_status, phase, payload),
        "stages": _build_stages(
            raw_status=raw_status,
            phase=phase,
            state=payload,
            run_dir=run_dir,
        ),
        "metadata": metadata,
    }


def delete_file_training_task(reader: Any, task_id: str) -> bool:
    task = get_file_training_task(reader, task_id)
    if task is None or is_active_training_task_phase(task.get("phase")):
        return False

    run_dir = _run_dir(reader, task_id)
    runs_dir = _runs_dir(reader)
    if not run_dir.is_dir() or run_dir.parent != runs_dir:
        return False

    try:
        shutil.rmtree(run_dir)
    except Exception:
        return False
    return True


def clear_finished_file_training_tasks(reader: Any) -> int:
    deleted = 0
    for task in list_file_training_tasks(reader, limit=None):
        if is_active_training_task_phase(task.get("phase")):
            continue
        if delete_file_training_task(reader, str(task.get("task_id") or "")):
            deleted += 1
    return deleted


def _task_error(raw_status: str, phase: str, state: dict[str, Any]) -> str:
    if phase != "failed":
        return ""
    message = state.get("error") or state.get("stop_reason") or ""
    return str(message or "Training run failed")


def _build_stages(
    *,
    raw_status: str,
    phase: str,
    state: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    queued_at = str(state.get("queued_at") or "")
    queue_position = _int_or_none(state.get("queue_position"))
    queue_message = str(state.get("queue_message") or "")
    base_detail = {
        "target_path": str(run_dir),
        "config_name": state.get("config_name") or "training",
        "model_type": state.get("model_type") or "",
        "training_type": state.get("training_type") or "",
    }

    if raw_status == "queued":
        detail = dict(base_detail)
        if queue_position is not None:
            detail["queue_position"] = queue_position
        if queue_message:
            detail["message"] = queue_message
        stages.append(
            {
                "code": "training.queued",
                "label_zh": "等待训练槽位",
                "label_en": "Waiting for training slot",
                "phase": "queued",
                "detail": detail,
                "timestamp": queued_at or str(state.get("updated_at") or state.get("started_at") or ""),
            }
        )
        return stages

    if queued_at or queue_position is not None:
        detail = dict(base_detail)
        if queue_position is not None:
            detail["queue_position"] = queue_position
        if queue_message:
            detail["message"] = queue_message
        stages.append(
            {
                "code": "training.queued",
                "label_zh": "排队完成",
                "label_en": "Queue completed",
                "phase": "succeeded",
                "detail": detail,
                "timestamp": queued_at,
            }
        )

    stage_phase = phase if phase in {"running", "failed", "cancelled", "succeeded"} else "running"
    detail = dict(base_detail)
    current_step = _int_or_none(state.get("current_step"))
    total_steps = _int_or_none(state.get("total_steps"))
    current_epoch = _int_or_none(state.get("current_epoch"))
    total_epochs = _int_or_none(state.get("total_epochs"))
    if current_step is not None:
        detail["current_step"] = current_step
    if total_steps is not None:
        detail["total_steps"] = total_steps
    if current_epoch is not None:
        detail["current_epoch"] = current_epoch
    if total_epochs is not None:
        detail["total_epochs"] = total_epochs
    if state.get("last_loss") is not None:
        detail["last_loss"] = state.get("last_loss")
    if state.get("last_lr") is not None:
        detail["last_lr"] = state.get("last_lr")
    if state.get("stop_reason"):
        detail["stop_reason"] = state.get("stop_reason")
    if state.get("error"):
        detail["error"] = state.get("error")

    label_zh, label_en = _run_stage_labels(raw_status=raw_status, phase=phase)
    stages.append(
        {
            "code": "training.run",
            "label_zh": label_zh,
            "label_en": label_en,
            "phase": stage_phase,
            "detail": detail,
            "timestamp": str(state.get("updated_at") or state.get("started_at") or ""),
        }
    )
    return stages


def _run_stage_labels(*, raw_status: str, phase: str) -> tuple[str, str]:
    if raw_status == "starting":
        return "启动训练进程", "Starting training worker"
    if raw_status == "paused":
        return "训练已暂停", "Training paused"
    if phase == "succeeded":
        return "训练完成", "Training completed"
    if phase == "failed":
        return "训练失败", "Training failed"
    if phase == "cancelled":
        return "训练已停止", "Training stopped"
    return "训练中", "Training running"


def _task_sort_key(task: dict[str, Any]) -> tuple[str, str]:
    phase = str(task.get("phase") or "")
    priority = "1" if is_active_training_task_phase(phase) else "0"
    timestamp = str(task.get("finished_at") or task.get("started_at") or "")
    return priority, timestamp


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _copy_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is None or value == "":
            continue
        result[key] = value
    return result


def _runs_dir(reader: Any) -> Path:
    return Path(getattr(reader, "_runs_dir")).resolve()


def _run_dir(reader: Any, run_id: str) -> Path:
    return (_runs_dir(reader) / str(run_id or "").strip()).resolve()
