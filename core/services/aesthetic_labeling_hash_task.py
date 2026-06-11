"""Background hash reindex tasks for aesthetic labeling."""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Callable


ProgressCallback = Callable[[int, int], None]
TaskRunner = Callable[[bool, ProgressCallback | None], dict[str, Any]]


class AestheticHashTaskManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}

    def start(self, runner: TaskRunner, *, hash_local: bool = True) -> dict[str, Any]:
        task_id = f"aesthetic_hash_{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": "running",
                "progress": 0.0,
                "processed": 0,
                "total": 1,
                "started_at": _now_iso(),
                "finished_at": "",
                "error": "",
                "result": None,
            }

        thread = threading.Thread(target=self._run, args=(task_id, runner, bool(hash_local)), daemon=True)
        thread.start()
        return self.get(task_id)

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._tasks.get(task_id) or {"task_id": task_id, "status": "not_found", "progress": 0.0})

    def _run(self, task_id: str, runner: TaskRunner, hash_local: bool) -> None:
        self._update(task_id, progress=0.05, processed=0, total=1)
        try:
            result = runner(hash_local, lambda processed, total: self._set_progress(task_id, processed, total))
            self._finish(task_id, status="completed", result=result)
        except Exception as exc:
            self._finish(task_id, status="failed", error=str(exc))

    def _update(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.update(fields)

    def _set_progress(self, task_id: str, processed: int, total: int) -> None:
        total = max(1, int(total or 1))
        processed = max(0, min(int(processed or 0), total))
        self._update(task_id, processed=processed, total=total, progress=processed / total)

    def _finish(self, task_id: str, *, status: str, result: dict[str, Any] | None = None, error: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.update({
                "status": status,
                "progress": 1.0 if status == "completed" else task.get("progress", 0.0),
                "processed": result.get("indexed_files", 0) if result else task.get("processed", 0),
                "total": result.get("indexed_files", 1) if result else task.get("total", 1),
                "finished_at": _now_iso(),
                "error": error,
                "result": result,
            })


_GLOBAL_HASH_TASKS = AestheticHashTaskManager()


def get_aesthetic_hash_task_manager() -> AestheticHashTaskManager:
    return _GLOBAL_HASH_TASKS


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
