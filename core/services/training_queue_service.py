from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from backend.core.services.training_queue_support import (
    QueueOperationResult,
    QueuedTrainingRun,
    build_worker_env,
    list_queued_runs,
    now_iso,
    prepare_queue_run_artifacts,
    refresh_queue_positions,
    write_run_state,
)

logger = logging.getLogger("lulynx.training_queue")

_ACTIVE_RUN_STATUSES = {"starting", "running", "paused"}
_TERMINAL_RUN_STATUSES = {"completed", "failed", "stopped", "cancelled", "canceled", "orphaned"}


class TrainingQueueService:
    """Coordinates single-GPU training runs as a persisted FIFO queue."""

    def __init__(
        self,
        *,
        backend_root: Path | None = None,
        poll_interval: float = 1.0,
        training_process_manager: Any | None = None,
        gpu_resource_manager: Any | None = None,
        telemetry_reader_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self._backend_root = backend_root or Path(__file__).resolve().parents[2]
        self._runs_dir = self._backend_root / ".runs"
        self._state_path = self._backend_root / "core" / "data" / "training_queue.json"
        self._entry_script = self._backend_root / "core" / "entry_train.py"
        self._poll_interval = max(float(poll_interval), 0.2)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._loaded = False
        self._state: dict[str, Any] = {
            "current_run_id": None,
            "queued_runs": [],
            "updated_at": now_iso(),
        }
        if training_process_manager is None:
            from backend.core.managers.training_manager import training_process_manager as _training_process_manager

            training_process_manager = _training_process_manager
        if gpu_resource_manager is None:
            from backend.core.gpu_resource_manager import gpu_resource_manager as _gpu_resource_manager

            gpu_resource_manager = _gpu_resource_manager
        if telemetry_reader_factory is None:
            from backend.core.telemetry_store import get_file_telemetry_reader

            telemetry_reader_factory = get_file_telemetry_reader
        self._training_process_manager = training_process_manager
        self._gpu_resource_manager = gpu_resource_manager
        self._telemetry_reader_factory = telemetry_reader_factory

    def ensure_monitor_started(self) -> None:
        with self._lock:
            self._load_state_locked()
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="training-queue-monitor",
                daemon=True,
            )
            self._monitor_thread.start()

    def enqueue_or_start(self, *, config: Any, resolved: Any) -> QueueOperationResult:
        self.ensure_monitor_started()
        with self._lock:
            self._sync_runtime_locked(start_next=True)
            prepared = self._prepare_run_locked(config=config, resolved=resolved)

            if self._can_start_immediately_locked() and self._start_prepared_run_locked(prepared):
                return QueueOperationResult(
                    status="training_started",
                    run_id=prepared.run_id,
                    config_name=prepared.config_name,
                    execution_profile_id=prepared.execution_profile_id,
                    requested_attention_backend=prepared.requested_attention_backend,
                    resolved_attention_backend=prepared.resolved_attention_backend,
                )

            message = self._busy_message_locked()
            queue_position = self._enqueue_prepared_run_locked(prepared, message=message)
            return QueueOperationResult(
                status="queued",
                run_id=prepared.run_id,
                config_name=prepared.config_name,
                execution_profile_id=prepared.execution_profile_id,
                requested_attention_backend=prepared.requested_attention_backend,
                resolved_attention_backend=prepared.resolved_attention_backend,
                queue_position=queue_position,
                queue_depth=len(self._queued_runs_locked()),
                message=message,
            )

    def stop_current_run(self) -> dict[str, Any]:
        self.ensure_monitor_started()
        with self._lock:
            self._sync_runtime_locked(start_next=False)
            current_run_id = self._current_run_id_locked()
            if not current_run_id:
                return {
                    "status": "no_training_running",
                    "stopped_runs": [],
                    "released_gpu_holder": self._release_training_gpu_lock_locked(),
                    "next_run_id": None,
                }

            self._attach_current_process_locked(current_run_id)
            result = self._training_process_manager.stop_worker()
            status = str(result.get("status") or "")
            if status == "error":
                return result
            if status == "no_process":
                return {
                    "status": "no_training_running",
                    "stopped_runs": [],
                    "released_gpu_holder": self._release_training_gpu_lock_locked(),
                    "next_run_id": None,
                }

            stopped_runs = self._mark_run_stopped_locked(current_run_id, message="Stopped by user")
            released_gpu_holder = self._release_training_gpu_lock_locked()
            self._set_current_run_id_locked(None)
            self._save_state_locked()
            next_run_id = self._start_next_if_possible_locked()
            return {
                **result,
                "stopped_runs": stopped_runs,
                "released_gpu_holder": released_gpu_holder,
                "next_run_id": next_run_id,
            }

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        self.ensure_monitor_started()
        with self._lock:
            self._sync_runtime_locked(start_next=False)
            target_run_id = str(run_id or "").strip()
            if not target_run_id:
                return None
            if target_run_id == self._current_run_id_locked():
                result = self.stop_current_run()
                status = str(result.get("status") or "")
                if status in {"stopped", "no_training_running"}:
                    return {
                        **result,
                        "stopped": status == "stopped",
                        "task_id": target_run_id,
                        "kind": "file_training_run",
                    }
                return None
            return self._cancel_queued_run_locked(target_run_id)

    def list_active_payload(self) -> dict[str, Any]:
        self.ensure_monitor_started()
        with self._lock:
            self._sync_runtime_locked(start_next=True)
            reader = self._reader()
            return {
                "runs": reader.get_active_runs(),
                "queued_runs": self._list_queued_runs_locked(),
                "current_run_id": self._current_run_id_locked(),
                "queue_depth": len(self._queued_runs_locked()),
            }

    def sync_runtime(self, *, start_next: bool = True) -> None:
        self.ensure_monitor_started()
        with self._lock:
            self._sync_runtime_locked(start_next=start_next)

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            try:
                with self._lock:
                    self._sync_runtime_locked(start_next=True)
            except Exception:
                logger.warning("Training queue monitor tick failed", exc_info=True)

    def _load_state_locked(self) -> None:
        if self._loaded:
            return
        if self._state_path.is_file():
            try:
                payload = json.loads(self._state_path.read_text(encoding="utf-8-sig"))
            except Exception:
                logger.warning("Failed to read training queue state", exc_info=True)
                payload = {}
            if isinstance(payload, dict):
                self._state["current_run_id"] = str(payload.get("current_run_id") or "") or None
                queued_runs = payload.get("queued_runs")
                if isinstance(queued_runs, list):
                    self._state["queued_runs"] = [
                        dict(item or {})
                        for item in queued_runs
                        if isinstance(item, dict)
                    ]
                self._state["updated_at"] = str(payload.get("updated_at") or now_iso())
        self._loaded = True
        self._sync_runtime_locked(start_next=False)

    def _save_state_locked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current_run_id": self._current_run_id_locked(),
            "queued_runs": self._queued_runs_locked(),
            "updated_at": now_iso(),
        }
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._state["updated_at"] = payload["updated_at"]

    def _reader(self):
        return self._telemetry_reader_factory(self._runs_dir)

    def _queued_runs_locked(self) -> list[dict[str, Any]]:
        value = self._state.get("queued_runs")
        return value if isinstance(value, list) else []

    def _current_run_id_locked(self) -> Optional[str]:
        value = str(self._state.get("current_run_id") or "").strip()
        return value or None

    def _set_current_run_id_locked(self, run_id: str | None) -> None:
        self._state["current_run_id"] = str(run_id or "").strip() or None

    def _sync_runtime_locked(self, *, start_next: bool) -> None:
        current_run_id = self._discover_current_run_locked()
        self._set_current_run_id_locked(current_run_id)
        if current_run_id and not self._is_run_still_active_locked(current_run_id):
            self._handle_run_exit_locked(current_run_id)
        if start_next:
            self._start_next_if_possible_locked()
        self._save_state_locked()

    def _discover_current_run_locked(self) -> Optional[str]:
        current_run_id = self._current_run_id_locked()
        if current_run_id and self._is_run_still_active_locked(current_run_id):
            return current_run_id

        reader = self._reader()
        active_runs = reader.get_active_runs()
        if active_runs:
            by_pid = {
                int(run.get("pid") or 0): str(run.get("run_id") or "")
                for run in active_runs
            }
            active_process = self._training_process_manager.get_process()
            process_pid = int(getattr(active_process, "pid", 0) or 0)
            if process_pid and by_pid.get(process_pid):
                return by_pid[process_pid]
            for run in active_runs:
                run_id = str(run.get("run_id") or "")
                if run_id and self._is_run_still_active_locked(run_id):
                    return run_id
            first_run_id = str(active_runs[0].get("run_id") or "")
            return first_run_id or None

        pid = int(self._training_process_manager.load_last_pid() or 0)
        if pid > 0:
            self._training_process_manager.attach_orphaned_process(pid)
            active_process = self._training_process_manager.get_process()
            process_pid = int(getattr(active_process, "pid", 0) or 0)
            if process_pid == pid:
                return self._find_active_run_by_pid_locked(pid)
        return None

    def _find_active_run_by_pid_locked(self, pid: int) -> Optional[str]:
        if pid <= 0:
            return None
        reader = self._reader()
        for run in reader.get_active_runs():
            if int(run.get("pid") or 0) == pid:
                run_id = str(run.get("run_id") or "")
                if run_id:
                    return run_id
        return None

    def _is_run_still_active_locked(self, run_id: str) -> bool:
        if not run_id:
            return False
        reader = self._reader()
        state = reader.get_state(run_id) or {}
        status = str(state.get("status") or "").strip().lower()
        if status in _TERMINAL_RUN_STATUSES:
            return False
        if status not in _ACTIVE_RUN_STATUSES:
            return False

        process = self._training_process_manager.get_process()
        if process is not None and getattr(process, "poll", None) is not None and process.poll() is None:
            return True

        pid = int(state.get("pid") or self._training_process_manager.load_last_pid() or 0)
        if pid <= 0:
            return False
        try:
            if self._training_process_manager.is_training_pid(pid):
                self._training_process_manager.attach_orphaned_process(pid)
                return True
        except Exception:
            logger.debug("Failed to verify training PID %s", pid, exc_info=True)
        try:
            return bool(self._training_process_manager._is_pid_alive(pid))
        except Exception:
            return False

    def _can_start_immediately_locked(self) -> bool:
        if self._current_run_id_locked():
            return False
        if self._queued_runs_locked():
            return False
        return self._gpu_resource_manager.get_lock_info() is None

    def _busy_message_locked(self) -> str:
        current_run_id = self._current_run_id_locked()
        if current_run_id:
            return (
                f"Training run {current_run_id} is still active. "
                "This run has been queued and will start automatically."
            )
        queue_depth = len(self._queued_runs_locked())
        if queue_depth > 0:
            return (
                f"There are {queue_depth} training run(s) ahead in the queue. "
                "This run has been queued and will start automatically."
            )
        lock_info = self._gpu_resource_manager.get_lock_info()
        holder_id = getattr(lock_info, "holder_id", "") if lock_info else ""
        if holder_id:
            return (
                f"GPU Resource is currently locked by: {holder_id}. "
                "This run has been queued and will start automatically."
            )
        return "Training run has been queued and will start automatically."

    def _prepare_run_locked(self, *, config: Any, resolved: Any) -> QueuedTrainingRun:
        run, config_dict, metadata = prepare_queue_run_artifacts(
            runs_dir=self._runs_dir,
            entry_script=self._entry_script,
            config=config,
            resolved=resolved,
        )
        write_run_state(
            runs_dir=self._runs_dir,
            run_id=run.run_id,
            config_name=run.config_name,
            model_type=run.model_type,
            training_type=run.training_type,
            execution_profile_id=run.execution_profile_id,
            schema_id=run.schema_id,
            requested_attention_backend=run.requested_attention_backend,
            resolved_attention_backend=run.resolved_attention_backend,
            status="starting",
            started_at=metadata["started_at"],
            updated_at=metadata["started_at"],
            pid=None,
            total_epochs=int(config_dict.get("max_train_epochs") or 0),
            total_steps=int(config_dict.get("max_train_steps") or 0),
        )
        return run

    def _enqueue_prepared_run_locked(self, run: QueuedTrainingRun, *, message: str) -> int:
        queued_runs = self._queued_runs_locked()
        queued_runs.append(run.__dict__.copy())
        queue_positions = refresh_queue_positions(
            reader=self._reader(),
            runs_dir=self._runs_dir,
            items=queued_runs,
            message_overrides={run.run_id: message},
        )
        self._save_state_locked()
        return queue_positions.get(run.run_id, len(queued_runs))

    def _start_next_if_possible_locked(self) -> Optional[str]:
        if self._current_run_id_locked():
            return None
        queued_runs = self._queued_runs_locked()
        if not queued_runs:
            return None

        while queued_runs:
            run = QueuedTrainingRun(**dict(queued_runs[0]))
            if not self._gpu_resource_manager.try_acquire_gpu(
                holder_id=run.holder_id,
                description=f"Training {run.model_type} (lulynx)",
            ):
                return None

            queued_runs.pop(0)
            self._set_current_run_id_locked(run.run_id)
            refresh_queue_positions(reader=self._reader(), runs_dir=self._runs_dir, items=queued_runs)
            if self._start_prepared_run_locked(run, already_acquired_lock=True):
                return run.run_id

        return None

    def _start_prepared_run_locked(
        self,
        run: QueuedTrainingRun,
        *,
        already_acquired_lock: bool = False,
    ) -> bool:
        if not already_acquired_lock:
            if not self._gpu_resource_manager.try_acquire_gpu(
                holder_id=run.holder_id,
                description=f"Training {run.model_type} (lulynx)",
            ):
                return False

        self._set_current_run_id_locked(run.run_id)
        run_dir = Path(run.run_dir)
        config_path = run_dir / "config.json"
        if not self._entry_script.exists():
            self._mark_run_failed_locked(run.run_id, f"Entry script not found at {self._entry_script}")
            self._set_current_run_id_locked(None)
            self._release_training_gpu_lock_locked(expected_holder_id=run.holder_id)
            self._save_state_locked()
            return False

        try:
            config_dict = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            self._mark_run_failed_locked(run.run_id, f"Failed to read queued config: {exc}")
            self._set_current_run_id_locked(None)
            self._release_training_gpu_lock_locked(expected_holder_id=run.holder_id)
            self._save_state_locked()
            return False

        env = build_worker_env(
            backend_root=self._backend_root,
            config_dict=config_dict,
            profile_env_vars=run.profile_env_vars,
        )

        try:
            process = self._training_process_manager.start_worker(
                [str(run.python_executable), "-u", str(self._entry_script), "--config", str(config_path)],
                env=env,
                cwd=str(self._backend_root),
                config_name=run.config_name,
                run_dir=str(run_dir),
                bufsize=1,
            )
        except Exception as exc:
            logger.error("Failed to launch queued training run %s", run.run_id, exc_info=True)
            self._mark_run_failed_locked(run.run_id, str(exc))
            self._training_process_manager.clear_pid()
            self._set_current_run_id_locked(None)
            self._release_training_gpu_lock_locked(expected_holder_id=run.holder_id)
            self._save_state_locked()
            return False

        started_at = now_iso()
        write_run_state(
            runs_dir=self._runs_dir,
            run_id=run.run_id,
            config_name=run.config_name,
            model_type=run.model_type,
            training_type=run.training_type,
            execution_profile_id=run.execution_profile_id,
            schema_id=run.schema_id,
            requested_attention_backend=run.requested_attention_backend,
            resolved_attention_backend=run.resolved_attention_backend,
            status="running",
            started_at=started_at,
            updated_at=started_at,
            pid=process.pid,
            total_epochs=int(config_dict.get("max_train_epochs") or 0),
            total_steps=int(config_dict.get("max_train_steps") or 0),
            queued_at=run.queued_at,
        )
        self._save_state_locked()
        logger.info(
            "[Training/Queue] Detached worker started: PID=%d run_id=%s profile=%s attention=%s",
            process.pid,
            run.run_id,
            run.execution_profile_id,
            run.resolved_attention_backend,
        )
        return True

    def _list_queued_runs_locked(self) -> list[dict[str, Any]]:
        return list_queued_runs(self._reader(), self._queued_runs_locked())

    def _handle_run_exit_locked(self, run_id: str) -> None:
        reader = self._reader()
        state = reader.get_state(run_id) or {}
        status = str(state.get("status") or "").strip().lower()
        if status not in _TERMINAL_RUN_STATUSES:
            write_run_state(
                runs_dir=self._runs_dir,
                run_id=run_id,
                config_name=str(state.get("config_name") or "training"),
                model_type=str(state.get("model_type") or ""),
                training_type=str(state.get("training_type") or ""),
                execution_profile_id=str(state.get("execution_profile_id") or ""),
                schema_id=str(state.get("schema_id") or ""),
                requested_attention_backend=str(state.get("requested_attention_backend") or ""),
                resolved_attention_backend=str(state.get("resolved_attention_backend") or ""),
                status="failed",
                started_at=str(state.get("started_at") or now_iso()),
                updated_at=now_iso(),
                pid=int(state.get("pid") or 0) or None,
                total_epochs=int(state.get("total_epochs") or 0),
                total_steps=int(state.get("total_steps") or 0),
                error=state.get("error") or "Training worker exited unexpectedly",
            )
        self._training_process_manager.cleanup_after_exit()
        self._training_process_manager.clear_pid()
        self._release_training_gpu_lock_locked()
        self._set_current_run_id_locked(None)

    def _attach_current_process_locked(self, run_id: str) -> None:
        process = self._training_process_manager.get_process()
        if process is not None and getattr(process, "poll", None) is not None and process.poll() is None:
            return
        reader = self._reader()
        state = reader.get_state(run_id) or {}
        pid = int(state.get("pid") or self._training_process_manager.load_last_pid() or 0)
        if pid > 0:
            self._training_process_manager.attach_orphaned_process(pid)

    def _release_training_gpu_lock_locked(self, *, expected_holder_id: str | None = None) -> str:
        lock_info = self._gpu_resource_manager.get_lock_info()
        holder_id = getattr(lock_info, "holder_id", "") if lock_info else ""
        if not holder_id.startswith("training_"):
            return ""
        if expected_holder_id and holder_id != expected_holder_id:
            return ""
        if self._gpu_resource_manager.release_gpu(holder_id):
            return holder_id
        return ""

    def _mark_run_stopped_locked(self, run_id: str, *, message: str) -> list[str]:
        if not run_id:
            return []
        reader = self._reader()
        state = reader.get_state(run_id)
        if not isinstance(state, dict):
            return []
        write_run_state(
            runs_dir=self._runs_dir,
            run_id=run_id,
            config_name=str(state.get("config_name") or "training"),
            model_type=str(state.get("model_type") or ""),
            training_type=str(state.get("training_type") or ""),
            execution_profile_id=str(state.get("execution_profile_id") or ""),
            schema_id=str(state.get("schema_id") or ""),
            requested_attention_backend=str(state.get("requested_attention_backend") or ""),
            resolved_attention_backend=str(state.get("resolved_attention_backend") or ""),
            status="stopped",
            started_at=str(state.get("started_at") or now_iso()),
            updated_at=now_iso(),
            pid=int(state.get("pid") or 0) or None,
            total_epochs=int(state.get("total_epochs") or 0),
            total_steps=int(state.get("total_steps") or 0),
            error=state.get("error"),
            stop_reason=message,
        )
        return [run_id]

    def _mark_run_failed_locked(self, run_id: str, error: str) -> None:
        reader = self._reader()
        state = reader.get_state(run_id) or {}
        write_run_state(
            runs_dir=self._runs_dir,
            run_id=run_id,
            config_name=str(state.get("config_name") or "training"),
            model_type=str(state.get("model_type") or ""),
            training_type=str(state.get("training_type") or ""),
            execution_profile_id=str(state.get("execution_profile_id") or ""),
            schema_id=str(state.get("schema_id") or ""),
            requested_attention_backend=str(state.get("requested_attention_backend") or ""),
            resolved_attention_backend=str(state.get("resolved_attention_backend") or ""),
            status="failed",
            started_at=str(state.get("started_at") or now_iso()),
            updated_at=now_iso(),
            pid=int(state.get("pid") or 0) or None,
            total_epochs=int(state.get("total_epochs") or 0),
            total_steps=int(state.get("total_steps") or 0),
            error=error,
        )

    def _cancel_queued_run_locked(self, run_id: str) -> dict[str, Any] | None:
        queued_runs = self._queued_runs_locked()
        if not queued_runs:
            return None
        removed_item = None
        remaining_runs: list[dict[str, Any]] = []
        for item in queued_runs:
            if removed_item is None and str(item.get("run_id") or "") == run_id:
                removed_item = dict(item)
                continue
            remaining_runs.append(item)
        if removed_item is None:
            return None

        self._state["queued_runs"] = remaining_runs
        refresh_queue_positions(reader=self._reader(), runs_dir=self._runs_dir, items=remaining_runs)
        reader = self._reader()
        state = reader.get_state(run_id) or {}
        write_run_state(
            runs_dir=self._runs_dir,
            run_id=run_id,
            config_name=str(state.get("config_name") or removed_item.get("config_name") or "training"),
            model_type=str(state.get("model_type") or removed_item.get("model_type") or ""),
            training_type=str(state.get("training_type") or removed_item.get("training_type") or ""),
            execution_profile_id=str(state.get("execution_profile_id") or removed_item.get("execution_profile_id") or ""),
            schema_id=str(state.get("schema_id") or removed_item.get("schema_id") or ""),
            requested_attention_backend=str(state.get("requested_attention_backend") or removed_item.get("requested_attention_backend") or ""),
            resolved_attention_backend=str(state.get("resolved_attention_backend") or removed_item.get("resolved_attention_backend") or ""),
            status="cancelled",
            started_at=str(state.get("started_at") or state.get("queued_at") or removed_item.get("queued_at") or now_iso()),
            updated_at=now_iso(),
            pid=None,
            total_epochs=int(state.get("total_epochs") or 0),
            total_steps=int(state.get("total_steps") or 0),
            error=state.get("error"),
            queued_at=str(state.get("queued_at") or removed_item.get("queued_at") or ""),
            stop_reason="Cancelled while queued",
        )
        self._save_state_locked()
        return {
            "status": "cancelled",
            "stopped": True,
            "task_id": run_id,
            "kind": "file_training_queue",
            "queue_depth": len(remaining_runs),
        }


_training_queue_service: TrainingQueueService | None = None


def get_training_queue_service() -> TrainingQueueService:
    global _training_queue_service
    if _training_queue_service is None:
        _training_queue_service = TrainingQueueService()
    return _training_queue_service
