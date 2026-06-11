"""Subprocess job adapter for Lulynx LAB runners."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.core.services.lab_artifacts import build_lab_artifacts, build_lab_run_result


class LabSubprocessSubmissionError(RuntimeError):
    """Error raised before a LAB subprocess job can be submitted."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class LabSubprocessJobSpec:
    """Prepared subprocess job descriptor passed from routes into runners."""

    name: str
    config: dict[str, Any]
    runtime_id: str
    runner_path: Path
    config_filename: str
    metadata: dict[str, Any] = field(default_factory=dict)
    runner_args: list[str] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LabSubprocessJobSpec":
        if not isinstance(value, Mapping):
            raise ValueError("lab_subprocess_job metadata must be an object")
        return cls(
            name=str(value.get("name") or "Lulynx LAB"),
            config=dict(value.get("config") or {}),
            runtime_id=str(value.get("runtime_id") or ""),
            runner_path=Path(str(value.get("runner_path") or "")),
            config_filename=str(value.get("config_filename") or "lab_config.json"),
            metadata=dict(value.get("metadata") or {}),
            runner_args=list(value.get("runner_args") or []) if value.get("runner_args") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": dict(self.config),
            "runtime_id": self.runtime_id,
            "runner_path": str(self.runner_path),
            "config_filename": self.config_filename,
            "metadata": dict(self.metadata),
            "runner_args": list(self.runner_args) if self.runner_args is not None else None,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_last_json(lines: list[str]) -> dict[str, Any]:
    for line in reversed(lines):
        text = str(line or "").strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def resolve_lab_runtime_python(
    runtime_id: str,
    config: dict[str, Any],
    *,
    project_root: Path,
    backend_root: Path,
) -> tuple[Path, str, dict[str, str]]:
    requested = str(runtime_id or config.get("runtime_id") or config.get("execution_profile_id") or "").strip()
    env_updates: dict[str, str] = {}
    if requested:
        try:
            from backend.core.execution_manifest import get_profile_entry, resolve_python_executable

            entry = get_profile_entry(requested)
        except Exception:
            entry = None
        if entry is None:
            raise LabSubprocessSubmissionError(f"Unknown runtime: {requested}", status_code=400)
        python_exe = resolve_python_executable(entry, backend_root).resolve()
        if not python_exe.is_file():
            raise LabSubprocessSubmissionError(f"Runtime Python not found: {python_exe}", status_code=400)
        env_updates.update(getattr(entry, "env_vars", {}) or {})
        return python_exe, entry.id, env_updates

    standard_python = (backend_root / "env" / "python" / ("python.exe" if os.name == "nt" else "bin/python")).resolve()
    if standard_python.is_file():
        return standard_python, "standard", {}
    return Path(sys.executable).resolve(), "current", {}


def submit_lab_subprocess_job(
    spec: LabSubprocessJobSpec,
    *,
    project_root: Path,
    backend_root: Path,
    job_manager: Any,
    telemetry_reader_factory: Callable[..., Any] | None = None,
    task_id_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Submit a prepared LAB subprocess job and return the legacy route payload."""

    from backend.core.job_manager import Job, JobType
    from backend.core.telemetry_store import get_file_telemetry_reader

    reader_factory = telemetry_reader_factory or get_file_telemetry_reader
    python_exe, resolved_runtime_id, env_updates = resolve_lab_runtime_python(
        spec.runtime_id,
        spec.config,
        project_root=project_root,
        backend_root=backend_root,
    )
    runner_path = spec.runner_path.resolve()
    if not runner_path.is_file():
        raise LabSubprocessSubmissionError(f"Runner not found: {runner_path}", status_code=500)

    task_id = (task_id_factory or (lambda: uuid.uuid4().hex[:12]))()
    reader = reader_factory()
    run_dir = Path(reader._runs_dir) / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "output.log"
    runner_config_path = run_dir / spec.config_filename
    runner_config_path.write_text(json.dumps(spec.config, ensure_ascii=False, indent=2), encoding="utf-8")

    job_metadata = {
        **spec.metadata,
        "runtime_id": resolved_runtime_id,
        "runner_path": str(runner_path),
        "config_path": str(runner_config_path),
        "log_path": str(log_file),
    }
    command_args = spec.runner_args or ["--config", str(runner_config_path)]
    command = [str(python_exe), str(runner_path), *command_args]
    job = Job(id=task_id, type=JobType.TRAINING, name=spec.name, metadata=job_metadata)

    def _run(progress_callback=None, cancel_check=None) -> None:
        _run_lab_subprocess_worker(
            command=command,
            name=spec.name,
            task_id=task_id,
            run_dir=run_dir,
            log_file=log_file,
            job=job,
            job_metadata=job_metadata,
            env_updates=env_updates,
            project_root=project_root,
            backend_root=backend_root,
            resolved_runtime_id=resolved_runtime_id,
            runner_path=runner_path,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    job_manager.submit(job, worker_func=_run)
    schema_id = str(spec.metadata.get("schema_id") or spec.config.get("schema_id") or "lulynx-lab")
    artifacts = build_lab_artifacts(
        schema_id=schema_id,
        run_id=task_id,
        config_path=str(runner_config_path),
        log_path=str(log_file),
        metadata=job_metadata,
    )
    run_result = build_lab_run_result(
        task_id=task_id,
        schema_id=schema_id,
        message=f"{spec.name} queued.",
        artifacts=artifacts,
        metadata=job_metadata,
    )
    job.metadata["artifacts"] = [artifact.model_dump(mode="json") for artifact in artifacts]
    job.metadata["run_result"] = run_result.model_dump(mode="json")
    return {
        "task_id": task_id,
        "runtime_id": resolved_runtime_id,
        "config_path": str(runner_config_path),
        "log_path": str(log_file),
        "command_args": command_args,
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        "run_result": run_result.model_dump(mode="json"),
        **{key: value for key, value in spec.metadata.items() if key.endswith("_path") or key in {"schema_id", "run_id"}},
    }


def _run_lab_subprocess_worker(
    *,
    command: list[str],
    name: str,
    task_id: str,
    run_dir: Path,
    log_file: Path,
    job: Any,
    job_metadata: dict[str, Any],
    env_updates: dict[str, str],
    project_root: Path,
    backend_root: Path,
    resolved_runtime_id: str,
    runner_path: Path,
    progress_callback: Any,
    cancel_check: Any,
) -> None:
    _write_state(
        run_dir,
        {
            "run_id": task_id,
            "status": "running",
            "started_at": utc_now(),
            "live_line": f"{name} started",
            "metadata": job_metadata,
        },
    )
    _write_text(log_file, f"[lulynx-lab] task_id={task_id}")
    _write_text(log_file, f"[lulynx-lab] runtime={resolved_runtime_id}")
    _write_text(log_file, f"[lulynx-lab] runner={runner_path}")
    if progress_callback:
        progress_callback(0, 1)

    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONPATH"] = os.pathsep.join([str(backend_root), str(project_root), env.get("PYTHONPATH", "")])
    popen_kwargs: dict[str, Any] = {
        "cwd": str(backend_root),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    proc = subprocess.Popen(command, **popen_kwargs)
    stdout_tail: list[str] = []
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            stdout_tail.append(line)
            stdout_tail = stdout_tail[-120:]
            _write_text(log_file, line)
            _write_state(
                run_dir,
                {
                    "run_id": task_id,
                    "status": "running",
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "live_line": line,
                    "metadata": job_metadata,
                },
            )
            if cancel_check and cancel_check():
                proc.terminate()
                raise RuntimeError("Task cancelled")
        returncode = proc.wait()
    finally:
        if proc.poll() is None and cancel_check and cancel_check():
            proc.terminate()

    tail_text = "\n".join(stdout_tail[-80:])
    job.metadata["stdout_tail"] = tail_text
    if returncode != 0:
        _write_state(
            run_dir,
            {
                "run_id": task_id,
                "status": "failed",
                "error": tail_text or f"Runner failed with exit code {returncode}",
                "live_line": f"{name} failed",
                "metadata": job.metadata,
            },
        )
        raise RuntimeError(tail_text or f"Runner failed with exit code {returncode}")

    result = parse_last_json(stdout_tail)
    if result:
        job.metadata["result"] = result
    if progress_callback:
        progress_callback(1, 1)
    _write_state(
        run_dir,
        {
            "run_id": task_id,
            "status": "completed",
            "finished_at": utc_now(),
            "live_line": f"{name} completed",
            "metadata": job.metadata,
        },
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    payload = {"updated_at": utc_now(), **state}
    (run_dir / "state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
