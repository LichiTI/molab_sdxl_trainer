"""Whitelisted clean-room tool execution helpers for compatibility routes."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ALLOWED_SCRIPT_TOOLS = {
    "tools/canny": "canny",
    "tools/face_crop": "face_crop",
    "tools/latent_upscale": "latent_upscale",
    "tools/diagnostic_card": "diagnostic_card",
}

ToolRunner = Callable[[str, dict[str, Any]], dict[str, Any]]
ThreadFactory = Callable[[Callable[[], None]], Any]


def resolve_script_action(script_name: str) -> str:
    if not script_name:
        raise ValueError("Missing script_name")
    if not script_name.endswith(".py"):
        raise ValueError(f"Script must be a .py file: {script_name}")
    tool_id = script_name[:-3].replace("\\", "/")
    action = ALLOWED_SCRIPT_TOOLS.get(tool_id)
    if not action:
        raise ValueError(f"Script not registered as a Lulynx clean-room tool: {script_name}")
    return action


def run_registered_tool(action: str, params: dict[str, Any]) -> dict[str, Any]:
    from backend.core.tools.image_utilities import canny_edges, face_crop_rotate, latent_upscale_preview

    input_path = str(params.get("input_path") or params.get("path") or "")
    output_path = str(params.get("output_path") or "") or None
    if action == "canny":
        return canny_edges(input_path, output_path, int(params.get("low", 80)), int(params.get("high", 160)))
    if action == "face_crop":
        return face_crop_rotate(
            input_path,
            output_path,
            int(params.get("target_size", 512)),
            float(params.get("rotate_degrees", 0)),
        )
    if action == "latent_upscale":
        return latent_upscale_preview(input_path, output_path, float(params.get("scale", 2.0)))
    if action == "diagnostic_card":
        return {"ready": True, "message": "Diagnostic card generation is exposed through /api/tools/diagnostic-card/check."}
    raise ValueError(f"Unknown registered tool: {action}")


def sample_prompt_payload(config: dict[str, Any]) -> dict[str, str]:
    prompts = config.get("sample_prompts", "")
    if isinstance(prompts, list):
        prompt = "\n".join(str(item) for item in prompts if item)
    elif isinstance(prompts, str):
        prompt = prompts
    else:
        prompt = str(prompts) if prompts else ""
    return {"prompt": prompt}


def submit_registered_tool_task(
    params: dict[str, Any],
    *,
    telemetry_reader: Any,
    job_manager: Any | None = None,
    tool_runner: ToolRunner = run_registered_tool,
    thread_factory: ThreadFactory | None = None,
    task_id: str | None = None,
) -> dict[str, str]:
    script_name = str(params.get("script_name", "") or "")
    action = resolve_script_action(script_name)
    run_id = task_id or str(uuid.uuid4())
    log_file = _prepare_log_file(telemetry_reader, run_id)

    def write_log(message: str) -> None:
        try:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except Exception:
            pass

    def run_task() -> None:
        job = _register_job(job_manager, run_id, script_name)
        write_log(f"[run_script] started: {script_name}")
        try:
            result = tool_runner(action, params)
            write_log(json.dumps(result, ensure_ascii=False))
            _finish_job(job, result)
            write_log("[run_script] finished")
        except Exception as exc:
            _fail_job(job, str(exc))
            write_log(f"[run_script] error: {exc}")

    factory = thread_factory or _default_thread_factory
    runner = factory(run_task)
    start = getattr(runner, "start", None)
    if callable(start):
        start()
    return {"task_id": run_id}


def _prepare_log_file(telemetry_reader: Any, task_id: str) -> Path:
    run_dir = Path(telemetry_reader._runs_dir) / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "output.log"


def _default_thread_factory(target: Callable[[], None]) -> threading.Thread:
    return threading.Thread(target=target, daemon=True)


def _register_job(job_manager: Any | None, task_id: str, script_name: str) -> Any | None:
    if not job_manager:
        return None
    from backend.core.job_manager import Job, JobStatus, JobType

    job = Job(id=task_id, type=JobType.GENERIC, name=f"Script: {script_name}", status=JobStatus.RUNNING)
    job_manager._jobs[task_id] = job
    return job


def _finish_job(job: Any | None, result: dict[str, Any]) -> None:
    if not job:
        return
    from backend.core.job_manager import JobStatus

    job.finished_at = datetime.now()
    job.status = JobStatus.COMPLETED
    job.progress = 1.0
    job.metadata["result"] = result


def _fail_job(job: Any | None, error: str) -> None:
    if not job:
        return
    from backend.core.job_manager import JobStatus

    job.status = JobStatus.FAILED
    job.error = error

