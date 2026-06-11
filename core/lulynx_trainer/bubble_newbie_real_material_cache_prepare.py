"""Prepare cache-ready Newbie real-material source slices for P60 canaries."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .bubble_real_material_canary import prepare_real_material_canary_source
from .bubble_real_material_source_scan import scan_real_material_source


NEWBIE_REAL_MATERIAL_CACHE_PREPARE_REPORT = "bubble_newbie_real_material_cache_prepare_v0"
NEWBIE_REAL_MATERIAL_CACHE_RUN_REPORT = "bubble_newbie_real_material_cache_run_v0"
_PIPE_DONE = object()


def _is_same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def prepare_newbie_real_material_cache_work_dir(
    source: Path,
    work_dir: Path,
    *,
    samples: int = 8,
    sample_offset: int = 0,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Copy a real-material window into a separate work dir before cache build.

    This helper deliberately does not build cache itself.  It prepares the exact
    image/caption/cache-sidecar slice that the P60 real-material canary will use,
    then scans the copy with the same readiness scanner.  Heavy model loading is
    kept in the opt-in devtool.
    """

    source = Path(source).resolve()
    work_dir = Path(work_dir).resolve()
    if _is_same_path(source, work_dir):
        raise ValueError("work_dir must be a separate copy, not the original source directory")
    if work_dir.exists() and any(work_dir.iterdir()) and not allow_existing:
        raise FileExistsError(f"work_dir is not empty: {work_dir}")

    manifest = prepare_real_material_canary_source(
        source,
        work_dir,
        family="newbie",
        samples=max(int(samples), 1),
        sample_offset=max(int(sample_offset), 0),
        native_cache_mode="cache_first",
        label="newbie_real_material_cache_prepare",
    )
    scan = scan_real_material_source(
        work_dir,
        samples=max(int(samples), 1),
        sample_offset=0,
        families=("sdxl", "anima", "newbie"),
    )
    newbie = next(
        (item for item in scan.get("family_readiness", []) if item.get("family") == "newbie"),
        {},
    )
    return {
        "schema_version": 1,
        "report": NEWBIE_REAL_MATERIAL_CACHE_PREPARE_REPORT,
        "status": "prepared_source_slice",
        "source_data": str(source),
        "work_dir": str(work_dir),
        "samples": int(manifest.get("samples") or 0),
        "sample_offset": int(manifest.get("sample_offset") or 0),
        "source_manifest_sha1": str(manifest.get("source_manifest_sha1") or ""),
        "prebuild_cache_state": str(manifest.get("cache_state") or ""),
        "prebuild_cache_has_family_cache": bool(manifest.get("cache_has_family_cache")),
        "prebuild_newbie_readiness": dict(newbie),
        "source_fixture": manifest,
        "scan": scan,
    }


def write_newbie_real_material_cache_prepare_report(report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_newbie_real_material_cache_prepare_command(
    *,
    python_exe: Path | str,
    prepare_script: Path,
    source: Path,
    work_dir: Path,
    prepare_report_path: Path,
    samples: int = 1,
    sample_offset: int = 0,
    model_dir: Path | None = None,
    resolution: int = 64,
    device: str = "cuda",
    trust_remote_code: bool = False,
    skip_clip: bool = False,
    allow_existing: bool = True,
    no_force_cache: bool = False,
) -> list[str]:
    """Build the heavy Newbie cache prepare command without running it."""

    command = [
        str(python_exe),
        str(Path(prepare_script)),
        "--source-data",
        str(Path(source)),
        "--work-dir",
        str(Path(work_dir)),
        "--samples",
        str(max(int(samples), 1)),
        "--sample-offset",
        str(max(int(sample_offset), 0)),
        "--resolution",
        str(max(int(resolution), 1)),
        "--device",
        str(device or "cuda"),
        "--out",
        str(Path(prepare_report_path)),
    ]
    if model_dir is not None:
        command.extend(["--model-dir", str(Path(model_dir))])
    if trust_remote_code:
        command.append("--trust-remote-code")
    if skip_clip:
        command.append("--skip-clip")
    if allow_existing:
        command.append("--allow-existing")
    if no_force_cache:
        command.append("--no-force-cache")
    return command


def build_newbie_real_material_runtime_cache_config(
    *,
    work_dir: Path,
    output_dir: Path,
    model_dir: Path,
    resolution: int = 64,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    """Build an entry_train config that warms Newbie cache through runtime.

    This uses the normal trainer boundary: ``native_cache_mode=rebuild_cache``
    prepares the cache, while ``newbie_force_cache_only`` prevents a real train
    loop after the cache-first dataset is prepared.
    """

    model_dir = Path(model_dir)
    return {
        "schema_id": "newbie-lora",
        "training_type": "lora",
        "model_type": "newbie",
        "model_arch": "newbie",
        "train_data_dir": str(Path(work_dir)),
        "output_dir": str(Path(output_dir)),
        "output_name": "newbie_real_material_warm_cache",
        "pretrained_model_name_or_path": str(model_dir),
        "newbie_transformer_path": str(model_dir / "transformer"),
        "newbie_gemma_model_path": str(model_dir / "text_encoder"),
        "newbie_clip_model_path": str(model_dir / "clip_model"),
        "newbie_vae_path": str(model_dir / "vae"),
        "native_cache_mode": "rebuild_cache",
        "newbie_rebuild_cache": True,
        "newbie_force_cache_only": True,
        "use_cache": True,
        "resolution": max(int(resolution), 1),
        "train_batch_size": 1,
        "max_train_steps": 1,
        "max_train_epochs": 1,
        "network_dim": 1,
        "trust_remote_code": bool(trust_remote_code),
    }


def write_newbie_real_material_runtime_cache_config(config: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_newbie_real_material_runtime_cache_command(
    *,
    python_exe: Path | str,
    entry_train_script: Path,
    config_path: Path,
) -> list[str]:
    return [str(python_exe), "-u", str(Path(entry_train_script)), "--config", str(Path(config_path))]


def _append_tail(lines: list[str], line: str, *, limit: int) -> None:
    lines.append(str(line).rstrip())
    del lines[: max(len(lines) - max(int(limit), 1), 0)]


def _pipe_reader(pipe: Any, output: "queue.Queue[str | object]") -> None:
    try:
        for line in pipe:
            output.put(str(line))
    finally:
        output.put(_PIPE_DONE)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
    except Exception:
        return {}
    return {}


def _tail_lines(path: Path, *, limit: int = 40) -> list[str]:
    try:
        if not path.is_file():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(int(limit), 1):]
    except Exception:
        return []


def _last_stage_event(stdout_tail: Sequence[str], progress_snapshot: dict[str, Any]) -> dict[str, Any]:
    last_event = progress_snapshot.get("last_event")
    if isinstance(last_event, dict) and last_event:
        return {"source": "progress_snapshot", **dict(last_event)}
    for line in reversed(list(stdout_tail)):
        text = str(line).strip()
        if "loader_stage=" in text or "load_" in text or "build_" in text:
            return {"source": "stdout_tail", "line": text}
    return {}


def _runner_status(
    *,
    timed_out: bool,
    returncode: int,
    child_report: dict[str, Any],
) -> str:
    if timed_out:
        return "timeout"
    child_status = str(child_report.get("status") or "").strip()
    if child_status:
        return child_status if returncode == 0 else f"process_failed_{child_status}"
    if returncode != 0:
        return "process_failed_report_missing"
    return "process_succeeded_report_missing"


def run_newbie_real_material_cache_prepare_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    prepare_report_path: Path,
    runner_report_path: Path,
    log_path: Path,
    progress_path: Path | None = None,
    state_path: Path | None = None,
    events_path: Path | None = None,
    timeout_seconds: int = 900,
    tail_limit: int = 160,
) -> dict[str, Any]:
    """Run the heavy Newbie cache prepare script in a timed subprocess.

    The caller still owns the actual cache build logic through ``command``.
    This wrapper only isolates the process, streams logs to disk, and writes an
    auditable runner report even when the child hangs during model loading.
    """

    command_list = [str(part) for part in command]
    cwd = Path(cwd)
    prepare_report_path = Path(prepare_report_path)
    runner_report_path = Path(runner_report_path)
    log_path = Path(log_path)
    runner_report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    last_output_at = start
    timed_out = False
    stdout_tail: list[str] = []
    returncode = -1
    output_queue: "queue.Queue[str | object]" = queue.Queue()

    with log_path.open("a", encoding="utf-8") as log:
        log.write("[bubble-newbie-cache-runner] command=" + json.dumps(command_list, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            command_list,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        reader = threading.Thread(target=_pipe_reader, args=(process.stdout, output_queue), daemon=True)
        reader.start()
        deadline = time.perf_counter() + max(int(timeout_seconds), 0) if int(timeout_seconds) > 0 else None
        stream_done = False
        while True:
            try:
                item = output_queue.get(timeout=0.2)
            except queue.Empty:
                item = None
            if item is _PIPE_DONE:
                stream_done = True
            elif item is not None:
                line = str(item)
                last_output_at = time.perf_counter()
                log.write(line)
                log.flush()
                _append_tail(stdout_tail, line, limit=max(int(tail_limit), 1))
            if deadline is not None and time.perf_counter() > deadline:
                timed_out = True
                _append_tail(
                    stdout_tail,
                    f"[bubble-newbie-cache-runner] timeout_after_seconds={timeout_seconds}",
                    limit=max(int(tail_limit), 1),
                )
                log.write(f"[bubble-newbie-cache-runner] timeout_after_seconds={timeout_seconds}\n")
                log.flush()
                process.kill()
                process.wait()
                break
            if process.poll() is not None and stream_done:
                break
        while not output_queue.empty():
            rest = output_queue.get_nowait()
            if rest is _PIPE_DONE:
                continue
            line = str(rest)
            last_output_at = time.perf_counter()
            log.write(line)
            _append_tail(stdout_tail, line, limit=max(int(tail_limit), 1))
        try:
            process.stdout.close()
        except Exception:
            pass
        reader.join(timeout=1.0)
        log.flush()
        returncode = int(process.returncode if process.returncode is not None else -9)

    child_report = _read_json(prepare_report_path)
    resolved_progress_path = Path(progress_path) if progress_path is not None else prepare_report_path.with_suffix(".progress.json")
    progress_snapshot = _read_json(resolved_progress_path)
    state_snapshot = _read_json(Path(state_path)) if state_path is not None else {}
    events_tail = _tail_lines(Path(events_path), limit=40) if events_path is not None else []
    status = _runner_status(timed_out=timed_out, returncode=returncode, child_report=child_report)
    report = {
        "schema_version": 1,
        "report": NEWBIE_REAL_MATERIAL_CACHE_RUN_REPORT,
        "status": status,
        "success": status == "cache_ready",
        "command": command_list,
        "cwd": str(cwd),
        "returncode": returncode,
        "timed_out": bool(timed_out),
        "timeout_seconds": int(timeout_seconds),
        "wall_seconds": round(time.perf_counter() - start, 3),
        "last_output_age_seconds": round(max(time.perf_counter() - last_output_at, 0.0), 3),
        "log_path": str(log_path),
        "prepare_report_path": str(prepare_report_path),
        "prepare_report_exists": prepare_report_path.is_file(),
        "prepare_report_status": str(child_report.get("status") or ""),
        "stdout_tail": stdout_tail,
        "last_stage_event": _last_stage_event(stdout_tail, progress_snapshot),
        "progress_path": str(resolved_progress_path),
        "progress_snapshot": progress_snapshot,
        "state_path": str(state_path) if state_path is not None else "",
        "state_snapshot": state_snapshot,
        "events_path": str(events_path) if events_path is not None else "",
        "events_tail": events_tail,
        "child_report": child_report,
    }
    runner_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


__all__ = [
    "NEWBIE_REAL_MATERIAL_CACHE_PREPARE_REPORT",
    "NEWBIE_REAL_MATERIAL_CACHE_RUN_REPORT",
    "build_newbie_real_material_cache_prepare_command",
    "build_newbie_real_material_runtime_cache_command",
    "build_newbie_real_material_runtime_cache_config",
    "prepare_newbie_real_material_cache_work_dir",
    "run_newbie_real_material_cache_prepare_subprocess",
    "write_newbie_real_material_runtime_cache_config",
    "write_newbie_real_material_cache_prepare_report",
]
