"""Adapter helpers for legacy telemetry/log compatibility routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from core.services.native_module_loader import load_lulynx_native, native_with_entrypoints
except ImportError:
    from backend.core.services.native_module_loader import load_lulynx_native, native_with_entrypoints


def native_telemetry_logs_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_TELEMETRY_LOGS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_native_telemetry_logs_api() -> Any:
    return load_lulynx_native()


load_native_telemetry_logs_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_telemetry_logs_api() -> Any:
    if native_telemetry_logs_disabled():
        return None
    return native_with_entrypoints("list_telemetry_log_dirs", "list_telemetry_log_files")


def list_log_directories(reader: Any) -> dict[str, Any]:
    native_payload = list_log_directories_native(reader)
    if native_payload is not None:
        return native_payload
    dirs = []
    for run in reader.list_runs():
        run_id = str(run.get("run_id", "") or "")
        run_dir = Path(reader._runs_dir) / run_id
        if not run_dir.is_dir():
            continue
        stat = run_dir.stat()
        has_events = any(file_path.name.startswith("events.out") for file_path in run_dir.iterdir() if file_path.is_file())
        dirs.append({"name": run_id, "time": int(stat.st_mtime * 1000), "hasEvents": has_events})
    dirs.sort(key=lambda item: item["time"], reverse=True)
    return {"dirs": dirs}


def get_log_detail(reader: Any, run_id: str) -> dict[str, Any]:
    if not run_id:
        raise ValueError("Missing dir parameter")
    run_dir = Path(reader._runs_dir) / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Log directory '{run_id}' not found")
    native_payload = get_log_detail_native(reader, run_id)
    if native_payload is not None:
        return native_payload
    files = []
    for file_path in run_dir.iterdir():
        if not file_path.is_file():
            continue
        stat = file_path.stat()
        files.append({"name": file_path.name, "size": stat.st_size, "time": int(stat.st_mtime * 1000)})
    files.sort(key=lambda item: item["name"])
    return {"dir": run_id, "files": files}


def list_log_directories_native(reader: Any) -> dict[str, Any] | None:
    native = native_telemetry_logs_api()
    if native is None:
        return None
    try:
        payload = native.list_telemetry_log_dirs(str(Path(reader._runs_dir)))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    dirs = payload.get("dirs", [])
    if not isinstance(dirs, list):
        return None
    normalized = []
    for item in dirs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "")
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "time": int(item.get("time", 0) or 0),
                "hasEvents": bool(item.get("hasEvents", False)),
            }
        )
    return {"dirs": normalized}


def get_log_detail_native(reader: Any, run_id: str) -> dict[str, Any] | None:
    native = native_telemetry_logs_api()
    if native is None:
        return None
    try:
        payload = native.list_telemetry_log_files(str(Path(reader._runs_dir)), str(run_id))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    files = payload.get("files", [])
    if not isinstance(files, list):
        return None
    normalized = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "")
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "size": int(item.get("size", 0) or 0),
                "time": int(item.get("time", 0) or 0),
            }
        )
    normalized.sort(key=lambda item: item["name"])
    return {"dir": str(payload.get("dir", "") or run_id), "files": normalized}


def get_task_output_payload(reader: Any, task_id: str, *, tail: int = 100) -> dict[str, Any]:
    log_file = Path(reader._runs_dir) / task_id / "output.log"
    state = reader.get_state(task_id) or {}
    native_tail = read_log_tail_native(log_file, tail=tail)
    if native_tail is not None:
        lines = native_tail["lines"]
        total = native_tail["total"]
    else:
        lines = reader.get_log_tail(task_id, tail)
        total = _count_log_lines(log_file, fallback=len(lines))
    live_line = str(state.get("live_line", "") or "")
    status = str(state.get("status", "unknown") or "unknown")
    error = str(state.get("error", "") or "")
    if status.lower() == "failed" or error:
        if log_file.is_file() and not lines:
            lines = _read_log_tail(log_file, tail=tail)
            total = _count_log_lines(log_file, fallback=len(lines))
        lines = _prepend_failure_summary(lines, error=error)
    return {"lines": lines, "total": total, "live_line": live_line, "status": status, "error": error}


def read_log_tail_native(log_file: Path, *, tail: int) -> dict[str, Any] | None:
    native = native_telemetry_logs_api()
    if native is None or not hasattr(native, "read_telemetry_log_tail"):
        return None
    try:
        payload = native.read_telemetry_log_tail(str(log_file), int(tail))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    lines = payload.get("lines", [])
    if not isinstance(lines, list):
        return None
    return {"lines": [str(line) for line in lines], "total": int(payload.get("total", 0) or 0)}


def _count_log_lines(log_file: Path, *, fallback: int = 0) -> int:
    if not log_file.is_file():
        return 0
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return fallback


def _read_log_tail(log_file: Path, *, tail: int) -> list[str]:
    try:
        raw_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return raw_lines[-min(max(int(tail), 20), 200):]


def _prepend_failure_summary(lines: list[str], *, error: str = "") -> list[str]:
    summary_lines = ["====== 训练失败摘要 / Training failure summary ======"]
    if error:
        summary_lines.append(f"ERROR: {error}")
    if lines:
        summary_lines.append("------ output.log tail ------")
    existing = "\n".join(lines[:8])
    if "Training failure summary" in existing or "训练失败摘要" in existing:
        return lines
    return summary_lines + lines


# Backwards-compatible private names used by older tests/patch points.
_load_native_telemetry_logs_api = load_native_telemetry_logs_api
_native_telemetry_logs_api = native_telemetry_logs_api
_native_telemetry_logs_disabled = native_telemetry_logs_disabled
