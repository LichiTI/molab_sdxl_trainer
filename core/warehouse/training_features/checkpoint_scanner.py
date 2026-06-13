"""
Pure checkpoint artifact scanning helpers for training resume UI/API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import importlib
import os
from pathlib import Path
import re
import sys
from typing import Any

from core.warehouse.training_features.resume_guard import ResumeGuard


_STATE_DIR_RE = re.compile(r"(?:^|-)state(?:-|$)", re.IGNORECASE)
_STEP_RE = re.compile(
    r"(?:step[s_-]*|checkpoint[s_-]*|epoch[s_-]*|ckpt[s_-]*)(\d+)",
    re.IGNORECASE,
)


def native_checkpoint_scan_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_CHECKPOINT_SCAN", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def load_native_checkpoint_scan_api() -> Any:
    artifact_dir = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if artifact_dir and artifact_dir not in sys.path and Path(artifact_dir).is_dir():
        sys.path.insert(0, artifact_dir)
    try:
        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def native_checkpoint_scan_api() -> Any:
    if native_checkpoint_scan_disabled():
        return None
    native = load_native_checkpoint_scan_api()
    if not hasattr(native, "scan_checkpoint_candidates"):
        return None
    return native


def extract_step_from_name(name: str) -> int | None:
    match = _STEP_RE.search(name)
    if match:
        return int(match.group(1))
    if name.isdigit():
        return int(name)
    return None


def scan_state_directories(output_path: Path, scheduler_optional: bool) -> list[dict[str, Any]]:
    guard = ResumeGuard()
    results: list[dict[str, Any]] = []
    if not output_path.is_dir():
        return results
    native_candidates = scan_checkpoint_candidates_native(output_path)
    state_candidates = list(native_candidates.get("state_dirs", []) or []) if native_candidates is not None else []
    if not state_candidates:
        state_candidates = [child for child in output_path.iterdir() if child.is_dir() and _STATE_DIR_RE.search(child.name)]
    for child in state_candidates:
        child_path = Path(str(child.get("path", "") or "")) if isinstance(child, dict) else Path(child)
        if not child_path.is_dir():
            continue
        validation = guard.validate_state_dir(child_path, scheduler_optional=scheduler_optional)
        try:
            timestamp = datetime.fromtimestamp(child_path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            timestamp = ""
        results.append(
            {
                "path": str(child_path),
                "name": child_path.name,
                "kind": "state_dir",
                "step": extract_step_from_name(child_path.name),
                "timestamp": timestamp,
                "has_state": True,
                "state_complete": validation.state_complete,
                "valid": validation.ok,
                "message": validation.message,
            }
        )
    results.sort(key=lambda item: (item.get("step") is not None, item.get("step") or 0, item.get("timestamp") or ""), reverse=True)
    return results


def scan_checkpoint_files(output_path: Path) -> list[dict[str, Any]]:
    guard = ResumeGuard()
    results: list[dict[str, Any]] = []
    native_candidates = scan_checkpoint_candidates_native(output_path)
    candidate_files = list(native_candidates.get("checkpoint_files", []) or []) if native_candidates is not None else []
    if candidate_files:
        checkpoint_entries = [
            {
                "path": Path(str(item.get("path", "") or "")),
                "step": item.get("step"),
                "size_bytes": int(item.get("size_bytes", 0) or 0),
            }
            for item in candidate_files
            if isinstance(item, dict) and str(item.get("path", "") or "")
        ]
    else:
        report = guard.scan(output_path)
        checkpoint_entries = [
            {"path": entry.path, "step": entry.step, "size_bytes": entry.size_bytes}
            for entry in report.all_checkpoints
        ]
    for entry in checkpoint_entries:
        entry_path = Path(entry["path"])
        state_dir = output_path / f"{entry_path.stem}-state"
        state_file = output_path / f"{entry_path.stem}-state.pt"
        try:
            timestamp = datetime.fromtimestamp(entry_path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            timestamp = ""
        results.append(
            {
                "path": str(entry_path),
                "name": entry_path.name,
                "kind": "checkpoint_file",
                "step": entry.get("step"),
                "timestamp": timestamp,
                "size_bytes": entry.get("size_bytes", 0),
                "has_state": state_dir.is_dir() or state_file.is_file(),
            }
        )
    results.sort(key=lambda item: (item.get("step") is not None, item.get("step") or 0, item.get("timestamp") or ""), reverse=True)
    return results


def collect_checkpoint_info(
    output_path: Path,
    *,
    resume_field_type: str | None,
    scheduler_optional: bool,
) -> dict[str, Any]:
    file_items = scan_checkpoint_files(output_path)
    state_items = scan_state_directories(output_path, scheduler_optional)
    preferred_items = state_items if resume_field_type == "folder" else file_items
    latest = preferred_items[0] if preferred_items else None
    return {
        "resume_field_type": resume_field_type,
        "checkpoints": preferred_items,
        "count": len(preferred_items),
        "latest": latest,
        "suggested_resume": latest.get("path") if latest else None,
        "artifacts": {
            "checkpoint_files": file_items,
            "state_dirs": state_items,
        },
    }


def scan_checkpoint_candidates_native(output_path: Path) -> dict[str, Any] | None:
    native = native_checkpoint_scan_api()
    if native is None:
        return None
    try:
        result = native.scan_checkpoint_candidates(str(output_path))
    except Exception:
        return None
    return result if isinstance(result, dict) else None


__all__ = [
    "collect_checkpoint_info",
    "extract_step_from_name",
    "scan_checkpoint_files",
    "scan_state_directories",
]


# Backwards-compatible private names used by older tests/patch points.
_load_native_checkpoint_scan_api = load_native_checkpoint_scan_api
_native_checkpoint_scan_api = native_checkpoint_scan_api
_native_checkpoint_scan_disabled = native_checkpoint_scan_disabled
