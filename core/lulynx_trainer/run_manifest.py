# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Run manifest and resume preflight helpers.

The manifest is a small JSON authority for native training runs.  It is not a
replacement for optimizer state; it records enough context to make resume
decisions explainable before any heavy torch objects are loaded.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


RUN_MANIFEST_VERSION = 1
RUN_MANIFEST_NAME = "run_manifest.json"

_CONFIG_COMPARE_KEYS = (
    "model_arch",
    "model_type",
    "training_type",
    "output_name",
    "train_data_dir",
    "eval_data_dir",
    "pretrained_model_name_or_path",
    "base_model_path",
    "model_train_type",
    "network_module",
    "lora_type",
    "newbie_adapter_type",
    "native_cache_mode",
)

_PATH_FINGERPRINT_KEYS = ("train_data_dir", "eval_data_dir", "reg_data_dir")


def _native_fingerprint_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_FINGERPRINT", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def _load_native_fingerprint_api() -> Any:
    artifact_dir = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if artifact_dir and artifact_dir not in sys.path and Path(artifact_dir).is_dir():
        sys.path.insert(0, artifact_dir)
    try:
        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def _native_fingerprint_api() -> Any:
    if _native_fingerprint_disabled():
        return None
    native = _load_native_fingerprint_api()
    if not hasattr(native, "fingerprint_path"):
        return None
    return native


@dataclass(frozen=True)
class ResumeManifestReport:
    manifest_path: Path
    found: bool
    ok: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    previous_global_step: int = 0
    previous_status: str = ""


def manifest_path_for(output_dir: str | Path) -> Path:
    return Path(output_dir) / RUN_MANIFEST_NAME


def write_run_manifest(
    output_dir: str | Path,
    *,
    config: Any,
    status: str,
    epoch: int = 0,
    global_step: int = 0,
    total_steps: int = 0,
    steps_per_epoch: int = 0,
    checkpoint_path: str = "",
    state_path: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write or update ``run_manifest.json`` in *output_dir*."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = manifest_path_for(out)
    config_snapshot = _config_snapshot(config)
    previous = _load_json(path)
    history = list(previous.get("history", [])) if isinstance(previous.get("history"), list) else []
    event = {
        "time": _now(),
        "status": str(status),
        "epoch": int(epoch or 0),
        "global_step": int(global_step or 0),
    }
    if checkpoint_path:
        event["checkpoint_path"] = str(checkpoint_path)
    if state_path:
        event["state_path"] = str(state_path)
    history.append(event)
    history = history[-100:]

    payload: Dict[str, Any] = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "updated_at": _now(),
        "status": str(status),
        "epoch": int(epoch or 0),
        "global_step": int(global_step or 0),
        "total_steps": int(total_steps or 0),
        "steps_per_epoch": int(steps_per_epoch or 0),
        "checkpoint_path": str(checkpoint_path or previous.get("checkpoint_path", "")),
        "state_path": str(state_path or previous.get("state_path", "")),
        "config": config_snapshot,
        "path_fingerprints": _path_fingerprints(config_snapshot),
        "history": history,
    }
    if extra:
        payload["extra"] = _jsonable(dict(extra))

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def find_run_manifest_for_resume(resume_path: str | Path) -> Path:
    """Find the run manifest associated with a resume file or directory."""

    path = Path(resume_path)
    if path.is_dir():
        direct = path / RUN_MANIFEST_NAME
        if direct.is_file():
            return direct
        parent = path.parent / RUN_MANIFEST_NAME
        return parent
    if path.is_file():
        return path.parent / RUN_MANIFEST_NAME
    return path.parent / RUN_MANIFEST_NAME if path.parent != Path(".") else manifest_path_for(path)


def validate_resume_manifest(
    resume_path: str | Path,
    *,
    config: Any,
    strict: bool = False,
) -> ResumeManifestReport:
    """Validate a resume target against its recorded run manifest."""

    manifest_path = find_run_manifest_for_resume(resume_path)
    if not manifest_path.is_file():
        return ResumeManifestReport(
            manifest_path=manifest_path,
            found=False,
            ok=not strict,
            warnings=(f"run manifest not found for resume target: {manifest_path}",),
            errors=("run manifest missing",) if strict else (),
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ResumeManifestReport(
            manifest_path=manifest_path,
            found=True,
            ok=False,
            errors=(f"run manifest cannot be read: {type(exc).__name__}: {exc}",),
        )

    errors = []
    warnings = []
    notes = []
    if int(manifest.get("manifest_version", 0) or 0) != RUN_MANIFEST_VERSION:
        warnings.append(
            f"run manifest version mismatch: expected {RUN_MANIFEST_VERSION}, got {manifest.get('manifest_version')!r}"
        )

    current = _config_snapshot(config)
    previous = manifest.get("config", {})
    if not isinstance(previous, dict):
        previous = {}
        errors.append("run manifest config snapshot is invalid")

    for key in _CONFIG_COMPARE_KEYS:
        old = _normalize_compare(previous.get(key))
        new = _normalize_compare(current.get(key))
        if old and new and old != new:
            message = f"resume config mismatch for {key}: manifest={old!r}, current={new!r}"
            if key in {"model_arch", "model_type", "training_type", "output_name"}:
                errors.append(message)
            else:
                warnings.append(message)

    previous_fps = manifest.get("path_fingerprints", {})
    current_fps = _path_fingerprints(current)
    if isinstance(previous_fps, dict):
        for key in _PATH_FINGERPRINT_KEYS:
            old_fp = previous_fps.get(key)
            new_fp = current_fps.get(key)
            if isinstance(old_fp, dict) and isinstance(new_fp, dict):
                if _fingerprint_changed(old_fp, new_fp):
                    warnings.append(f"resume path fingerprint changed for {key}: {new_fp.get('path', '')}")
    else:
        warnings.append("run manifest path_fingerprints section is invalid")

    previous_step = int(manifest.get("global_step", 0) or 0)
    previous_status = str(manifest.get("status", "") or "")
    notes.append(f"run manifest found: step={previous_step}, status={previous_status or 'unknown'}")
    return ResumeManifestReport(
        manifest_path=manifest_path,
        found=True,
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        notes=tuple(notes),
        previous_global_step=previous_step,
        previous_status=previous_status,
    )


def _config_snapshot(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        data = config.model_dump(mode="json")
    elif hasattr(config, "dict"):
        data = config.dict()
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        data = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    return _jsonable(data)


def _path_fingerprints(config_snapshot: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for key in _PATH_FINGERPRINT_KEYS:
        raw = str(config_snapshot.get(key, "") or "").strip()
        if not raw:
            continue
        result[key] = fingerprint_path(raw)
    return result


def fingerprint_path(path: str | Path, *, max_files: int = 20000) -> Dict[str, Any]:
    """Return a cheap recursive filesystem fingerprint for a file or directory."""

    native = _native_fingerprint_api()
    if native is not None:
        return native.fingerprint_path(str(path), int(max_files))

    p = Path(path)
    result: Dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir(),
        "file_count": 0,
        "total_size": 0,
        "latest_mtime_ns": 0,
        "truncated": False,
    }
    if not p.exists():
        return result
    files: Iterable[Path]
    if p.is_file():
        files = (p,)
    else:
        files = (child for child in p.rglob("*") if child.is_file())
    for index, file_path in enumerate(files):
        if index >= max_files:
            result["truncated"] = True
            break
        try:
            stat = file_path.stat()
        except OSError:
            continue
        result["file_count"] += 1
        result["total_size"] += int(stat.st_size)
        result["latest_mtime_ns"] = max(int(result["latest_mtime_ns"]), int(stat.st_mtime_ns))
    return result


def _fingerprint_changed(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    for key in ("exists", "is_dir", "file_count", "total_size", "latest_mtime_ns"):
        if old.get(key) != new.get(key):
            return True
    return False


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_compare(value: Any) -> str:
    return str(_jsonable(value) or "").strip().replace("\\", "/")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
