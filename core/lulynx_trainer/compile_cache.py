"""Local torch.compile cache layout helpers for Lulynx.

This module keeps compile-cache management deliberately local-first:

- cache assets stay on the current machine
- directory layout is stable and human-readable
- manifests explain why a cache bucket is reused or bypassed
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

DEFAULT_COMPILE_CACHE_ROOT = "model.cache"


def _slugify(value: Any, *, default: str = "unknown", max_len: int = 96) -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    if not text:
        text = default
    return text[:max_len]


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _normalize_model_name(model_path: str, *, route: str) -> str:
    raw = Path(str(model_path or "")).stem
    if not raw:
        raw = f"{route or 'unknown'}-model"
    return _slugify(raw, default=f"{route or 'unknown'}-model")


def _torch_cuda_tag() -> str:
    torch_ver = _slugify(getattr(torch, "__version__", "unknown"), default="torch")
    cuda_ver = _slugify(getattr(torch.version, "cuda", "cpu"), default="cpu")
    return f"torch-{torch_ver}_cu-{cuda_ver}"


def _gpu_tag(device: str) -> str:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            gpu_name = torch.cuda.get_device_name(torch.device(device))
        except Exception:
            try:
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            except Exception:
                gpu_name = "cuda-gpu"
        return f"gpu-{_slugify(gpu_name, default='cuda-gpu')}"
    return "gpu-cpu"


def _shape_bucket(config: Any) -> str:
    model_type = str(getattr(getattr(config, "model_type", ""), "value", getattr(config, "model_type", "")) or "").lower()
    batch = max(int(getattr(config, "train_batch_size", getattr(config, "batch_size", 1)) or 1), 1)
    resolution = str(getattr(config, "resolution", "unknown") or "unknown").replace(",", "x").replace(" ", "")
    pieces = [f"bs-{batch}", f"res-{_slugify(resolution, default='unknown', max_len=48)}"]
    shape_strategy = _slugify(str(getattr(config, "compile_shape_strategy", "auto") or "auto"), default="auto", max_len=32)
    target_strategy = _slugify(str(getattr(config, "compile_target_strategy", "auto") or "auto"), default="auto", max_len=32)
    pieces.append(f"shape-{shape_strategy}")
    pieces.append(f"target-{target_strategy}")
    if model_type == "anima":
        pieces.append(f"txt-{max(int(getattr(config, 'anima_fixed_text_tokens', 0) or 0), 0)}")
        pieces.append(f"vis-{max(int(getattr(config, 'anima_fixed_visual_tokens', 0) or 0), 0)}")
    elif model_type == "newbie":
        pieces.append(f"txt-{max(int(getattr(config, 'newbie_fixed_text_tokens', 0) or 0), 0)}")
        pieces.append(f"vis-{max(int(getattr(config, 'newbie_fixed_visual_tokens', 0) or 0), 0)}")
    else:
        pieces.append(f"txt-{max(int(getattr(config, 'max_token_length', 0) or 0), 0)}")
    return "_".join(pieces)


def _compile_fingerprint(config: Any) -> Dict[str, Any]:
    return {
        "torch_compile": _boolish(getattr(config, "torch_compile", False), default=False),
        "backend": str(getattr(config, "torch_compile_backend", "inductor") or "inductor"),
        "mode": str(getattr(config, "torch_compile_mode", "default") or "default"),
        "dynamic": _boolish(getattr(config, "torch_compile_dynamic", False), default=False),
        "fullgraph": _boolish(getattr(config, "torch_compile_fullgraph", False), default=False),
        "scope": str(getattr(config, "torch_compile_scope", "") or ""),
        "anima_scope": str(getattr(config, "anima_compile_scope", "") or ""),
        "shape_strategy": str(getattr(config, "compile_shape_strategy", "auto") or "auto"),
        "target_strategy": str(getattr(config, "compile_target_strategy", "auto") or "auto"),
    }


def _fingerprint_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


@dataclass(frozen=True)
class CompileCacheLayout:
    root: Path
    model_dir: Path
    env_dir: Path
    compile_dir: Path
    inductor_dir: Path
    triton_dir: Path
    manifest_path: Path
    manifest: Dict[str, Any]

    def log_lines(self) -> Iterable[str]:
        yield (
            "[compile-cache] "
            f"root={self.root} model_dir={self.model_dir.name} env_dir={self.env_dir.name} "
            f"compile_dir={self.compile_dir.name}"
        )
        yield (
            "[compile-cache] "
            f"TORCHINDUCTOR_CACHE_DIR={self.inductor_dir} TRITON_CACHE_DIR={self.triton_dir}"
        )


def _python_dev_artifacts() -> Dict[str, Path]:
    base = Path(sys.executable).resolve().parent
    include_dir = base / "Include"
    libs_dir = base / "libs"
    libs_dir_alt = base / "Libs"
    version_tag = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    return {
        "include_dir": include_dir,
        "python_h": include_dir / "Python.h",
        "libs_dir": libs_dir,
        "libs_dir_alt": libs_dir_alt,
        "python_lib": libs_dir / version_tag,
        "python_lib_alt": libs_dir_alt / version_tag,
    }


def build_compile_cache_layout(
    config: Any,
    *,
    route: str,
    model_path: str,
    device: str,
) -> CompileCacheLayout:
    root = Path(str(getattr(config, "compile_cache_root", DEFAULT_COMPILE_CACHE_ROOT) or DEFAULT_COMPILE_CACHE_ROOT)).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    route_name = str(route or "unknown").strip().lower()
    model_name = _normalize_model_name(model_path, route=route_name)
    env_tag = _torch_cuda_tag()
    gpu_tag = _gpu_tag(device)
    compile_fp = _compile_fingerprint(config)
    compile_sig = {
        "route": route_name,
        "compile": compile_fp,
        "shape": _shape_bucket(config),
    }
    compile_hash = _fingerprint_hash(compile_sig)
    compile_dir_name = (
        f"route-{_slugify(route_name, default='unknown')}_"
        f"backend-{_slugify(compile_fp['backend'], default='inductor')}_"
        f"scope-{_slugify(compile_fp['anima_scope'] or compile_fp['scope'] or 'default', default='default')}_"
        f"{_slugify(compile_sig['shape'], default='shape')}_"
        f"{compile_hash}"
    )

    model_dir = root / model_name
    env_dir = model_dir / env_tag / gpu_tag
    compile_dir = env_dir / compile_dir_name
    inductor_dir = compile_dir / "inductor"
    triton_dir = compile_dir / "triton"
    manifest_path = compile_dir / "manifest.json"
    manifest = {
        "schema": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "route": route_name,
        "model_name": model_name,
        "model_path": str(model_path or ""),
        "device": str(device or ""),
        "gpu_name": None if gpu_tag == "gpu-cpu" else gpu_tag[len("gpu-"):],
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(torch.version, "cuda", "cpu"),
        "compile": compile_fp,
        "shape_bucket": compile_sig["shape"],
        "fingerprint": compile_hash,
        "env": {
            "TORCHINDUCTOR_CACHE_DIR": str(inductor_dir),
            "TRITON_CACHE_DIR": str(triton_dir),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        },
    }
    return CompileCacheLayout(
        root=root,
        model_dir=model_dir,
        env_dir=env_dir,
        compile_dir=compile_dir,
        inductor_dir=inductor_dir,
        triton_dir=triton_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def prepare_compile_cache_environment(layout: CompileCacheLayout, *, reuse: bool = True) -> Dict[str, str]:
    layout.compile_dir.mkdir(parents=True, exist_ok=True)
    layout.inductor_dir.mkdir(parents=True, exist_ok=True)
    layout.triton_dir.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(
        json.dumps(layout.manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env_updates = {
        "TORCHINDUCTOR_CACHE_DIR": str(layout.inductor_dir),
        "TRITON_CACHE_DIR": str(layout.triton_dir),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
    if _boolish(reuse, default=True):
        env_updates.setdefault("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC", "1")
    for key, value in env_updates.items():
        os.environ[key] = value
    return env_updates


def compile_cache_status(layout: CompileCacheLayout) -> Dict[str, Any]:
    manifest_exists = layout.manifest_path.exists()
    inductor_exists = layout.inductor_dir.exists()
    triton_exists = layout.triton_dir.exists()
    inductor_files = sum(1 for _ in layout.inductor_dir.rglob("*")) if inductor_exists else 0
    triton_files = sum(1 for _ in layout.triton_dir.rglob("*")) if triton_exists else 0
    hit = manifest_exists and (inductor_files > 0 or triton_files > 0)
    return {
        "hit": hit,
        "manifest_exists": manifest_exists,
        "inductor_files": inductor_files,
        "triton_files": triton_files,
    }


def compile_cache_profile(
    layout: CompileCacheLayout,
    *,
    status_before: Dict[str, Any],
    status_after: Optional[Dict[str, Any]] = None,
    reuse: bool = True,
    env_updates: Optional[Dict[str, str]] = None,
    blocker: Optional[str] = None,
) -> Dict[str, Any]:
    before_hit = bool(status_before.get("hit", False))
    after = dict(status_after or status_before)
    state = "blocked" if blocker else ("hot" if before_hit else "cold")
    return {
        "enabled": True,
        "state": state,
        "reuse_requested": bool(reuse),
        "hit_before_prepare": before_hit,
        "hit_after_prepare": bool(after.get("hit", False)),
        "manifest_exists_before": bool(status_before.get("manifest_exists", False)),
        "manifest_exists_after": bool(after.get("manifest_exists", False)),
        "inductor_files_before": int(status_before.get("inductor_files", 0) or 0),
        "triton_files_before": int(status_before.get("triton_files", 0) or 0),
        "inductor_files_after": int(after.get("inductor_files", 0) or 0),
        "triton_files_after": int(after.get("triton_files", 0) or 0),
        "root": str(layout.root),
        "compile_dir": str(layout.compile_dir),
        "manifest_path": str(layout.manifest_path),
        "fingerprint": str(layout.manifest.get("fingerprint", "") or ""),
        "shape_bucket": str(layout.manifest.get("shape_bucket", "") or ""),
        "route": str(layout.manifest.get("route", "") or ""),
        "backend": str((layout.manifest.get("compile") or {}).get("backend", "") or ""),
        "scope": str(
            (layout.manifest.get("compile") or {}).get("anima_scope")
            or (layout.manifest.get("compile") or {}).get("scope")
            or ""
        ),
        "env": dict(env_updates or {}),
        "blocker": blocker or "",
    }


def compile_cache_cold_bucket_blocker(layout: CompileCacheLayout) -> Optional[str]:
    artifacts = _python_dev_artifacts()
    if not artifacts["python_h"].exists():
        return f"portable Python headers missing: {artifacts['python_h']}"
    if not artifacts["python_lib"].exists() and not artifacts["python_lib_alt"].exists():
        return (
            "portable Python import library missing: "
            f"{artifacts['python_lib']} or {artifacts['python_lib_alt']}"
        )
    return None
