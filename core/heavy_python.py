from __future__ import annotations

import json
import os
import sys
from pathlib import Path


HEAVY_ENV_CANDIDATES: tuple[str, ...] = (
    "python-sageattention-blackwell",
    "python_sagebwd_nvidia",
    "python-sagebwd-nvidia",
    "python-flexattention-blackwell",
    "python_flexattention_blackwell",
    "python_blackwell",
    "python-flashattention",
    "python_flashattention",
    "python-sageattention2",
    "python_sageattention2",
    "python-sageattention",
    "python_sageattention",
    "python_xpu_intel_sage",
    "python_xpu_intel",
    "python_rocm_amd_sage2",
    "python_rocm_amd",
    "python",
    "python-spargeattn2",
    "python_spargeattn2",
)


def resolve_heavy_python(backend_root: Path | str) -> Path:
    """Resolve the Python used by backend heavy worker routes.

    The old router helper only looked at active_env.json and then fell back to
    the current interpreter. In WPF mode that current interpreter can be the
    lightweight launcher Python, so explicit runtime env candidates come before
    sys.executable.
    """
    backend_root = Path(backend_root).resolve()
    active = _active_env_python(backend_root)
    if active is not None:
        return active

    env_root = backend_root / "env"
    for name in HEAVY_ENV_CANDIDATES:
        candidate = _python_in_env(env_root / name)
        if candidate is not None:
            return candidate

    legacy = _python_in_env(backend_root / "venv")
    if legacy is not None:
        return legacy
    return Path(sys.executable)


def resolve_heavy_env_dir(backend_root: Path | str) -> Path:
    python = resolve_heavy_python(backend_root)
    if python.parent.name.lower() in {"scripts", "bin"}:
        return python.parent.parent
    return python.parent


def _active_env_python(backend_root: Path) -> Path | None:
    pointer_file = backend_root / "active_env.json"
    active_path = ""
    if pointer_file.is_file():
        try:
            data = json.loads(pointer_file.read_text(encoding="utf-8"))
            active_path = str(data.get("active_path") or "").strip()
        except Exception:
            active_path = ""
    if not active_path:
        return None
    active_root = Path(active_path)
    if not active_root.is_absolute():
        active_root = backend_root / active_root
    return _python_in_env(active_root)


def _python_in_env(env_dir: Path) -> Path | None:
    candidates = (
        env_dir / "python.exe",
        env_dir / "Scripts" / "python.exe",
        env_dir / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def pip_for_python(python_exe: Path | str) -> Path | None:
    python_exe = Path(python_exe)
    pip_name = "pip.exe" if os.name == "nt" else "pip"
    candidates = [python_exe.with_name(pip_name)]
    if python_exe.parent.name.lower() not in {"scripts", "bin"}:
        candidates.append(python_exe.parent / "Scripts" / pip_name)
        candidates.append(python_exe.parent / "bin" / pip_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
