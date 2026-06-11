from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_path(project_root: Path) -> Path:
    """Ensure the real repository root is importable.

    In a PyInstaller build ``__file__`` may point inside the temporary
    extraction directory instead of the copied project.  Deriving the repo root
    from this source file at import time can therefore add the wrong directory
    to ``sys.path`` and make startup fail with ``No module named 'scripts'``.
    Use the runtime ``project_root`` supplied by the launcher instead, and do
    this lazily so the launcher can boot even when refresh-only helper scripts
    are not packaged into the executable.
    """

    repo_root = Path(project_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _load_refresh_first_release_evidence(project_root: Path):
    _ensure_repo_root_on_path(project_root)
    module = importlib.import_module("scripts.refresh_first_release_evidence")
    return module.refresh_first_release_evidence


def _preferred_python(project_root: Path) -> str:
    root = Path(project_root).resolve()
    candidates = [
        root / "backend" / "env" / "python-flashattention" / "python.exe",
        root / "backend" / "env" / "python_launcher" / "python.exe",
    ]
    if os_name() != "nt":
        candidates.extend(
            [
                root / "backend" / "env" / "python-flashattention" / "bin" / "python",
                root / "backend" / "env" / "python_launcher" / "bin" / "python",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def os_name() -> str:
    try:
        import os

        return os.name
    except Exception:
        return ""


def refresh_first_release_evidence_status(project_root: Path) -> dict[str, Any]:
    repo_root = _ensure_repo_root_on_path(project_root)
    refresh_first_release_evidence = _load_refresh_first_release_evidence(repo_root)
    return refresh_first_release_evidence(
        python=_preferred_python(repo_root),
        release_smoke_json="temp/lulynx_release_smoke.json",
        readiness_json="temp/lulynx_first_release_readiness.json",
        current_readiness_json="temp/lulynx_first_release_readiness_current.json",
    )


__all__ = ["refresh_first_release_evidence_status"]
