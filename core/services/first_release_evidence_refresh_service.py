from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.refresh_first_release_evidence import refresh_first_release_evidence


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
    repo_root = Path(project_root).resolve()
    return refresh_first_release_evidence(
        python=_preferred_python(repo_root),
        release_smoke_json="temp/lulynx_release_smoke.json",
        readiness_json="temp/lulynx_first_release_readiness.json",
        current_readiness_json="temp/lulynx_first_release_readiness_current.json",
    )


__all__ = ["refresh_first_release_evidence_status"]
