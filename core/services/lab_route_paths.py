"""Path resolution helpers for LAB WebUI compatibility routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class LabRoutePathError(ValueError):
    """Raised when a LAB route path cannot be safely resolved."""


def resolve_lab_input_path(
    value: Any,
    *,
    project_root: Path,
    required: bool,
    label: str,
    must_exist: bool = True,
) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        if required:
            raise LabRoutePathError(f"{label} is required")
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise LabRoutePathError(f"{label} is invalid: {exc}") from exc
    if must_exist and not resolved.exists():
        raise LabRoutePathError(f"{label} does not exist: {resolved}")
    return resolved


def resolve_lab_project_output_path(
    value: Any,
    *,
    project_root: Path,
    default: str,
    suffix: str | None = None,
    label: str = "Output",
) -> Path:
    raw = str(value or default).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    try:
        resolved = path.resolve()
        root = project_root.resolve()
    except OSError as exc:
        raise LabRoutePathError(f"{label} path is invalid: {exc}") from exc
    if not _is_relative_to(resolved, root):
        raise LabRoutePathError(f"{label} path must stay inside the project directory")
    if suffix and resolved.suffix.lower() != suffix:
        raise LabRoutePathError(f"{label} path must end with {suffix}")
    return resolved


def require_lab_artifact_file(path: Path, *, label: str) -> Path:
    """Require a resolved LAB artifact path to exist as a file."""

    if not path.is_file():
        raise LabRoutePathError(f"{label} does not exist: {path}")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
