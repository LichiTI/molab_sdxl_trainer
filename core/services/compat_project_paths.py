"""Project path helpers for compatibility route adapters."""

from __future__ import annotations

from pathlib import Path


def backend_root_from_compat_router(router_file: str | Path) -> Path:
    """Resolve backend root from resources/web/routers/compat.py."""

    return Path(router_file).resolve().parent.parent.parent.parent / "backend"


def project_root_from_backend(backend_root: str | Path) -> Path:
    return Path(backend_root).resolve().parent


def output_root_from_project(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "output"
