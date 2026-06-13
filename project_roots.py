"""Shared project-root resolution helpers for local source layouts.

These helpers intentionally anchor to the repository layout instead of the
caller's current working directory so copied installs and launcher entrypoints
remain stable after moving the project.
"""

from __future__ import annotations

from pathlib import Path


def _normalize_candidate(path: Path | str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.exists():
        return candidate.parent if candidate.is_file() else candidate
    return candidate.parent if candidate.suffix else candidate


def _candidate_chain(path: Path | str) -> tuple[Path, ...]:
    start = _normalize_candidate(path)
    return (start, *start.parents)


def is_project_root(path: Path | str) -> bool:
    candidate = _normalize_candidate(path)
    return (candidate / "backend").is_dir() and (candidate / "resources").is_dir()


def is_backend_root(path: Path | str) -> bool:
    candidate = _normalize_candidate(path)
    return candidate.name == "backend" and (candidate / "lulynx_launcher").is_dir()


def resolve_project_root(
    path: Path | str | None = None,
    *,
    source_file: Path | str | None = None,
) -> Path:
    """Resolve the canonical project root for this source checkout.

    ``path`` may point at the project root itself, the backend root, or a file
    nested under either. ``source_file`` provides a stable fallback chain when
    the caller has no explicit path and should usually be set to ``__file__``.
    """

    seen: set[Path] = set()
    inputs = [item for item in (path, source_file, __file__) if item is not None]
    for raw in inputs:
        for candidate in _candidate_chain(raw):
            if candidate in seen:
                continue
            seen.add(candidate)
            if is_project_root(candidate):
                return candidate
            if is_backend_root(candidate):
                return candidate.parent.resolve()
    return Path(__file__).resolve().parent.parent


def resolve_backend_root(
    path: Path | str | None = None,
    *,
    source_file: Path | str | None = None,
) -> Path:
    """Resolve the canonical backend root for this source checkout."""

    if path is not None:
        candidate = _normalize_candidate(path)
        if is_backend_root(candidate):
            return candidate
    return resolve_project_root(path, source_file=source_file) / "backend"


__all__ = [
    "is_backend_root",
    "is_project_root",
    "resolve_backend_root",
    "resolve_project_root",
]
