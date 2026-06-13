"""Script lifecycle management for lulynx training runs.

Provides environment preparation, path resolution, and controlled
script execution with cleanup.

This module is a Warehouse design.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator


@dataclass(frozen=True)
class ScriptDescriptor:
    """Declarative description of a training script to execute."""
    script_path: Path
    working_dir: Path | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    python_path_extra: list[Path] = field(default_factory=list)
    direct_python: bool = False
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScriptResult:
    """Outcome of a script execution."""
    script_path: Path
    success: bool
    exit_code: int = 0
    error: str = ""
    duration_seconds: float = 0.0


def resolve_script_path(
    script_path: str | Path,
    search_dirs: list[Path] | None = None,
) -> Path:
    """Resolve a script path, searching directories if needed.

    Raises FileNotFoundError if the script cannot be found.
    """
    p = Path(script_path)
    if p.is_file():
        return p.resolve()
    if search_dirs:
        for d in search_dirs:
            candidate = d / script_path
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Script not found: {script_path}")


@contextmanager
def prepared_environment(
    descriptor: ScriptDescriptor,
) -> Generator[None, None, None]:
    """Context manager that sets up and tears down the script environment.

    - Sets environment variables from descriptor
    - Adds extra paths to sys.path
    - Changes working directory if specified
    - Restores everything on exit
    """
    saved_env: dict[str, str | None] = {}
    saved_cwd = Path.cwd()
    saved_path = list(sys.path)

    try:
        for key, val in descriptor.env_vars.items():
            saved_env[key] = os.environ.get(key)
            os.environ[key] = val

        for extra in descriptor.python_path_extra:
            sp = str(extra.resolve())
            if sp not in sys.path:
                sys.path.insert(0, sp)

        parent = str(descriptor.script_path.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)

        if descriptor.working_dir:
            descriptor.working_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(descriptor.working_dir)

        yield

    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        for key, old_val in saved_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


def execute_script(
    descriptor: ScriptDescriptor,
    *,
    namespace: dict[str, Any] | None = None,
) -> ScriptResult:
    """Execute a training script in a prepared environment.

    Uses runpy.run_path for direct execution.
    """
    import runpy
    import time

    resolved = resolve_script_path(descriptor.script_path)
    start = time.monotonic()

    try:
        with prepared_environment(descriptor):
            runpy.run_path(
                str(resolved),
                init_globals=namespace,
                run_name="__main__",
            )
        return ScriptResult(
            script_path=resolved,
            success=True,
            duration_seconds=time.monotonic() - start,
        )
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return ScriptResult(
            script_path=resolved,
            success=(code == 0),
            exit_code=code,
            duration_seconds=time.monotonic() - start,
        )
    except Exception as exc:
        return ScriptResult(
            script_path=resolved,
            success=False,
            error=str(exc),
            duration_seconds=time.monotonic() - start,
        )

