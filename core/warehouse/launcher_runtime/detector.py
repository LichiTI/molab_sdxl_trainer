"""Read-only filesystem probing for runtime environments.

Given a *base directory* (the root under which each runtime's env folder
lives) and a :class:`RuntimeDef`, the detector walks the filesystem to
determine whether the runtime is installed and healthy.

No files are created, modified, or deleted.
"""

from __future__ import annotations

from pathlib import Path

from .contracts.runtime import RuntimeDef
from .contracts.status import IntegrityReport, RuntimeStatus
from .registry import RuntimeRegistry


class RuntimeDetector:
    """Stateless probe that reads disk to build :class:`RuntimeStatus` snapshots."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir).resolve()

    @property
    def base_dir(self) -> Path:
        return self._base

    # ── public API ─────────────────────────────────────────────

    def detect(self, rd: RuntimeDef) -> RuntimeStatus:
        """Probe a single runtime definition against the filesystem."""
        env_dir = self._resolve_env_dir(rd)
        python_path = self._find_python(env_dir, rd) if env_dir else None
        python_exists = python_path is not None and python_path.is_file()
        deps_installed = self._check_deps(env_dir) if env_dir else False
        installed = env_dir is not None and env_dir.is_dir()
        integrity = self._check_integrity(rd, env_dir, python_exists, deps_installed)

        return RuntimeStatus(
            runtime_id=rd.id,
            python_exists=python_exists,
            deps_installed=deps_installed,
            installed=installed,
            python_path=python_path,
            env_dir=env_dir,
            integrity=integrity,
        )

    def detect_all(self, registry: RuntimeRegistry) -> list[RuntimeStatus]:
        """Probe every runtime in *registry*."""
        return [self.detect(rd) for rd in registry.all()]

    # ── internal probing ───────────────────────────────────────

    def _resolve_env_dir(self, rd: RuntimeDef) -> Path | None:
        """Return the first existing env directory for *rd*, or ``None``."""
        for name in rd.env_dir_names:
            candidate = self._base / name
            if candidate.is_dir():
                return candidate
        return None

    def _find_python(self, env_dir: Path | None, rd: RuntimeDef) -> Path | None:
        """Return the python executable path if it exists, else ``None``."""
        if env_dir is None:
            return None
        return env_dir / rd.python_rel_path

    def _check_deps(self, env_dir: Path | None) -> bool:
        """Heuristic: deps are installed if *site-packages* is non-empty."""
        if env_dir is None:
            return False
        # Look for Lib/site-packages (Windows) or lib/python*/site-packages (Unix)
        sp_win = env_dir / "Lib" / "site-packages"
        if sp_win.is_dir() and any(sp_win.iterdir()):
            return True
        lib_dir = env_dir / "lib"
        if lib_dir.is_dir():
            for child in lib_dir.iterdir():
                if child.is_dir() and child.name.startswith("python"):
                    sp = child / "site-packages"
                    if sp.is_dir() and any(sp.iterdir()):
                        return True
        return False

    def _check_integrity(
        self,
        rd: RuntimeDef,
        env_dir: Path | None,
        python_exists: bool,
        deps_installed: bool,
    ) -> IntegrityReport:
        """Build an :class:`IntegrityReport` from probe results."""
        if env_dir is None or not env_dir.is_dir():
            return IntegrityReport.broken(
                "env_missing",
                f"Environment directory not found for '{rd.id}'.",
                f"未找到 '{rd.id}' 的环境目录。",
            )
        if not python_exists:
            return IntegrityReport.broken(
                "python_missing",
                f"Python executable not found at expected path for '{rd.id}'.",
                f"未找到 '{rd.id}' 的 Python 可执行文件。",
            )
        if not deps_installed:
            return IntegrityReport(
                ok=False,
                bootstrap_ready=True,
                issue_code="deps_missing",
                message_en=f"Environment '{rd.id}' exists but dependencies are not installed.",
                message_zh=f"环境 '{rd.id}' 已存在但依赖尚未安装。",
            )
        return IntegrityReport.healthy()
