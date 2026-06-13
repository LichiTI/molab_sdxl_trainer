"""Runtime detection status contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegrityReport:
    """Result of probing a portable Python environment for health."""

    ok: bool
    bootstrap_ready: bool
    issue_code: str | None = None
    message_en: str = ""
    message_zh: str = ""

    # ── convenience constructors ─────────────────────────────

    @classmethod
    def healthy(cls) -> IntegrityReport:
        return cls(ok=True, bootstrap_ready=True)

    @classmethod
    def broken(cls, code: str, en: str, zh: str) -> IntegrityReport:
        return cls(ok=False, bootstrap_ready=False, issue_code=code, message_en=en, message_zh=zh)


@dataclass
class RuntimeStatus:
    """Snapshot of one runtime's installation state on disk."""

    runtime_id: str
    python_exists: bool
    deps_installed: bool
    installed: bool
    python_path: Path | None = None
    env_dir: Path | None = None
    integrity: IntegrityReport | None = None

    # ── helpers ──────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self.installed and self.integrity is not None and self.integrity.ok
