"""Diagnostics and preflight contracts."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class CheckSeverity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class DiagnosticCheck:
    """A single named health check."""

    id: str
    name_en: str
    name_zh: str


@dataclass
class DiagnosticFinding:
    """Result of running one check."""

    check_id: str
    passed: bool
    severity: CheckSeverity
    message_en: str = ""
    message_zh: str = ""


@dataclass
class HealthReport:
    """Aggregated report from all health checks."""

    findings: list[DiagnosticFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[DiagnosticFinding]:
        return [f for f in self.findings if not f.passed and f.severity == CheckSeverity.ERROR]

    @property
    def warnings(self) -> list[DiagnosticFinding]:
        return [f for f in self.findings if not f.passed and f.severity == CheckSeverity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class PreflightResult:
    """Go/no-go verdict before launching."""

    can_launch: bool
    blockers: list[DiagnosticFinding] = field(default_factory=list)
    advisories: list[DiagnosticFinding] = field(default_factory=list)
