"""Installation result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class InstallOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class InstallSectionSpec:
    """Named progress section inside an install script output.

    Maps regex patterns to percentage ranges so a log-line parser
    can estimate overall progress.
    """

    name: str
    pattern: str
    percent_start: int
    percent_end: int


# Default section specs for a typical PS1 install script.
DEFAULT_INSTALL_SECTIONS: tuple[InstallSectionSpec, ...] = (
    InstallSectionSpec("provisioning", r"(?i)provisioning|creating env", 0, 10),
    InstallSectionSpec("pip_tooling", r"(?i)upgrading pip|installing setuptools", 10, 20),
    InstallSectionSpec("torch_stack", r"(?i)installing torch|pytorch", 20, 50),
    InstallSectionSpec("xformers", r"(?i)xformers", 50, 60),
    InstallSectionSpec("requirements", r"(?i)requirements\.txt|pip install", 60, 90),
    InstallSectionSpec("finalizing", r"(?i)finaliz|marker|deps_installed", 90, 100),
)


@dataclass
class InstallProgress:
    """Streaming progress update during an install."""

    percent: int = 0
    section: str = ""
    line: str = ""


@dataclass
class InstallResult:
    """Final outcome of an install run."""

    outcome: InstallOutcome = InstallOutcome.SUCCESS
    runtime_id: str = ""
    elapsed_seconds: float = 0.0
    error_message: str = ""
    log_lines: list[str] = field(default_factory=list)
