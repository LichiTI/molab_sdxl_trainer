"""Update-check contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class UpdateChannel(Enum):
    STABLE = "stable"
    BETA = "beta"


class UpdateOutcome(Enum):
    UPDATED = "updated"
    ALREADY_CURRENT = "already_current"
    FAILED = "failed"


@dataclass(frozen=True)
class UpdateInfo:
    """Result of checking for a newer version."""

    has_update: bool
    current_version: str
    latest_version: str = ""
    release_url: str = ""
    release_notes: str = ""
    channel: UpdateChannel = UpdateChannel.STABLE


@dataclass
class UpdateResult:
    """Outcome of running the updater script."""

    outcome: UpdateOutcome = UpdateOutcome.ALREADY_CURRENT
    new_version: str = ""
    error_message: str = ""
    log_lines: list[str] = field(default_factory=list)
