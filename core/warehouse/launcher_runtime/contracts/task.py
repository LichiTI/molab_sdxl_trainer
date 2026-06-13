"""Task state machine contracts."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class TaskState(enum.Enum):
    """Lifecycle states of an orchestrated task."""

    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskKind(enum.Enum):
    """What kind of operation the task wraps."""

    LAUNCH = "launch"
    INSTALL = "install"
    INITIALIZE = "initialize"
    UNINSTALL = "uninstall"
    UPDATE = "update"


@dataclass(frozen=True)
class ProgressEvent:
    """Emitted during task execution for UI consumption."""

    task_id: str
    percent: int
    section: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    """Final outcome produced by a completed task."""

    task_id: str
    state: TaskState
    kind: TaskKind
    runtime_id: str
    elapsed_seconds: float = 0.0
    error_message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResultRecord:
    """Persisted history entry for a past task."""

    task_id: str
    kind: TaskKind
    runtime_id: str
    state: TaskState
    started_at: float
    finished_at: float
    elapsed_seconds: float
    error_message: str = ""


def new_task_id() -> str:
    """Generate a short unique task identifier."""
    return uuid.uuid4().hex[:12]
