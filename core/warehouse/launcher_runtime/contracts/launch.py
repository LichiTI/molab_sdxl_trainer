"""Launch plan contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PlanStepKind(Enum):
    """Discrete step inside a launch/install plan."""

    CHECK_ENV = "check_env"
    SET_ENV_VAR = "set_env_var"
    CLEAR_ENV_VAR = "clear_env_var"
    COMPOSE_COMMAND = "compose_command"
    SPAWN_PROCESS = "spawn_process"
    WAIT_PROCESS = "wait_process"
    INSTALL_DEPS = "install_deps"
    INITIALIZE_ENV = "initialize_env"


@dataclass(frozen=True)
class PlanStep:
    """One human-readable step in an execution plan."""

    kind: PlanStepKind
    label_en: str
    label_zh: str
    detail: str = ""


@dataclass(frozen=True)
class LaunchOptions:
    """User-facing options that influence launch command/env composition."""

    runtime_id: str
    safe_mode: bool = False
    cn_mirror: bool = False
    attention_policy: str = "auto"
    host: str = "127.0.0.1"
    port: int = 7860
    listen: bool = False
    extra_args: tuple[str, ...] = ()


@dataclass
class LaunchPlan:
    """Result of composing a launch: ready-to-run command + environment."""

    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    steps: list[PlanStep] = field(default_factory=list)
    cwd: str = ""
