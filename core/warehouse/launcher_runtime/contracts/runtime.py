"""Runtime definition contracts.

Pure data types describing supported runtime environments.
No I/O, no side-effects.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class RuntimeCategory(enum.Enum):
    """Broad hardware vendor / acceleration category."""

    NVIDIA = "nvidia"
    NVIDIA_FRONTIER = "nvidia_frontier"
    INTEL = "intel"
    AMD = "amd"
    CPU = "cpu"


class CompatibilityStatus(enum.Enum):
    """How well a runtime suits a given model family."""

    RECOMMENDED = "recommended"
    SUPPORTED = "supported"
    CAUTION = "caution"
    NOT_RECOMMENDED = "not_recommended"


@dataclass(frozen=True)
class CapabilityTag:
    """A discrete capability a runtime may advertise (e.g. 'bf16', 'flash_attn')."""

    name: str


@dataclass(frozen=True)
class CompatibilityRule:
    """Pairing verdict for one runtime + one model family."""

    status: CompatibilityStatus
    reason_en: str
    reason_zh: str


@dataclass(frozen=True)
class RuntimeDef:
    """Immutable definition of a single supported runtime environment.

    Loaded from external definition files (TOML/YAML/JSON).
    """

    id: str
    name_en: str
    name_zh: str
    description_en: str
    description_zh: str
    category: RuntimeCategory
    experimental: bool = False
    env_dir_names: tuple[str, ...] = ()
    python_rel_path: str = "python.exe"
    env_vars: dict[str, str] = field(default_factory=dict)
    install_scripts: tuple[str, ...] = ()
    attention_policy_default: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    # ── helpers ──────────────────────────────────────────────

    def env_var(self, key: str, default: str = "") -> str:
        return self.env_vars.get(key, default)

    @property
    def primary_env_dir(self) -> str | None:
        return self.env_dir_names[0] if self.env_dir_names else None
