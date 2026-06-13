"""
Hook and capability contract definitions for plugin interoperability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set


class HookPhase(str, Enum):
    """Lifecycle phase at which a hook fires."""

    PRE = "pre"
    MAIN = "main"
    POST = "post"
    CLEANUP = "cleanup"


@dataclass(frozen=True)
class HookDefinition:
    """Declares a single extension point that plugins may implement.

    Attributes:
        name: Unique identifier (e.g. ``"training.before_epoch"``).
        description: Human-readable purpose.
        phase: Lifecycle phase.
        required_args: Names the host *must* pass to the hook.
        optional_args: Names the host *may* pass.
        returns_value: Whether the hook is expected to return a result.
        category: Logical grouping for filtering (e.g. ``"training"``).
    """

    name: str
    description: str = ""
    phase: HookPhase = HookPhase.MAIN
    required_args: Sequence[str] = field(default_factory=tuple)
    optional_args: Sequence[str] = field(default_factory=tuple)
    returns_value: bool = False
    category: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.category}.{self.name}" if self.category else self.name


class HookCatalog:
    """Registry of known hook definitions.

    Plugins consult the catalog to discover which extension points exist;
    hosts consult it to validate that a hook name is well-defined before
    attempting to invoke it.
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, HookDefinition] = {}

    def register(self, hook: HookDefinition) -> None:
        """Add a hook definition to the catalog."""
        key = hook.qualified_name
        if key in self._hooks:
            raise ValueError(f"Hook '{key}' is already registered")
        self._hooks[key] = hook

    def register_many(self, hooks: Sequence[HookDefinition]) -> int:
        """Bulk-register hook definitions.  Returns the count added."""
        for h in hooks:
            self.register(h)
        return len(hooks)

    def get(self, qualified_name: str) -> Optional[HookDefinition]:
        return self._hooks.get(qualified_name)

    def __contains__(self, qualified_name: str) -> bool:
        return qualified_name in self._hooks

    def by_category(self, category: str) -> List[HookDefinition]:
        return [h for h in self._hooks.values() if h.category == category]

    def by_phase(self, phase: HookPhase) -> List[HookDefinition]:
        return [h for h in self._hooks.values() if h.phase == phase]

    @property
    def all(self) -> List[HookDefinition]:
        return list(self._hooks.values())

    def __len__(self) -> int:
        return len(self._hooks)


# ── Capability contracts ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CapabilityDefinition:
    """Declares a discrete capability a plugin may advertise.

    Attributes:
        name: Unique identifier (e.g. ``"training.lora"``).
        description: What this capability enables.
        version: Semver-ish string for the capability contract.
        required_hooks: Hook names that must be present for this capability.
        tags: Free-form labels for filtering.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    required_hooks: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)


class CapabilityRegistry:
    """Tracks which capabilities are available in the system.

    Designed for lookups like "does any loaded plugin provide X?" without
    coupling to plugin loading internals.
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilityDefinition] = {}

    def register(self, capability: CapabilityDefinition) -> None:
        if capability.name in self._capabilities:
            raise ValueError(
                f"Capability '{capability.name}' is already registered"
            )
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Optional[CapabilityDefinition]:
        return self._capabilities.get(name)

    def has(self, name: str) -> bool:
        return name in self._capabilities

    def names(self) -> Set[str]:
        return set(self._capabilities.keys())

    def tags_index(self) -> Dict[str, List[str]]:
        """Build a tag -> capability-name mapping."""
        index: Dict[str, List[str]] = {}
        for cap in self._capabilities.values():
            for tag in cap.tags:
                index.setdefault(tag, []).append(cap.name)
        return index

    def __contains__(self, name: str) -> bool:
        return name in self._capabilities

    def __len__(self) -> int:
        return len(self._capabilities)
