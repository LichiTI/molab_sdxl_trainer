"""Trainer Registry — a generic, data-driven registry for training backends.

Trainers are registered by a unique string key and carry metadata about
their supported architectures, capabilities, and entry-point information.
The registry itself stores only data; it does not import, load, or execute
trainer code.  Pure-stdlib, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from .types import Capability, ModelArchitecture


# ---------------------------------------------------------------------------
# Trainer entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainerEntry:
    """Immutable metadata describing a single registered trainer."""

    key: str
    display_name: str
    architectures: frozenset[ModelArchitecture] = field(default_factory=frozenset)
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def supports_architecture(self, arch: ModelArchitecture) -> bool:
        """Return True if this trainer handles *arch*."""
        return arch in self.architectures

    def supports_capability(self, cap: Capability) -> bool:
        """Return True if this trainer advertises *cap*."""
        return cap in self.capabilities

    def has_all_capabilities(self, caps: set[Capability]) -> bool:
        """Return True if this trainer advertises every capability in *caps*."""
        return caps <= self.capabilities


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TrainerRegistry:
    """A generic, mutable registry of :class:`TrainerEntry` objects.

    Typical usage::

        registry = TrainerRegistry()
        registry.register(TrainerEntry(
            key="native_sd15",
            display_name="Native SD1.5",
            architectures=frozenset({ModelArchitecture.SD15}),
        ))

        entry = registry.get("native_sd15")
        matches = registry.query(architecture=ModelArchitecture.SD15)
    """

    def __init__(self) -> None:
        self._entries: dict[str, TrainerEntry] = {}

    # -- mutation -----------------------------------------------------------

    def register(self, entry: TrainerEntry) -> None:
        """Add or replace a trainer entry."""
        self._entries[entry.key] = entry

    def unregister(self, key: str) -> bool:
        """Remove a trainer by key.  Returns True if it existed."""
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    # -- lookup -------------------------------------------------------------

    def get(self, key: str) -> TrainerEntry | None:
        """Return the entry for *key*, or None."""
        return self._entries.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[TrainerEntry]:
        return iter(self._entries.values())

    @property
    def keys(self) -> list[str]:
        """All registered keys."""
        return list(self._entries.keys())

    @property
    def entries(self) -> list[TrainerEntry]:
        """All registered entries."""
        return list(self._entries.values())

    # -- query --------------------------------------------------------------

    def query(
        self,
        *,
        architecture: ModelArchitecture | None = None,
        capability: Capability | None = None,
        all_capabilities: set[Capability] | None = None,
    ) -> list[TrainerEntry]:
        """Return entries matching the given filters (AND semantics)."""
        result: list[TrainerEntry] = []
        for entry in self._entries.values():
            if architecture is not None and not entry.supports_architecture(architecture):
                continue
            if capability is not None and not entry.supports_capability(capability):
                continue
            if all_capabilities is not None and not entry.has_all_capabilities(all_capabilities):
                continue
            result.append(entry)
        return result

    def first(
        self,
        *,
        architecture: ModelArchitecture | None = None,
        capability: Capability | None = None,
    ) -> TrainerEntry | None:
        """Return the first matching entry, or None."""
        matches = self.query(architecture=architecture, capability=capability)
        return matches[0] if matches else None
