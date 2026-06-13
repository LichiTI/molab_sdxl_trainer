"""In-memory catalog of known runtime definitions.

Registry is populated at startup (from TOML/YAML/JSON definition files
loaded elsewhere) and queried by detector, recommender, and UI layers.
Immutable after freeze; no I/O performed here.
"""

from __future__ import annotations

from typing import Iterable

from .contracts.runtime import RuntimeCategory, RuntimeDef


class RuntimeRegistry:
    """Ordered, id-keyed collection of :class:`RuntimeDef` entries."""

    def __init__(self) -> None:
        self._defs: dict[str, RuntimeDef] = {}

    # ── mutation ───────────────────────────────────────────────

    def register(self, rd: RuntimeDef) -> None:
        """Add a runtime definition.  Overwrites if *rd.id* already present."""
        self._defs[rd.id] = rd

    def register_many(self, defs: Iterable[RuntimeDef]) -> None:
        for rd in defs:
            self.register(rd)

    def clear(self) -> None:
        self._defs.clear()

    # ── lookup ─────────────────────────────────────────────────

    def get(self, runtime_id: str) -> RuntimeDef | None:
        return self._defs.get(runtime_id)

    def __contains__(self, runtime_id: str) -> bool:
        return runtime_id in self._defs

    def __len__(self) -> int:
        return len(self._defs)

    # ── queries ────────────────────────────────────────────────

    def all(self) -> list[RuntimeDef]:
        """All registered runtimes in insertion order."""
        return list(self._defs.values())

    def ids(self) -> list[str]:
        return list(self._defs.keys())

    def by_category(self, category: RuntimeCategory) -> list[RuntimeDef]:
        return [rd for rd in self._defs.values() if rd.category == category]

    def experimental(self) -> list[RuntimeDef]:
        return [rd for rd in self._defs.values() if rd.experimental]

    def stable(self) -> list[RuntimeDef]:
        return [rd for rd in self._defs.values() if not rd.experimental]
