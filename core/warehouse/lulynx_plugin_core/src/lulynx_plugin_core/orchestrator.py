"""Plugin orchestrator for the lulynx plugin runtime.

Manages plugin lifecycle (register, activate, deactivate) and dispatches
events through an injected EventBus.  Provides query helpers for
introspecting the current plugin state.

This module is a Warehouse design: it does not import, copy, or
paraphrase any upstream or fork code.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field
from typing import Any

from lulynx_plugin_core.event_bus import EventBus


class PluginState(str, enum.Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginDescriptor:
    """Mutable state for a registered plugin."""
    plugin_id: str
    display_name: str = ""
    version: str = "0.0.0"
    state: PluginState = PluginState.REGISTERED
    tier: int = 0
    capabilities: tuple[str, ...] = ()
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PluginOrchestrator:
    """Coordinate plugin lifecycle and event dispatch.

    Thread-safe for concurrent registration and dispatch.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._bus = event_bus or EventBus()
        self._plugins: dict[str, PluginDescriptor] = {}
        self._lock = threading.Lock()

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    def register(self, desc: PluginDescriptor) -> None:
        with self._lock:
            self._plugins[desc.plugin_id] = desc

    def activate(self, plugin_id: str) -> bool:
        with self._lock:
            desc = self._plugins.get(plugin_id)
            if desc is None:
                return False
            desc.state = PluginState.ACTIVE
            return True

    def deactivate(self, plugin_id: str) -> bool:
        with self._lock:
            desc = self._plugins.get(plugin_id)
            if desc is None:
                return False
            desc.state = PluginState.DISABLED
            return True

    def dispatch(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit *event* on the bus and return the dispatch report."""
        return self._bus.emit(event, payload)

    def dispatch_training_event(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Emit a training-loop event, injecting plugin metadata."""
        enriched = dict(payload)
        active = self.query_active()
        enriched["_active_plugin_ids"] = [p.plugin_id for p in active]
        return self._bus.emit(event, enriched)

    def get(self, plugin_id: str) -> PluginDescriptor | None:
        with self._lock:
            return self._plugins.get(plugin_id)

    def query_all(self) -> list[PluginDescriptor]:
        with self._lock:
            return list(self._plugins.values())

    def query_active(self) -> list[PluginDescriptor]:
        with self._lock:
            return [p for p in self._plugins.values()
                    if p.state == PluginState.ACTIVE]

    def query_by_tier(self, tier: int) -> list[PluginDescriptor]:
        with self._lock:
            return [p for p in self._plugins.values() if p.tier == tier]

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all plugin states."""
        with self._lock:
            return {
                "plugin_count": len(self._plugins),
                "active_count": sum(
                    1 for p in self._plugins.values()
                    if p.state == PluginState.ACTIVE
                ),
                "plugins": {
                    pid: {
                        "state": desc.state.value,
                        "tier": desc.tier,
                        "display_name": desc.display_name,
                        "error": desc.error,
                    }
                    for pid, desc in self._plugins.items()
                },
            }

