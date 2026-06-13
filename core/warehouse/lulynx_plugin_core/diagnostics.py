"""Dispatch diagnostics for the lulynx plugin event system.

Provides per-event and per-handler performance metrics collection.
Designed for lightweight integration into the training-loop hot path.

This module is a Warehouse design.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HandlerMetrics:
    """Cumulative metrics for a single handler."""
    handler_id: str = ""
    invocations: int = 0
    errors: int = 0
    total_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    slow_count: int = 0

    @property
    def avg_duration_ms(self) -> float:
        return (self.total_duration_ms / self.invocations
                if self.invocations > 0 else 0.0)

    def record(self, duration_ms: float, error: bool = False,
               slow_threshold_ms: float = 25.0) -> None:
        self.invocations += 1
        self.total_duration_ms += duration_ms
        if duration_ms > self.max_duration_ms:
            self.max_duration_ms = duration_ms
        if error:
            self.errors += 1
        if slow_threshold_ms > 0 and duration_ms >= slow_threshold_ms:
            self.slow_count += 1


@dataclass
class EventMetrics:
    """Cumulative metrics for a single event type."""
    event: str = ""
    dispatches: int = 0
    total_handlers_invoked: int = 0
    total_errors: int = 0
    total_duration_ms: float = 0.0
    exclusive_conflicts: int = 0
    mutations: int = 0

    @property
    def avg_duration_ms(self) -> float:
        return (self.total_duration_ms / self.dispatches
                if self.dispatches > 0 else 0.0)

    @property
    def error_rate(self) -> float:
        return (self.total_errors / self.dispatches
                if self.dispatches > 0 else 0.0)

    def record(self, report: dict[str, Any]) -> None:
        self.dispatches += 1
        self.total_handlers_invoked += report.get("handled", 0)
        self.total_errors += len(report.get("errors", []))
        self.total_duration_ms += report.get("elapsed_ms", 0.0)
        if report.get("exclusive_conflict"):
            self.exclusive_conflicts += 1
        if report.get("mutated"):
            self.mutations += 1


class DiagnosticsCollector:
    """Collect and query dispatch diagnostics across all events.

    Thread-safe.  Call :meth:`record` after each ``EventBus.emit()``.
    """

    def __init__(self, slow_threshold_ms: float = 25.0) -> None:
        self._slow_threshold = slow_threshold_ms
        self._events: dict[str, EventMetrics] = {}
        self._handlers: dict[str, HandlerMetrics] = {}
        self._lock = threading.Lock()

    def record(self, report: dict[str, Any]) -> None:
        """Ingest a dispatch report from EventBus.emit()."""
        event = report.get("event", "")
        with self._lock:
            em = self._events.get(event)
            if em is None:
                em = EventMetrics(event=event)
                self._events[event] = em
            em.record(report)
            for detail in report.get("handlers", []):
                hid = f"{detail.get('plugin_id', '')}:{detail.get('handler', '')}"
                hm = self._handlers.get(hid)
                if hm is None:
                    hm = HandlerMetrics(handler_id=hid)
                    self._handlers[hid] = hm
                hm.record(
                    detail.get("duration_ms", 0.0),
                    error=(detail.get("status") == "error"),
                    slow_threshold_ms=self._slow_threshold,
                )

    def get_event_metrics(self, event: str) -> EventMetrics | None:
        with self._lock:
            return self._events.get(event)

    def get_handler_metrics(self, handler_id: str) -> HandlerMetrics | None:
        with self._lock:
            return self._handlers.get(handler_id)

    def get_slow_handlers(self, min_count: int = 1) -> list[HandlerMetrics]:
        with self._lock:
            return [h for h in self._handlers.values() if h.slow_count >= min_count]

    def get_error_handlers(self, min_count: int = 1) -> list[HandlerMetrics]:
        with self._lock:
            return [h for h in self._handlers.values() if h.errors >= min_count]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "event_count": len(self._events),
                "handler_count": len(self._handlers),
                "total_dispatches": sum(
                    e.dispatches for e in self._events.values()
                ),
                "total_errors": sum(
                    e.total_errors for e in self._events.values()
                ),
                "events": {
                    name: {
                        "dispatches": m.dispatches,
                        "avg_ms": round(m.avg_duration_ms, 3),
                        "errors": m.total_errors,
                        "mutations": m.mutations,
                    }
                    for name, m in self._events.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._handlers.clear()

