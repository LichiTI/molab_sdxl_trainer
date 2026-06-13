"""
Synchronous in-process event bus with priority ordering and dispatch reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class HandlerConstraints:
    """Guards that must be satisfied before a handler is invoked."""

    min_priority: int = 0
    max_priority: int = 9999
    required_tags: frozenset[str] = field(default_factory=frozenset)
    blocked_tags: frozenset[str] = field(default_factory=frozenset)
    max_invocations: Optional[int] = None

    def accepts(self, priority: int, tags: Set[str]) -> bool:
        """Return True if the handler's priority and tags satisfy these constraints."""
        if priority < self.min_priority or priority > self.max_priority:
            return False
        if self.required_tags and not self.required_tags.issubset(tags):
            return False
        if self.blocked_tags & tags:
            return False
        return True


@dataclass
class HandlerRegistration:
    """A single event handler bound to an event type."""

    event_type: str
    handler: Callable[..., Any]
    priority: int = 100
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    invocation_count: int = 0

    def matches(self, constraints: HandlerConstraints) -> bool:
        if constraints.max_invocations is not None:
            if self.invocation_count >= constraints.max_invocations:
                return False
        return constraints.accepts(self.priority, self.tags)


@dataclass
class DispatchReport:
    """Result of dispatching one event through the bus."""

    event_type: str
    handlers_invoked: int = 0
    handlers_failed: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    handler_timings: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.handlers_failed == 0


class EventBus:
    """Priority-ordered, synchronous event dispatcher.

    Handlers execute highest-priority-first (lower number = higher priority).
    Each dispatch returns a ``DispatchReport`` summarising what happened.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[HandlerRegistration]] = {}
        self._global_constraints = HandlerConstraints()

    # -- Registration ----------------------------------------------------------

    def register(
        self,
        event_type: str,
        handler: Callable[..., Any],
        *,
        priority: int = 100,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HandlerRegistration:
        """Subscribe *handler* to *event_type* and return its registration."""
        reg = HandlerRegistration(
            event_type=event_type,
            handler=handler,
            priority=priority,
            tags=tags or set(),
            metadata=metadata or {},
        )
        bucket = self._handlers.setdefault(event_type, [])
        bucket.append(reg)
        bucket.sort(key=lambda r: r.priority)
        return reg

    def unregister(self, registration: HandlerRegistration) -> bool:
        """Remove a previously registered handler.  Returns True if found."""
        bucket = self._handlers.get(registration.event_type)
        if bucket is None:
            return False
        try:
            bucket.remove(registration)
            return True
        except ValueError:
            return False

    def handlers_for(
        self,
        event_type: str,
        constraints: Optional[HandlerConstraints] = None,
    ) -> List[HandlerRegistration]:
        """Return matching handlers sorted by priority."""
        constraints = constraints or self._global_constraints
        return [
            reg
            for reg in self._handlers.get(event_type, [])
            if reg.matches(constraints)
        ]

    # -- Dispatch --------------------------------------------------------------

    def dispatch(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        constraints: Optional[HandlerConstraints] = None,
    ) -> DispatchReport:
        """Fire all matching handlers and return a dispatch report."""
        constraints = constraints or self._global_constraints
        report = DispatchReport(event_type=event_type)
        started = time.perf_counter()

        for reg in self._handlers.get(event_type, []):
            if not reg.matches(constraints):
                continue

            report.handlers_invoked += 1
            reg.invocation_count += 1
            label = getattr(reg.handler, "__qualname__", repr(reg.handler))

            t0 = time.perf_counter()
            try:
                reg.handler(payload or {})
            except Exception as exc:
                report.handlers_failed += 1
                report.errors.append({"handler": label, "error": str(exc)})
            finally:
                report.handler_timings[label] = (time.perf_counter() - t0) * 1000

        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return report

    # -- Introspection ---------------------------------------------------------

    @property
    def event_types(self) -> List[str]:
        """All event types that have at least one registered handler."""
        return list(self._handlers.keys())

    @property
    def handler_count(self) -> int:
        """Total number of handler registrations across all events."""
        return sum(len(v) for v in self._handlers.values())

    def set_global_constraints(self, constraints: HandlerConstraints) -> None:
        """Replace the default constraints applied to every dispatch."""
        self._global_constraints = constraints

    def clear(self, event_type: Optional[str] = None) -> None:
        """Remove all handlers, or only those for *event_type*."""
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)
