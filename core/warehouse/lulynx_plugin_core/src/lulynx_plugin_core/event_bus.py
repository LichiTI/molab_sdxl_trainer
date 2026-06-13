"""Thread-safe event bus with priority ordering, mutation tracking,
and exclusive-hook conflict detection.

The bus is designed for the training-loop hook dispatch path: handlers
are registered once, then the bus is queried at each training step.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from lulynx_plugin_core.hooks import get_hook


@dataclass
class HandlerRegistration:
    """A single handler registration on the event bus."""

    plugin_id: str
    event: str
    handler_name: str
    handler: Callable[[Any], Any]
    priority: int = 0
    mutable: bool = False
    predicate: Callable[[Any], bool] | None = None


def _freeze(obj: Any) -> Any:
    """Deep-freeze a dict/list payload into read-only proxy types."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(item) for item in obj)
    return obj


class EventBus:
    """Thread-safe pub/sub event bus with priority dispatch.

    Handlers for the same event are sorted by descending priority.
    Exclusive hooks (as declared in the hook catalog) allow only one
    handler; additional registrations are skipped with a report flag.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[HandlerRegistration]] = {}
        self._lock = threading.RLock()

    def clear(self) -> None:
        """Remove all handler registrations."""
        with self._lock:
            self._handlers.clear()

    def register(self, reg: HandlerRegistration) -> None:
        """Register a handler.  Raises if event name is empty."""
        if not reg.event.strip():
            raise ValueError("event must not be empty")
        with self._lock:
            self._handlers.setdefault(reg.event, []).append(reg)

    def has_handlers(self, event: str) -> bool:
        """Return True if at least one handler is registered for *event*."""
        with self._lock:
            return bool(self._handlers.get(event))

    def emit(
        self,
        event: str,
        payload: dict | None = None,
        *,
        slow_threshold_ms: float = 25.0,
        capture_result: bool = False,
    ) -> dict:
        """Dispatch *event* to all registered handlers and return a report.

        The report dict contains ``event``, ``handled``, ``errors``,
        ``skipped``, ``exclusive_conflict``, ``mutated``, ``elapsed_ms``,
        ``slow_handlers``, and ``handlers`` (per-handler detail list).
        """
        event = str(event or "").strip()
        hook_def = get_hook(event)
        with self._lock:
            handlers = sorted(
                list(self._handlers.get(event, [])),
                key=lambda h: h.priority,
                reverse=True,
            )

        report: dict[str, Any] = {
            "event": event,
            "handled": 0,
            "errors": [],
            "skipped": [],
            "exclusive_conflict": False,
            "mutated": False,
            "elapsed_ms": 0.0,
            "slow_handlers": 0,
            "handlers": [],
        }
        if not handlers:
            return report

        # Exclusive hook conflict detection
        if hook_def and hook_def.exclusive and len(handlers) > 1:
            report["exclusive_conflict"] = True
            for skipped in handlers[1:]:
                report["skipped"].append({
                    "plugin_id": skipped.plugin_id,
                    "handler": skipped.handler_name,
                    "reason": "exclusive_conflict",
                })
            handlers = handlers[:1]

        dispatch_start = time.perf_counter()
        current_payload = copy.deepcopy(payload or {})

        for h in handlers:
            h_start = time.perf_counter()
            status = "ok"
            error_msg = ""
            try:
                if callable(h.predicate) and not h.predicate(current_payload):
                    status = "skipped"
                    report["skipped"].append({
                        "plugin_id": h.plugin_id,
                        "handler": h.handler_name,
                        "reason": "predicate_filtered",
                    })
                else:
                    # Decide frozen vs mutable dispatch
                    if hook_def and (hook_def.read_only or not hook_def.allows_mutation):
                        handler_input = _freeze(current_payload)
                        h.handler(handler_input)
                    elif not h.mutable:
                        handler_input = _freeze(current_payload)
                        h.handler(handler_input)
                    else:
                        before = copy.deepcopy(current_payload)
                        result = h.handler(current_payload)
                        if isinstance(result, dict):
                            current_payload = result
                        if current_payload != before:
                            report["mutated"] = True
                    report["handled"] += 1
            except Exception as exc:
                status = "error"
                error_msg = str(exc)
                report["errors"].append({
                    "plugin_id": h.plugin_id,
                    "handler": h.handler_name,
                    "error": error_msg,
                })

            dur_ms = round((time.perf_counter() - h_start) * 1000, 3)
            is_slow = slow_threshold_ms > 0 and dur_ms >= slow_threshold_ms
            if is_slow:
                report["slow_handlers"] += 1
            detail: dict[str, Any] = {
                "plugin_id": h.plugin_id,
                "handler": h.handler_name,
                "priority": h.priority,
                "status": status,
                "duration_ms": dur_ms,
                "slow": is_slow,
            }
            if error_msg:
                detail["error"] = error_msg
            report["handlers"].append(detail)

        report["elapsed_ms"] = round((time.perf_counter() - dispatch_start) * 1000, 3)
        if capture_result:
            report["result_payload"] = copy.deepcopy(current_payload)
        return report
