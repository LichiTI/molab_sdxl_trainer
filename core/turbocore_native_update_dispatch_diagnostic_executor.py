"""Diagnostic-only executor adapters for TurboCore native update dispatch.

These adapters are callable slots for runtime plumbing tests.  They replay
already-collected shadow evidence and never launch a new training update or
mutate real training parameters.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping


def build_shadow_owner_native_diagnostic_executor(
    shadow_report: Mapping[str, Any] | None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Return a safe diagnostic executor backed by owner-native shadow evidence."""

    shadow = _as_dict(shadow_report)
    owner_native = _as_dict(shadow.get("owner_native_launch_probe"))

    def _executor(request: Mapping[str, Any]) -> dict[str, Any]:
        blocked = _blocked_reasons(request=request, owner_native=owner_native)
        ok = not blocked
        return {
            "schema_version": 1,
            "executor": "turbocore_shadow_owner_native_diagnostic_executor_v0",
            "ok": ok,
            "reason": "shadow_owner_native_evidence_replayed" if ok else blocked[0],
            "diagnostic_replay": True,
            "shadow_owner_native_kernel_evidence": bool(owner_native.get("kernel_executed", False)),
            "shadow_owner_native_parity_ok": bool(owner_native.get("parity_ok", False)),
            "shadow_owner_native_event_chain_verified": bool(owner_native.get("event_chain_verified", False)),
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_step_executed": False,
            "native_kernel_launched": False,
            "training_parameters_mutated": False,
            "should_call_pytorch_optimizer_step": True,
            "blocked_reasons": blocked,
        }

    return _executor


def _blocked_reasons(
    *,
    request: Mapping[str, Any],
    owner_native: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if bool(request.get("training_dispatch", False)) or bool(request.get("training_path_enabled", False)):
        blocked.append("diagnostic_executor_received_training_dispatch_request")
    if not owner_native:
        blocked.append("owner_native_launch_probe_missing")
        return blocked
    if not bool(owner_native.get("ok", False)):
        blocked.append(str(owner_native.get("reason", "owner_native_launch_probe_not_ok") or "owner_native_launch_probe_not_ok"))
    if not bool(owner_native.get("kernel_executed", False)):
        blocked.append("owner_native_kernel_not_executed")
    if not bool(owner_native.get("parity_ok", False)):
        blocked.append("owner_native_launch_parity_failed")
    if bool(owner_native.get("persistent_owner_mutated", False)):
        blocked.append("owner_native_probe_mutated_persistent_owner")
    return _dedupe(blocked)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_shadow_owner_native_diagnostic_executor"]
