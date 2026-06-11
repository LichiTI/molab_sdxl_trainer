"""Fail-closed gate for Lulynx multi-batch execution strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_MULTI_BATCH_EXECUTION_STRATEGY_GATE = "lulynx_multi_batch_execution_strategy_gate_v0"
_ALLOWED_STRATEGIES = {
    "native_batch_forward",
    "microbatch_forward_diagnostic",
    "single_item_forward_debug",
}


def build_lulynx_multi_batch_execution_strategy_gate(
    *,
    execution_strategy: Mapping[str, Any] | None,
    requested_strategy: str = "",
    internal_gate_enabled: bool = False,
    allow_diagnostic_strategy: bool = False,
) -> dict[str, Any]:
    """Validate a strategy before any future runtime switch can use it."""

    strategy = execution_strategy if isinstance(execution_strategy, Mapping) else {}
    strategy_name = str(strategy.get("strategy") or "")
    requested = str(requested_strategy or strategy_name or "")
    diagnostic = bool(strategy.get("diagnostic_only")) or strategy_name != "native_batch_forward"
    blockers: list[str] = []
    cautions: list[str] = []

    if str(strategy.get("contract") or "") != "lulynx_multi_batch_execution_strategy_v0":
        blockers.append("missing_multi_batch_execution_strategy")
    if strategy_name and strategy_name not in _ALLOWED_STRATEGIES:
        blockers.append("unknown_multi_batch_execution_strategy")
    if requested and strategy_name and requested != strategy_name:
        blockers.append("requested_strategy_differs_from_contract_strategy")
    if bool(strategy.get("release_claim_allowed")):
        blockers.append("strategy_release_claim_allowed_must_remain_false")
    if diagnostic and not allow_diagnostic_strategy:
        blockers.append("diagnostic_strategy_requires_explicit_allow")
    if diagnostic:
        cautions.append("diagnostic_strategy_not_release_claim")
    if not bool(internal_gate_enabled):
        blockers.append("internal_execution_strategy_gate_disabled")

    non_gate_blockers = [item for item in blockers if item != "internal_execution_strategy_gate_disabled"]
    ready_behind_gate = not non_gate_blockers and not bool(internal_gate_enabled)
    can_route = not blockers and bool(internal_gate_enabled)
    if can_route:
        status = "ready_to_route_strategy"
    elif ready_behind_gate:
        status = "ready_behind_disabled_internal_gate"
    else:
        status = "blocked"

    return {
        "schema_version": 1,
        "gate": LULYNX_MULTI_BATCH_EXECUTION_STRATEGY_GATE,
        "status": status,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_change_forward_path": True,
        "runtime_switch_enabled": bool(internal_gate_enabled),
        "can_route_requested_strategy": can_route,
        "ready_behind_disabled_internal_gate": ready_behind_gate,
        "requested_strategy": requested,
        "strategy": strategy_name,
        "diagnostic_only": diagnostic,
        "allow_diagnostic_strategy": bool(allow_diagnostic_strategy),
        "blockers": _dedupe(blockers),
        "cautions": _dedupe(cautions),
        "recommended_next_actions": _recommended_next_actions(
            can_route=can_route,
            ready_behind_gate=ready_behind_gate,
            blockers=blockers,
        ),
    }


def _recommended_next_actions(*, can_route: bool, ready_behind_gate: bool, blockers: Sequence[str]) -> list[str]:
    if can_route:
        return ["route_strategy_through_existing_training_loop_boundary"]
    if ready_behind_gate:
        return ["keep_strategy_report_only_until_parity_smoke_passes"]
    actions = ["fix_execution_strategy_gate_blockers"]
    if "diagnostic_strategy_requires_explicit_allow" in blockers:
        actions.append("use_diagnostic_strategy_only_for_failure_isolation")
    if "internal_execution_strategy_gate_disabled" in blockers:
        actions.append("keep_internal_strategy_gate_disabled_by_default")
    return actions


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_MULTI_BATCH_EXECUTION_STRATEGY_GATE",
    "build_lulynx_multi_batch_execution_strategy_gate",
]
