"""Forward strategy switch reports for Lulynx multi-batch execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_MULTI_BATCH_FORWARD_STRATEGY_SWITCH = "lulynx_multi_batch_forward_strategy_switch_v0"
_SUPPORTED_RUNTIME_ROUTES = {"existing_eager_forward_path", "diagnostic_microbatch_forward_path"}


def build_lulynx_multi_batch_forward_strategy_switch(
    *,
    execution_strategy: Mapping[str, Any] | None,
    execution_strategy_gate: Mapping[str, Any] | None,
    requested_strategy: str = "",
    implemented_runtime_routes: Sequence[str] = ("existing_eager_forward_path",),
) -> dict[str, Any]:
    """Describe the selected forward route without changing model execution."""

    strategy = execution_strategy if isinstance(execution_strategy, Mapping) else {}
    gate = execution_strategy_gate if isinstance(execution_strategy_gate, Mapping) else {}
    strategy_name = str(strategy.get("strategy") or "")
    requested = str(requested_strategy or strategy_name or "")
    implemented = {str(item) for item in implemented_runtime_routes if item}
    blockers: list[str] = []
    cautions: list[str] = []

    if str(strategy.get("contract") or "") != "lulynx_multi_batch_execution_strategy_v0":
        blockers.append("missing_multi_batch_execution_strategy")
    if str(gate.get("gate") or "") != "lulynx_multi_batch_execution_strategy_gate_v0":
        blockers.append("missing_multi_batch_execution_strategy_gate")
    if requested and strategy_name and requested != strategy_name:
        blockers.append("requested_strategy_differs_from_contract_strategy")
    if bool(strategy.get("diagnostic_only")):
        cautions.append("diagnostic_strategy_not_release_claim")
    gate_can_route = bool(gate.get("can_route_requested_strategy"))
    gate_ready_report_only = bool(gate.get("ready_behind_disabled_internal_gate"))
    if not gate_can_route and not gate_ready_report_only:
        blockers.append("execution_strategy_gate_not_ready")

    selected_route = "existing_eager_forward_path"
    switch_mode = "report_only_existing_path"
    if gate_can_route and strategy_name == "native_batch_forward":
        selected_route = "existing_eager_forward_path"
        switch_mode = "native_strategy_uses_existing_eager_forward_path"
    elif gate_can_route and strategy_name == "microbatch_forward_diagnostic":
        selected_route = "diagnostic_microbatch_forward_path"
        switch_mode = "diagnostic_microbatch_forward_path"
        if selected_route not in implemented:
            blockers.append("diagnostic_forward_route_not_implemented")
    elif gate_can_route and strategy_name == "single_item_forward_debug":
        selected_route = "diagnostic_microbatch_forward_path"
        switch_mode = "single_item_debug_uses_diagnostic_microbatch_forward_path"
        if selected_route not in implemented:
            blockers.append("diagnostic_forward_route_not_implemented")
    if selected_route and selected_route not in implemented and selected_route not in _SUPPORTED_RUNTIME_ROUTES:
        blockers.append("selected_forward_route_not_implemented")

    return {
        "schema_version": 1,
        "switch": LULYNX_MULTI_BATCH_FORWARD_STRATEGY_SWITCH,
        "status": "ready_to_use_selected_forward_route" if not blockers else "blocked",
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_change_forward_path": selected_route == "existing_eager_forward_path",
        "requested_strategy": requested,
        "strategy": strategy_name,
        "gate_status": str(gate.get("status") or ""),
        "switch_mode": switch_mode,
        "selected_forward_route": selected_route,
        "implemented_runtime_routes": sorted(implemented),
        "blockers": _dedupe(blockers),
        "cautions": _dedupe(cautions),
    }


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
    "LULYNX_MULTI_BATCH_FORWARD_STRATEGY_SWITCH",
    "build_lulynx_multi_batch_forward_strategy_switch",
]
