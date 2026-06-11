"""Default-off dispatch request planner for TurboCore native updates."""

from __future__ import annotations

from typing import Any, Mapping


RUNTIME_BLOCKER = "native_dispatch_runtime_not_implemented"


def build_native_update_dispatch_request(
    *,
    mode: str,
    dispatch_enabled: bool,
    gate_report: Mapping[str, Any] | None = None,
    dispatch_contract: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-step native dispatch request plan without executing it."""

    normalized = _normalize_mode(mode)
    gate = _as_dict(gate_report)
    contract = _as_dict(dispatch_contract)
    context = _as_dict(runtime_context)
    requested = bool(dispatch_enabled and normalized == "native_experimental")
    training_path_request = _training_path_request(
        requested=requested,
        context=context,
    )
    blocked = _blocked_reasons(
        requested=requested,
        mode=normalized,
        gate=gate,
        contract=contract,
        context=context,
    )
    unique_blocked = _dedupe(blocked)
    dispatch_allowed = bool(
        requested
        and not unique_blocked
        and gate.get("would_enable_native_update", False)
        and contract.get("would_allow_native_dispatch", False)
        and training_path_request.get("explicit_training_path_requested", False)
        and training_path_request.get("runtime_dispatch_available", False)
        and training_path_request.get("training_mutation_guard_enabled", False)
    )
    return {
        "schema_version": 1,
        "request": "turbocore_native_update_dispatch_request_v0",
        "mode": normalized,
        "dispatch_enabled": bool(dispatch_enabled),
        "requested": requested,
        "training_dispatch": dispatch_allowed,
        "training_path_enabled": dispatch_allowed,
        "dispatch_allowed": dispatch_allowed,
        "runtime_dispatch_available": bool(training_path_request.get("runtime_dispatch_available", False)),
        "pytorch_optimizer_authoritative": not dispatch_allowed,
        "fallback_to_pytorch_required": not dispatch_allowed,
        "plan": _dispatch_plan(requested=requested, contract=contract, dispatch_allowed=dispatch_allowed),
        "training_path_request": training_path_request,
        "evidence": {
            "gate_present": bool(gate),
            "gate_would_enable_native_update": bool(gate.get("would_enable_native_update", False)),
            "contract_present": bool(contract),
            "contract_rehearsal_ready": bool(contract.get("dispatch_rehearsal_ready", False)),
            "contract_would_allow_native_dispatch": bool(contract.get("would_allow_native_dispatch", False)),
            "native_kernel_present": bool(contract.get("native_kernel_present", gate.get("native_kernel_present", False))),
            "stream_lifetime_bound": bool(contract.get("stream_lifetime_bound", gate.get("stream_lifetime_bound", False))),
            "performance_test_ready": bool(contract.get("performance_test_ready", gate.get("performance_test_ready", False))),
            "training_path_request_boundary_ready": bool(training_path_request.get("request_boundary_ready", False)),
            "training_path_explicitly_requested": bool(training_path_request.get("explicit_training_path_requested", False)),
            "training_path_default_off": bool(training_path_request.get("default_off", True)),
        },
        "blocked_reasons": unique_blocked,
    }


def _blocked_reasons(
    *,
    requested: bool,
    mode: str,
    gate: Mapping[str, Any],
    contract: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if not requested:
        blocked.append("native_dispatch_not_requested")
    if mode != "native_experimental":
        blocked.append("native_dispatch_requires_native_experimental_mode")
    if not gate:
        blocked.append("native_update_gate_report_missing")
    elif not bool(gate.get("would_enable_native_update", False)):
        blocked.append("native_update_gate_not_enabled")
    if not contract:
        blocked.append("native_dispatch_contract_missing")
    else:
        if not bool(contract.get("dispatch_rehearsal_ready", False)):
            blocked.append("native_dispatch_rehearsal_not_ready")
        if not bool(contract.get("would_allow_native_dispatch", False)):
            blocked.append("native_dispatch_contract_not_allowing_dispatch")
        blocked.extend(_strings(contract.get("blocked_reasons")))
    if bool(context.get("gradient_release_active", False)):
        blocked.append("gradient_release_not_supported")
    if bool(context.get("multi_gpu", False)) or int(context.get("num_processes", 1) or 1) > 1:
        blocked.append("distributed_not_supported")
    if bool(context.get("deepspeed", False)):
        blocked.append("deepspeed_not_supported")
    if not bool(context.get("native_update_runtime_dispatch_available", False)):
        blocked.append(RUNTIME_BLOCKER)
    return blocked


def _dispatch_plan(*, requested: bool, contract: Mapping[str, Any], dispatch_allowed: bool = False) -> dict[str, Any]:
    sequence = contract.get("dispatch_sequence") if isinstance(contract.get("dispatch_sequence"), list) else []
    return {
        "planner": "turbocore_native_update_dispatch_plan_v0",
        "requested": bool(requested),
        "execute_native_step": bool(dispatch_allowed),
        "mutate_training_parameters": bool(dispatch_allowed),
        "call_pytorch_optimizer_step": not bool(dispatch_allowed),
        "call_python_scheduler": True,
        "zero_grad_owner_buffers": bool(dispatch_allowed),
        "sequence": [
            {**dict(item), "enabled": bool(dispatch_allowed and item.get("planned", False))}
            for item in sequence
            if isinstance(item, Mapping)
        ],
    }


def _training_path_request(*, requested: bool, context: Mapping[str, Any]) -> dict[str, Any]:
    training_dispatch_flag = bool(context.get("native_update_training_dispatch_enabled", False))
    training_path_flag = bool(context.get("training_path_enabled", False))
    runtime_available = bool(context.get("native_update_runtime_dispatch_available", False))
    mutation_guard = bool(context.get("native_update_training_mutation_guard_enabled", False))
    explicit = bool(requested and training_dispatch_flag and training_path_flag)
    blocked: list[str] = []
    if not explicit:
        blocked.append("native_dispatch_training_path_default_off")
    if explicit and not runtime_available:
        blocked.append("native_dispatch_runtime_not_implemented")
    if explicit and not mutation_guard:
        blocked.append("native_dispatch_training_mutation_guard_disabled")
    return {
        "schema_version": 1,
        "request": "turbocore_native_update_training_path_request_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "request_boundary_ready": True,
        "explicit_training_path_requested": explicit,
        "training_dispatch_flag_enabled": training_dispatch_flag,
        "training_path_flag_enabled": training_path_flag,
        "runtime_dispatch_available": runtime_available,
        "training_mutation_guard_enabled": mutation_guard,
        "default_off": not explicit,
        "blocked_reasons": _dedupe(blocked),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _normalize_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower().replace("-", "_")
    return normalized if normalized in {"off", "profile", "native_experimental"} else "off"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result



__all__ = ["build_native_update_dispatch_request"]
