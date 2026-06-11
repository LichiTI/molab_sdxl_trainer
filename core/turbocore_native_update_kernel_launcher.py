"""Report-only kernel launcher plan for TurboCore native AdamW updates."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_stream_descriptor import current_torch_stream_descriptor


def build_native_update_kernel_launch_plan(
    *,
    dispatch_request: Mapping[str, Any] | None = None,
    dispatch_contract: Mapping[str, Any] | None = None,
    owner_native_launch_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the future native AdamW launch without executing it."""

    request = _as_dict(dispatch_request)
    contract = _as_dict(dispatch_contract)
    owner_probe = _as_dict(owner_native_launch_probe)
    requested = bool(request.get("requested", False))
    launch_evidence = _launch_evidence(owner_probe)
    blocked = _blocked_reasons(request=request, contract=contract, owner_probe=owner_probe)
    unique_blocked = _dedupe(blocked)
    launch_allowed = bool(
        requested
        and not unique_blocked
        and request.get("dispatch_allowed", False)
        and contract.get("would_allow_native_dispatch", False)
    )
    return {
        "schema_version": 1,
        "launcher": "turbocore_native_update_kernel_launcher_v0",
        "kernel": "adamw_flat_fp32_cuda_kernel_v0",
        "launch_plan": "adamw_flat_fp32_launch_plan_v0",
        "requested": requested,
        "training_dispatch": launch_allowed,
        "training_path_enabled": launch_allowed,
        "launch_allowed": launch_allowed,
        "launch_attempted": False,
        "kernel_executed": False,
        "owner_buffer_launch_only": True,
        "mutates_training_parameters": launch_allowed,
        "requires_owner_buffers": True,
        "requires_tensor_binding_session": True,
        "requires_runtime_session": True,
        "requires_stream_event_chain": True,
        "evidence": launch_evidence,
        "sequence": _sequence(requested=requested, launch_evidence=launch_evidence, launch_allowed=launch_allowed),
        "blocked_reasons": unique_blocked,
    }


def build_native_update_adamw_launch_config(
    owner: Any,
    *,
    max_numel: int,
    event_chain_probe: bool = False,
    capture_stage: str = "native_update_kernel_launcher",
) -> dict[str, Any]:
    """Build the shared native AdamW launch config for probes and dispatch."""

    cfg = owner.config
    return {
        "contract": "turbocore_native_update_adamw_launch_config_v0",
        "kernel": "adamw_flat_fp32_cuda_kernel_v0",
        "launch_plan": "adamw_flat_fp32_launch_plan_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "lr": float(cfg.lr),
        "betas": [float(cfg.betas[0]), float(cfg.betas[1])],
        "eps": float(cfg.eps),
        "weight_decay": float(cfg.weight_decay),
        "step_index": int(owner.step_index),
        "block_size": int(cfg.block_size),
        "max_numel": max(int(max_numel or 0), int(owner.param_flat.numel()), 1),
        "stream_guard_descriptor": current_torch_stream_descriptor(
            owner.param_flat.device,
            capture_stage=str(capture_stage or "native_update_kernel_launcher"),
            request_event_chain=event_chain_probe,
        ),
    }


def _launch_evidence(owner_probe: Mapping[str, Any]) -> dict[str, Any]:
    launch = _as_dict(owner_probe.get("launch"))
    return {
        "owner_native_probe_present": bool(owner_probe),
        "owner_native_probe_ok": bool(owner_probe.get("ok", False)) if owner_probe else None,
        "runtime_session_id": int(owner_probe.get("runtime_session_id", 0) or 0) if owner_probe else 0,
        "binding_session_id": int(owner_probe.get("binding_session_id", 0) or 0) if owner_probe else 0,
        "runtime_session_reused": bool(owner_probe.get("runtime_session_reused", False)) if owner_probe else False,
        "binding_session_reused": bool(owner_probe.get("binding_session_reused", False)) if owner_probe else False,
        "diagnostic_kernel_executed": bool(owner_probe.get("kernel_executed", False) or launch.get("kernel_executed", False)),
        "diagnostic_parity_ok": bool(owner_probe.get("parity_ok", False)) if owner_probe else False,
        "event_chain_verified": bool(owner_probe.get("event_chain_verified", False)) if owner_probe else False,
        "persistent_owner_mutated": bool(owner_probe.get("persistent_owner_mutated", False)) if owner_probe else False,
        "diagnostic_elapsed_ms": _float_or_none(owner_probe.get("elapsed_ms")) if owner_probe else None,
    }


def _sequence(*, requested: bool, launch_evidence: Mapping[str, Any], launch_allowed: bool = False) -> list[dict[str, Any]]:
    has_sessions = bool(launch_evidence.get("runtime_session_id", 0) and launch_evidence.get("binding_session_id", 0))
    return [
        {"step": "validate_dispatch_request", "planned": bool(requested), "enabled": bool(launch_allowed)},
        {"step": "reuse_or_create_runtime_session", "planned": bool(requested), "enabled": bool(launch_allowed)},
        {"step": "reuse_or_create_tensor_binding_session", "planned": bool(requested), "enabled": bool(launch_allowed)},
        {"step": "verify_stream_event_chain", "planned": bool(requested), "enabled": bool(launch_allowed)},
        {"step": "launch_adamw_flat_fp32_cuda_kernel", "planned": bool(requested and has_sessions), "enabled": bool(launch_allowed and has_sessions)},
        {"step": "record_launch_report", "planned": True, "enabled": bool(launch_allowed)},
    ]


def _blocked_reasons(
    *,
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    owner_probe: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if not request:
        blocked.append("dispatch_request_missing")
    elif not bool(request.get("dispatch_allowed", False)):
        blocked.append("dispatch_request_not_allowed")
    if not contract:
        blocked.append("dispatch_contract_missing")
    elif not bool(contract.get("would_allow_native_dispatch", False)):
        blocked.append("dispatch_contract_not_allowing_launch")
    if not owner_probe:
        blocked.append("owner_native_launch_probe_missing")
    elif not bool(owner_probe.get("ok", False)):
        blocked.append("owner_native_launch_probe_not_ok")
    if owner_probe and not bool(owner_probe.get("kernel_executed", False)):
        blocked.append("diagnostic_kernel_not_executed")
    if owner_probe and not bool(owner_probe.get("parity_ok", False)):
        blocked.append("owner_native_launch_parity_failed")
    if owner_probe and bool(owner_probe.get("persistent_owner_mutated", False)):
        blocked.append("owner_native_probe_mutated_persistent_owner")
    if owner_probe and bool(owner_probe.get("event_chain_probe_requested", False)) and not bool(owner_probe.get("event_chain_verified", False)):
        blocked.append("owner_native_event_chain_not_verified")
    if not bool(request.get("runtime_dispatch_available", False)):
        blocked.append("native_dispatch_runtime_not_implemented")
    if not bool(request.get("training_path_enabled", False)):
        blocked.append("native_dispatch_training_path_disabled")
    return blocked


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result



__all__ = ["build_native_update_adamw_launch_config", "build_native_update_kernel_launch_plan"]
