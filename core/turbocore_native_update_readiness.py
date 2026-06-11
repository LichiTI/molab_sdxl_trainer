"""Cheap readiness facts for future TurboCore native update dispatch.

This module intentionally does not run benchmarks or launch native kernels.  It
turns current capability metadata and trainer context into a small report that
can be emitted beside the native-update gate.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch

try:  # pragma: no cover - import layout differs in direct smoke runs
    from core.turbocore_capabilities import probe_native_training_bridge
    from core.turbocore_native_abi import validate_native_optimizer_stateful_capability
except Exception:  # pragma: no cover
    probe_native_training_bridge = None  # type: ignore[assignment]
    validate_native_optimizer_stateful_capability = None  # type: ignore[assignment]


def build_native_update_readiness_report(
    *,
    optimizer: Any,
    params: Iterable[torch.nn.Parameter],
    runtime_context: Mapping[str, Any] | None = None,
    shadow_config: Mapping[str, Any] | None = None,
    native_update_mode: str = "off",
) -> dict[str, Any]:
    """Return report-only readiness facts for native AdamW update dispatch."""

    param_list = [param for param in params if isinstance(param, torch.nn.Parameter) and param.requires_grad]
    context = dict(runtime_context or {})
    shadow = dict(shadow_config or {})
    bridge = _probe_bridge()
    validation = _validate_optimizer_capability(bridge)
    features = _as_dict(bridge.get("features"))
    native_optimizer = _as_dict(features.get("native_optimizer"))
    flat_owner = _as_dict(native_optimizer.get("flat_owner"))
    binding = _as_dict(flat_owner.get("binding_request"))
    registry = _as_dict(binding.get("kernel_registry"))

    static_checks = _static_checks(optimizer, param_list, context, shadow)
    owner_checks = _owner_checks(param_list, flat_owner, shadow, context)
    native_checks = _native_checks(bridge, validation, flat_owner, binding, registry, shadow, context)
    blocked = _dedupe(
        static_checks["blocked_reasons"]
        + owner_checks["blocked_reasons"]
        + native_checks["blocked_reasons"]
    )
    return {
        "schema_version": 1,
        "report": "turbocore_native_update_readiness_v0",
        "mode": _normalize_mode(native_update_mode),
        "ok": not blocked,
        "training_path_enabled": False,
        "native_kernel_present": bool(native_checks["native_kernel_present"]),
        "training_dispatch_kernel_present": bool(native_checks["training_dispatch_kernel_present"]),
        "diagnostic_runtime_available": bool(native_checks["diagnostic_runtime_available"]),
        "performance_test_ready": bool(native_checks.get("performance_test_ready", False)),
        "short_training_dispatch_performance_ready": bool(
            native_checks.get("short_training_dispatch_performance_ready", False)
        ),
        "stream_lifetime_bound": bool(native_checks["stream_lifetime_bound"]),
        "stream_lifetime_ownership_bound": bool(native_checks["stream_lifetime_ownership_bound"]),
        "stream_ordering_verified": bool(native_checks["stream_ordering_verified"]),
        "event_chain_verified": bool(native_checks["event_chain_verified"]),
        "static_checks": static_checks,
        "owner_checks": owner_checks,
        "native_checks": native_checks,
        "blocked_reasons": blocked,
    }


def _static_checks(
    optimizer: Any,
    params: list[torch.nn.Parameter],
    context: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer_name = type(optimizer).__name__ if optimizer is not None else ""
    devices = sorted({str(param.device) for param in params})
    dtypes = sorted({str(param.dtype).replace("torch.", "") for param in params})
    shadow_mode = str(shadow.get("mode", "off") or "off")
    blocked: list[str] = []
    if "adamw" not in optimizer_name.lower():
        blocked.append("optimizer_not_adamw")
    if not params:
        blocked.append("no_trainable_params")
    if len(devices) > 1:
        blocked.append("trainable_params_multi_device")
    if bool(context.get("multi_gpu", False)) or int(context.get("num_processes", 1) or 1) > 1:
        blocked.append("distributed_not_supported")
    if int(context.get("num_machines", 1) or 1) > 1:
        blocked.append("distributed_not_supported")
    if bool(context.get("deepspeed", False)):
        blocked.append("deepspeed_not_supported")
    if bool(context.get("gradient_release_active", False)):
        blocked.append("gradient_release_not_supported")
    if shadow_mode not in {"profile", "shadow"}:
        blocked.append("shadow_mode_not_enabled")
    return {
        "ok": not blocked,
        "optimizer": optimizer_name,
        "parameter_tensors": len(params),
        "parameter_numel": int(sum(param.numel() for param in params)),
        "devices": devices,
        "dtypes": dtypes,
        "shadow_mode": shadow_mode,
        "shadow_direct_grad_requested": bool(shadow.get("direct_grad", False)),
        "runtime_context": {
            "multi_gpu": bool(context.get("multi_gpu", False)),
            "num_processes": int(context.get("num_processes", 1) or 1),
            "num_machines": int(context.get("num_machines", 1) or 1),
            "deepspeed": bool(context.get("deepspeed", False)),
            "gradient_release_active": bool(context.get("gradient_release_active", False)),
        },
        "blocked_reasons": _dedupe(blocked),
    }


def _owner_checks(
    params: list[torch.nn.Parameter],
    flat_owner: Mapping[str, Any],
    shadow: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    blocked: list[str] = []
    has_params = bool(params)
    same_device = len({str(param.device) for param in params}) <= 1
    flat_layout_possible = bool(has_params and same_device)
    if not flat_layout_possible:
        blocked.append("flat_owner_layout_not_ready")
    if not bool(flat_owner.get("owns_parameter_buffer", False)):
        blocked.append("native_owner_parameter_buffer_contract_missing")
    if not bool(flat_owner.get("owns_gradient_buffer", False)):
        blocked.append("native_owner_gradient_buffer_contract_missing")
    direct_gradient_write_boundary_ready = bool(flat_owner.get("owns_gradient_buffer", False))
    direct_gradient_write_native_supported = bool(flat_owner.get("supports_direct_gradient_write", False))
    explicit_training_context = bool(
        context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    direct_gradient_write_default_off = bool(direct_gradient_write_boundary_ready and not explicit_training_context)
    direct_lifecycle_integrated = bool(shadow.get("direct_grad_lifecycle_integrated", False))
    checkpoint_metadata_integrated = bool(shadow.get("checkpoint_metadata_integrated", False))
    checkpoint_owner_state_enabled = bool(shadow.get("checkpoint_owner_state_enabled", False))
    copyback_probe_integrated = bool(
        shadow.get("copyback_scratch_probe_integrated", False)
        or shadow.get("copyback_probe_integrated", False)
        or shadow.get("copyback_scratch_validated", False)
    )
    copyback_scratch_validated = bool(shadow.get("copyback_scratch_validated", False))
    copyback_dispatch_requested = bool(shadow.get("copyback_dispatch_experimental_enabled", False))
    copyback_dispatch_validated = bool(shadow.get("copyback_dispatch_validated", False))
    copyback_dispatch_enabled = bool(copyback_dispatch_requested and copyback_dispatch_validated)
    native_tensor_binding_probe_integrated = bool(shadow.get("native_tensor_binding_probe_integrated", False))
    owner_native_launch_probe_integrated = bool(shadow.get("owner_native_launch_probe_integrated", False))
    owner_native_event_chain_probe_requested = bool(shadow.get("owner_native_event_chain_probe_requested", False))
    owner_gradient_sync_boundary_ready = bool(flat_layout_possible and flat_owner.get("owns_parameter_buffer", False) and flat_owner.get("owns_gradient_buffer", False))
    owner_gradient_sync_supported = owner_gradient_sync_boundary_ready
    owner_gradient_sync_default_off = bool(owner_gradient_sync_boundary_ready and not explicit_training_context)
    owner_gradient_sync_training_integrated = bool(copyback_dispatch_enabled and checkpoint_metadata_integrated)
    owner_gradient_sync_guard_enabled = bool(context.get("native_update_owner_gradient_sync_guard_enabled", False))
    owner_gradient_sync_bound = bool(context.get("native_update_owner_gradient_sync_bound", False))
    direct_gradient_write_optional_blocked: list[str] = []
    if direct_gradient_write_default_off:
        direct_gradient_write_optional_blocked.append("direct_gradient_write_default_off")
    elif not direct_gradient_write_native_supported:
        direct_gradient_write_optional_blocked.append("direct_gradient_write_not_native_supported")
    elif not direct_lifecycle_integrated:
        direct_gradient_write_optional_blocked.append("direct_gradient_write_not_training_integrated")
    if not owner_gradient_sync_boundary_ready:
        blocked.append("owner_gradient_sync_boundary_missing")
    elif owner_gradient_sync_default_off:
        blocked.append("owner_gradient_sync_default_off")
    elif not owner_gradient_sync_supported:
        blocked.append("owner_gradient_sync_not_supported")
    elif not owner_gradient_sync_training_integrated:
        blocked.append("owner_gradient_sync_not_training_integrated")
    elif not owner_gradient_sync_guard_enabled:
        blocked.append("owner_gradient_sync_guard_disabled")
    elif not owner_gradient_sync_bound:
        blocked.append("owner_gradient_sync_not_promoted")
    if not copyback_probe_integrated:
        blocked.append("parameter_owner_copyback_not_integrated")
    elif not copyback_dispatch_requested:
        blocked.append("parameter_owner_copyback_dispatch_disabled")
    elif not copyback_dispatch_validated:
        blocked.append("parameter_owner_copyback_dispatch_not_validated")
    else:
        copyback_scratch_validated = True
    if not checkpoint_metadata_integrated:
        blocked.append("trainer_checkpoint_integration_missing")
    if not checkpoint_owner_state_enabled:
        blocked.append("trainer_checkpoint_owner_state_not_enabled")
    return {
        "ok": not blocked,
        "persistent_flat_owner_contract_present": bool(flat_owner),
        "flat_layout_possible": flat_layout_possible,
        "python_owner_state_dict_available": True,
        "trainer_checkpoint_integration": checkpoint_metadata_integrated,
        "trainer_checkpoint_owner_state_enabled": checkpoint_owner_state_enabled,
        "direct_gradient_write_boundary_ready": direct_gradient_write_boundary_ready,
        "direct_gradient_write_native_supported": direct_gradient_write_native_supported,
        "direct_gradient_write_default_off": direct_gradient_write_default_off,
        "direct_gradient_write_optional": True,
        "direct_gradient_write_optional_blocked_reasons": _dedupe(direct_gradient_write_optional_blocked),
        "explicit_training_context_requested": explicit_training_context,
        "direct_gradient_write_training_integrated": direct_lifecycle_integrated,
        "owner_gradient_sync_boundary_ready": owner_gradient_sync_boundary_ready,
        "owner_gradient_sync_supported": owner_gradient_sync_supported,
        "owner_gradient_sync_default_off": owner_gradient_sync_default_off,
        "owner_gradient_sync_training_integrated": owner_gradient_sync_training_integrated,
        "owner_gradient_sync_guard_enabled": owner_gradient_sync_guard_enabled,
        "owner_gradient_sync_bound": owner_gradient_sync_bound,
        "owner_gradient_sync_preconditions_ready": bool(
            explicit_training_context
            and owner_gradient_sync_boundary_ready
            and owner_gradient_sync_supported
            and owner_gradient_sync_training_integrated
            and owner_gradient_sync_guard_enabled
            and owner_gradient_sync_bound
        ),
        "parameter_owner_copyback_probe_integrated": copyback_probe_integrated,
        "parameter_owner_copyback_scratch_validated": copyback_scratch_validated,
        "parameter_owner_copyback_dispatch_experimental_enabled": copyback_dispatch_requested,
        "parameter_owner_copyback_dispatch_validated": copyback_dispatch_validated,
        "parameter_owner_copyback_integrated": copyback_dispatch_enabled,
        "parameter_owner_copyback_dispatch_enabled": copyback_dispatch_enabled,
        "native_tensor_binding_probe_integrated": native_tensor_binding_probe_integrated,
        "owner_native_launch_probe_integrated": owner_native_launch_probe_integrated,
        "owner_native_event_chain_probe_requested": owner_native_event_chain_probe_requested,
        "blocked_reasons": _dedupe(blocked),
    }


def _native_checks(
    bridge: Mapping[str, Any],
    validation: Mapping[str, Any],
    flat_owner: Mapping[str, Any],
    binding: Mapping[str, Any],
    registry: Mapping[str, Any],
    shadow: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostic = _as_dict(bridge.get("diagnostic"))
    features = _as_dict(bridge.get("features"))
    runtime = _as_dict(features.get("cuda_adamw_runtime"))
    flat_validation = _as_dict(_as_dict(_as_dict(validation.get("features")).get("native_optimizer")).get("flat_owner"))
    binding_validation = _as_dict(flat_validation.get("binding_request"))
    kernel_contract = _as_dict(registry.get("kernel_contract"))
    tensor_object_session_supported = bool(binding.get("supports_tensor_object_sessions", False))
    external_tensor_handles_supported = bool(binding.get("supports_external_tensor_handles", False))
    current_process_binding_supported = bool(tensor_object_session_supported)
    diagnostic_runtime_available = bool(runtime.get("available", False) and runtime.get("runtime_session", False))
    diagnostic_runtime_probe_supported = _has_entrypoint(runtime, "tensor_binding_session_cuda_adamw_runtime_probe")
    diagnostic_runtime_benchmark_supported = _has_entrypoint(runtime, "benchmark_adamw_cuda_kernel_runtime_session_py")
    diagnostic_kernel_contract_present = str(kernel_contract.get("contract", "") or "") == "adamw_flat_fp32_cuda_kernel_v0"
    flat_owner_contract_ready = bool(flat_validation.get("ok", False))
    reference_flat_owner_ready = bool(flat_owner_contract_ready and flat_owner.get("reference_owner"))
    runtime_owner_backend = str(context.get("native_update_owner_backend", "") or "")
    runtime_owner_promoted = bool(
        context.get("native_update_flat_owner_runtime_promoted", False)
        or runtime_owner_backend == "rust_cuda_adamw_v0"
    )
    training_flat_owner_promoted = bool(
        (flat_owner.get("available", False) and flat_owner.get("training_path_enabled", False))
        or runtime_owner_promoted
    )
    explicit_training_context = bool(
        context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    training_flat_owner_default_off = bool(flat_owner_contract_ready and not explicit_training_context)
    kernel_contract_ready = bool(
        diagnostic_kernel_contract_present
        and str(kernel_contract.get("launch_plan", "") or "") == "adamw_flat_fp32_launch_plan_v0"
    )
    runtime_kernel_present = bool(
        context.get("native_update_training_dispatch_kernel_runtime_present", False)
        or runtime_owner_backend == "rust_cuda_adamw_v0"
    )
    training_dispatch_kernel_present = runtime_kernel_present or bool(flat_owner.get("native_kernel_present", False)) or bool(
        binding.get("native_kernel_present", False)
    ) or bool(registry.get("native_kernel_present", False)) or bool(kernel_contract.get("native_kernel_present", False))
    training_dispatch_kernel_default_off = bool(kernel_contract_ready and not explicit_training_context)
    stream_lifetime_ownership_bound = bool(
        shadow.get("native_binding_stream_lifetime_bound", False)
        or context.get("native_update_stream_lifetime_ownership_runtime_bound", False)
    )
    event_chain_verified = bool(
        shadow.get("native_binding_event_chain_verified", False)
        or shadow.get("owner_native_event_chain_verified", False)
    )
    stream_ordering_verified = bool(
        event_chain_verified
        or shadow.get("native_binding_pre_launch_ordering_verified", False)
        or shadow.get("native_binding_post_launch_ordering_verified", False)
        or shadow.get("native_binding_stream_wait_event_verified", False)
        or shadow.get("owner_native_pre_launch_ordering_verified", False)
        or shadow.get("owner_native_post_launch_ordering_verified", False)
        or shadow.get("owner_native_stream_wait_event_verified", False)
    )
    stream_lifetime_bound = bool(stream_lifetime_ownership_bound or stream_ordering_verified)
    short_training_dispatch_performance_ready = bool(
        context.get("native_update_allow_short_training_dispatch_evidence", False)
        and context.get("native_update_short_training_dispatch_performance_ready", False)
    )
    blocked: list[str] = []
    if not bool(validation.get("ok", False)):
        blocked.append("native_optimizer_schema_incomplete")
    if not flat_owner_contract_ready:
        blocked.append("native_training_flat_owner_contract_incomplete")
    elif training_flat_owner_default_off:
        blocked.append("native_training_flat_owner_default_off")
    elif not training_flat_owner_promoted:
        blocked.append("native_training_flat_owner_not_promoted")
    if not tensor_object_session_supported:
        blocked.append("tensor_binding_session_unsupported")
    if not bool(current_process_binding_supported or external_tensor_handles_supported):
        blocked.append("external_tensor_handles_unsupported")
    if not bool(registry.get("dry_run_launch_supported", False)):
        blocked.append("native_launch_plan_registry_missing")
    if not diagnostic_runtime_available:
        blocked.append("native_diagnostic_runtime_unavailable")
    if diagnostic_runtime_available and not diagnostic_runtime_probe_supported:
        blocked.append("native_diagnostic_runtime_probe_missing")
    if not kernel_contract_ready:
        blocked.append("native_training_dispatch_kernel_contract_missing")
    elif training_dispatch_kernel_default_off:
        blocked.append("native_training_dispatch_kernel_default_off")
    elif not training_dispatch_kernel_present:
        blocked.append("native_training_dispatch_kernel_not_promoted")
    if not stream_lifetime_bound:
        blocked.append("stream_lifetime_unbound")
    elif not stream_lifetime_ownership_bound:
        blocked.append("stream_lifetime_ownership_not_promoted")
    if not short_training_dispatch_performance_ready:
        blocked.append("representative_performance_gate_missing")
    return {
        "ok": not blocked,
        "provider": str(diagnostic.get("provider", "unknown") or "unknown"),
        "importable": bool(diagnostic.get("importable", False)),
        "bridge_status": str(bridge.get("status", "unknown") or "unknown"),
        "native_optimizer_schema_ok": bool(validation.get("ok", False)),
        "flat_owner_available": bool(flat_owner.get("available", False)),
        "flat_owner_status": str(flat_owner.get("status", "unknown") or "unknown"),
        "flat_owner_contract_ready": flat_owner_contract_ready,
        "training_flat_owner_boundary_ready": flat_owner_contract_ready,
        "reference_flat_owner_ready": reference_flat_owner_ready,
        "training_flat_owner_promoted": training_flat_owner_promoted,
        "training_flat_owner_runtime_promoted": runtime_owner_promoted,
        "training_flat_owner_default_off": training_flat_owner_default_off,
        "explicit_training_context_requested": explicit_training_context,
        "runtime_owner_backend": runtime_owner_backend,
        "flat_owner_reason": str(flat_owner.get("reason", "") or ""),
        "binding_request_schema_ok": bool(binding_validation.get("ok", False)),
        "tensor_binding_session_supported": tensor_object_session_supported,
        "external_tensor_handles_supported": external_tensor_handles_supported,
        "launch_plan_registry_supported": bool(registry.get("dry_run_launch_supported", False)),
        "current_process_tensor_object_session_supported": current_process_binding_supported,
        "diagnostic_kernel_contract_present": diagnostic_kernel_contract_present,
        "training_dispatch_kernel_contract_ready": kernel_contract_ready,
        "training_dispatch_kernel_promoted": training_dispatch_kernel_present,
        "training_dispatch_kernel_runtime_present": runtime_kernel_present,
        "training_dispatch_kernel_default_off": training_dispatch_kernel_default_off,
        "diagnostic_runtime_available": diagnostic_runtime_available,
        "diagnostic_runtime_status": str(runtime.get("status", "unknown") or "unknown"),
        "diagnostic_runtime_session_supported": bool(runtime.get("runtime_session", False)),
        "diagnostic_runtime_probe_supported": diagnostic_runtime_probe_supported,
        "diagnostic_runtime_benchmark_supported": diagnostic_runtime_benchmark_supported,
        "stream_lifetime_bound": stream_lifetime_bound,
        "stream_lifetime_ownership_bound": stream_lifetime_ownership_bound,
        "stream_ordering_verified": stream_ordering_verified,
        "event_chain_verified": event_chain_verified,
        "native_kernel_present": training_dispatch_kernel_present,
        "training_dispatch_kernel_present": training_dispatch_kernel_present,
        "runtime_recovery_policy_defined": True,
        "runtime_recovery_dispatch_integrated": True,
        "performance_test_ready": short_training_dispatch_performance_ready,
        "short_training_dispatch_performance_ready": short_training_dispatch_performance_ready,
        "blocked_reasons": _dedupe(blocked),
    }


def _probe_bridge() -> dict[str, Any]:
    if probe_native_training_bridge is None:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "training_path_enabled": False,
            "features": {},
            "diagnostic": {"provider": "unavailable", "reason": "probe_import_failed"},
        }
    try:
        report = probe_native_training_bridge()
    except Exception as exc:  # pragma: no cover - defensive metadata only
        return {
            "schema_version": 1,
            "status": "unavailable",
            "training_path_enabled": False,
            "features": {},
            "diagnostic": {
                "provider": "unavailable",
                "reason": "probe_failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
    return dict(report) if isinstance(report, Mapping) else {}


def _validate_optimizer_capability(bridge: Mapping[str, Any]) -> dict[str, Any]:
    if validate_native_optimizer_stateful_capability is None:
        return {
            "schema_version": 1,
            "validator": "turbocore_native_optimizer_stateful_capability",
            "ok": False,
            "training_path_enabled": False,
            "features": {},
        }
    try:
        return validate_native_optimizer_stateful_capability({"features": _as_dict(bridge.get("features"))})
    except Exception as exc:  # pragma: no cover - defensive metadata only
        return {
            "schema_version": 1,
            "validator": "turbocore_native_optimizer_stateful_capability",
            "ok": False,
            "training_path_enabled": False,
            "error": f"{type(exc).__name__}: {exc}",
            "features": {},
        }


def _normalize_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower().replace("-", "_")
    return normalized if normalized in {"off", "profile", "native_experimental"} else "off"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _has_entrypoint(report: Mapping[str, Any], name: str) -> bool:
    entrypoints = report.get("entrypoints") if isinstance(report.get("entrypoints"), list) else []
    return str(name) in {str(item) for item in entrypoints}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_native_update_readiness_report"]
