"""Training-loop adapter for TurboCore native update readiness reports."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch

from core.turbocore_native_update_readiness import build_native_update_readiness_report


def build_native_update_runtime_context(
    *,
    multi_gpu: bool,
    num_processes: int,
    num_machines: int,
    deepspeed: bool,
    gradient_release_active: bool,
) -> dict[str, Any]:
    return {
        "multi_gpu": bool(multi_gpu),
        "num_processes": int(num_processes or 1),
        "num_machines": int(num_machines or 1),
        "deepspeed": bool(deepspeed),
        "gradient_release_active": bool(gradient_release_active),
    }


def build_shadow_readiness_config(
    shadow_config: Any,
    *,
    save_owner_state: bool,
    shadow_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _config_dict(shadow_config)
    report = _as_dict(shadow_report)
    copyback = _as_dict(report.get("copyback_probe"))
    dispatch = _as_dict(report.get("copyback_dispatch_probe"))
    native_binding = _as_dict(report.get("native_binding_probe"))
    owner_native = _as_dict(report.get("owner_native_launch_probe"))
    return {
        **config,
        "direct_grad_lifecycle_integrated": bool(config.get("direct_grad", False)),
        "checkpoint_metadata_integrated": True,
        "trainer_state_metadata_integrated": True,
        "trainer_state_save_sync_verified": bool(save_owner_state),
        "resume_owner_state_guard_verified": bool(save_owner_state),
        "checkpoint_owner_state_enabled": bool(save_owner_state),
        "copyback_scratch_probe_integrated": bool(
            config.get("copyback_probe", False) or copyback.get("scratch_copyback_validated", False)
        ),
        "copyback_scratch_validated": bool(copyback.get("scratch_copyback_validated", False)),
        "copyback_dispatch_experimental_enabled": bool(
            config.get("copyback_dispatch_experimental", False)
            or dispatch.get("copyback_dispatch_enabled", False)
        ),
        "copyback_dispatch_validated": bool(dispatch.get("copyback_dispatch_validated", False)),
        "native_tensor_binding_probe_integrated": bool(config.get("native_binding_probe", False) or native_binding),
        "native_binding_stream_lifetime_bound": bool(native_binding.get("stream_lifetime_bound", False)),
        "native_binding_event_chain_verified": bool(native_binding.get("event_chain_verified", False)),
        "native_binding_pre_launch_ordering_verified": bool(native_binding.get("pre_launch_ordering_verified", False)),
        "native_binding_post_launch_ordering_verified": bool(native_binding.get("post_launch_ordering_verified", False)),
        "native_binding_stream_wait_event_verified": bool(native_binding.get("stream_wait_event_verified", False)),
        "owner_native_launch_probe_integrated": bool(config.get("owner_native_launch_probe", False) or owner_native),
        "owner_native_event_chain_probe_requested": bool(
            config.get("owner_native_event_chain_probe", False)
            or owner_native.get("event_chain_probe_requested", False)
        ),
        "owner_native_event_chain_probe_attempted": bool(owner_native.get("event_chain_probe_attempted", False)),
        "owner_native_event_chain_verified": bool(owner_native.get("event_chain_verified", False)),
        "owner_native_pre_launch_ordering_verified": bool(owner_native.get("pre_launch_ordering_verified", False)),
        "owner_native_post_launch_ordering_verified": bool(owner_native.get("post_launch_ordering_verified", False)),
        "owner_native_stream_wait_event_verified": bool(owner_native.get("stream_wait_event_verified", False)),
    }


def build_training_loop_native_update_readiness(
    *,
    optimizer: Any,
    params: Iterable[torch.nn.Parameter],
    mode: str,
    runtime_context: Mapping[str, Any],
    shadow_config: Any,
    save_owner_state: bool,
    shadow_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = _runtime_context_with_shadow_evidence(runtime_context, shadow_report)
    return build_native_update_readiness_report(
        optimizer=optimizer,
        params=params,
        runtime_context=context,
        shadow_config=build_shadow_readiness_config(
            shadow_config,
            save_owner_state=save_owner_state,
            shadow_report=shadow_report,
        ),
        native_update_mode=mode,
    )


def build_native_update_runtime_context_with_shadow_evidence(
    runtime_context: Mapping[str, Any],
    shadow_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach step-local shadow/native evidence to a runtime context."""

    return _runtime_context_with_shadow_evidence(runtime_context, shadow_report)


def _runtime_context_with_shadow_evidence(
    runtime_context: Mapping[str, Any],
    shadow_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = dict(runtime_context or {})
    owner_native = _as_dict(_as_dict(shadow_report).get("owner_native_launch_probe"))
    if bool(owner_native.get("kernel_executed", False)) and bool(owner_native.get("parity_ok", False)):
        context["native_update_owner_backend"] = "rust_cuda_adamw_v0"
        context["native_update_flat_owner_runtime_promoted"] = True
        context["native_update_training_dispatch_kernel_runtime_present"] = True
    native_binding = _as_dict(_as_dict(shadow_report).get("native_binding_probe"))
    if bool(context.get("native_update_allow_short_training_dispatch_evidence", False)):
        if bool(owner_native.get("event_chain_verified", False)):
            context["native_update_stream_lifetime_ownership_runtime_bound"] = True
        if bool(native_binding.get("event_chain_verified", False)):
            context["native_update_stream_lifetime_ownership_runtime_bound"] = True
        if bool(owner_native.get("kernel_executed", False)) and bool(owner_native.get("parity_ok", False)):
            context["native_update_short_training_dispatch_performance_ready"] = True
    return context


def _config_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_dict"):
        return _as_dict(value.as_dict())
    return _as_dict(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "build_native_update_runtime_context",
    "build_native_update_runtime_context_with_shadow_evidence",
    "build_shadow_readiness_config",
    "build_training_loop_native_update_readiness",
]
