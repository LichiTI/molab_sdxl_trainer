"""Report-only dispatch contract for future TurboCore native updates."""

from __future__ import annotations

from typing import Any, Mapping


REQUIRED_CONTRACT_BLOCKERS = (
    "native_dispatch_runtime_not_implemented",
    "native_dispatch_training_path_disabled",
)


def build_native_update_dispatch_contract(
    *,
    mode: str,
    requested: bool,
    readiness_report: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    dispatch_preflight: Mapping[str, Any] | None = None,
    fallback_policy: Mapping[str, Any] | None = None,
    gate_blocked_reasons: list[str] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable rehearsal report without enabling native dispatch."""

    readiness = _as_dict(readiness_report)
    shadow = _as_dict(shadow_report)
    preflight = _as_dict(dispatch_preflight)
    context = _as_dict(runtime_context)
    fallback = _as_dict(fallback_policy) or _as_dict(shadow.get("fallback_policy"))
    owner_native = _as_dict(shadow.get("owner_native_launch_probe"))
    copyback = _as_dict(shadow.get("copyback_dispatch_probe"))
    binding = _as_dict(shadow.get("native_binding_probe"))
    runtime_recovery = _as_dict(fallback.get("runtime_recovery"))
    performance = _as_dict(_as_dict(preflight.get("evidence")).get("performance"))
    direct_gradient_write = _direct_gradient_write_summary(context=context, readiness=readiness, requested=bool(requested))
    owner_gradient_sync = _owner_gradient_sync_summary(context=context, readiness=readiness, requested=bool(requested))
    training_flat_owner = _training_flat_owner_summary(context=context, readiness=readiness, requested=bool(requested))
    training_dispatch_kernel = _training_dispatch_kernel_summary(
        context=context,
        readiness=readiness,
        owner_native=owner_native,
        requested=bool(requested),
    )
    training_executor = _training_executor_summary(context=context, requested=bool(requested))
    stream_lifetime = _stream_lifetime_ownership_summary(
        context=context,
        readiness=readiness,
        preflight=preflight,
        binding=binding,
        owner_native=owner_native,
        requested=bool(requested),
    )

    rehearsal = _rehearsal_status(
        requested=bool(requested),
        owner_native=owner_native,
        copyback=copyback,
        binding=binding,
        fallback=fallback,
    )
    blocked = _collect_blockers(
        preflight=preflight,
        fallback=fallback,
        runtime_recovery=runtime_recovery,
        performance=performance,
        gate_blocked_reasons=gate_blocked_reasons,
    )
    if not requested:
        blocked.append("native_update_not_requested")
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
        and context.get("native_update_runtime_dispatch_available", False)
    )
    training_contract_ready = bool(
        explicit_training_context
        and owner_gradient_sync.get("owner_gradient_sync_preconditions_ready", False)
        and training_flat_owner.get("training_flat_owner_preconditions_ready", False)
        and training_dispatch_kernel.get("training_dispatch_kernel_preconditions_ready", False)
        and training_executor.get("training_executor_preconditions_ready", False)
        and stream_lifetime.get("stream_lifetime_ownership_preconditions_ready", False)
        and _recovery_summary(fallback, runtime_recovery).get("training_dispatch_recovery_ready", False)
    )
    if not training_contract_ready:
        blocked.extend(REQUIRED_CONTRACT_BLOCKERS)
    unique_blocked = _dedupe(blocked)
    dispatch_allowed = bool(training_contract_ready and not unique_blocked)

    return {
        "schema_version": 1,
        "contract": "turbocore_native_update_dispatch_contract_v0",
        "mode": _normalize_mode(mode),
        "requested": bool(requested),
        "training_dispatch": dispatch_allowed,
        "training_path_enabled": dispatch_allowed,
        "dispatch_rehearsal_ready": dispatch_allowed,
        "would_allow_native_dispatch": dispatch_allowed,
        "pytorch_optimizer_authoritative": not dispatch_allowed,
        "native_mutation_allowed": dispatch_allowed,
        "training_parameter_mutation_allowed": dispatch_allowed,
        "scheduler_stays_python_side": True,
        "native_kernel_present": bool(
            readiness.get("native_kernel_present", False)
            or owner_native.get("kernel_executed", False)
            or preflight.get("native_kernel_present", False)
        ),
        "stream_lifetime_bound": bool(
            readiness.get("stream_lifetime_bound", False)
            or binding.get("stream_lifetime_bound", False)
            or preflight.get("stream_lifetime_bound", False)
            or preflight.get("stream_ordering_verified", False)
        ),
        "stream_lifetime_ownership_bound": bool(
            readiness.get("stream_lifetime_ownership_bound", False)
            or binding.get("stream_lifetime_bound", False)
            or preflight.get("stream_lifetime_ownership_bound", False)
        ),
        "stream_ordering_verified": bool(
            readiness.get("stream_ordering_verified", False)
            or binding.get("event_chain_verified", False)
            or owner_native.get("event_chain_verified", False)
            or preflight.get("stream_ordering_verified", False)
        ),
        "performance_test_ready": bool(
            readiness.get("performance_test_ready", False)
            or preflight.get("performance_test_ready", False)
            or performance.get("performance_test_ready", False)
        ),
        "training_dispatch_performance_gate_ready": bool(
            preflight.get("training_dispatch_performance_gate_ready", False)
            or performance.get("training_dispatch_performance_gate_ready", False)
            or performance.get("representative_performance_gate_ready", False)
        ),
        "rehearsal": rehearsal,
        "dispatch_sequence": _dispatch_sequence(rehearsal),
        "recovery": _recovery_summary(fallback, runtime_recovery),
        "direct_gradient_write": direct_gradient_write,
        "owner_gradient_sync": owner_gradient_sync,
        "training_flat_owner": training_flat_owner,
        "training_dispatch_kernel": training_dispatch_kernel,
        "training_executor": training_executor,
        "stream_lifetime_ownership": stream_lifetime,
        "evidence": _evidence(preflight, fallback, runtime_recovery, performance, owner_native, copyback, binding),
        "actions_required": _actions_required(unique_blocked),
        "blocked_reasons": unique_blocked,
    }


def _rehearsal_status(
    *,
    requested: bool,
    owner_native: Mapping[str, Any],
    copyback: Mapping[str, Any],
    binding: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    owner_kernel_ok = bool(owner_native.get("kernel_executed", False) and owner_native.get("parity_ok", False))
    copyback_target = str(copyback.get("copyback_dispatch_target", "") or "")
    copyback_ok = bool(copyback.get("copyback_dispatch_validated", False))
    event_verified = bool(binding.get("event_chain_verified", False) or owner_native.get("event_chain_verified", False))
    stream_ready = bool(
        event_verified
        or binding.get("pre_launch_ordering_verified", False)
        or binding.get("post_launch_ordering_verified", False)
        or binding.get("stream_wait_event_verified", False)
        or owner_native.get("pre_launch_ordering_verified", False)
        or owner_native.get("post_launch_ordering_verified", False)
        or owner_native.get("stream_wait_event_verified", False)
    )
    would_copyback = bool(copyback_ok and copyback_target == "training_parameters")
    return {
        "stage": "report_only_rehearsal",
        "would_use_owner_buffers": bool(requested and owner_native),
        "would_sync_gradients_to_owner": bool(requested),
        "would_bind_training_tensors": bool(requested and binding),
        "would_launch_native_kernel": bool(owner_kernel_ok),
        "would_copyback_to_training_parameters": would_copyback,
        "would_zero_owner_grad": bool(owner_kernel_ok and would_copyback),
        "would_step_scheduler_after_successful_dispatch": False,
        "would_fallback_to_pytorch_on_error": bool(fallback.get("fallback_to_pytorch_enabled", True)),
        "strict_fail_fast": bool(fallback.get("strict", False)),
        "copyback_dispatch_target": copyback_target or None,
        "stream_ordering_ready": stream_ready,
    }


def _dispatch_sequence(rehearsal: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"step": "validate_gate_and_preflight", "planned": True, "enabled": False},
        {"step": "bind_owner_buffers_and_training_tensors", "planned": bool(rehearsal.get("would_bind_training_tensors", False)), "enabled": False},
        {"step": "sync_gradients_to_owner", "planned": bool(rehearsal.get("would_sync_gradients_to_owner", False)), "enabled": False},
        {"step": "wait_for_training_stream_event_chain", "planned": bool(rehearsal.get("stream_ordering_ready", False)), "enabled": False},
        {"step": "launch_native_adamw_kernel", "planned": bool(rehearsal.get("would_launch_native_kernel", False)), "enabled": False},
        {"step": "copy_owner_params_to_training_parameters", "planned": bool(rehearsal.get("would_copyback_to_training_parameters", False)), "enabled": False},
        {"step": "zero_owner_gradients", "planned": bool(rehearsal.get("would_zero_owner_grad", False)), "enabled": False},
        {"step": "return_control_to_python_scheduler", "planned": True, "enabled": False},
    ]


def _recovery_summary(fallback: Mapping[str, Any], runtime_recovery: Mapping[str, Any]) -> dict[str, Any]:
    integration = _as_dict(runtime_recovery.get("integration"))
    return {
        "fallback_to_pytorch_enabled": bool(fallback.get("fallback_to_pytorch_enabled", True)),
        "fail_fast_if_strict": bool(fallback.get("fail_fast_if_strict", fallback.get("strict", False))),
        "disable_native_update_for_run": bool(runtime_recovery.get("disable_native_update_for_run", False)),
        "policy_observation_integrated": bool(runtime_recovery.get("policy_observation_integrated", False)),
        "run_disable_latch_integrated": bool(runtime_recovery.get("run_disable_latch_integrated", False)),
        "pre_step_arming_observes_latch": bool(runtime_recovery.get("pre_step_arming_observes_latch", False)),
        "default_off_recovery_bridge_ready": bool(runtime_recovery.get("default_off_recovery_bridge_ready", False)),
        "recovery_observation_bridge_ready": bool(runtime_recovery.get("recovery_observation_bridge_ready", False)),
        "training_dispatch_recovery_ready": bool(runtime_recovery.get("training_dispatch_recovery_ready", False)),
        "training_dispatch_recovery_blocked": bool(runtime_recovery.get("training_dispatch_recovery_blocked", False)),
        "training_dispatch_recovery_blocker": str(runtime_recovery.get("training_dispatch_recovery_blocker", "") or ""),
        "requires_shadow_parity_revalidation": "require_shadow_parity_revalidation_after_recovery" in _strings(runtime_recovery.get("actions")),
        "integration": integration,
        "actions": _strings(runtime_recovery.get("actions")),
    }


def _direct_gradient_write_summary(
    *,
    context: Mapping[str, Any],
    readiness: Mapping[str, Any],
    requested: bool,
) -> dict[str, Any]:
    owner_checks = _as_dict(readiness.get("owner_checks"))
    boundary_ready = bool(owner_checks.get("direct_gradient_write_boundary_ready", False))
    native_supported = bool(owner_checks.get("direct_gradient_write_native_supported", False))
    lifecycle_integrated = bool(owner_checks.get("direct_gradient_write_training_integrated", False))
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    guard_enabled = bool(context.get("native_update_direct_gradient_write_guard_enabled", False))
    binding_enabled = bool(context.get("native_update_direct_gradient_write_bound", False))
    bound_to_training_path = bool(
        explicit_training_context
        and boundary_ready
        and native_supported
        and lifecycle_integrated
        and guard_enabled
        and binding_enabled
    )
    blocked: list[str] = []
    if not boundary_ready:
        blocked.append("direct_gradient_write_boundary_missing")
    elif not explicit_training_context:
        blocked.append("direct_gradient_write_default_off")
    elif not native_supported:
        blocked.append("direct_gradient_write_not_native_supported")
    elif not lifecycle_integrated:
        blocked.append("direct_gradient_write_not_training_integrated")
    elif not guard_enabled:
        blocked.append("direct_gradient_write_guard_disabled")
    elif not binding_enabled:
        blocked.append("direct_gradient_write_not_promoted")
    return {
        "schema_version": 1,
        "contract": "turbocore_native_update_direct_gradient_write_contract_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "write_boundary_ready": boundary_ready,
        "native_supported": native_supported,
        "training_lifecycle_integrated": lifecycle_integrated,
        "explicit_training_context_requested": explicit_training_context,
        "write_guard_enabled": guard_enabled,
        "write_binding_enabled": binding_enabled,
        "bound_to_training_path": bound_to_training_path,
        "direct_gradient_write_preconditions_ready": bound_to_training_path,
        "default_off": not bound_to_training_path,
        "blocked_reasons": _dedupe(blocked),
    }


def _owner_gradient_sync_summary(
    *,
    context: Mapping[str, Any],
    readiness: Mapping[str, Any],
    requested: bool,
) -> dict[str, Any]:
    owner_checks = _as_dict(readiness.get("owner_checks"))
    boundary_ready = bool(owner_checks.get("owner_gradient_sync_boundary_ready", False))
    supported = bool(owner_checks.get("owner_gradient_sync_supported", boundary_ready))
    lifecycle_integrated = bool(owner_checks.get("owner_gradient_sync_training_integrated", False))
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    guard_enabled = bool(context.get("native_update_owner_gradient_sync_guard_enabled", False))
    binding_enabled = bool(context.get("native_update_owner_gradient_sync_bound", False))
    bound_to_training_path = bool(
        explicit_training_context
        and boundary_ready
        and supported
        and lifecycle_integrated
        and guard_enabled
        and binding_enabled
    )
    blocked: list[str] = []
    if not boundary_ready:
        blocked.append("owner_gradient_sync_boundary_missing")
    elif not explicit_training_context:
        blocked.append("owner_gradient_sync_default_off")
    elif not supported:
        blocked.append("owner_gradient_sync_not_supported")
    elif not lifecycle_integrated:
        blocked.append("owner_gradient_sync_not_training_integrated")
    elif not guard_enabled:
        blocked.append("owner_gradient_sync_guard_disabled")
    elif not binding_enabled:
        blocked.append("owner_gradient_sync_not_promoted")
    return {
        "schema_version": 1,
        "contract": "turbocore_native_update_owner_gradient_sync_contract_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "sync_boundary_ready": boundary_ready,
        "native_supported": supported,
        "training_lifecycle_integrated": lifecycle_integrated,
        "explicit_training_context_requested": explicit_training_context,
        "sync_guard_enabled": guard_enabled,
        "sync_binding_enabled": binding_enabled,
        "bound_to_training_path": bound_to_training_path,
        "owner_gradient_sync_preconditions_ready": bound_to_training_path,
        "default_off": not bound_to_training_path,
        "blocked_reasons": _dedupe(blocked),
    }


def _training_flat_owner_summary(
    *,
    context: Mapping[str, Any],
    readiness: Mapping[str, Any],
    requested: bool,
) -> dict[str, Any]:
    native_checks = _as_dict(readiness.get("native_checks"))
    contract_ready = bool(native_checks.get("flat_owner_contract_ready", False))
    reference_owner_ready = bool(native_checks.get("reference_flat_owner_ready", False))
    promoted = bool(native_checks.get("training_flat_owner_promoted", False))
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    owner_guard_enabled = bool(context.get("native_update_flat_owner_training_guard_enabled", False))
    owner_binding_enabled = bool(context.get("native_update_flat_owner_bound", False))
    bound_to_training_path = bool(
        explicit_training_context
        and contract_ready
        and reference_owner_ready
        and promoted
        and owner_guard_enabled
        and owner_binding_enabled
    )
    blocked: list[str] = []
    if not contract_ready:
        blocked.append("native_training_flat_owner_contract_incomplete")
    elif not explicit_training_context:
        blocked.append("native_training_flat_owner_default_off")
    elif promoted and not owner_guard_enabled:
        blocked.append("native_training_flat_owner_guard_disabled")
    elif not owner_binding_enabled or not promoted:
        blocked.append("native_training_flat_owner_not_promoted")
    return {
        "schema_version": 1,
        "contract": "turbocore_native_update_training_flat_owner_contract_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "owner_boundary_ready": contract_ready,
        "reference_owner_ready": reference_owner_ready,
        "training_flat_owner_promoted": promoted,
        "explicit_training_context_requested": explicit_training_context,
        "owner_guard_enabled": owner_guard_enabled,
        "owner_binding_enabled": owner_binding_enabled,
        "bound_to_training_path": bound_to_training_path,
        "training_flat_owner_preconditions_ready": bound_to_training_path,
        "default_off": not bound_to_training_path,
        "blocked_reasons": _dedupe(blocked),
    }


def _training_dispatch_kernel_summary(
    *,
    context: Mapping[str, Any],
    readiness: Mapping[str, Any],
    owner_native: Mapping[str, Any],
    requested: bool,
) -> dict[str, Any]:
    native_checks = _as_dict(readiness.get("native_checks"))
    contract_ready = bool(native_checks.get("training_dispatch_kernel_contract_ready", False))
    kernel_present = bool(
        readiness.get("training_dispatch_kernel_present", False)
        or readiness.get("native_kernel_present", False)
        or native_checks.get("training_dispatch_kernel_present", False)
        or owner_native.get("kernel_executed", False)
    )
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    kernel_guard_enabled = bool(context.get("native_update_training_dispatch_kernel_guard_enabled", False))
    kernel_binding_enabled = bool(context.get("native_update_training_dispatch_kernel_bound", False))
    bound_to_training_path = bool(
        explicit_training_context
        and contract_ready
        and kernel_present
        and kernel_guard_enabled
        and kernel_binding_enabled
    )
    blocked: list[str] = []
    if not contract_ready:
        blocked.append("native_training_dispatch_kernel_contract_missing")
    elif not explicit_training_context:
        blocked.append("native_training_dispatch_kernel_default_off")
    elif kernel_present and not kernel_guard_enabled:
        blocked.append("native_training_dispatch_kernel_guard_disabled")
    elif not kernel_binding_enabled or not kernel_present:
        blocked.append("native_training_dispatch_kernel_not_promoted")
    return {
        "schema_version": 1,
        "contract": "turbocore_native_update_training_dispatch_kernel_contract_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "kernel_boundary_ready": contract_ready,
        "kernel_present_evidence": kernel_present,
        "explicit_training_context_requested": explicit_training_context,
        "kernel_guard_enabled": kernel_guard_enabled,
        "kernel_binding_enabled": kernel_binding_enabled,
        "bound_to_training_path": bound_to_training_path,
        "training_dispatch_kernel_preconditions_ready": bound_to_training_path,
        "default_off": not bound_to_training_path,
        "blocked_reasons": _dedupe(blocked),
    }


def _training_executor_summary(*, context: Mapping[str, Any], requested: bool) -> dict[str, Any]:
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    executor_handle_present = bool(context.get("native_update_executor_present", False))
    runtime_guard = bool(context.get("native_update_runtime_execution_guard_enabled", False))
    mutation_guard = bool(context.get("native_update_training_mutation_guard_enabled", False))
    runtime_available = bool(context.get("native_update_runtime_dispatch_available", False))
    bound_to_training_path = bool(
        explicit_training_context
        and executor_handle_present
        and runtime_guard
        and mutation_guard
        and runtime_available
    )
    blocked: list[str] = []
    if not explicit_training_context:
        blocked.append("native_dispatch_training_runtime_executor_default_off")
    if explicit_training_context and not executor_handle_present:
        blocked.append("native_dispatch_runtime_executor_missing")
    if explicit_training_context and not runtime_guard:
        blocked.append("native_dispatch_runtime_execution_guard_disabled")
    if explicit_training_context and not mutation_guard:
        blocked.append("native_dispatch_training_mutation_guard_disabled")
    if explicit_training_context and not runtime_available:
        blocked.append("native_dispatch_runtime_not_implemented")
    return {
        "schema_version": 1,
        "contract": "turbocore_native_update_training_executor_contract_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "callable_slot_integrated": True,
        "executor_factory_available": True,
        "executor_boundary_ready": True,
        "explicit_training_context_requested": explicit_training_context,
        "executor_handle_present": executor_handle_present,
        "runtime_execution_guard_enabled": runtime_guard,
        "training_mutation_guard_enabled": mutation_guard,
        "runtime_dispatch_available": runtime_available,
        "bound_to_training_path": bound_to_training_path,
        "training_executor_preconditions_ready": bound_to_training_path,
        "default_off": not bound_to_training_path,
        "blocked_reasons": _dedupe(blocked),
    }


def _stream_lifetime_ownership_summary(
    *,
    context: Mapping[str, Any],
    readiness: Mapping[str, Any],
    preflight: Mapping[str, Any],
    binding: Mapping[str, Any],
    owner_native: Mapping[str, Any],
    requested: bool,
) -> dict[str, Any]:
    ordering_verified = bool(
        readiness.get("stream_ordering_verified", False)
        or preflight.get("stream_ordering_verified", False)
        or preflight.get("event_chain_verified", False)
        or binding.get("event_chain_verified", False)
        or owner_native.get("event_chain_verified", False)
        or binding.get("pre_launch_ordering_verified", False)
        or binding.get("post_launch_ordering_verified", False)
        or binding.get("stream_wait_event_verified", False)
        or owner_native.get("pre_launch_ordering_verified", False)
        or owner_native.get("post_launch_ordering_verified", False)
        or owner_native.get("stream_wait_event_verified", False)
    )
    ownership_bound_evidence = bool(
        readiness.get("stream_lifetime_ownership_bound", False)
        or binding.get("stream_lifetime_bound", False)
        or preflight.get("stream_lifetime_ownership_bound", False)
    )
    explicit_training_context = bool(
        requested
        and context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
    )
    ownership_guard_enabled = bool(context.get("native_update_stream_lifetime_ownership_guard_enabled", False))
    ownership_binding_enabled = bool(context.get("native_update_stream_lifetime_ownership_bound", False))
    bound_to_training_path = bool(
        explicit_training_context
        and ordering_verified
        and ownership_bound_evidence
        and ownership_guard_enabled
        and ownership_binding_enabled
    )
    blocked: list[str] = []
    if not ordering_verified:
        blocked.append("stream_event_chain_validation_missing")
    elif not explicit_training_context:
        blocked.append("stream_lifetime_ownership_default_off")
    elif not ownership_guard_enabled:
        blocked.append("stream_lifetime_ownership_guard_disabled")
    elif not ownership_binding_enabled or not ownership_bound_evidence:
        blocked.append("stream_lifetime_ownership_not_promoted")
    return {
        "schema_version": 1,
        "contract": "turbocore_native_update_stream_lifetime_ownership_contract_v0",
        "training_dispatch": False,
        "training_path_enabled": False,
        "ownership_boundary_ready": True,
        "ordering_verified": ordering_verified,
        "ownership_bound_evidence": ownership_bound_evidence,
        "explicit_training_context_requested": explicit_training_context,
        "ownership_guard_enabled": ownership_guard_enabled,
        "ownership_binding_enabled": ownership_binding_enabled,
        "bound_to_training_path": bound_to_training_path,
        "stream_lifetime_ownership_preconditions_ready": bound_to_training_path,
        "default_off": not bound_to_training_path,
        "blocked_reasons": _dedupe(blocked),
    }


def _evidence(
    preflight: Mapping[str, Any],
    fallback: Mapping[str, Any],
    runtime_recovery: Mapping[str, Any],
    performance: Mapping[str, Any],
    owner_native: Mapping[str, Any],
    copyback: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "preflight_present": bool(preflight),
        "preflight_passed": bool(preflight.get("dispatch_preflight_passed", False)),
        "fallback_policy_present": bool(fallback),
        "runtime_recovery_policy_defined": bool(runtime_recovery.get("policy_defined", False)),
        "runtime_recovery_policy_observation_integrated": bool(runtime_recovery.get("policy_observation_integrated", False)),
        "runtime_recovery_latch_integrated": bool(runtime_recovery.get("run_disable_latch_integrated", False)),
        "runtime_recovery_default_off_bridge_ready": bool(runtime_recovery.get("default_off_recovery_bridge_ready", False)),
        "runtime_recovery_observation_bridge_ready": bool(runtime_recovery.get("recovery_observation_bridge_ready", False)),
        "runtime_recovery_training_dispatch_ready": bool(runtime_recovery.get("training_dispatch_recovery_ready", False)),
        "runtime_recovery_training_dispatch_blocked": bool(runtime_recovery.get("training_dispatch_recovery_blocked", False)),
        "runtime_recovery_dispatch_integrated": bool(runtime_recovery.get("dispatch_integration_ready", False)),
        "representative_performance_gate_ready": bool(performance.get("representative_performance_gate_ready", False)),
        "training_dispatch_performance_gate_ready": bool(
            performance.get("training_dispatch_performance_gate_ready", False)
            or performance.get("representative_performance_gate_ready", False)
            or preflight.get("training_dispatch_performance_gate_ready", False)
        ),
        "owner_native_launch_probe_present": bool(owner_native),
        "owner_native_launch_ok": bool(owner_native.get("ok", False)) if owner_native else None,
        "copyback_dispatch_probe_present": bool(copyback),
        "copyback_dispatch_validated": bool(copyback.get("copyback_dispatch_validated", False)) if copyback else None,
        "native_binding_probe_present": bool(binding),
        "stream_lifetime_ownership_bound": bool(
            binding.get("stream_lifetime_bound", False)
            or preflight.get("stream_lifetime_ownership_bound", False)
        ),
        "stream_ordering_verified": bool(
            preflight.get("stream_ordering_verified", False)
            or preflight.get("event_chain_verified", False)
            or binding.get("event_chain_verified", False)
            or owner_native.get("event_chain_verified", False)
            or binding.get("pre_launch_ordering_verified", False)
            or binding.get("post_launch_ordering_verified", False)
            or binding.get("stream_wait_event_verified", False)
            or owner_native.get("pre_launch_ordering_verified", False)
            or owner_native.get("post_launch_ordering_verified", False)
            or owner_native.get("stream_wait_event_verified", False)
        ),
        "event_chain_verified": bool(binding.get("event_chain_verified", False) or owner_native.get("event_chain_verified", False)),
    }


def _collect_blockers(
    *,
    preflight: Mapping[str, Any],
    fallback: Mapping[str, Any],
    runtime_recovery: Mapping[str, Any],
    performance: Mapping[str, Any],
    gate_blocked_reasons: list[str] | None,
) -> list[str]:
    blocked = _strings(preflight.get("blocked_reasons"))
    blocked.extend(_strings(runtime_recovery.get("blocked_reasons")))
    blocked.extend(_strings(performance.get("blocked_reasons")))
    blocked.extend(str(item) for item in (gate_blocked_reasons or []) if str(item))
    if not preflight:
        blocked.append("dispatch_preflight_missing")
    if fallback and "keep_training_dispatch_disabled_until_recovery_integrated" in _strings(runtime_recovery.get("actions")):
        blocked.append("native_recovery_keeps_dispatch_disabled")
    return blocked


def _actions_required(blocked: list[str]) -> list[str]:
    actions: list[str] = []
    if "native_dispatch_runtime_not_implemented" in blocked:
        actions.append("implement_native_update_dispatch_runtime")
    if "native_dispatch_training_path_disabled" in blocked:
        actions.append("add_explicit_default_off_training_dispatch_flag")
    if any("recovery" in item for item in blocked):
        actions.append("integrate_native_runtime_recovery_with_dispatch")
    if any("performance" in item or "representative_training_matrix" in item for item in blocked):
        actions.append("collect_representative_native_dispatch_benchmark_matrix")
    if any("stream" in item or "event_chain" in item for item in blocked):
        actions.append("validate_stream_lifetime_and_event_chain")
    if any("copyback" in item for item in blocked):
        actions.append("validate_training_parameter_copyback_dispatch")
    return _dedupe(actions)


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


__all__ = ["build_native_update_dispatch_contract"]
