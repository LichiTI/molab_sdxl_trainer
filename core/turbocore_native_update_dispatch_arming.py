"""Pre-step arming state for future TurboCore native update dispatch."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_native_update_probe_cache import probe_cache_summary


class TurboCoreNativeUpdateDispatchArmer:
    """Carry post-step gate evidence into the next step's pre-step decision."""

    def __init__(self) -> None:
        self._last_gate_report: dict[str, Any] = {}
        self._last_decision: dict[str, Any] = {}
        self._retained_probe_evidence_steps = 0

    def prepare_before_optimizer(
        self,
        *,
        step: int,
        runtime_context: Mapping[str, Any] | None = None,
        runtime_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        gate = self._last_gate_report
        request = _as_dict(gate.get("dispatch_request"))
        contract = _as_dict(gate.get("dispatch_contract"))
        kernel_launch = _as_dict(gate.get("kernel_launch_plan"))
        context = _as_dict(runtime_context)
        runtime = _as_dict(runtime_state)
        preconditions = _promotion_preconditions(
            gate=gate,
            request=request,
            contract=contract,
            kernel_launch=kernel_launch,
            context=context,
            runtime_state=runtime,
        )
        probe_cache = probe_cache_summary(gate)
        retained_probe_evidence = bool(probe_cache.get("retained", False) or self._retained_probe_evidence_steps > 0)
        probe_cache_source = str(probe_cache.get("source", "") or "")
        if retained_probe_evidence and not probe_cache_source:
            probe_cache_source = "previous_step_gate_report"
        probe_cache_reused_steps = max(
            int(probe_cache.get("reused_steps", 0) or 0),
            int(self._retained_probe_evidence_steps),
        )
        training_promotion_ready = bool(preconditions.get("training_promotion_ready", False))
        blocked = _blocked_reasons(
            gate=gate,
            request=request,
            contract=contract,
            kernel_launch=kernel_launch,
            context=context,
            runtime_state=runtime,
            preconditions=preconditions,
        )
        decision = {
            "schema_version": 1,
            "decision": "turbocore_native_update_dispatch_arming_v0",
            "step": int(step),
            "source": "previous_step_gate_report",
            "training_dispatch": training_promotion_ready,
            "training_path_enabled": training_promotion_ready,
            "armed_for_native_dispatch": training_promotion_ready,
            "execute_native_step": training_promotion_ready,
            "call_pytorch_optimizer_step": not training_promotion_ready,
            "native_mutation_allowed": training_promotion_ready,
            "training_parameter_mutation_allowed": training_promotion_ready,
            "previous_gate_present": bool(gate),
            "previous_request_requested": bool(request.get("requested", False)),
            "previous_request_allowed": bool(request.get("dispatch_allowed", False)),
            "previous_contract_ready": bool(contract.get("dispatch_rehearsal_ready", False)),
            "previous_contract_rehearsal_evidence_ready": bool(preconditions.get("contract_rehearsal_evidence_ready", False)),
            "previous_kernel_launch_plan_present": bool(kernel_launch),
            "previous_kernel_launch_allowed": bool(kernel_launch.get("launch_allowed", False)),
            "previous_kernel_launch_rehearsal_evidence_ready": bool(preconditions.get("kernel_launch_rehearsal_evidence_ready", False)),
            "native_dispatch_rehearsal_evidence_ready": bool(preconditions.get("rehearsal_evidence_ready", False)),
            "native_dispatch_training_promotion_preconditions_ready": training_promotion_ready,
            "promotion_preconditions": preconditions,
            "retained_probe_evidence": retained_probe_evidence,
            "probe_cache_source": probe_cache_source,
            "probe_cache_reused_steps": probe_cache_reused_steps,
            "runtime_disabled_for_run": bool(runtime.get("disabled_for_run", False)),
            "runtime_disable_reason": str(runtime.get("disable_reason", "") or ""),
            "blocked_reasons": _dedupe(blocked),
        }
        self._last_decision = decision
        return dict(decision)

    def observe_after_optimizer(self, gate_report: Mapping[str, Any] | None) -> dict[str, Any]:
        incoming = _as_dict(gate_report)
        retained_previous = _should_retain_previous_gate_after_shadow_autostop(
            incoming=incoming,
            previous=self._last_gate_report,
        )
        if not retained_previous:
            self._last_gate_report = incoming
        active_gate = self._last_gate_report
        active_request = _as_dict(active_gate.get("dispatch_request"))
        active_contract = _as_dict(active_gate.get("dispatch_contract"))
        active_kernel = _as_dict(active_gate.get("kernel_launch_plan"))
        active_cache = probe_cache_summary(active_gate)
        incoming_cache = probe_cache_summary(incoming)
        if bool(active_cache.get("retained", False)):
            self._retained_probe_evidence_steps = int(active_cache.get("reused_steps", 0) or 0)
        elif retained_previous:
            self._retained_probe_evidence_steps += 1
        else:
            self._retained_probe_evidence_steps = 0
        return {
            "schema_version": 1,
            "state": "turbocore_native_update_dispatch_arming_observation_v0",
            "observed_gate_report": bool(incoming),
            "observed_request": bool(_as_dict(incoming.get("dispatch_request"))),
            "observed_contract": bool(_as_dict(incoming.get("dispatch_contract"))),
            "observed_kernel_launch_plan": bool(_as_dict(incoming.get("kernel_launch_plan"))),
            "retained_previous_gate_after_shadow_autostop": retained_previous,
            "active_gate_report_retained": bool(retained_previous and active_gate),
            "retained_probe_evidence": bool(active_cache.get("retained", False) or retained_previous),
            "incoming_retained_probe_evidence": bool(incoming_cache.get("retained", False)),
            "probe_cache_source": str(active_cache.get("source", "") or ("previous_step_gate_report" if retained_previous else "")),
            "probe_cache_reused_steps": int(self._retained_probe_evidence_steps),
            "next_step_can_consider_native_dispatch": bool(
                active_request.get("dispatch_allowed", False)
                and active_contract.get("dispatch_rehearsal_ready", False)
                and active_kernel.get("launch_allowed", False)
            ),
            "training_path_enabled": bool(active_request.get("training_path_enabled", False)),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "state": "turbocore_native_update_dispatch_arming_state_v0",
            "last_gate_present": bool(self._last_gate_report),
            "last_decision": dict(self._last_decision),
            "retained_probe_evidence_steps": int(self._retained_probe_evidence_steps),
            "training_path_enabled": bool(self._last_decision.get("training_path_enabled", False)),
        }

    def last_gate_report(self) -> dict[str, Any]:
        return dict(self._last_gate_report)


def _blocked_reasons(
    *,
    gate: Mapping[str, Any],
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    kernel_launch: Mapping[str, Any],
    context: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if bool(runtime_state.get("disabled_for_run", False)):
        blocked.append("native_dispatch_disabled_for_run")
    if not gate:
        blocked.append("previous_gate_report_missing")
    if not request:
        blocked.append("previous_dispatch_request_missing")
    elif not bool(request.get("dispatch_allowed", False)):
        blocked.append("previous_dispatch_request_not_allowed")
    if not contract:
        blocked.append("previous_dispatch_contract_missing")
    elif not bool(contract.get("dispatch_rehearsal_ready", False)):
        blocked.append("previous_dispatch_contract_not_ready")
    if not kernel_launch:
        blocked.append("previous_kernel_launch_plan_missing")
    elif not bool(kernel_launch.get("launch_allowed", False)):
        blocked.append("previous_kernel_launch_not_allowed")
    training_promotion_ready = bool(preconditions.get("training_promotion_ready", False))
    if bool(preconditions.get("rehearsal_evidence_ready", False)) and not training_promotion_ready:
        blocked.append("native_dispatch_rehearsal_evidence_only")
    if bool(context.get("gradient_release_active", False)):
        blocked.append("gradient_release_not_supported")
    if not training_promotion_ready:
        blocked.append("native_dispatch_runtime_not_implemented")
        blocked.append("native_dispatch_training_path_disabled")
    return blocked


def _promotion_preconditions(
    *,
    gate: Mapping[str, Any],
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    kernel_launch: Mapping[str, Any],
    context: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    contract_evidence = _as_dict(contract.get("evidence"))
    contract_rehearsal = _as_dict(contract.get("rehearsal"))
    kernel_evidence = _as_dict(kernel_launch.get("evidence"))
    recovery = _as_dict(contract.get("recovery"))
    recovery_integration = _as_dict(recovery.get("integration"))
    direct_gradient_write = _as_dict(contract.get("direct_gradient_write"))
    owner_gradient_sync = _as_dict(contract.get("owner_gradient_sync"))
    training_flat_owner = _as_dict(contract.get("training_flat_owner"))
    training_dispatch_kernel = _as_dict(contract.get("training_dispatch_kernel"))
    training_executor = _as_dict(contract.get("training_executor"))
    stream_lifetime = _as_dict(contract.get("stream_lifetime_ownership"))
    training_path_request = _as_dict(request.get("training_path_request"))
    request_requested = bool(request.get("requested", False))
    owner_kernel_ready = bool(
        contract_rehearsal.get("would_launch_native_kernel", False)
        or kernel_evidence.get("diagnostic_kernel_executed", False)
        or contract_evidence.get("owner_native_launch_ok", False)
    )
    owner_parity_ready = bool(kernel_evidence.get("diagnostic_parity_ok", False))
    copyback_ready = bool(contract_evidence.get("copyback_dispatch_validated", False))
    event_chain_ready = bool(contract_evidence.get("event_chain_verified", False))
    stream_ordering_ready = bool(contract_evidence.get("stream_ordering_verified", event_chain_ready))
    stream_lifetime_ownership_evidence_bound = bool(contract_evidence.get("stream_lifetime_ownership_bound", False))
    stream_lifetime_ownership_ready = bool(
        stream_lifetime.get("stream_lifetime_ownership_preconditions_ready", False)
        or stream_lifetime.get("bound_to_training_path", False)
        or (not stream_lifetime and stream_lifetime_ownership_evidence_bound)
    )
    stream_lifetime_boundary_ready = bool(
        stream_lifetime.get("ownership_boundary_ready", False)
        or stream_lifetime.get("contract") == "turbocore_native_update_stream_lifetime_ownership_contract_v0"
    )
    stream_lifetime_default_off = bool(
        stream_lifetime_boundary_ready
        and not stream_lifetime.get("bound_to_training_path", False)
        and "stream_lifetime_ownership_default_off" in _strings(stream_lifetime.get("blocked_reasons"))
    )
    recovery_observation_ready = bool(
        recovery.get("default_off_recovery_bridge_ready", False)
        or recovery.get("recovery_observation_bridge_ready", False)
        or recovery.get("policy_observation_integrated", False)
        or recovery.get("run_disable_latch_integrated", False)
        or recovery_integration.get("default_off_recovery_bridge_ready", False)
        or recovery_integration.get("recovery_observation_bridge_ready", False)
    )
    training_dispatch_recovery_ready = bool(
        recovery.get("training_dispatch_recovery_ready", False)
        or recovery_integration.get("training_dispatch_recovery_ready", False)
    )
    direct_gradient_write_boundary_ready = bool(
        direct_gradient_write.get("write_boundary_ready", False)
        or direct_gradient_write.get("contract") == "turbocore_native_update_direct_gradient_write_contract_v0"
    )
    direct_gradient_write_native_supported = bool(direct_gradient_write.get("native_supported", False))
    direct_gradient_write_lifecycle_ready = bool(direct_gradient_write.get("training_lifecycle_integrated", False))
    direct_gradient_write_bound = bool(
        direct_gradient_write.get("direct_gradient_write_preconditions_ready", False)
        or direct_gradient_write.get("bound_to_training_path", False)
    )
    direct_gradient_write_default_off = bool(
        direct_gradient_write_boundary_ready
        and not direct_gradient_write.get("bound_to_training_path", False)
        and "direct_gradient_write_default_off" in _strings(direct_gradient_write.get("blocked_reasons"))
    )
    owner_gradient_sync_boundary_ready = bool(
        owner_gradient_sync.get("sync_boundary_ready", False)
        or owner_gradient_sync.get("contract") == "turbocore_native_update_owner_gradient_sync_contract_v0"
    )
    owner_gradient_sync_native_supported = bool(owner_gradient_sync.get("native_supported", False))
    owner_gradient_sync_lifecycle_ready = bool(owner_gradient_sync.get("training_lifecycle_integrated", False))
    owner_gradient_sync_bound = bool(
        owner_gradient_sync.get("owner_gradient_sync_preconditions_ready", False)
        or owner_gradient_sync.get("bound_to_training_path", False)
    )
    owner_gradient_sync_default_off = bool(
        owner_gradient_sync_boundary_ready
        and not owner_gradient_sync.get("bound_to_training_path", False)
        and "owner_gradient_sync_default_off" in _strings(owner_gradient_sync.get("blocked_reasons"))
    )
    training_flat_owner_boundary_ready = bool(
        training_flat_owner.get("owner_boundary_ready", False)
        or training_flat_owner.get("contract") == "turbocore_native_update_training_flat_owner_contract_v0"
    )
    training_flat_owner_reference_ready = bool(training_flat_owner.get("reference_owner_ready", False))
    training_flat_owner_bound = bool(
        training_flat_owner.get("training_flat_owner_preconditions_ready", False)
        or training_flat_owner.get("bound_to_training_path", False)
    )
    training_flat_owner_default_off = bool(
        training_flat_owner_boundary_ready
        and not training_flat_owner.get("bound_to_training_path", False)
        and "native_training_flat_owner_default_off" in _strings(training_flat_owner.get("blocked_reasons"))
    )
    training_dispatch_kernel_boundary_ready = bool(
        training_dispatch_kernel.get("kernel_boundary_ready", False)
        or training_dispatch_kernel.get("contract") == "turbocore_native_update_training_dispatch_kernel_contract_v0"
    )
    training_dispatch_kernel_evidence_present = bool(training_dispatch_kernel.get("kernel_present_evidence", False))
    training_dispatch_kernel_bound = bool(
        training_dispatch_kernel.get("training_dispatch_kernel_preconditions_ready", False)
        or training_dispatch_kernel.get("bound_to_training_path", False)
    )
    training_dispatch_kernel_default_off = bool(
        training_dispatch_kernel_boundary_ready
        and not training_dispatch_kernel.get("bound_to_training_path", False)
        and "native_training_dispatch_kernel_default_off" in _strings(training_dispatch_kernel.get("blocked_reasons"))
    )
    training_executor_boundary_ready = bool(
        training_executor.get("executor_boundary_ready", False)
        or training_executor.get("callable_slot_integrated", False)
    )
    training_executor_bound = bool(
        training_executor.get("training_executor_preconditions_ready", False)
        or training_executor.get("bound_to_training_path", False)
    )
    training_path_request_boundary_ready = bool(training_path_request.get("request_boundary_ready", False))
    explicit_training_path_requested = bool(training_path_request.get("explicit_training_path_requested", False))
    performance_ready = bool(
        contract_evidence.get("representative_performance_gate_ready", False)
        or contract_evidence.get("training_dispatch_performance_gate_ready", False)
    )
    unsupported_context = bool(
        context.get("gradient_release_active", False)
        or context.get("multi_gpu", False)
        or int(context.get("num_processes", 1) or 1) > 1
        or context.get("deepspeed", False)
        or runtime_state.get("disabled_for_run", False)
    )
    rehearsal_ready = bool(request_requested and owner_kernel_ready and owner_parity_ready and copyback_ready)
    missing = []
    if not request_requested:
        missing.append("dispatch_request_not_requested")
    if not owner_kernel_ready:
        missing.append("owner_native_kernel_rehearsal_missing")
    if not owner_parity_ready:
        missing.append("owner_native_kernel_parity_missing")
    if not copyback_ready:
        missing.append("copyback_dispatch_validation_missing")
    if not stream_ordering_ready:
        missing.append("stream_event_chain_validation_missing")
    elif not stream_lifetime_boundary_ready:
        missing.append("stream_lifetime_ownership_boundary_missing")
    elif stream_lifetime_default_off:
        missing.append("stream_lifetime_ownership_default_off")
    elif not stream_lifetime_ownership_ready:
        missing.append("stream_lifetime_ownership_not_promoted")
    if not recovery_observation_ready:
        missing.append("training_dispatch_recovery_missing")
    elif not training_dispatch_recovery_ready:
        missing.append("training_dispatch_recovery_default_off")
    if not owner_gradient_sync_boundary_ready:
        missing.append("owner_gradient_sync_boundary_missing")
    elif owner_gradient_sync_default_off:
        missing.append("owner_gradient_sync_default_off")
    elif not owner_gradient_sync_native_supported:
        missing.append("owner_gradient_sync_not_supported")
    elif not owner_gradient_sync_lifecycle_ready:
        missing.append("owner_gradient_sync_not_training_integrated")
    elif not owner_gradient_sync_bound:
        missing.append("owner_gradient_sync_not_promoted")
    if not training_flat_owner_boundary_ready:
        missing.append("native_training_flat_owner_boundary_missing")
    elif training_flat_owner_default_off:
        missing.append("native_training_flat_owner_default_off")
    elif not training_flat_owner_bound:
        missing.append("native_training_flat_owner_not_promoted")
    if not training_dispatch_kernel_boundary_ready:
        missing.append("native_training_dispatch_kernel_boundary_missing")
    elif training_dispatch_kernel_default_off:
        missing.append("native_training_dispatch_kernel_default_off")
    elif not training_dispatch_kernel_bound:
        missing.append("native_training_dispatch_kernel_not_promoted")
    if not performance_ready:
        missing.append("representative_performance_gate_missing")
    if unsupported_context:
        missing.append("unsupported_runtime_context")
    if not training_executor_boundary_ready:
        missing.append("native_dispatch_training_runtime_executor_missing")
    elif not training_executor_bound:
        missing.append("native_dispatch_training_runtime_executor_default_off")
    if not training_path_request_boundary_ready:
        missing.append("native_dispatch_training_path_request_missing")
    elif not explicit_training_path_requested:
        missing.append("native_dispatch_training_path_default_off")
    elif not bool(request.get("dispatch_allowed", False)):
        missing.append("native_dispatch_training_path_disabled")
    training_promotion_ready = bool(rehearsal_ready and not missing)
    return {
        "schema_version": 1,
        "preconditions": "turbocore_native_update_dispatch_arming_preconditions_v0",
        "previous_gate_present": bool(gate),
        "dispatch_request_requested": request_requested,
        "dispatch_request_allowed": bool(request.get("dispatch_allowed", False)),
        "contract_rehearsal_ready": bool(contract.get("dispatch_rehearsal_ready", False)),
        "contract_rehearsal_evidence_ready": bool(rehearsal_ready),
        "kernel_launch_allowed": bool(kernel_launch.get("launch_allowed", False)),
        "kernel_launch_rehearsal_evidence_ready": bool(owner_kernel_ready and owner_parity_ready),
        "owner_native_kernel_ready": owner_kernel_ready,
        "owner_native_parity_ready": owner_parity_ready,
        "copyback_dispatch_ready": copyback_ready,
        "stream_event_chain_ready": event_chain_ready,
        "stream_ordering_ready": stream_ordering_ready,
        "stream_lifetime_ownership_boundary_ready": stream_lifetime_boundary_ready,
        "stream_lifetime_ownership_evidence_bound": stream_lifetime_ownership_evidence_bound,
        "stream_lifetime_ownership_default_off": stream_lifetime_default_off,
        "stream_lifetime_ownership_ready": stream_lifetime_ownership_ready,
        "recovery_observation_bridge_ready": recovery_observation_ready,
        "training_dispatch_recovery_ready": training_dispatch_recovery_ready,
        "training_dispatch_recovery_blocked": bool(recovery_observation_ready and not training_dispatch_recovery_ready),
        "direct_gradient_write_boundary_ready": direct_gradient_write_boundary_ready,
        "direct_gradient_write_native_supported": direct_gradient_write_native_supported,
        "direct_gradient_write_lifecycle_ready": direct_gradient_write_lifecycle_ready,
        "direct_gradient_write_bound": direct_gradient_write_bound,
        "direct_gradient_write_default_off": direct_gradient_write_default_off,
        "owner_gradient_sync_boundary_ready": owner_gradient_sync_boundary_ready,
        "owner_gradient_sync_native_supported": owner_gradient_sync_native_supported,
        "owner_gradient_sync_lifecycle_ready": owner_gradient_sync_lifecycle_ready,
        "owner_gradient_sync_bound": owner_gradient_sync_bound,
        "owner_gradient_sync_default_off": owner_gradient_sync_default_off,
        "training_flat_owner_boundary_ready": training_flat_owner_boundary_ready,
        "training_flat_owner_reference_ready": training_flat_owner_reference_ready,
        "training_flat_owner_bound": training_flat_owner_bound,
        "training_flat_owner_default_off": training_flat_owner_default_off,
        "training_dispatch_kernel_boundary_ready": training_dispatch_kernel_boundary_ready,
        "training_dispatch_kernel_evidence_present": training_dispatch_kernel_evidence_present,
        "training_dispatch_kernel_bound": training_dispatch_kernel_bound,
        "training_dispatch_kernel_default_off": training_dispatch_kernel_default_off,
        "training_runtime_executor_boundary_ready": training_executor_boundary_ready,
        "training_runtime_executor_bound": training_executor_bound,
        "training_runtime_executor_default_off": bool(training_executor_boundary_ready and not training_executor_bound),
        "training_path_request_boundary_ready": training_path_request_boundary_ready,
        "explicit_training_path_requested": explicit_training_path_requested,
        "training_path_default_off": bool(training_path_request_boundary_ready and not explicit_training_path_requested),
        "representative_performance_ready": performance_ready,
        "unsupported_runtime_context": unsupported_context,
        "rehearsal_evidence_ready": rehearsal_ready,
        "training_promotion_ready": training_promotion_ready,
        "missing_for_training_promotion": _dedupe(missing),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _should_retain_previous_gate_after_shadow_autostop(
    *,
    incoming: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> bool:
    if not incoming or not previous:
        return False
    previous_request = _as_dict(previous.get("dispatch_request"))
    previous_contract = _as_dict(previous.get("dispatch_contract"))
    previous_kernel = _as_dict(previous.get("kernel_launch_plan"))
    if not (
        previous_request.get("dispatch_allowed", False)
        and previous_contract.get("dispatch_rehearsal_ready", False)
        and previous_kernel.get("launch_allowed", False)
    ):
        return False
    config = _as_dict(incoming.get("config"))
    if str(config.get("mode", incoming.get("mode", "")) or "") != "native_experimental":
        return False
    if not bool(config.get("dispatch_enabled", False)):
        return False
    shadow = _as_dict(incoming.get("shadow"))
    fallback = _as_dict(incoming.get("fallback_policy"))
    recovery = _as_dict(fallback.get("runtime_recovery"))
    if bool(recovery.get("runtime_error_observed", False)):
        return False
    if bool(recovery.get("disable_native_update_for_run", False)):
        return False
    shadow_blockers = set(_strings(shadow.get("blocked_reasons")))
    readiness = _as_dict(incoming.get("readiness"))
    readiness_blockers = set(_strings(readiness.get("blocked_reasons")))
    incoming_blockers = set(_strings(incoming.get("blocked_reasons")))
    autostop_reason = str(shadow.get("reason", "") or "")
    auto_stopped = (
        autostop_reason == "auto_stopped_after_consecutive_passes"
        or "shadow_not_compared" in shadow_blockers
        or "shadow_not_compared" in incoming_blockers
    )
    if not auto_stopped:
        return False
    allowed_regressions = {
        "shadow_not_compared",
        "shadow_parity_not_ok",
        "shadow_max_abs_diff_too_high",
        "shadow_mean_abs_diff_too_high",
        "shadow_warmup_not_satisfied",
        "owner_gradient_sync_not_training_integrated",
        "parameter_owner_copyback_dispatch_not_validated",
        "native_training_flat_owner_not_promoted",
        "native_training_dispatch_kernel_not_promoted",
        "stream_lifetime_unbound",
        "representative_performance_gate_missing",
        "copyback_probe_missing",
        "copyback_dispatch_probe_missing",
        "owner_native_launch_probe_missing",
        "native_binding_probe_missing",
        "owner_backed_native_kernel_evidence_missing",
        "native_kernel_missing",
    }
    non_shadow_regressions = (incoming_blockers | readiness_blockers) - allowed_regressions
    return not non_shadow_regressions


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["TurboCoreNativeUpdateDispatchArmer"]
