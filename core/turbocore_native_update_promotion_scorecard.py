"""Promotion scorecard for TurboCore native optimizer update dispatch."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from core.turbocore_native_update_dispatch_arming import TurboCoreNativeUpdateDispatchArmer
from core.turbocore_native_update_dispatch_diagnostic_executor import build_shadow_owner_native_diagnostic_executor
from core.turbocore_native_update_dispatch_runtime import TurboCoreNativeUpdateDispatchRuntime
from core.turbocore_native_update_performance import build_native_update_performance_gate
from core.turbocore_native_update_promotion_blockers import split_promotion_blockers
from core.turbocore_native_update_readiness import build_native_update_readiness_report
from core.turbocore_native_update_review_evidence import (
    compact_product_exposure_decision,
    compact_release_review_package,
    product_exposure_blockers,
    release_review_blockers,
)
from core.turbocore_native_update_training_executor import build_native_update_training_executor
from core.turbocore_update_gate import TurboCoreNativeUpdateGate, build_native_update_gate_config


def build_native_update_promotion_scorecard(
    *,
    optimizer: Any,
    params: list[torch.nn.Parameter],
    shadow_report: Mapping[str, Any] | None = None,
    performance_report: Mapping[str, Any] | None = None,
    readiness_report: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    mode: str = "native_experimental",
    dispatch_enabled: bool = True,
    required_shadow_passes: int = 1,
    allow_missing_native_kernel: bool = False,
    strict: bool = True,
    diagnostic_executor_replay: bool = False,
) -> dict[str, Any]:
    """Build a report-only native update dispatch scorecard.

    Passing the default scorecard is deliberately weaker than promotion.  It
    proves the existing gates agree on fallback and evidence shape.  A synthetic
    explicit training-dispatch context can exercise the executor slot for smoke
    coverage, but promotion still requires the recovery, stream, native-kernel,
    and representative-performance gates to stay clean.
    """

    context = dict(runtime_context or {})
    shadow = dict(shadow_report or {})
    readiness_base = _readiness_source(
        optimizer=optimizer,
        params=params,
        runtime_context=context,
        shadow=shadow,
        mode=mode,
        readiness_report=readiness_report,
    )
    performance_source = _performance_source(shadow, performance_report)
    performance_gate = _performance_gate(readiness=readiness_base, shadow=shadow, performance_report=performance_source)
    readiness = _readiness_with_performance(readiness_base, performance_gate)
    gate_shadow = _shadow_with_performance(shadow, performance_source, performance_gate)
    product_exposure = _product_exposure_source(context)
    release_review = _release_review_with_owner_record(
        _release_review_source(context),
        _owner_release_review_record_source(context),
    )
    gate = TurboCoreNativeUpdateGate(
        build_native_update_gate_config(
            mode,
            required_shadow_passes=required_shadow_passes,
            allow_missing_native_kernel=allow_missing_native_kernel,
            strict=strict,
            dispatch_enabled=dispatch_enabled,
        )
    )
    gate_report = gate.update(
        shadow_report=gate_shadow,
        optimizer=optimizer,
        trainable_param_count=len([param for param in params if param.requires_grad]),
        runtime_context=context,
        readiness_report=readiness,
    )
    armer = TurboCoreNativeUpdateDispatchArmer()
    arming_observation = armer.observe_after_optimizer(gate_report)
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    runtime_observation = runtime.observe_recovery_policy(_as_dict(gate_report.get("fallback_policy")).get("runtime_recovery"))
    promotion_dispatch = _promotion_dispatch_requested(context)
    arming = armer.prepare_before_optimizer(
        step=1,
        runtime_context=context,
        runtime_state=runtime.snapshot(),
    )
    kernel_launch_plan = _as_dict(gate_report.get("kernel_launch_plan"))
    native_executor = None
    if promotion_dispatch and bool(arming.get("execute_native_step", False)) and bool(kernel_launch_plan.get("launch_allowed", False)):
        native_executor = build_native_update_training_executor(
            optimizer=optimizer,
            params=params,
            config=_as_dict(context.get("native_update_training_executor_config")),
        )
    runtime_report = runtime.prepare_step(
        step=0,
        arming_report=arming,
        kernel_launch_plan=kernel_launch_plan,
        runtime_context=context,
        native_executor=native_executor,
    )
    if native_executor is not None:
        native_executor.close()
    diagnostic_replay_report = _build_diagnostic_replay_runtime_report(
        shadow=shadow,
        enabled=diagnostic_executor_replay,
    )
    checks = _scorecard_checks(
        readiness=readiness,
        gate=gate_report,
        runtime_report=runtime_report,
        performance_gate=performance_gate,
        promotion_dispatch=promotion_dispatch,
    )
    blockers = _dedupe(
        _strings(readiness.get("blocked_reasons"))
        + _strings(performance_gate.get("blocked_reasons"))
        + _strings(gate_report.get("blocked_reasons"))
        + _strings(_as_dict(gate_report.get("dispatch_preflight")).get("blocked_reasons"))
        + _strings(_as_dict(gate_report.get("dispatch_contract")).get("blocked_reasons"))
        + _strings(_as_dict(gate_report.get("dispatch_request")).get("blocked_reasons"))
        + _strings(_as_dict(gate_report.get("kernel_launch_plan")).get("blocked_reasons"))
        + _strings(_as_dict(arming.get("promotion_preconditions")).get("missing_for_training_promotion"))
        + _strings(runtime_report.get("blocked_reasons"))
        + product_exposure_blockers(product_exposure)
        + release_review_blockers(release_review)
        + checks["failed_checks"]
    )
    blockers = _filter_promotion_blockers(
        blockers,
        promotion_dispatch=promotion_dispatch,
        runtime_report=runtime_report,
        performance_gate=performance_gate,
    )
    promotion_blockers = _promotion_blockers(
        blockers,
        performance_gate=performance_gate,
        promotion_dispatch=promotion_dispatch,
        runtime_report=runtime_report,
        product_exposure=product_exposure,
        release_review=release_review,
    )
    blocker_layers = split_promotion_blockers(
        promotion_blockers,
        promotion_dispatch=promotion_dispatch,
        native_step_executed=bool(runtime_report.get("native_step_executed", False)),
    )
    promotion_ready = bool(checks["scorecard_evidence_coherent"] and promotion_dispatch and not promotion_blockers)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_update_promotion_scorecard_v0",
        "ok": bool(checks["scorecard_evidence_coherent"]),
        "debug_only": True,
        "shadow_run": True,
        "mode": _normalize_mode(mode),
        "requested": True,
        "dispatch_enabled": bool(dispatch_enabled),
        "promotion_ready": promotion_ready,
        "training_dispatch": bool(runtime_report.get("training_dispatch", False)),
        "training_path_enabled": bool(runtime_report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(runtime_report.get("native_step_executed", False)),
        "native_step_executed": bool(runtime_report.get("native_step_executed", False)),
        "pytorch_optimizer_authoritative": not bool(runtime_report.get("native_step_executed", False)),
        "fallback_to_pytorch_required": bool(runtime_report.get("fallback_to_pytorch_required", True)),
        "should_call_pytorch_optimizer_step": bool(runtime_report.get("should_call_pytorch_optimizer_step", True)),
        "native_mutation_allowed": bool(runtime_report.get("native_step_executed", False)),
        "training_parameter_mutation_allowed": bool(runtime_report.get("native_step_executed", False)),
        "checks": checks,
        "promotion_blockers": promotion_blockers,
        "primary_promotion_blockers": blocker_layers["primary_promotion_blockers"],
        "derived_promotion_blockers": blocker_layers["derived_promotion_blockers"],
        "blocked_reasons": blockers,
        "readiness": _compact_readiness(readiness),
        "performance_gate": _compact_performance_gate(performance_gate),
        "gate": _compact_gate(gate_report),
        "dispatch_preflight": _compact_preflight(_as_dict(gate_report.get("dispatch_preflight"))),
        "dispatch_contract": _compact_contract(_as_dict(gate_report.get("dispatch_contract"))),
        "dispatch_request": _compact_request(_as_dict(gate_report.get("dispatch_request"))),
        "kernel_launch_plan": _compact_kernel_launch(_as_dict(gate_report.get("kernel_launch_plan"))),
        "dispatch_arming_observation": arming_observation,
        "dispatch_arming": arming,
        "runtime_recovery_observation": runtime_observation,
        "dispatch_runtime": runtime_report,
        "dispatch_execution_plan": _compact_execution_plan(_as_dict(runtime_report.get("execution_plan"))),
        "dispatch_executor_probe": _compact_executor_probe(_as_dict(runtime_report.get("executor_probe"))),
        "dispatch_training_executor": _compact_training_executor(_as_dict(runtime_report.get("training_executor"))),
        "dispatch_runtime_diagnostic_replay": diagnostic_replay_report,
        "product_exposure_decision": compact_product_exposure_decision(product_exposure),
        "release_review_package": compact_release_review_package(release_review),
        "dispatch_diagnostic_execution_plan": _compact_execution_plan(
            _as_dict(diagnostic_replay_report.get("execution_plan"))
        ),
        "dispatch_diagnostic_executor_probe": _compact_executor_probe(
            _as_dict(diagnostic_replay_report.get("executor_probe"))
        ),
    }


def _build_diagnostic_replay_runtime_report(*, shadow: Mapping[str, Any], enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "schema_version": 1,
            "runtime": "turbocore_native_update_dispatch_runtime_diagnostic_replay_v0",
            "enabled": False,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_step_executed": False,
            "should_call_pytorch_optimizer_step": True,
            "blocked_reasons": ["native_dispatch_diagnostic_executor_replay_disabled"],
        }
    runtime = TurboCoreNativeUpdateDispatchRuntime()
    return runtime.prepare_step(
        step=-1,
        arming_report={
            "previous_request_requested": True,
            "armed_for_native_dispatch": True,
            "execute_native_step": True,
        },
        kernel_launch_plan={"launch_allowed": True, "launch_attempted": False},
        runtime_context={
            "native_update_runtime_execution_guard_enabled": True,
            "native_update_diagnostic_executor_call_enabled": True,
            "native_update_diagnostic_clone_context_enabled": True,
            "training_path_enabled": False,
        },
        native_executor=build_shadow_owner_native_diagnostic_executor(shadow),
    )


def _readiness_source(
    *,
    optimizer: Any,
    params: list[torch.nn.Parameter],
    runtime_context: Mapping[str, Any],
    shadow: Mapping[str, Any],
    mode: str,
    readiness_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    explicit = _as_dict(readiness_report)
    if explicit:
        return explicit
    return build_native_update_readiness_report(
        optimizer=optimizer,
        params=params,
        runtime_context=runtime_context,
        shadow_config=_shadow_config_from_report(shadow, mode=mode),
        native_update_mode=mode,
    )


def _promotion_dispatch_requested(context: Mapping[str, Any]) -> bool:
    return bool(
        context.get("native_update_training_dispatch_enabled", False)
        and context.get("training_path_enabled", False)
        and context.get("native_update_runtime_dispatch_available", False)
    )


def _shadow_config_from_report(shadow: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    copyback = _as_dict(shadow.get("copyback_probe"))
    copyback_dispatch = _as_dict(shadow.get("copyback_dispatch_probe"))
    binding = _as_dict(shadow.get("native_binding_probe"))
    owner = _as_dict(shadow.get("owner_native_launch_probe"))
    return {
        "mode": "shadow" if _normalize_mode(mode) == "native_experimental" else "profile",
        "direct_grad_lifecycle_integrated": bool(shadow.get("direct_grad_lifecycle_integrated", True)),
        "checkpoint_metadata_integrated": bool(shadow.get("checkpoint_metadata_integrated", True)),
        "checkpoint_owner_state_enabled": bool(shadow.get("checkpoint_owner_state_enabled", True)),
        "copyback_scratch_probe_integrated": bool(copyback),
        "copyback_scratch_validated": bool(copyback.get("scratch_copyback_validated", False)),
        "copyback_dispatch_experimental_enabled": bool(copyback_dispatch.get("copyback_dispatch_enabled", False)),
        "copyback_dispatch_validated": bool(copyback_dispatch.get("copyback_dispatch_validated", False)),
        "native_tensor_binding_probe_integrated": bool(binding),
        "native_binding_stream_lifetime_bound": bool(binding.get("stream_lifetime_bound", False)),
        "native_binding_event_chain_verified": bool(binding.get("event_chain_verified", False)),
        "native_binding_pre_launch_ordering_verified": bool(binding.get("pre_launch_ordering_verified", False)),
        "native_binding_post_launch_ordering_verified": bool(binding.get("post_launch_ordering_verified", False)),
        "native_binding_stream_wait_event_verified": bool(binding.get("stream_wait_event_verified", False)),
        "owner_native_launch_probe_integrated": bool(owner),
        "owner_native_event_chain_probe_requested": bool(owner.get("event_chain_probe_requested", False)),
        "owner_native_event_chain_probe_attempted": bool(owner.get("event_chain_probe_attempted", False)),
        "owner_native_event_chain_verified": bool(owner.get("event_chain_verified", False)),
        "owner_native_pre_launch_ordering_verified": bool(owner.get("pre_launch_ordering_verified", False)),
        "owner_native_post_launch_ordering_verified": bool(owner.get("post_launch_ordering_verified", False)),
        "owner_native_stream_wait_event_verified": bool(owner.get("stream_wait_event_verified", False)),
    }


def _performance_source(shadow: Mapping[str, Any], performance_report: Mapping[str, Any] | None) -> dict[str, Any]:
    explicit = _as_dict(performance_report)
    if explicit:
        return explicit
    for key in ("native_update_performance_report", "performance_report"):
        value = _as_dict(shadow.get(key))
        if value:
            return value
    return {}


def _product_exposure_source(runtime_context: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("native_update_product_exposure_decision", "product_exposure_decision"):
        value = _as_dict(runtime_context.get(key))
        if value:
            return value
    return {}


def _release_review_source(runtime_context: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("native_update_release_review_package", "release_review_package"):
        value = _as_dict(runtime_context.get(key))
        if value:
            return value
    return {}


def _owner_release_review_record_source(runtime_context: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("native_update_owner_release_review_record", "owner_release_review_record"):
        value = _as_dict(runtime_context.get(key))
        if value:
            return value
    return {}


def _release_review_with_owner_record(
    release_review: Mapping[str, Any],
    owner_record: Mapping[str, Any],
) -> dict[str, Any]:
    report = dict(release_review)
    if owner_record:
        report["owner_release_review_record"] = dict(owner_record)
    return report


def _performance_gate(
    *,
    readiness: Mapping[str, Any],
    shadow: Mapping[str, Any],
    performance_report: Mapping[str, Any],
) -> dict[str, Any]:
    if performance_report.get("gate") == "turbocore_native_update_performance_gate_v0":
        return dict(performance_report)
    nested = _as_dict(performance_report.get("performance_gate"))
    if nested.get("gate") == "turbocore_native_update_performance_gate_v0":
        return nested
    return build_native_update_performance_gate(
        readiness_report=readiness,
        shadow_report=shadow,
        performance_report=performance_report,
    )


def _readiness_with_performance(readiness: Mapping[str, Any], performance_gate: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(readiness)
    evidence_present = _performance_evidence_present(performance_gate)
    performance_ready = bool(performance_gate.get("representative_performance_gate_ready", False))
    if evidence_present:
        report["blocked_reasons"] = _without_reason(_strings(report.get("blocked_reasons")), "representative_performance_gate_missing")
        native_checks = _as_dict(report.get("native_checks"))
        if native_checks:
            native_checks["blocked_reasons"] = _without_reason(
                _strings(native_checks.get("blocked_reasons")),
                "representative_performance_gate_missing",
            )
            native_checks["performance_test_ready"] = performance_ready
            report["native_checks"] = native_checks
    report["performance_test_ready"] = performance_ready
    report["ok"] = not _strings(report.get("blocked_reasons"))
    return report


def _shadow_with_performance(
    shadow: Mapping[str, Any],
    performance_source: Mapping[str, Any],
    performance_gate: Mapping[str, Any],
) -> dict[str, Any]:
    report = dict(shadow)
    for key in ("optimizer_performance_gate", "native_update_optimizer_performance_gate", "benchmark_matrix", "native_update_benchmark_matrix", "update_benchmark_matrix"):
        value = _as_dict(performance_source.get(key))
        if value and key not in report:
            report[key] = value
    if performance_source.get("matrix") == "turbocore_update_benchmark_matrix_v0" and "benchmark_matrix" not in report:
        report["benchmark_matrix"] = dict(performance_source)
    optimizer_gate = _optimizer_gate_from_native_performance_gate(performance_gate)
    if optimizer_gate and "optimizer_performance_gate" not in report:
        report["optimizer_performance_gate"] = optimizer_gate
    nested_matrix = _as_dict(performance_source.get("benchmark_matrix"))
    if nested_matrix and "benchmark_matrix" not in report:
        report["benchmark_matrix"] = nested_matrix
    report["native_update_performance_gate"] = dict(performance_gate)
    return report


def _optimizer_gate_from_native_performance_gate(performance_gate: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _as_dict(performance_gate.get("evidence"))
    optimizer = _as_dict(evidence.get("optimizer_microbenchmark"))
    if not optimizer:
        return {}
    return {
        "gate": str(optimizer.get("source_gate", "") or "turbocore_optimizer_performance_gate"),
        "ok": bool(optimizer.get("ok", False)),
        "promotion_gate_ok": bool(optimizer.get("promotion_gate_ok", False)),
        "runtime_dispatch_allowed": False,
        "evidence_quality": str(optimizer.get("evidence_quality", "") or ""),
        "best_candidate": {
            "optimizer": str(optimizer.get("best_candidate_optimizer", "") or ""),
            "speedup_vs_baseline": optimizer.get("best_speedup_vs_baseline"),
        },
    }


def _scorecard_checks(
    *,
    readiness: Mapping[str, Any],
    gate: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
    performance_gate: Mapping[str, Any],
    promotion_dispatch: bool,
) -> dict[str, Any]:
    preflight = _as_dict(gate.get("dispatch_preflight"))
    contract = _as_dict(gate.get("dispatch_contract"))
    request = _as_dict(gate.get("dispatch_request"))
    launch = _as_dict(gate.get("kernel_launch_plan"))
    if promotion_dispatch:
        execution_expected = bool(
            gate.get("would_enable_native_update", False)
            and preflight.get("would_allow_native_dispatch", False)
            and contract.get("would_allow_native_dispatch", False)
            and request.get("dispatch_allowed", False)
            and launch.get("launch_allowed", False)
        )
        required = {
            "readiness_report_present": bool(readiness),
            "dispatch_preflight_present": bool(preflight),
            "dispatch_contract_present": bool(contract),
            "dispatch_request_present": bool(request),
            "kernel_launch_plan_present": bool(launch),
            "runtime_report_present": bool(runtime_report),
            "performance_gate_report_present": bool(performance_gate),
            "representative_performance_gate_ready": bool(performance_gate.get("representative_performance_gate_ready", False)),
            "training_path_enabled": bool(runtime_report.get("training_path_enabled", False)),
        }
        if execution_expected:
            required.update(
                {
                    "runtime_executes_native_step": bool(runtime_report.get("native_step_executed", False)),
                    "runtime_skips_pytorch_step": bool(runtime_report.get("should_call_pytorch_optimizer_step", True) is False),
                    "runtime_fallback_not_required": bool(runtime_report.get("fallback_to_pytorch_required", True) is False),
                }
            )
        else:
            required.update(
                {
                    "gate_blocks_unready_dispatch": bool(gate.get("would_enable_native_update", True) is False),
                    "preflight_blocks_unready_dispatch": bool(preflight.get("would_allow_native_dispatch", True) is False),
                    "contract_blocks_unready_dispatch": bool(contract.get("would_allow_native_dispatch", True) is False),
                    "request_blocks_unready_dispatch": bool(request.get("dispatch_allowed", True) is False),
                    "kernel_launch_blocks_unready_dispatch": bool(launch.get("launch_allowed", True) is False),
                    "runtime_keeps_pytorch_step": bool(runtime_report.get("should_call_pytorch_optimizer_step", False)),
                    "runtime_does_not_execute_native_step": bool(runtime_report.get("native_step_executed", True) is False),
                }
            )
    else:
        required = {
            "readiness_report_present": bool(readiness),
            "dispatch_preflight_present": bool(preflight),
            "dispatch_contract_present": bool(contract),
            "dispatch_request_present": bool(request),
            "kernel_launch_plan_present": bool(launch),
            "runtime_report_present": bool(runtime_report),
            "performance_gate_report_present": bool(performance_gate),
            "performance_gate_blocks_runtime_dispatch": bool(performance_gate.get("runtime_dispatch_allowed", True) is False),
            "preflight_blocks_dispatch": bool(preflight.get("would_allow_native_dispatch", True) is False),
            "contract_blocks_dispatch": bool(contract.get("would_allow_native_dispatch", True) is False),
            "request_blocks_dispatch": bool(request.get("dispatch_allowed", True) is False),
            "kernel_launch_blocks_execution": bool(launch.get("launch_allowed", True) is False),
            "runtime_keeps_pytorch_step": bool(runtime_report.get("should_call_pytorch_optimizer_step", False)),
            "runtime_does_not_execute_native_step": bool(runtime_report.get("native_step_executed", True) is False),
            "training_path_disabled": bool(gate.get("training_path_enabled", True) is False),
        }
    failed = [key for key, passed in required.items() if not passed]
    return {
        "schema_version": 1,
        "scorecard_evidence_coherent": not failed,
        "required": required,
        "failed_checks": failed,
        "representative_performance_gate_ready": bool(performance_gate.get("representative_performance_gate_ready", False)),
        "representative_performance_evidence_present": _performance_evidence_present(performance_gate),
        "training_path_enabled": bool(runtime_report.get("training_path_enabled", False)),
    }


def _compact_readiness(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "mode": str(report.get("mode", "") or ""),
        "native_kernel_present": bool(report.get("native_kernel_present", False)),
        "performance_test_ready": bool(report.get("performance_test_ready", False)),
        "stream_lifetime_bound": bool(report.get("stream_lifetime_bound", False)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": False,
    }


def _compact_performance_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _as_dict(report.get("evidence"))
    optimizer = _as_dict(evidence.get("optimizer_microbenchmark"))
    owner = _as_dict(evidence.get("owner_native_kernel"))
    matrix = _as_dict(evidence.get("training_matrix"))
    return {
        "gate": str(report.get("gate", "") or ""),
        "present": bool(report),
        "performance_test_ready": bool(report.get("performance_test_ready", False)),
        "representative_performance_gate_ready": bool(report.get("representative_performance_gate_ready", False)),
        "promotion_gate_ok": bool(report.get("promotion_gate_ok", False)),
        "runtime_dispatch_allowed": False,
        "training_path_enabled": False,
        "optimizer_microbenchmark": {
            "ok": bool(optimizer.get("ok", False)),
            "present": bool(optimizer.get("present", False)),
            "evidence_quality": optimizer.get("evidence_quality"),
            "best_candidate_optimizer": optimizer.get("best_candidate_optimizer"),
            "best_speedup_vs_baseline": optimizer.get("best_speedup_vs_baseline"),
            "blocked_reasons": _strings(optimizer.get("blocked_reasons")),
        },
        "owner_native_kernel": {
            "ok": bool(owner.get("ok", False)),
            "present": bool(owner.get("present", False)),
            "kernel_executed": bool(owner.get("kernel_executed", False)),
            "parity_ok": bool(owner.get("parity_ok", False)),
            "blocked_reasons": _strings(owner.get("blocked_reasons")),
        },
        "training_matrix": {
            "ok": bool(matrix.get("ok", False)),
            "present": bool(matrix.get("present", False)),
            "native_case": matrix.get("native_case"),
            "native_dispatch_executed": matrix.get("native_dispatch_executed"),
            "representative_steps": int(matrix.get("representative_steps", 0) or 0),
            "end_to_end_speedup": matrix.get("end_to_end_speedup"),
            "blocked_reasons": _strings(matrix.get("blocked_reasons")),
        },
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gate": str(report.get("gate", "") or ""),
        "mode": str(report.get("mode", "") or ""),
        "requested": bool(report.get("requested", False)),
        "would_enable_native_update": bool(report.get("would_enable_native_update", False)),
        "consecutive_shadow_passes": int(report.get("consecutive_shadow_passes", 0) or 0),
        "required_shadow_passes": int(report.get("required_shadow_passes", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
    }


def _compact_preflight(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preflight": str(report.get("preflight", "") or ""),
        "dispatch_preflight_passed": bool(report.get("dispatch_preflight_passed", False)),
        "would_allow_native_dispatch": bool(report.get("would_allow_native_dispatch", False)),
        "native_kernel_present": bool(report.get("native_kernel_present", False)),
        "performance_test_ready": bool(report.get("performance_test_ready", False)),
        "stream_lifetime_bound": bool(report.get("stream_lifetime_bound", False)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
    }


def _compact_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": str(report.get("contract", "") or ""),
        "dispatch_rehearsal_ready": bool(report.get("dispatch_rehearsal_ready", False)),
        "would_allow_native_dispatch": bool(report.get("would_allow_native_dispatch", False)),
        "pytorch_optimizer_authoritative": bool(report.get("pytorch_optimizer_authoritative", True)),
        "native_mutation_allowed": bool(report.get("native_mutation_allowed", False)),
        "training_parameter_mutation_allowed": bool(report.get("training_parameter_mutation_allowed", False)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "actions_required": _strings(report.get("actions_required")),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
    }


def _compact_request(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request": str(report.get("request", "") or ""),
        "requested": bool(report.get("requested", False)),
        "dispatch_allowed": bool(report.get("dispatch_allowed", False)),
        "runtime_dispatch_available": bool(report.get("runtime_dispatch_available", False)),
        "pytorch_optimizer_authoritative": bool(report.get("pytorch_optimizer_authoritative", True)),
        "fallback_to_pytorch_required": bool(report.get("fallback_to_pytorch_required", True)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
    }


def _compact_kernel_launch(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "launcher": str(report.get("launcher", "") or ""),
        "kernel": str(report.get("kernel", "") or ""),
        "requested": bool(report.get("requested", False)),
        "launch_allowed": bool(report.get("launch_allowed", False)),
        "launch_attempted": False,
        "kernel_executed": False,
        "mutates_training_parameters": bool(report.get("mutates_training_parameters", False)),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
    }


def _compact_execution_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan": str(report.get("plan", "") or ""),
        "native_executor_present": bool(report.get("native_executor_present", False)),
        "executor_preconditions_ready": bool(report.get("executor_preconditions_ready", False)),
        "training_executor_preconditions_ready": bool(report.get("training_executor_preconditions_ready", False)),
        "diagnostic_executor_preconditions_ready": bool(report.get("diagnostic_executor_preconditions_ready", False)),
        "diagnostic_executor_probe_allowed": bool(report.get("diagnostic_executor_probe_allowed", False)),
        "execution_allowed": bool(report.get("execution_allowed", False)),
        "would_call_native_executor": bool(report.get("would_call_native_executor", False)),
        "should_call_pytorch_optimizer_step": bool(report.get("should_call_pytorch_optimizer_step", True)),
        "diagnostic_executor_blocked_reasons": _strings(report.get("diagnostic_executor_blocked_reasons")),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
    }


def _compact_executor_probe(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "probe": str(report.get("probe", "") or ""),
        "attempted": bool(report.get("attempted", False)),
        "called": bool(report.get("called", False)),
        "ok": bool(report.get("ok", False)),
        "native_step_executed": False,
        "training_parameters_mutated": False,
        "should_call_pytorch_optimizer_step": bool(report.get("should_call_pytorch_optimizer_step", True)),
        "result": _as_dict(report.get("result")),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": False,
    }


def _compact_training_executor(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "executor": str(report.get("executor", "") or ""),
        "attempted": bool(report.get("attempted", False)),
        "called": bool(report.get("called", False)),
        "ok": bool(report.get("ok", False)),
        "native_step_executed": bool(report.get("native_step_executed", False)),
        "training_parameters_mutated": bool(report.get("training_parameters_mutated", False)),
        "should_call_pytorch_optimizer_step": bool(report.get("should_call_pytorch_optimizer_step", True)),
        "result": _as_dict(report.get("result")),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
    }


def _filter_promotion_blockers(
    blockers: list[str],
    *,
    promotion_dispatch: bool,
    runtime_report: Mapping[str, Any],
    performance_gate: Mapping[str, Any],
) -> list[str]:
    resolved: set[str] = set()
    if bool(performance_gate.get("representative_performance_gate_ready", False)):
        resolved.update(
            {
                "representative_performance_gate_missing",
                "representative_training_matrix_missing",
                "representative_training_matrix_not_executed",
                "representative_training_matrix_failed",
                "representative_training_steps_missing",
                "representative_training_steps_too_low",
                "native_dispatch_benchmark_case_missing",
                "native_dispatch_not_executed_in_benchmark_case",
                "optimizer_microbenchmark_missing",
                "optimizer_microbenchmark_gate_not_ok",
                "optimizer_microbenchmark_promotion_gate_not_ok",
                "optimizer_microbenchmark_not_promotion_grade",
                "optimizer_microbenchmark_speedup_missing",
                "optimizer_microbenchmark_speedup_below_promotion",
                "end_to_end_speedup_missing",
                "end_to_end_speedup_below_threshold",
            }
        )
    if not promotion_dispatch or not bool(runtime_report.get("native_step_executed", False)):
        return _dedupe([item for item in blockers if item not in resolved])
    native_runtime_ready = _native_training_execution_ready(runtime_report)
    resolved.update(
        {
            "native_dispatch_runtime_not_implemented",
            "native_dispatch_training_path_disabled",
            "native_dispatch_runtime_executor_missing",
            "native_dispatch_runtime_execution_guard_disabled",
            "native_dispatch_training_mutation_guard_disabled",
            "native_dispatch_training_path_not_requested",
            "native_dispatch_runtime_default_off",
            "native_dispatch_diagnostic_executor_call_disabled",
        }
    )
    if native_runtime_ready:
        resolved.update(
            {
                "owner_gradient_sync_default_off",
                "owner_gradient_sync_not_supported",
                "owner_gradient_sync_not_training_integrated",
                "owner_gradient_sync_guard_disabled",
                "owner_gradient_sync_not_promoted",
                "native_training_flat_owner_unavailable",
                "native_training_flat_owner_default_off",
                "native_training_flat_owner_not_promoted",
                "native_training_dispatch_kernel_missing",
                "native_training_dispatch_kernel_default_off",
                "native_training_dispatch_kernel_not_promoted",
                "stream_lifetime_unbound",
                "stream_lifetime_ownership_default_off",
                "stream_lifetime_ownership_not_promoted",
                "training_dispatch_recovery_default_off",
                "native_runtime_recovery_training_dispatch_disabled",
                "native_recovery_keeps_dispatch_disabled",
                "native_update_gate_not_enabled",
                "native_dispatch_rehearsal_not_ready",
                "native_dispatch_contract_not_allowing_dispatch",
                "dispatch_request_not_allowed",
                "dispatch_contract_not_allowing_launch",
                "dispatch_not_armed",
                "kernel_launch_not_allowed",
                "native_step_execution_disabled",
                "native_dispatch_training_path_not_requested",
                "native_dispatch_training_runtime_executor_default_off",
                "native_dispatch_training_path_default_off",
            }
        )
    if bool(performance_gate.get("representative_performance_gate_ready", False)):
        resolved.add("representative_performance_gate_missing")
    return _dedupe([item for item in blockers if item not in resolved])


def _promotion_blockers(
    blockers: list[str],
    *,
    performance_gate: Mapping[str, Any],
    promotion_dispatch: bool,
    runtime_report: Mapping[str, Any],
    product_exposure: Mapping[str, Any],
    release_review: Mapping[str, Any],
) -> list[str]:
    required: list[str] = []
    native_runtime_ready = _native_training_execution_ready(runtime_report)
    if not promotion_dispatch:
        required.extend(
            [
                "native_dispatch_runtime_not_implemented",
                "native_dispatch_training_path_disabled",
                "native_runtime_recovery_training_dispatch_disabled",
            ]
        )
    elif not native_runtime_ready:
        required.append("native_dispatch_native_kernel_not_promoted")
    if not bool(performance_gate.get("representative_performance_gate_ready", False)):
        required.append("representative_performance_gate_missing")
    required.extend(product_exposure_blockers(product_exposure))
    required.extend(release_review_blockers(release_review))
    return _dedupe(required + blockers)


def _native_training_execution_ready(runtime_report: Mapping[str, Any]) -> bool:
    training_executor = _as_dict(runtime_report.get("training_executor"))
    result = _as_dict(training_executor.get("result"))
    update_report = _as_dict(result.get("update_report"))
    optimizer_sync = _as_dict(result.get("optimizer_state_sync"))
    return bool(
        runtime_report.get("native_step_executed", False)
        and runtime_report.get("native_kernel_launched", False)
        and training_executor.get("native_kernel_launched", False)
        and training_executor.get("training_parameters_mutated", False)
        and result.get("pytorch_optimizer_state_synced", False)
        and optimizer_sync.get("synced", False)
        and update_report.get("owner_backend") == "rust_cuda_adamw_v0"
        and runtime_report.get("fallback_to_pytorch_required", True) is False
        and runtime_report.get("should_call_pytorch_optimizer_step", True) is False
    )


def _performance_evidence_present(report: Mapping[str, Any]) -> bool:
    evidence = _as_dict(report.get("evidence"))
    optimizer = _as_dict(evidence.get("optimizer_microbenchmark"))
    owner = _as_dict(evidence.get("owner_native_kernel"))
    matrix = _as_dict(evidence.get("training_matrix"))
    return bool(
        report.get("gate") == "turbocore_native_update_performance_gate_v0"
        and optimizer.get("present")
        and owner.get("present")
        and matrix.get("present")
    )


def _without_reason(values: list[str], reason: str) -> list[str]:
    return [value for value in values if value != reason]


def _normalize_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower().replace("-", "_")
    return normalized if normalized in {"off", "profile", "native_experimental"} else "off"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_native_update_promotion_scorecard"]
