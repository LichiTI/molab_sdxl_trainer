"""Promotion scorecard for TurboCore LoRA fused-kernel candidates."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch

from core.turbocore_candidates import list_turbocore_candidates
from core.turbocore_lora_forward_preflight import build_lora_forward_dispatch_preflight
from core.turbocore_lora_native_abi import probe_lora_fused_native_abi
from core.turbocore_triton_lora import triton_lora_delta_v3_decision_for_shape
from core.lulynx_trainer.turbocore_lora_candidate_policy import decide_lora_candidate_for_shape
from core.lulynx_trainer.turbocore_lora_fused_benchmark import SHAPE_PRESETS


DEFAULT_PRESETS = ("tiny", "sdxl_short", "dit_short")
DEFAULT_RANKS = (4, 8, 16)
MIN_PROMOTION_CASES = 4
MIN_AVG_SPEEDUP = 1.05


def build_lora_fused_promotion_scorecard(
    *,
    candidate_scorecard: Mapping[str, Any] | None = None,
    benchmark_matrix: Mapping[str, Any] | None = None,
    native_abi_report: Mapping[str, Any] | None = None,
    native_scratch_report: Mapping[str, Any] | None = None,
    native_training_report: Mapping[str, Any] | None = None,
    forward_preflight_report: Mapping[str, Any] | None = None,
    target_candidate: str = "rust_cuda_lora_delta_v0",
    research_candidate: str = "triton_lora_delta_v3_dispatch",
    presets: Sequence[str] = DEFAULT_PRESETS,
    ranks: Sequence[int] = DEFAULT_RANKS,
    dtype: torch.dtype = torch.float16,
    shape_policy: str = "auto",
) -> dict[str, Any]:
    """Build a report-only scorecard for Rust/CUDA LoRA promotion.

    This does not execute kernels itself or attach to training.  It describes
    whether current Triton/native evidence is strong enough to promote into a
    Rust/CUDA ABI and keeps training dispatch closed until that is true.
    """

    native_abi = _native_abi_status(native_abi_report, target_candidate=target_candidate)
    native_scratch = _native_scratch_kernel_status(native_scratch_report)
    native_training = _native_training_status(native_training_report)
    registry = _registry_status(
        target_candidate=target_candidate,
        research_candidate=research_candidate,
        native_abi=native_abi,
    )
    policy = _shape_policy_status(
        research_candidate=research_candidate,
        presets=presets,
        ranks=ranks,
        dtype=dtype,
        shape_policy=shape_policy,
    )
    forward_preflight = _forward_preflight_status(
        forward_preflight_report,
        native_abi=native_abi,
        native_scratch=native_scratch,
        native_training=native_training,
        dtype=dtype,
    )
    native_training_ready = _native_training_ready(native_training)
    candidate_scorecard_report = _candidate_scorecard_with_native_validation(candidate_scorecard, native_training)
    candidate_evidence = _candidate_scorecard_status(candidate_scorecard_report, target_candidate=target_candidate)
    benchmark = _benchmark_status(benchmark_matrix, research_candidate=research_candidate)
    benchmark_ready = _benchmark_ready(benchmark)
    forward_ready = bool(forward_preflight.get("native_dispatch_allowed", False))
    abi_ready = bool(native_abi.get("abi_contract_available", False) and native_abi.get("native_kernel_present", False))
    base_required = _base_required_blockers(
        native_training_ready=native_training_ready,
        benchmark_ready=benchmark_ready,
        forward_ready=forward_ready,
        abi_ready=abi_ready,
    )
    blockers = _dedupe(
        registry["blocked_reasons"]
        + native_abi["blocked_reasons"]
        + native_scratch["blocked_reasons"]
        + native_training["blocked_reasons"]
        + forward_preflight["blocked_reasons"]
        + policy["blocked_reasons"]
        + candidate_evidence["blocked_reasons"]
        + benchmark["blocked_reasons"]
        + base_required
    )
    blockers = _with_scratch_kernel_context(blockers, native_scratch, native_training_ready=native_training_ready)
    promotion_blockers = _promotion_blockers(blockers, native_training_ready=native_training_ready)
    promotion_ready = bool(native_training_ready and benchmark_ready and forward_ready and abi_ready and not promotion_blockers)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_lora_fused_promotion_scorecard_v0",
        "ok": True,
        "debug_only": True,
        "shadow_run": True,
        "target_candidate": str(target_candidate),
        "research_candidate": str(research_candidate),
        "promotion_ready": promotion_ready,
        "training_dispatch": bool(native_training_ready and native_training.get("training_dispatch", False)),
        "training_path_enabled": bool(native_training_ready and native_training.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(native_training_ready and forward_ready),
        "native_kernel_present": bool(native_abi.get("native_kernel_present", False) or native_training.get("native_kernel_present", False)),
        "native_scratch_kernel_present": bool(native_scratch.get("scratch_kernel_present", False)),
        "scratch_kernel_probe_available": bool(native_scratch.get("scratch_kernel_probe_available", False)),
        "scratch_kernel_probe_ok": bool(native_scratch.get("ok", False)),
        "returns_training_tensor_payloads": bool(native_training_ready and native_training.get("training_tensor_binding", False)),
        "pytorch_lora_path_authoritative": not native_training_ready,
        "triton_research_path_authoritative": False,
        "fallback_to_pytorch_lora": not native_training_ready,
        "fallback_to_existing_training_path": not native_training_ready,
        "registry": registry,
        "native_abi": native_abi,
        "native_scratch_kernel": native_scratch,
        "native_training_dispatch": native_training,
        "forward_preflight": forward_preflight,
        "shape_policy": policy,
        "candidate_scorecard": candidate_evidence,
        "benchmark_matrix": benchmark,
        "promotion_blockers": promotion_blockers,
        "blocked_reasons": blockers,
    }


def _forward_preflight_status(
    report: Mapping[str, Any] | None,
    *,
    native_abi: Mapping[str, Any],
    native_scratch: Mapping[str, Any],
    native_training: Mapping[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    payload = _as_dict(report)
    if not payload:
        payload = build_lora_forward_dispatch_preflight(
            x_shape=(2, 64, 320),
            dtype=str(dtype).replace("torch.", ""),
            rank=4,
            native_abi_report={
                "schema_version": 1,
                "ok": bool(native_abi.get("ok", False)),
                "abi_contract_available": bool(native_abi.get("abi_contract_available", False)),
                "native_kernel_present": bool(native_abi.get("native_kernel_present", False)),
                "training_path_enabled": bool(native_abi.get("training_path_enabled", False)),
                "launch_plan": {
                    "plan_kind": native_abi.get("launch_plan_kind", ""),
                    "shape_contract_ok": native_abi.get("launch_plan_shape_contract_ok", False),
                    "training_path_enabled": False,
                },
                "blocked_reasons": list(native_abi.get("blocked_reasons", []) or []),
            },
            native_scratch_report=native_scratch,
            native_training_report=native_training,
            request_training_dispatch=_native_training_ready(native_training),
            allow_experimental_native=_native_training_ready(native_training),
        )
    blocked = _strings(payload.get("blocked_reasons"))
    return {
        "schema_version": 1,
        "present": bool(payload),
        "ok": bool(payload.get("ok", False)),
        "shape_contract_ok": bool(payload.get("shape_contract_ok", False)),
        "abi_contract_available": bool(payload.get("abi_contract_available", False)),
        "native_validation_ok": bool(payload.get("native_validation_ok", False)),
        "native_candidate_repeated_validation_seen": bool(payload.get("native_candidate_repeated_validation_seen", False)),
        "would_allow_native_forward": bool(payload.get("would_allow_native_forward", False)),
        "native_dispatch_allowed": bool(payload.get("native_dispatch_allowed", False)),
        "training_dispatch": bool(payload.get("training_dispatch", False)),
        "training_path_enabled": bool(payload.get("training_path_enabled", False)),
        "fallback_to_pytorch_lora": bool(payload.get("fallback_to_pytorch_lora", True)),
        "promotion_blockers": list(payload.get("promotion_blockers", []) or []),
        "blocked_reasons": _dedupe(blocked),
    }


def _native_training_status(report: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _as_dict(report)
    blocked = _strings(payload.get("blocked_reasons"))
    if not payload:
        blocked.append("native_lora_training_dispatch_report_missing")
    if payload and not bool(payload.get("ok", False)):
        blocked.append("native_lora_training_dispatch_probe_failed")
    required_flags = {
        "native_kernel_present": "native_lora_training_kernel_not_promoted",
        "kernel_executed": "lora_fused_training_dispatch_not_implemented",
        "output_mutated": "lora_forward_output_not_mutated",
        "training_tensor_binding": "lora_training_tensor_binding_missing",
        "training_dispatch": "lora_training_dispatch_not_integrated",
        "training_path_enabled": "lora_training_path_not_enabled",
        "autograd_binding": "lora_forward_autograd_binding_missing",
        "forward_backward_training_integration": "lora_forward_backward_training_integration_missing",
        "forward_parity_ok": "lora_forward_parity_failed",
        "backward_parity_ok": "lora_backward_parity_failed",
        "stream_lifetime_bound": "lora_forward_stream_lifetime_unbound",
        "runtime_recovery_ready": "lora_forward_runtime_recovery_missing",
    }
    for key, reason in required_flags.items():
        if payload and not bool(payload.get(key, False)):
            blocked.append(reason)
    if payload and bool(payload.get("fallback_to_pytorch_lora", True)):
        blocked.append("lora_native_training_fell_back_to_pytorch")
    return {
        "schema_version": 1,
        "present": bool(payload),
        "ok": bool(payload.get("ok", False)),
        "candidate": str(payload.get("candidate", "rust_cuda_lora_delta_v0") or "rust_cuda_lora_delta_v0"),
        "native_kernel_present": bool(payload.get("native_kernel_present", False)),
        "kernel_executed": bool(payload.get("kernel_executed", False)),
        "kernel_launch_count": int(payload.get("kernel_launch_count", 0) or 0),
        "output_mutated": bool(payload.get("output_mutated", False)),
        "training_tensor_binding": bool(payload.get("training_tensor_binding", False)),
        "training_dispatch": bool(payload.get("training_dispatch", False)),
        "training_path_enabled": bool(payload.get("training_path_enabled", False)),
        "autograd_binding": bool(payload.get("autograd_binding", False)),
        "forward_backward_training_integration": bool(payload.get("forward_backward_training_integration", False)),
        "forward_parity_ok": bool(payload.get("forward_parity_ok", False)),
        "backward_parity_ok": bool(payload.get("backward_parity_ok", False)),
        "max_abs_forward_diff": _float_or_none(payload.get("max_abs_forward_diff")),
        "max_abs_grad_diff": _float_or_none(payload.get("max_abs_grad_diff")),
        "stream_lifetime_bound": bool(payload.get("stream_lifetime_bound", False)),
        "runtime_recovery_ready": bool(payload.get("runtime_recovery_ready", False)),
        "fallback_to_pytorch_lora": bool(payload.get("fallback_to_pytorch_lora", True)),
        "pytorch_lora_path_authoritative": bool(payload.get("pytorch_lora_path_authoritative", True)),
        "dtype": str(payload.get("dtype", "") or ""),
        "device": str(payload.get("device", "") or ""),
        "blocked_reasons": _dedupe(blocked),
    }


def _native_scratch_kernel_status(report: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _as_dict(report)
    blocked = _strings(payload.get("blocked_reasons"))
    if not payload:
        blocked.append("lora_cuda_scratch_kernel_probe_missing")
    if payload and not bool(payload.get("ok", False)):
        blocked.append("lora_cuda_scratch_kernel_probe_not_passed")
    if payload and bool(payload.get("training_path_enabled", False)):
        blocked.append("lora_cuda_scratch_probe_unexpectedly_enabled_training_path")
    if payload and bool(payload.get("training_dispatch", False)):
        blocked.append("lora_cuda_scratch_probe_unexpectedly_enabled_dispatch")
    if payload and bool(payload.get("training_tensor_binding", False)):
        blocked.append("lora_cuda_scratch_probe_bound_training_tensors")
    if payload and bool(payload.get("training_parameters_mutated", False)):
        blocked.append("lora_cuda_scratch_probe_mutated_training_parameters")
    return {
        "schema_version": 1,
        "present": bool(payload),
        "ok": bool(payload.get("ok", False)),
        "scratch_kernel_probe_available": bool(payload.get("scratch_kernel_probe_available", False)),
        "scratch_kernel_present": bool(payload.get("scratch_kernel_present", False)),
        "native_kernel_present": bool(payload.get("native_kernel_present", False)),
        "kernel_executed": bool(payload.get("kernel_executed", False)),
        "case_count": int(payload.get("case_count", 0) or 0),
        "passed_case_count": int(payload.get("passed_case_count", 0) or 0),
        "kernel_executed_count": int(payload.get("kernel_executed_count", 0) or 0),
        "rank_count": int(payload.get("rank_count", 0) or 0),
        "native_candidate_repeated_validation_seen": bool(payload.get("native_candidate_repeated_validation_seen", False)),
        "scratch_matrix_representative": bool(payload.get("scratch_matrix_representative", False)),
        "parity_ok": bool(payload.get("parity_ok", False)),
        "max_abs_diff": _float_or_none(payload.get("max_abs_diff")),
        "scratch_buffers_only": bool(payload.get("scratch_buffers_only", False)),
        "training_tensor_binding": bool(payload.get("training_tensor_binding", False)),
        "training_dispatch": bool(payload.get("training_dispatch", False)),
        "training_path_enabled": bool(payload.get("training_path_enabled", False)),
        "performance_test_ready": bool(payload.get("performance_test_ready", False)),
        "blocked_reasons": _dedupe(blocked),
    }


def _with_scratch_kernel_context(
    blockers: list[str],
    native_scratch: Mapping[str, Any],
    *,
    native_training_ready: bool,
) -> list[str]:
    if not bool(native_scratch.get("ok", False)):
        return _dedupe(blockers)
    contextualized = [
        value for value in blockers
        if value != "native_lora_kernel_not_registered"
    ]
    if not native_training_ready:
        contextualized.append("native_lora_training_kernel_not_promoted")
    if not bool(native_scratch.get("scratch_matrix_representative", False)) and not native_training_ready:
        contextualized.append("lora_scratch_kernel_not_representative")
    if not bool(native_scratch.get("native_candidate_repeated_validation_seen", False)):
        contextualized.append("native_lora_candidate_has_not_passed_repeated_validation")
    return _dedupe(contextualized)


def _registry_status(*, target_candidate: str, research_candidate: str, native_abi: Mapping[str, Any]) -> dict[str, Any]:
    rows = list_turbocore_candidates("lora_fused").get("lora_fused", [])
    by_name = {str(row.get("name", "") or ""): dict(row) for row in rows if isinstance(row, Mapping)}
    target = by_name.get(str(target_candidate), {})
    research = by_name.get(str(research_candidate), {})
    blocked: list[str] = []
    if not target:
        blocked.append("rust_cuda_lora_candidate_missing")
    elif not bool(target.get("native", False)):
        blocked.append("rust_cuda_lora_candidate_not_marked_native")
    abi_contract_available = bool(native_abi.get("abi_contract_available", False))
    if target and not bool(target.get("available", False)) and not abi_contract_available:
        blocked.append("rust_cuda_lora_native_abi_not_available")
    if not research:
        blocked.append("triton_research_candidate_missing")
    elif not bool(research.get("experimental", False)):
        blocked.append("triton_research_candidate_not_marked_experimental")
    return {
        "schema_version": 1,
        "target_registered": bool(target),
        "target_native": bool(target.get("native", False)) if target else False,
        "target_available": bool(target.get("available", False)) if target else False,
        "target_abi_contract_available": abi_contract_available,
        "target_reason": str(target.get("reason", "") or "") if target else "missing",
        "research_registered": bool(research),
        "research_available": bool(research.get("available", False)) if research else False,
        "research_reason": str(research.get("reason", "") or "") if research else "missing",
        "reserved_native_candidates": [str(row.get("name")) for row in rows if bool(row.get("native", False))],
        "training_path_enabled": bool(native_abi.get("training_path_enabled", False)),
        "blocked_reasons": _dedupe(blocked),
    }


def _native_abi_status(report: Mapping[str, Any] | None, *, target_candidate: str) -> dict[str, Any]:
    payload = _as_dict(report)
    if not payload:
        try:
            payload = probe_lora_fused_native_abi()
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "report": "turbocore_lora_native_abi_probe_v0",
                "ok": False,
                "abi_contract_available": False,
                "blocked_reasons": [f"lora_native_abi_probe_failed:{type(exc).__name__}: {exc}"],
            }
    contract = _as_dict(payload.get("contract"))
    launch_plan = _as_dict(payload.get("launch_plan"))
    validation = _as_dict(payload.get("launch_plan_validation"))
    blocked = _strings(payload.get("blocked_reasons"))
    abi_contract_available = bool(payload.get("abi_contract_available", False))
    if not abi_contract_available:
        blocked.append("rust_cuda_lora_abi_contract_not_available")
    if not bool(payload.get("native_kernel_present", False)):
        blocked.append("native_lora_kernel_not_registered")
    return {
        "schema_version": 1,
        "present": bool(payload),
        "ok": bool(payload.get("ok", False)),
        "candidate": str(payload.get("candidate", target_candidate) or target_candidate),
        "abi_contract_available": abi_contract_available,
        "contract": str(contract.get("contract", "") or ""),
        "launch_plan_kind": str(launch_plan.get("plan_kind", "") or ""),
        "launch_plan_shape_contract_ok": bool(launch_plan.get("shape_contract_ok", False)),
        "launch_plan_validation_ok": bool(validation.get("ok", False)),
        "native_kernel_present": bool(payload.get("native_kernel_present", False)),
        "native_dispatch_allowed": bool(payload.get("native_dispatch_allowed", False)),
        "training_path_enabled": bool(payload.get("training_path_enabled", False)),
        "blocked_reasons": _dedupe(blocked),
    }


def _shape_policy_status(
    *,
    research_candidate: str,
    presets: Sequence[str],
    ranks: Sequence[int],
    dtype: torch.dtype,
    shape_policy: str,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    route_counts: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}
    allowed = 0
    skipped = 0
    for preset in presets:
        for batch, tokens, width in SHAPE_PRESETS.get(str(preset), []):
            for rank in ranks:
                decision = decide_lora_candidate_for_shape(
                    candidate=research_candidate,
                    preset=str(preset),
                    batch=int(batch),
                    tokens=int(tokens),
                    width=int(width),
                    rank=int(rank),
                    shape_policy=shape_policy,
                ).as_dict()
                route = triton_lora_delta_v3_decision_for_shape(
                    dtype=dtype,
                    out_features=int(width),
                    rank=int(rank),
                ) if str(research_candidate) == "triton_lora_delta_v3_dispatch" else {}
                if decision["should_run"]:
                    allowed += 1
                else:
                    skipped += 1
                    skipped_reasons[str(decision.get("reason", "unknown"))] = skipped_reasons.get(str(decision.get("reason", "unknown")), 0) + 1
                path = str(route.get("path", "policy_only") or "policy_only")
                route_counts[path] = route_counts.get(path, 0) + 1
                decisions.append({**decision, "route": route})
    blocked: list[str] = []
    if not decisions:
        blocked.append("lora_shape_policy_matrix_empty")
    if allowed <= 0:
        blocked.append("no_research_shapes_allowed_by_policy")
    if route_counts.get("pytorch_explicit", 0) > 0:
        blocked.append("research_dispatcher_falls_back_to_pytorch_for_representative_shapes")
    if skipped > 0:
        blocked.append("shape_policy_skips_representative_cases")
    return {
        "schema_version": 1,
        "research_candidate": str(research_candidate),
        "dtype": str(dtype).replace("torch.", ""),
        "shape_policy": str(shape_policy),
        "preset_count": len(list(presets)),
        "rank_count": len(list(ranks)),
        "case_count": len(decisions),
        "allowed_case_count": allowed,
        "skipped_case_count": skipped,
        "route_counts": route_counts,
        "skipped_reasons": skipped_reasons,
        "sample_decisions": decisions[:8],
        "training_path_enabled": False,
        "blocked_reasons": _dedupe(blocked),
    }


def _candidate_scorecard_status(scorecard: Mapping[str, Any] | None, *, target_candidate: str) -> dict[str, Any]:
    payload = _as_dict(scorecard)
    summary = _as_dict(payload.get("summary"))
    native_validation = _as_dict(payload.get("native_validation"))
    rows = [dict(row) for row in payload.get("rows", [])] if isinstance(payload.get("rows"), list) else []
    target_rows = [row for row in rows if str(_as_dict(row.get("candidate")).get("name", "")) == str(target_candidate)]
    native_validated = bool(native_validation.get("native_candidate_repeated_validation_seen", False)) or any(
        str(row.get("gate", "")) == "native_candidate_needs_repeated_validation" for row in target_rows
    )
    blocked: list[str] = []
    if not payload:
        blocked.append("lora_candidate_scorecard_missing")
    elif bool(summary.get("ready_for_training_activation", False)):
        blocked.append("candidate_scorecard_unexpectedly_claims_training_ready")
    if payload and not native_validated:
        blocked.append("native_lora_candidate_has_not_passed_repeated_validation")
    return {
        "schema_version": 1,
        "present": bool(payload),
        "prototype": str(payload.get("prototype", "") or "") if payload else "",
        "ready_for_training_activation": False,
        "target_candidate_rows": len(target_rows),
        "native_candidate_repeated_validation_seen": native_validated,
        "native_validation": native_validation,
        "gate_counts": dict(summary.get("gate_counts", {}) or {}) if summary else {},
        "training_activation_blockers": list(summary.get("training_activation_blockers", []) or []) if summary else [],
        "training_path_enabled": False,
        "blocked_reasons": _dedupe(blocked),
    }


def _candidate_scorecard_with_native_validation(
    scorecard: Mapping[str, Any] | None,
    native_training: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if not scorecard:
        return None
    payload = dict(scorecard)
    if bool(native_training.get("ok", False)):
        candidate = str(native_training.get("candidate", "rust_cuda_lora_delta_v0") or "rust_cuda_lora_delta_v0")
        payload["native_validation"] = {
            "schema_version": 1,
            "source": "lora_native_training_dispatch_probe",
            "native_candidate": candidate,
            "native_candidate_repeated_validation_seen": True,
            "case_count": int(native_training.get("case_count", 4) or 4),
            "passed_case_count": int(native_training.get("passed_case_count", 4) or 4),
            "rank_count": int(native_training.get("rank_count", 4) or 4),
            "scratch_only": False,
            "training_path_enabled": bool(native_training.get("training_path_enabled", False)),
        }
        rows = [dict(row) for row in payload.get("rows", [])] if isinstance(payload.get("rows"), list) else []
        for row in rows:
            if str(_as_dict(row.get("candidate")).get("name", "")) == candidate:
                row["gate"] = "native_candidate_needs_repeated_validation"
        if rows:
            payload["rows"] = rows
    return payload


def _benchmark_status(matrix: Mapping[str, Any] | None, *, research_candidate: str) -> dict[str, Any]:
    payload = _as_dict(matrix)
    summary = _as_dict(payload.get("summary"))
    quality = _as_dict(summary.get("quality"))
    candidates = [dict(item) for item in summary.get("candidate_summaries", [])] if isinstance(summary.get("candidate_summaries"), list) else []
    row = next((item for item in candidates if str(item.get("candidate")) == str(research_candidate)), {})
    blocked: list[str] = []
    if not payload:
        blocked.append("representative_lora_benchmark_matrix_missing")
    if payload and quality.get("evidence_level") != "benchmark":
        blocked.append("representative_lora_benchmark_not_promotion_grade")
    if payload and int(row.get("case_count", 0) or 0) < MIN_PROMOTION_CASES:
        blocked.append("representative_lora_case_count_too_low")
    avg_speedup = _float_or_none(row.get("avg_speedup_vs_reference")) if row else None
    if payload and avg_speedup is None:
        blocked.append("representative_lora_speedup_missing")
    elif avg_speedup is not None and avg_speedup < MIN_AVG_SPEEDUP:
        blocked.append("representative_lora_speedup_below_threshold")
    if payload and int(row.get("loss_count", 0) or 0) > 0:
        blocked.append("representative_lora_matrix_has_loss_cases")
    return {
        "schema_version": 1,
        "present": bool(payload),
        "benchmark": str(payload.get("benchmark", "") or "") if payload else "",
        "research_candidate": str(research_candidate),
        "evidence_level": str(quality.get("evidence_level", "") or "") if quality else "",
        "smoke_only": bool(quality.get("smoke_only", False)) if quality else False,
        "case_count": int(row.get("case_count", 0) or 0) if row else 0,
        "avg_speedup_vs_reference": avg_speedup,
        "best_speedup_vs_reference": _float_or_none(row.get("best_speedup_vs_reference")) if row else None,
        "worst_speedup_vs_reference": _float_or_none(row.get("worst_speedup_vs_reference")) if row else None,
        "win_count": int(row.get("win_count", 0) or 0) if row else 0,
        "loss_count": int(row.get("loss_count", 0) or 0) if row else 0,
        "ready_for_training_activation": _benchmark_ready({
            "present": bool(payload),
            "evidence_level": str(quality.get("evidence_level", "") or "") if quality else "",
            "case_count": int(row.get("case_count", 0) or 0) if row else 0,
            "avg_speedup_vs_reference": avg_speedup,
            "loss_count": int(row.get("loss_count", 0) or 0) if row else 0,
            "blocked_reasons": blocked,
        }),
        "training_path_enabled": False,
        "blocked_reasons": _dedupe(blocked),
    }


def _base_required_blockers(
    *,
    native_training_ready: bool,
    benchmark_ready: bool,
    forward_ready: bool,
    abi_ready: bool,
) -> list[str]:
    required: list[str] = []
    if not abi_ready:
        required.append("rust_cuda_lora_abi_not_promoted")
    if not native_training_ready:
        required.extend([
            "native_lora_training_kernel_not_promoted",
            "lora_fused_training_dispatch_not_implemented",
            "lora_forward_backward_training_integration_missing",
        ])
    if not benchmark_ready:
        required.append("representative_lora_step_throughput_matrix_not_passed")
    if not forward_ready:
        required.append("lora_training_dispatch_not_integrated")
    return required


def _promotion_blockers(blockers: list[str], *, native_training_ready: bool) -> list[str]:
    if native_training_ready and not blockers:
        return []
    required = [
        "lora_fused_training_dispatch_not_implemented",
        "representative_lora_step_throughput_matrix_not_passed",
        "lora_forward_backward_training_integration_missing",
    ]
    if "rust_cuda_lora_abi_not_promoted" in blockers:
        required.insert(0, "rust_cuda_lora_abi_not_promoted")
    else:
        required.insert(0, "native_lora_training_kernel_not_promoted")
    return _dedupe(required + blockers)


def _native_training_ready(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("ok", False)
        and report.get("native_kernel_present", False)
        and report.get("kernel_executed", False)
        and report.get("output_mutated", False)
        and report.get("training_tensor_binding", False)
        and report.get("training_dispatch", False)
        and report.get("training_path_enabled", False)
        and report.get("autograd_binding", False)
        and report.get("forward_backward_training_integration", False)
        and report.get("forward_parity_ok", False)
        and report.get("backward_parity_ok", False)
        and report.get("stream_lifetime_bound", False)
        and report.get("runtime_recovery_ready", False)
        and report.get("fallback_to_pytorch_lora", True) is False
    )


def _benchmark_ready(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("present", False)
        and report.get("evidence_level") == "benchmark"
        and int(report.get("case_count", 0) or 0) >= MIN_PROMOTION_CASES
        and (_float_or_none(report.get("avg_speedup_vs_reference")) or 0.0) >= MIN_AVG_SPEEDUP
        and int(report.get("loss_count", 0) or 0) == 0
        and not list(report.get("blocked_reasons", []) or [])
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_lora_fused_promotion_scorecard"]
