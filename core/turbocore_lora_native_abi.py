"""Read-only ABI contract probes for future Rust/CUDA LoRA fused kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.services.native_module_loader import native_with_entrypoints
from core.turbocore_capabilities import probe_native_training_bridge


REQUIRED_ENTRYPOINTS = (
    "get_lora_fused_kernel_contract",
    "build_lora_fused_launch_plan",
    "validate_lora_fused_launch_plan",
    "create_lora_cuda_kernel_runtime_session_py",
    "lora_cuda_kernel_runtime_session_snapshot_py",
    "launch_lora_delta_f32_runtime_session_py",
    "destroy_lora_cuda_kernel_runtime_session_py",
    "create_lora_cuda_f16_kernel_runtime_session_py",
    "lora_cuda_f16_kernel_runtime_session_snapshot_py",
    "launch_lora_delta_f16_runtime_session_py",
    "destroy_lora_cuda_f16_kernel_runtime_session_py",
)
SCRATCH_KERNEL_ENTRYPOINTS = (
    "probe_lora_cuda_scratch_launch_py",
    "probe_lora_cuda_scratch_matrix_py",
)


def probe_lora_fused_native_abi(
    *,
    x_shape: Sequence[int] = (2, 64, 320),
    rank: int = 4,
    out_features: int = 320,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Return native LoRA ABI evidence without executing a kernel."""

    bridge = probe_native_training_bridge()
    features = bridge.get("features") if isinstance(bridge.get("features"), Mapping) else {}
    capability = dict(features.get("lora_fused", {}) or {}) if isinstance(features, Mapping) else {}
    native = native_with_entrypoints(*REQUIRED_ENTRYPOINTS)
    if native is None:
        return _closed_report(
            capability=capability,
            reason="lora_fused_native_abi_entrypoints_missing",
            contract={},
            launch_plan={},
            validation={},
        )

    contract = _as_dict(native.get_lora_fused_kernel_contract())
    resolved_rank = max(int(rank), 1)
    resolved_x_shape = [max(int(dim), 1) for dim in x_shape]
    in_features = resolved_x_shape[-1] if resolved_x_shape else 1
    batch = resolved_x_shape[0] if len(resolved_x_shape) >= 1 else 1
    tokens = resolved_x_shape[1] if len(resolved_x_shape) >= 2 else 1
    down_shape = [resolved_rank, in_features]
    up_shape = [max(int(out_features), 1), resolved_rank]
    base_shape = [batch, tokens, max(int(out_features), 1)]
    launch_plan = _as_dict(native.build_lora_fused_launch_plan(
        json.dumps(resolved_x_shape),
        json.dumps(down_shape),
        json.dumps(up_shape),
        json.dumps(base_shape),
        str(dtype),
        resolved_rank,
        1.0,
    ))
    validation = _as_dict(native.validate_lora_fused_launch_plan(json.dumps(launch_plan)))
    abi_ready = bool(
        capability.get("abi_contract_available", False)
        and contract.get("contract") == "lora_delta_add_cuda_kernel_v0"
        and launch_plan.get("plan_kind") == "lora_delta_add_launch_plan_v0"
        and validation.get("ok", False)
    )
    blocked = _dedupe(
        _strings(capability.get("blocked_reasons"))
        + _strings(contract.get("blocked_reasons"))
        + _strings(launch_plan.get("blocked_reasons"))
    )
    if not abi_ready:
        blocked.append("lora_fused_native_abi_contract_not_ready")
    native_kernel_present = bool(contract.get("native_kernel_present", False))
    native_dispatch_allowed = bool(launch_plan.get("launch_allowed", False))
    training_path_enabled = bool(launch_plan.get("training_path_enabled", False))
    if not native_kernel_present:
        blocked.append("native_lora_kernel_not_registered")
    return {
        "schema_version": 1,
        "report": "turbocore_lora_native_abi_probe_v0",
        "ok": abi_ready,
        "abi_contract_available": abi_ready,
        "candidate": "rust_cuda_lora_delta_v0",
        "native_kernel_present": native_kernel_present,
        "native_dispatch_allowed": native_dispatch_allowed,
        "training_dispatch": bool(launch_plan.get("training_dispatch", False)),
        "training_path_enabled": training_path_enabled,
        "capability": capability,
        "contract": contract,
        "launch_plan": launch_plan,
        "launch_plan_validation": validation,
        "blocked_reasons": _dedupe(blocked),
    }


def probe_lora_cuda_scratch_kernel(
    *,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
    matrix: bool = True,
) -> dict[str, Any]:
    """Execute the scratch-only LoRA CUDA probe when the native entrypoint exists.

    This probe uses synthetic fp32 buffers owned by the native module.  It never
    binds training tensors and never promotes LoRA training dispatch.
    """

    bridge = probe_native_training_bridge()
    features = bridge.get("features") if isinstance(bridge.get("features"), Mapping) else {}
    capability = dict(features.get("cuda_lora_scratch_probe", {}) or {}) if isinstance(features, Mapping) else {}
    native = native_with_entrypoints(*SCRATCH_KERNEL_ENTRYPOINTS)
    if native is None:
        return _closed_scratch_report(
            capability=capability,
            reason="lora_cuda_scratch_entrypoint_missing",
            raw={},
        )

    root = Path(workspace_root) if workspace_root is not None else Path(__file__).resolve().parents[2]
    try:
        if matrix:
            raw = _as_dict(native.probe_lora_cuda_scratch_matrix_py(str(root), arch))
        else:
            raw = _as_dict(native.probe_lora_cuda_scratch_launch_py(str(root), arch))
    except Exception as exc:
        return _closed_scratch_report(
            capability=capability,
            reason=f"lora_cuda_scratch_probe_failed:{type(exc).__name__}: {exc}",
            raw={},
        )
    return _normalize_scratch_report(capability=capability, raw=raw)


def _closed_report(
    *,
    capability: Mapping[str, Any],
    reason: str,
    contract: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report": "turbocore_lora_native_abi_probe_v0",
        "ok": False,
        "abi_contract_available": False,
        "candidate": "rust_cuda_lora_delta_v0",
        "native_kernel_present": False,
        "native_dispatch_allowed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "capability": dict(capability),
        "contract": dict(contract),
        "launch_plan": dict(launch_plan),
        "launch_plan_validation": dict(validation),
        "blocked_reasons": [reason],
    }


def _closed_scratch_report(
    *,
    capability: Mapping[str, Any],
    reason: str,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report": "turbocore_lora_cuda_scratch_kernel_probe_v0",
        "ok": False,
        "present": False,
        "entrypoint_present": False,
        "scratch_kernel_probe_available": bool(capability.get("available", False)),
        "scratch_kernel_present": False,
        "native_kernel_present": False,
        "kernel_executed": False,
        "case_count": 0,
        "passed_case_count": 0,
        "kernel_executed_count": 0,
        "rank_count": 0,
        "native_candidate_repeated_validation_seen": False,
        "parity_ok": False,
        "max_abs_diff": None,
        "scratch_buffers_only": True,
        "training_tensor_binding": False,
        "parameters_mutated": False,
        "training_parameters_mutated": False,
        "native_dispatch_allowed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "performance_test_ready": False,
        "capability": dict(capability),
        "raw_probe": dict(raw),
        "blocked_reasons": [reason],
    }


def _normalize_scratch_report(*, capability: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    reason = str(raw.get("reason", "") or "")
    ok = bool(raw.get("ok", False))
    kernel_executed = bool(raw.get("kernel_executed", False))
    case_count = int(raw.get("case_count", 1 if kernel_executed else 0) or 0)
    passed_case_count = int(raw.get("passed_case_count", 1 if ok else 0) or 0)
    kernel_executed_count = int(raw.get("kernel_executed_count", 1 if kernel_executed else 0) or 0)
    parity_ok = bool(raw.get("parity_ok", False))
    training_path_enabled = bool(raw.get("training_path_enabled", False))
    training_dispatch = bool(raw.get("training_dispatch", False))
    training_tensor_binding = bool(raw.get("training_tensor_binding", False))
    training_parameters_mutated = bool(raw.get("training_parameters_mutated", False))
    scratch_buffers_only = bool(raw.get("scratch_buffers_only", False))
    blocked: list[str] = []
    if not ok:
        blocked.append(reason or "lora_cuda_scratch_probe_not_ready")
    if ok and kernel_executed_count <= 0:
        blocked.append("lora_cuda_scratch_kernel_not_executed")
    if kernel_executed_count > 0 and not parity_ok:
        blocked.append("lora_cuda_scratch_parity_failed")
    if training_path_enabled or training_dispatch or training_tensor_binding or training_parameters_mutated:
        blocked.append("lora_cuda_scratch_probe_unexpected_training_side_effect")
    return {
        "schema_version": 1,
        "report": "turbocore_lora_cuda_scratch_kernel_probe_v0",
        "ok": bool(ok and not blocked),
        "present": True,
        "entrypoint_present": True,
        "scratch_kernel_probe_available": True,
        "scratch_kernel_present": bool(raw.get("native_kernel_present", False)) and bool(kernel_executed or kernel_executed_count > 0),
        "native_kernel_present": bool(raw.get("native_kernel_present", False)),
        "kernel_name": str(raw.get("kernel_name", "") or ""),
        "kernel_executed": bool(kernel_executed or kernel_executed_count > 0),
        "case_count": case_count,
        "passed_case_count": passed_case_count,
        "kernel_executed_count": kernel_executed_count,
        "rank_count": int(raw.get("rank_count", 0) or 0),
        "native_candidate_repeated_validation_seen": bool(raw.get("native_candidate_repeated_validation_seen", False)),
        "scratch_matrix_representative": bool(raw.get("scratch_matrix_representative", False)),
        "parity_ok": parity_ok,
        "max_abs_diff": _float_or_none(raw.get("max_abs_diff")),
        "scratch_buffers_only": scratch_buffers_only,
        "training_tensor_binding": training_tensor_binding,
        "parameters_mutated": bool(raw.get("parameters_mutated", False)),
        "training_parameters_mutated": training_parameters_mutated,
        "native_dispatch_allowed": False,
        "training_dispatch": training_dispatch,
        "training_path_enabled": training_path_enabled,
        "performance_test_ready": bool(raw.get("performance_test_ready", False)),
        "compile_probe": _as_dict(raw.get("compile_probe")),
        "capability": dict(capability),
        "raw_probe": dict(raw),
        "blocked_reasons": _dedupe(blocked),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "REQUIRED_ENTRYPOINTS",
    "SCRATCH_KERNEL_ENTRYPOINTS",
    "probe_lora_cuda_scratch_kernel",
    "probe_lora_fused_native_abi",
]
