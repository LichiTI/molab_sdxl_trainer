"""Report-only preflight for future native LoRA forward dispatch."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from core.turbocore_lora_native_abi import probe_lora_fused_native_abi


SUPPORTED_RANKS = {4, 8, 16, 32}
SUPPORTED_DTYPES = {"float32", "float16"}


def build_lora_forward_dispatch_preflight(
    *,
    x_shape: Sequence[int] = (2, 64, 320),
    down_shape: Sequence[int] | None = None,
    up_shape: Sequence[int] | None = None,
    base_output_shape: Sequence[int] | None = None,
    dtype: str = "float16",
    rank: int = 4,
    native_abi_report: Mapping[str, Any] | None = None,
    native_scratch_report: Mapping[str, Any] | None = None,
    native_training_report: Mapping[str, Any] | None = None,
    request_training_dispatch: bool = False,
    allow_experimental_native: bool = False,
) -> dict[str, Any]:
    """Build a closed preflight contract for native LoRA forward dispatch.

    The report intentionally refuses real dispatch.  It records shape/ABI/kernel
    evidence so training integration can later consume the same gate without
    adding a new training entrypoint.
    """

    shape = _shape_contract(
        x_shape=x_shape,
        down_shape=down_shape,
        up_shape=up_shape,
        base_output_shape=base_output_shape,
        dtype=dtype,
        rank=rank,
    )
    native_abi = dict(native_abi_report or probe_lora_fused_native_abi(
        x_shape=shape["x_shape"],
        rank=shape["rank"],
        out_features=shape["out_features"],
        dtype=shape["dtype"],
    ))
    scratch = dict(native_scratch_report or {})
    training = dict(native_training_report or {})
    training_ready = _native_training_ready(training)
    blockers = _dedupe(
        list(shape["blocked_reasons"])
        + _strings(native_abi.get("blocked_reasons"))
        + _strings(scratch.get("blocked_reasons"))
        + _strings(training.get("blocked_reasons"))
    )
    if not bool(native_abi.get("abi_contract_available", False)):
        blockers.append("lora_forward_native_abi_contract_missing")
    if not bool(native_abi.get("native_kernel_present", False)):
        blockers.append("native_lora_kernel_not_registered")
    if not bool(scratch.get("ok", False)) and not training_ready:
        blockers.append("lora_forward_native_kernel_validation_missing")
    if bool(scratch.get("ok", False)) and not bool(scratch.get("scratch_matrix_representative", False)) and not training_ready:
        blockers.append("lora_forward_scratch_kernel_not_representative")
    if not bool(request_training_dispatch):
        blockers.append("lora_forward_training_dispatch_not_requested")
    if not bool(allow_experimental_native):
        blockers.append("lora_forward_native_experimental_gate_disabled")
    if not training_ready:
        blockers.extend([
            "lora_forward_backward_training_integration_missing",
            "lora_forward_autograd_binding_missing",
            "lora_forward_stream_lifetime_unbound",
            "lora_forward_runtime_recovery_missing",
        ])
    blockers = _dedupe(blockers)
    dispatch_allowed = bool(
        shape["ok"]
        and request_training_dispatch
        and allow_experimental_native
        and training_ready
        and not blockers
    )
    launch_plan = _as_dict(native_abi.get("launch_plan"))
    return {
        "schema_version": 1,
        "preflight": "turbocore_lora_forward_dispatch_preflight_v0",
        "ok": True,
        "debug_only": True,
        "shadow_run": True,
        "shape_contract_ok": bool(shape["ok"]),
        "abi_contract_available": bool(native_abi.get("abi_contract_available", False)),
        "native_validation_ok": bool(scratch.get("ok", False)),
        "native_candidate_repeated_validation_seen": bool(scratch.get("native_candidate_repeated_validation_seen", False)),
        "native_kernel_present": bool(native_abi.get("native_kernel_present", False)),
        "scratch_kernel_present": bool(scratch.get("scratch_kernel_present", False)),
        "would_allow_native_forward": dispatch_allowed,
        "native_dispatch_allowed": dispatch_allowed,
        "training_dispatch": bool(dispatch_allowed and training.get("training_dispatch", False)),
        "training_path_enabled": bool(dispatch_allowed and training.get("training_path_enabled", False)),
        "pytorch_lora_path_authoritative": not dispatch_allowed,
        "fallback_to_pytorch_lora": not dispatch_allowed,
        "request_training_dispatch": bool(request_training_dispatch),
        "allow_experimental_native": bool(allow_experimental_native),
        "shape": shape,
        "native_abi": {
            "ok": bool(native_abi.get("ok", False)),
            "abi_contract_available": bool(native_abi.get("abi_contract_available", False)),
            "launch_plan_kind": str(launch_plan.get("plan_kind", "") or ""),
            "launch_plan_shape_contract_ok": bool(launch_plan.get("shape_contract_ok", False)),
            "training_path_enabled": bool(native_abi.get("training_path_enabled", False)),
        },
        "native_scratch_kernel": {
            "ok": bool(scratch.get("ok", False)),
            "case_count": int(scratch.get("case_count", 0) or 0),
            "passed_case_count": int(scratch.get("passed_case_count", 0) or 0),
            "rank_count": int(scratch.get("rank_count", 0) or 0),
            "max_abs_diff": _float_or_none(scratch.get("max_abs_diff")),
            "scratch_matrix_representative": bool(scratch.get("scratch_matrix_representative", False)),
            "training_path_enabled": bool(scratch.get("training_path_enabled", False)),
        },
        "native_training_dispatch": {
            "ok": bool(training.get("ok", False)),
            "native_kernel_present": bool(training.get("native_kernel_present", False)),
            "kernel_executed": bool(training.get("kernel_executed", False)),
            "output_mutated": bool(training.get("output_mutated", False)),
            "autograd_binding": bool(training.get("autograd_binding", False)),
            "forward_backward_training_integration": bool(training.get("forward_backward_training_integration", False)),
            "forward_parity_ok": bool(training.get("forward_parity_ok", False)),
            "backward_parity_ok": bool(training.get("backward_parity_ok", False)),
            "stream_lifetime_bound": bool(training.get("stream_lifetime_bound", False)),
            "runtime_recovery_ready": bool(training.get("runtime_recovery_ready", False)),
            "training_path_enabled": bool(training.get("training_path_enabled", False)),
        },
        "dispatch_contract": {
            "schema_version": 1,
            "kind": "lora_forward_dispatch_contract_v0",
            "candidate": "rust_cuda_lora_delta_v0",
            "launch_plan_kind": str(launch_plan.get("plan_kind", "lora_delta_add_launch_plan_v0") or "lora_delta_add_launch_plan_v0"),
            "shape_contract_ok": bool(shape["ok"]),
            "would_launch_kernel": dispatch_allowed,
            "fallback_route": "none" if dispatch_allowed else "pytorch_lora_delta",
            "training_path_enabled": dispatch_allowed,
        },
        "promotion_blockers": _promotion_blockers(blockers),
        "blocked_reasons": blockers,
    }


def _shape_contract(
    *,
    x_shape: Sequence[int],
    down_shape: Sequence[int] | None,
    up_shape: Sequence[int] | None,
    base_output_shape: Sequence[int] | None,
    dtype: str,
    rank: int,
) -> dict[str, Any]:
    x = [int(dim) for dim in x_shape]
    resolved_rank = max(int(rank), 1)
    batch = x[0] if len(x) >= 1 else 0
    tokens = x[1] if len(x) >= 2 else 0
    in_features = x[2] if len(x) >= 3 else 0
    down = [int(dim) for dim in (down_shape or (resolved_rank, in_features))]
    out_features = int(up_shape[0]) if up_shape and len(up_shape) >= 1 else in_features
    up = [int(dim) for dim in (up_shape or (out_features, resolved_rank))]
    base = [int(dim) for dim in (base_output_shape or (batch, tokens, out_features))]
    normalized_dtype = _normalize_dtype(dtype)
    blocked: list[str] = []
    if len(x) != 3:
        blocked.append("lora_forward_x_shape_must_be_rank3")
    if len(down) != 2:
        blocked.append("lora_forward_down_shape_must_be_rank2")
    if len(up) != 2:
        blocked.append("lora_forward_up_shape_must_be_rank2")
    if len(base) != 3:
        blocked.append("lora_forward_base_output_shape_must_be_rank3")
    if resolved_rank not in SUPPORTED_RANKS:
        blocked.append("lora_forward_unsupported_rank")
    if normalized_dtype not in SUPPORTED_DTYPES:
        blocked.append("lora_forward_unsupported_dtype")
    if len(down) == 2 and down != [resolved_rank, in_features]:
        blocked.append("lora_forward_down_shape_mismatch")
    if len(up) == 2 and up[1] != resolved_rank:
        blocked.append("lora_forward_up_rank_mismatch")
    if len(base) == 3 and len(x) == 3 and (base[0] != batch or base[1] != tokens):
        blocked.append("lora_forward_base_batch_tokens_mismatch")
    if len(up) == 2 and len(base) == 3 and up[0] != base[2]:
        blocked.append("lora_forward_up_out_features_mismatch")
    if any(dim <= 0 for dim in x + down + up + base):
        blocked.append("lora_forward_shape_dimensions_must_be_positive")
    return {
        "ok": not blocked,
        "x_shape": x,
        "down_shape": down,
        "up_shape": up,
        "base_output_shape": base,
        "dtype": normalized_dtype,
        "rank": resolved_rank,
        "batch": batch,
        "tokens": tokens,
        "in_features": in_features,
        "out_features": base[2] if len(base) == 3 else 0,
        "blocked_reasons": _dedupe(blocked),
    }


def _normalize_dtype(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "fp16": "float16",
        "half": "float16",
        "torch.float16": "float16",
        "bf16": "bfloat16",
        "torch.bfloat16": "bfloat16",
    }.get(normalized, normalized)


def _promotion_blockers(blockers: list[str]) -> list[str]:
    if not blockers:
        return []
    required = [
        "lora_forward_backward_training_integration_missing",
        "lora_forward_training_dispatch_not_requested",
        "lora_forward_native_experimental_gate_disabled",
    ]
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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def main() -> int:
    print(json.dumps(build_lora_forward_dispatch_preflight(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_lora_forward_dispatch_preflight"]
