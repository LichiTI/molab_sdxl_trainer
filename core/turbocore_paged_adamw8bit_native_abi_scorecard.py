"""Report-only native ABI sketch for PagedAdamW8bit.

This scorecard turns the P8B residency contract into a launch-plan contract.
It deliberately stops before checkpoint adapter, runtime dispatch, or CUDA
kernel implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.configs import OptimizerType
from core.turbocore_paged_adamw8bit_residency_scorecard import (
    REQUIRED_LIVE_KEYS,
    build_paged_adamw8bit_residency_scorecard,
)


OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"
LAUNCH_PLAN = "paged_adamw8bit_flat_quantized_launch_plan_v0"
TARGET_OPTIMIZER = OptimizerType.PAGED_ADAMW_8BIT

_REQUIRED_ABI_ROLES = (
    "param_flat",
    "grad_flat",
    "state1_uint8",
    "state2_uint8",
    "qmap1_fp32",
    "qmap2_fp32",
    "absmax1_fp32",
    "absmax2_fp32",
    "step",
    "lr",
    "beta1",
    "beta2",
    "eps",
    "weight_decay",
)


def build_paged_adamw8bit_native_abi_scorecard(
    *,
    residency_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build a request-shaped ABI sketch without native dispatch."""

    residency = dict(
        residency_report
        or build_paged_adamw8bit_residency_scorecard(run_live_probe=False)
    )
    mode = _normalize_mode(native_training_mode)
    buffer_contract = _buffer_contract()
    checkpoint_adapter = _checkpoint_adapter_contract(residency)
    validations = _validations(residency, buffer_contract, checkpoint_adapter, mode)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    abi_ready = not failed and bool(residency.get("residency_contract_ready", False))
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_native_abi_scorecard_v0",
        "gate": "paged_adamw8bit_native_abi_sketch",
        "ok": abi_ready,
        "promotion_ready": False,
        "abi_sketch_ready": abi_ready,
        "checkpoint_adapter_contract_ready": bool(checkpoint_adapter["contract_ready"]),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "launch_plan": LAUNCH_PLAN,
        "request_contract": {
            "optimizer_type": TARGET_OPTIMIZER.value,
            "optimizer_kind": OPTIMIZER_KIND,
            "optimizer_family": OPTIMIZER_FAMILY,
            "native_training_mode": mode,
            "launch_plan": LAUNCH_PLAN,
            "route_policy": "observe_or_canary_only_until_kernel_and_adapter_exist",
        },
        "buffer_contract": buffer_contract,
        "checkpoint_adapter_contract": checkpoint_adapter,
        "route_decision": _route_decision(mode, abi_ready),
        "validations": validations,
        "residency_summary": dict(residency.get("summary") or {}),
        "summary": {
            "buffer_role_count": len(buffer_contract),
            "required_abi_role_count": len(_REQUIRED_ABI_ROLES),
            "validation_count": len(validations),
            "passed_validation_count": len(validations) - len(failed),
            "checkpoint_adapter_required": bool(checkpoint_adapter["adapter_required"]),
            "checkpoint_adapter_implemented": bool(checkpoint_adapter["adapter_implemented"]),
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_checkpoint_adapter_missing",
                "paged_adamw8bit_native_kernel_missing",
                "runtime_canary_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement report-only bnb quant-state checkpoint adapter proof"
            if abi_ready
            else "fix PagedAdamW8bit native ABI sketch blockers"
        ),
        "notes": [
            "This scorecard names the native ABI but never enables dispatch.",
            "PagedAdamW8bit cannot reuse exact AdamW buffers because live state is quantized and packed for checkpoints.",
            "A future kernel must prove dequant/update/requant parity before runtime canary.",
        ],
    }


def _buffer_contract() -> list[dict[str, Any]]:
    return [
        _buffer("param_flat", "parameter values", "float32|float16|bfloat16", True, "trainer_param"),
        _buffer("grad_flat", "gradient values", "float32|float16|bfloat16", False, "trainer_grad"),
        _buffer("state1_uint8", "first moment quantized state", "uint8", True, "bnb_live_state.state1"),
        _buffer("state2_uint8", "second moment quantized state", "uint8", True, "bnb_live_state.state2"),
        _buffer("qmap1_fp32", "first moment quantization map", "float32", False, "bnb_live_state.qmap1"),
        _buffer("qmap2_fp32", "second moment quantization map", "float32", False, "bnb_live_state.qmap2"),
        _buffer("absmax1_fp32", "first moment block scales", "float32", True, "bnb_live_state.absmax1"),
        _buffer("absmax2_fp32", "second moment block scales", "float32", True, "bnb_live_state.absmax2"),
        _scalar("step", "optimizer step", "int64"),
        _scalar("lr", "learning rate", "float32"),
        _scalar("beta1", "Adam beta1", "float32"),
        _scalar("beta2", "Adam beta2", "float32"),
        _scalar("eps", "Adam epsilon", "float32"),
        _scalar("weight_decay", "decoupled weight decay", "float32"),
    ]


def _buffer(role: str, meaning: str, dtype: str, mutable: bool, source: str) -> dict[str, Any]:
    return {
        "role": role,
        "kind": "tensor",
        "meaning": meaning,
        "dtype": dtype,
        "mutable": mutable,
        "source": source,
        "required": True,
    }


def _scalar(role: str, meaning: str, dtype: str) -> dict[str, Any]:
    return {
        "role": role,
        "kind": "scalar",
        "meaning": meaning,
        "dtype": dtype,
        "mutable": False,
        "source": "optimizer_param_group",
        "required": True,
    }


def _checkpoint_adapter_contract(residency: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(residency.get("summary") or {})
    static = dict(residency.get("static_contract") or {})
    policy = dict(static.get("residency_policy") or {})
    return {
        "schema_version": 1,
        "contract_ready": bool(policy.get("native_route_requires_checkpoint_adapter", True)),
        "adapter_required": True,
        "adapter_implemented": False,
        "checkpoint_layout": "bnb_state_dict_with___bnb_optimizer_quant_state__",
        "live_layout": "state1/state2/qmap1/qmap2/absmax1/absmax2",
        "required_transitions": [
            "unpack_bnb_quant_state_to_live_buffers",
            "pack_live_buffers_to_bnb_quant_state",
            "roundtrip_resume_without_parameter_drift",
        ],
        "observed_checkpoint_packs_quant_state": bool(summary.get("checkpoint_packs_quant_state", False)),
        "native_route_requires_checkpoint_adapter": bool(
            policy.get("native_route_requires_checkpoint_adapter", True)
        ),
        "blocked_reasons": ["paged_adamw8bit_checkpoint_adapter_missing"],
    }


def _validations(
    residency: Mapping[str, Any],
    buffer_contract: Sequence[Mapping[str, Any]],
    checkpoint_adapter: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    static = dict(residency.get("static_contract") or {})
    required_from_p8b = [str(item) for item in static.get("native_abi_requirements", []) or []]
    buffer_roles = [str(item.get("role", "")) for item in buffer_contract]
    return [
        _validation(
            "p8b_residency_contract_ready",
            bool(residency.get("residency_contract_ready", False)),
            "paged_adamw8bit_residency_contract_missing",
        ),
        _validation(
            "native_abi_roles_match_residency_contract",
            _same_roles(required_from_p8b, _REQUIRED_ABI_ROLES) and _same_roles(buffer_roles, _REQUIRED_ABI_ROLES),
            "paged_adamw8bit_native_abi_role_mismatch",
            {
                "p8b_required_roles": required_from_p8b,
                "buffer_roles": buffer_roles,
            },
        ),
        _validation(
            "live_state_roles_named",
            all(role in buffer_roles for role in _live_buffer_roles()),
            "paged_adamw8bit_live_roles_missing_from_abi",
            {"required_live_keys": list(REQUIRED_LIVE_KEYS)},
        ),
        _validation(
            "checkpoint_adapter_contract_named",
            bool(checkpoint_adapter.get("adapter_required", False))
            and bool(checkpoint_adapter.get("contract_ready", False)),
            "paged_adamw8bit_checkpoint_adapter_contract_missing",
        ),
        _validation(
            "runtime_mode_report_only",
            mode in {"off", "observe", "canary", "auto"},
            "paged_adamw8bit_native_training_mode_invalid",
            {"native_training_mode": mode},
        ),
    ]


def _validation(name: str, ok: bool, blocker: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _route_decision(mode: str, abi_ready: bool) -> dict[str, Any]:
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif mode == "observe" and abi_ready:
        decision = "would_native_shadow_but_blocked"
        reason = "abi_sketch_ready_kernel_and_adapter_missing"
    elif mode in {"canary", "auto"} and abi_ready:
        decision = "blocked_before_canary"
        reason = "checkpoint_adapter_and_native_kernel_missing"
    else:
        decision = "fallback"
        reason = "abi_sketch_not_ready"
    return {
        "schema_version": 1,
        "decision": decision,
        "reason": reason,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": mode in {"observe", "canary", "auto"},
        "request_fields": {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_kind": OPTIMIZER_KIND,
            "native_training_mode": mode,
            "launch_plan": LAUNCH_PLAN,
        },
    }


def _live_buffer_roles() -> tuple[str, ...]:
    return (
        "state1_uint8",
        "state2_uint8",
        "qmap1_fp32",
        "qmap2_fp32",
        "absmax1_fp32",
        "absmax2_fp32",
    )


def _same_roles(left: Sequence[str], right: Sequence[str]) -> bool:
    return set(left) == set(right)


def _normalize_mode(value: str) -> str:
    normalized = str(value or "observe").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "observe"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_paged_adamw8bit_native_abi_scorecard"]
