"""Report-only quantized update parity for PagedAdamW8bit.

This scorecard proves that the P8C ABI buffers can be consumed by the
bitsandbytes blockwise 8-bit Adam update oracle.  It does not implement a
Lulynx native kernel or enable training dispatch.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Mapping, Sequence

import torch

from core.turbocore_paged_adamw8bit_checkpoint_adapter_scorecard import (
    build_paged_adamw8bit_checkpoint_adapter_scorecard,
)
from core.turbocore_paged_adamw8bit_native_abi_scorecard import (
    LAUNCH_PLAN,
    OPTIMIZER_FAMILY,
    OPTIMIZER_KIND,
    TARGET_OPTIMIZER,
    build_paged_adamw8bit_native_abi_scorecard,
)
from core.turbocore_paged_adamw8bit_residency_scorecard import (
    REQUIRED_LIVE_KEYS,
    _first_live_state,
    _make_trainer,
    _max_abs,
    _step,
)


SCRATCH_UPDATE_KIND = "bnb_blockwise_quantized_update_oracle_v0"
_STATE_COMPARE_KEYS = REQUIRED_LIVE_KEYS


def build_paged_adamw8bit_quantized_update_scorecard(
    *,
    native_abi_report: Mapping[str, Any] | None = None,
    checkpoint_adapter_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
) -> dict[str, Any]:
    """Build a report-only PagedAdamW8bit quantized update parity proof."""

    abi = dict(native_abi_report or build_paged_adamw8bit_native_abi_scorecard())
    adapter = dict(
        checkpoint_adapter_report
        or build_paged_adamw8bit_checkpoint_adapter_scorecard(
            native_abi_report=abi,
            run_live_probe=False,
            numel=numel,
        )
    )
    contract = _scratch_update_contract(numel=numel)
    live_probe = _live_probe(numel=numel) if run_live_probe else _skipped_live_probe("live_probe_disabled")
    validations = _validations(abi, adapter, contract, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    ready = not failed
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_quantized_update_scorecard_v0",
        "gate": "paged_adamw8bit_quantized_scratch_update_parity",
        "ok": ready,
        "promotion_ready": False,
        "quantized_update_contract_ready": ready,
        "bnb_oracle_probe_ready": str(live_probe.get("status", "unknown")) in {"passed", "skipped"},
        "bnb_oracle_parity_ready": bool(live_probe.get("bnb_oracle_parity_passed", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "launch_plan": LAUNCH_PLAN,
        "scratch_update_kind": SCRATCH_UPDATE_KIND,
        "scratch_update_contract": contract,
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "required_live_key_count": len(REQUIRED_LIVE_KEYS),
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "bnb_oracle_parity_passed": bool(live_probe.get("bnb_oracle_parity_passed", False)),
            "max_param_diff": live_probe.get("max_param_diff"),
            "max_state_float_diff": live_probe.get("max_state_float_diff"),
            "state_uint8_mismatch_count": live_probe.get("state_uint8_mismatch_count"),
            "native_kernel_ready": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_native_quantized_update_kernel_missing",
                "paged_adamw8bit_checkpoint_adapter_runtime_missing",
                "runtime_canary_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement PagedAdamW8bit native dequant/update/requant scratch kernel behind report-only canary"
            if ready
            else "fix PagedAdamW8bit quantized update parity blockers"
        ),
        "notes": [
            "This gate uses bitsandbytes.functional.optimizer_update_8bit_blockwise as an oracle.",
            "Passing this gate means the ABI buffers are sufficient for the quantized update boundary.",
            "It is not a Lulynx native kernel and does not change optimizer dispatch.",
        ],
    }


def _scratch_update_contract(*, numel: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scratch_update_kind": SCRATCH_UPDATE_KIND,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "functional_oracle": "bitsandbytes.functional.optimizer_update_8bit_blockwise",
        "optimizer_name": "adam",
        "required_live_buffers": list(REQUIRED_LIVE_KEYS),
        "required_scalars": [
            "step",
            "lr",
            "beta1",
            "beta2",
            "eps",
            "weight_decay",
            "gnorm_scale",
            "skip_zeros",
        ],
        "shape_contract": {
            "param_numel": int(numel),
            "quant_state_numel": int(numel),
            "quant_map_numel": 256,
            "absmax_block_size": 256,
        },
        "parity_contract": {
            "reference": "PagedAdamW8bit.step",
            "scratch": "direct optimizer_update_8bit_blockwise call on P8C buffers",
            "param_tolerance": 0.0,
            "float_state_tolerance": 0.0,
            "uint8_state_exact": True,
        },
        "runtime_policy": {
            "report_only": True,
            "native_kernel_implemented": False,
            "native_dispatch_allowed": False,
        },
    }


def _live_probe(*, numel: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    if importlib.util.find_spec("bitsandbytes") is None:
        return _skipped_live_probe("bitsandbytes_unavailable")
    try:
        import bitsandbytes.functional as F

        value = torch.linspace(-1.0, 1.0, steps=max(int(numel), 1), device="cuda")
        trainer = _make_trainer(value)
        optimizer = trainer._create_optimizer()
        optimizer_name = type(optimizer).__name__
        if optimizer_name == "AdamW":
            return {
                "schema_version": 1,
                "status": "failed",
                "reason": "resolved_to_fallback_adamw",
                "optimizer_class": optimizer_name,
                "blocked_reasons": ["paged_adamw8bit_resolved_to_fallback_adamw"],
            }

        param = trainer.lora_injector.param
        grad1 = torch.linspace(-0.1, 0.1, steps=param.numel(), device=param.device)
        grad2 = torch.linspace(0.05, -0.05, steps=param.numel(), device=param.device)
        _step(param, optimizer, grad1)
        checkpoint = copy.deepcopy(optimizer.state_dict())
        saved_param = param.detach().clone()

        reference = _make_trainer(saved_param)
        reference_optimizer = reference._create_optimizer()
        reference_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        _step(reference.lora_injector.param, reference_optimizer, grad2)

        scratch = _make_trainer(saved_param)
        scratch_optimizer = scratch._create_optimizer()
        scratch_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        scratch_param = scratch.lora_injector.param
        scratch_state = _first_live_state(scratch_optimizer)
        group = scratch_optimizer.param_groups[0]
        beta1, beta2 = group["betas"][:2]
        next_step = _step_int(scratch_state.get("step")) + 1
        scratch_param.grad = grad2.detach().clone()

        F.optimizer_update_8bit_blockwise(
            "adam",
            scratch_param.grad,
            scratch_param,
            scratch_state["state1"],
            scratch_state["state2"],
            float(beta1),
            float(beta2),
            0.0,
            float(group.get("alpha", 0.0) or 0.0),
            float(group["eps"]),
            next_step,
            float(group["lr"]),
            scratch_state["qmap1"],
            scratch_state["qmap2"],
            scratch_state["absmax1"],
            scratch_state["absmax2"],
            float(group["weight_decay"]),
            gnorm_scale=1.0,
            skip_zeros=False,
        )
        scratch_state["step"] = next_step

        reference_state = _first_live_state(reference_optimizer)
        param_compare = _compare_tensor(
            "param",
            reference.lora_injector.param.detach(),
            scratch_param.detach(),
            tolerance=0.0,
        )
        state_compare = _compare_states(reference_state, scratch_state, _STATE_COMPARE_KEYS)
        step_match = _step_int(reference_state.get("step")) == _step_int(scratch_state.get("step"))
        ok = bool(param_compare["ok"]) and bool(state_compare["ok"]) and step_match
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "optimizer_class": optimizer_name,
            "device": str(param.device),
            "functional_oracle": "bitsandbytes.functional.optimizer_update_8bit_blockwise",
            "bnb_oracle_parity_passed": ok,
            "direct_functional_call_used": True,
            "reference_step": _step_int(reference_state.get("step")),
            "scratch_step": _step_int(scratch_state.get("step")),
            "step_match": step_match,
            "param_compare": param_compare,
            "state_compare": state_compare,
            "max_param_diff": param_compare.get("max_diff"),
            "max_state_float_diff": state_compare.get("max_float_diff"),
            "state_uint8_mismatch_count": state_compare.get("uint8_mismatch_count"),
            "reference_state_signature": _state_signature(reference_state, _STATE_COMPARE_KEYS),
            "scratch_state_signature": _state_signature(scratch_state, _STATE_COMPARE_KEYS),
            "blocked_reasons": [] if ok else _live_blockers(param_compare, state_compare, step_match),
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"paged_adamw8bit_quantized_update_probe_failed:{type(exc).__name__}"],
        }


def _compare_tensor(name: str, left: torch.Tensor, right: torch.Tensor, *, tolerance: float) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    max_diff = _max_abs(left, right) if same_shape else float("inf")
    ok = same_shape and same_dtype and max_diff <= float(tolerance)
    return {
        "schema_version": 1,
        "name": name,
        "ok": ok,
        "shape_match": same_shape,
        "dtype_match": same_dtype,
        "left_dtype": str(left.dtype).replace("torch.", ""),
        "right_dtype": str(right.dtype).replace("torch.", ""),
        "max_diff": max_diff,
        "tolerance": float(tolerance),
    }


def _compare_states(left: Mapping[str, Any], right: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    mismatches: list[str] = []
    max_diffs: dict[str, float | int] = {}
    uint8_mismatch_count = 0
    max_float_diff = 0.0
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if not torch.is_tensor(left_value) or not torch.is_tensor(right_value):
            mismatches.append(f"{key}:missing")
            continue
        if left_value.shape != right_value.shape:
            mismatches.append(f"{key}:shape")
            continue
        if left_value.dtype != right_value.dtype:
            mismatches.append(f"{key}:dtype")
            continue
        if left_value.dtype == torch.uint8:
            diff = int((left_value.detach() != right_value.detach()).sum().cpu())
            max_diffs[key] = diff
            uint8_mismatch_count += diff
            if diff != 0:
                mismatches.append(f"{key}:uint8_value")
        else:
            diff = _max_abs(left_value.detach(), right_value.detach())
            max_diffs[key] = diff
            max_float_diff = max(max_float_diff, diff)
            if diff != 0.0:
                mismatches.append(f"{key}:float_value")
    return {
        "schema_version": 1,
        "ok": not mismatches,
        "mismatches": mismatches,
        "max_diffs": max_diffs,
        "max_float_diff": max_float_diff,
        "uint8_mismatch_count": uint8_mismatch_count,
    }


def _state_signature(state: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for key in keys:
        value = state.get(key)
        if not torch.is_tensor(value):
            rows.append({"role": key, "present": False})
            continue
        rows.append(
            {
                "role": key,
                "present": True,
                "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device),
                "shape": list(value.shape),
                "numel": int(value.numel()),
                "bytes": int(value.numel() * value.element_size()),
            }
        )
    return rows


def _validations(
    abi: Mapping[str, Any],
    adapter: Mapping[str, Any],
    contract: Mapping[str, Any],
    live_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy = contract.get("runtime_policy", {}) if isinstance(contract.get("runtime_policy"), Mapping) else {}
    return [
        _validation(
            "p8c_native_abi_sketch_ready",
            bool(abi.get("abi_sketch_ready", False)),
            "paged_adamw8bit_native_abi_sketch_missing",
        ),
        _validation(
            "p8d_checkpoint_adapter_proof_ready",
            bool(adapter.get("checkpoint_adapter_proof_ready", False)),
            "paged_adamw8bit_checkpoint_adapter_proof_missing",
        ),
        _validation(
            "quantized_update_oracle_contract_named",
            contract.get("functional_oracle") == "bitsandbytes.functional.optimizer_update_8bit_blockwise"
            and _same_roles(contract.get("required_live_buffers", []), REQUIRED_LIVE_KEYS),
            "paged_adamw8bit_quantized_update_contract_missing",
        ),
        _validation(
            "bnb_oracle_probe_or_skip",
            str(live_probe.get("status", "unknown")) in {"passed", "skipped"},
            "paged_adamw8bit_quantized_update_oracle_probe_failed",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(policy.get("native_kernel_implemented", True))
            and not bool(policy.get("native_dispatch_allowed", True)),
            "paged_adamw8bit_quantized_update_runtime_enabled_too_early",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _live_blockers(
    param_compare: Mapping[str, Any],
    state_compare: Mapping[str, Any],
    step_match: bool,
) -> list[str]:
    blockers = []
    if not bool(param_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_quantized_update_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_quantized_update_state_parity_failed")
    if not step_match:
        blockers.append("paged_adamw8bit_quantized_update_step_mismatch")
    return blockers


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "bnb_oracle_parity_passed": False,
        "max_param_diff": None,
        "max_state_float_diff": None,
        "state_uint8_mismatch_count": None,
        "blocked_reasons": [],
    }


def _step_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value or 0)


def _same_roles(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return {str(item) for item in left} == {str(item) for item in right}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "SCRATCH_UPDATE_KIND",
    "build_paged_adamw8bit_quantized_update_scorecard",
]
