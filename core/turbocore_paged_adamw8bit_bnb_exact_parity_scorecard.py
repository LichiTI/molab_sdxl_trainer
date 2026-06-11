"""Report-only bnb live-buffer parity probe for PagedAdamW8bit.

This gate compares the native scratch-kernel formula against real
bitsandbytes live buffers.  It intentionally does not launch a native kernel
on training tensors and does not enable optimizer dispatch.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Mapping, Sequence

import torch

from core.turbocore_paged_adamw8bit_native_scratch_kernel_scorecard import (
    build_paged_adamw8bit_native_scratch_kernel_scorecard,
)
from core.turbocore_paged_adamw8bit_residency_scorecard import (
    REQUIRED_LIVE_KEYS,
    _first_live_state,
    _make_trainer,
    _step,
)


PROBE_KIND = "paged_adamw8bit_bnb_exact_native_formula_parity_v0"
BLOCK_SIZE = 256
FLOAT_TOLERANCE = 5e-6


def build_paged_adamw8bit_bnb_exact_parity_scorecard(
    *,
    native_kernel_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
) -> dict[str, Any]:
    """Build a report-only bnb live-buffer parity scorecard."""

    kernel = dict(native_kernel_report or build_paged_adamw8bit_native_scratch_kernel_scorecard())
    contract = _parity_contract(numel=numel)
    live_probe = _live_probe(numel=numel) if run_live_probe else _skipped_live_probe("live_probe_disabled")
    validations = _validations(kernel, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_bnb_exact_parity_scorecard_v0",
        "gate": "paged_adamw8bit_bnb_exact_native_formula_parity",
        "ok": ready,
        "promotion_ready": False,
        "bnb_exact_native_formula_parity_ready": ready,
        "bnb_exact_native_launch_parity_ready": False,
        "native_live_tensor_launch_ready": False,
        "training_tensor_binding_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "paged_adamw8bit",
        "optimizer_family": "adamw_quantized_paged",
        "probe_kind": PROBE_KIND,
        "parity_contract": contract,
        "native_kernel_summary": dict(kernel.get("summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "bnb_exact_native_formula_parity_passed": bool(
                live_probe.get("bnb_exact_native_formula_parity_passed", False)
            ),
            "native_scratch_kernel_parity_ready": bool(
                kernel.get("native_scratch_kernel_parity_ready", False)
            ),
            "max_param_diff": live_probe.get("max_param_diff"),
            "max_state_float_diff": live_probe.get("max_state_float_diff"),
            "state_uint8_mismatch_count": live_probe.get("state_uint8_mismatch_count"),
            "native_launch_used": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_native_live_tensor_launch_missing",
                "paged_adamw8bit_training_tensor_binding_missing",
                "paged_adamw8bit_checkpoint_adapter_runtime_missing",
                "runtime_canary_e2e_no_regression_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add report-only native live-buffer launch probe for PagedAdamW8bit"
            if ready
            else "fix PagedAdamW8bit bnb live-buffer formula parity blockers"
        ),
        "notes": [
            "This probe consumes real bitsandbytes live buffers but runs the native-equivalent formula in torch.",
            "It proves whether P8F math and bnb qmap/absmax semantics agree before tensor binding.",
            "It does not launch a native kernel on live training tensors and does not change dispatch.",
        ],
    }


def _parity_contract(*, numel: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe_kind": PROBE_KIND,
        "reference": "bitsandbytes PagedAdamW8bit.step",
        "candidate": "P8F native scratch formula on real bnb live buffers",
        "required_live_buffers": list(REQUIRED_LIVE_KEYS),
        "shape_contract": {
            "param_numel": int(numel),
            "block_size": BLOCK_SIZE,
            "quant_map_numel": 256,
        },
        "tolerances": {
            "param_max_abs": FLOAT_TOLERANCE,
            "float_state_max_abs": FLOAT_TOLERANCE,
            "uint8_state_exact": True,
        },
        "runtime_policy": {
            "report_only": True,
            "native_launch_used": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
    }


def _live_probe(*, numel: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    if importlib.util.find_spec("bitsandbytes") is None:
        return _skipped_live_probe("bitsandbytes_unavailable")
    try:
        value = torch.linspace(-1.0, 1.0, steps=max(int(numel), 1), device="cuda")
        trainer = _make_trainer(value)
        optimizer = trainer._create_optimizer()
        optimizer_name = type(optimizer).__name__
        if optimizer_name == "AdamW":
            return _failed_live_probe(
                "resolved_to_fallback_adamw",
                ["paged_adamw8bit_resolved_to_fallback_adamw"],
                optimizer_class=optimizer_name,
            )

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

        candidate = _make_trainer(saved_param)
        candidate_optimizer = candidate._create_optimizer()
        candidate_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        candidate_param = candidate.lora_injector.param
        candidate_state = _first_live_state(candidate_optimizer)
        group = candidate_optimizer.param_groups[0]
        _native_formula_update(candidate_param, grad2, candidate_state, group)

        reference_state = _first_live_state(reference_optimizer)
        param_compare = _compare_tensor(
            "param",
            reference.lora_injector.param.detach(),
            candidate_param.detach(),
            tolerance=FLOAT_TOLERANCE,
        )
        state_compare = _compare_states(reference_state, candidate_state, REQUIRED_LIVE_KEYS)
        step_match = _step_int(reference_state.get("step")) == _step_int(candidate_state.get("step"))
        ok = bool(param_compare["ok"]) and bool(state_compare["ok"]) and step_match
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "optimizer_class": optimizer_name,
            "device": str(param.device),
            "probe_kind": PROBE_KIND,
            "bnb_exact_native_formula_parity_passed": ok,
            "native_formula_uses_live_bnb_qmap": True,
            "native_launch_used": False,
            "training_tensor_binding": False,
            "reference_step": _step_int(reference_state.get("step")),
            "candidate_step": _step_int(candidate_state.get("step")),
            "step_match": step_match,
            "param_compare": param_compare,
            "state_compare": state_compare,
            "max_param_diff": param_compare.get("max_diff"),
            "max_state_float_diff": state_compare.get("max_float_diff"),
            "state_uint8_mismatch_count": state_compare.get("uint8_mismatch_count"),
            "reference_state_signature": _state_signature(reference_state, REQUIRED_LIVE_KEYS),
            "candidate_state_signature": _state_signature(candidate_state, REQUIRED_LIVE_KEYS),
            "blocked_reasons": [] if ok else _live_blockers(param_compare, state_compare, step_match),
        }
    except Exception as exc:
        return _failed_live_probe(
            f"{type(exc).__name__}: {exc}",
            [f"paged_adamw8bit_bnb_exact_formula_probe_failed:{type(exc).__name__}"],
        )


def _native_formula_update(
    param: torch.nn.Parameter,
    grad: torch.Tensor,
    state: Mapping[str, Any],
    group: Mapping[str, Any],
) -> None:
    with torch.no_grad():
        beta1, beta2 = group["betas"][:2]
        lr = float(group["lr"])
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        numel = int(param.numel())
        device = param.device
        indices = torch.arange(numel, device=device)
        block_ids = torch.div(indices, BLOCK_SIZE, rounding_mode="floor")

        state1 = state["state1"].view(-1)
        state2 = state["state2"].view(-1)
        qmap1 = state["qmap1"].view(-1)
        qmap2 = state["qmap2"].view(-1)
        absmax1 = state["absmax1"].view(-1)
        absmax2 = state["absmax2"].view(-1)
        g = grad.detach().to(device=device, dtype=torch.float32).view(-1)

        next_m = qmap1[state1.long()] * absmax1[block_ids]
        next_v = qmap2[state2.long()] * absmax2[block_ids]
        next_m = next_m * float(beta1) + g * (1.0 - float(beta1))
        next_v = next_v * float(beta2) + g * g * (1.0 - float(beta2))

        if weight_decay != 0.0:
            param.data.mul_(1.0 - lr * weight_decay)
        step = _step_int(state.get("step")) + 1
        bias_correction1 = 1.0 - float(beta1) ** step
        bias_correction2 = 1.0 - float(beta2) ** step
        denom = next_v.sqrt() / (bias_correction2**0.5) + eps
        param.data.view(-1).addcdiv_(next_m, denom, value=-(lr / bias_correction1))

        blocks = int(absmax1.numel())
        new_absmax1 = torch.empty_like(absmax1)
        new_absmax2 = torch.empty_like(absmax2)
        new_state1 = torch.empty_like(state1)
        new_state2 = torch.empty_like(state2)
        for block in range(blocks):
            start = block * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, numel)
            m_block = next_m[start:end]
            v_block = next_v[start:end]
            scale1 = m_block.abs().max().clamp_min(1.0e-20)
            scale2 = v_block.abs().max().clamp_min(1.0e-20)
            new_absmax1[block] = scale1
            new_absmax2[block] = scale2
            new_state1[start:end] = _nearest_codes(m_block / scale1, qmap1)
            new_state2[start:end] = _nearest_codes(v_block / scale2, qmap2)
        state1.copy_(new_state1)
        state2.copy_(new_state2)
        absmax1.copy_(new_absmax1)
        absmax2.copy_(new_absmax2)
        if isinstance(state, dict):
            state["step"] = step


def _nearest_codes(values: torch.Tensor, qmap: torch.Tensor) -> torch.Tensor:
    diff = (qmap.view(-1, 1) - values.view(1, -1)).abs()
    return torch.argmin(diff, dim=0).to(dtype=torch.uint8)


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
            if diff > FLOAT_TOLERANCE:
                mismatches.append(f"{key}:float_value")
    return {
        "schema_version": 1,
        "ok": not mismatches,
        "mismatches": mismatches,
        "max_diffs": max_diffs,
        "max_float_diff": max_float_diff,
        "uint8_mismatch_count": uint8_mismatch_count,
        "float_tolerance": FLOAT_TOLERANCE,
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
            }
        )
    return rows


def _validations(kernel: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    live_status = str(live_probe.get("status", "unknown"))
    live_ready = live_status == "skipped" or bool(
        live_probe.get("bnb_exact_native_formula_parity_passed", False)
    )
    return [
        _validation(
            "p8f_native_scratch_kernel_ready",
            bool(kernel.get("native_scratch_kernel_parity_ready", False)),
            "paged_adamw8bit_native_scratch_kernel_parity_missing",
        ),
        _validation(
            "bnb_exact_formula_probe_or_skip",
            live_ready,
            "paged_adamw8bit_bnb_exact_native_formula_parity_failed",
        ),
        _validation(
            "native_live_launch_not_enabled",
            not bool(live_probe.get("native_launch_used", True))
            and not bool(live_probe.get("training_tensor_binding", True)),
            "paged_adamw8bit_native_live_launch_enabled_too_early",
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
        blockers.append("paged_adamw8bit_bnb_exact_param_formula_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("paged_adamw8bit_bnb_exact_state_formula_parity_failed")
    if not step_match:
        blockers.append("paged_adamw8bit_bnb_exact_step_mismatch")
    return blockers


def _failed_live_probe(reason: str, blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "bnb_exact_native_formula_parity_passed": False,
        "native_launch_used": False,
        "training_tensor_binding": False,
        "blocked_reasons": blockers,
        **extra,
    }


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "bnb_exact_native_formula_parity_passed": False,
        "native_launch_used": False,
        "training_tensor_binding": False,
        "blocked_reasons": [],
    }


def _step_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value or 0)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "PROBE_KIND",
    "build_paged_adamw8bit_bnb_exact_parity_scorecard",
]
