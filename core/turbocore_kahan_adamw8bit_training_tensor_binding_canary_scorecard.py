"""Report-only training tensor binding canary for KahanAdamW8bit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.lulynx_trainer.kahan_adamw8bit import KahanAdamW8bit
from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_kahan_adamw8bit_runtime_canary_scorecard import (
    build_kahan_adamw8bit_runtime_canary_scorecard,
)


ENTRYPOINT = "probe_kahan_adamw8bit_training_tensor_binding_canary_py"
PROBE_KIND = "kahan_adamw8bit_training_tensor_binding_canary_v0"
FLOAT_TOLERANCE = 5e-6
BF16_PARAM_TOLERANCE = 5e-5
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_kahan_adamw8bit_training_tensor_binding_canary_scorecard(
    *,
    runtime_canary_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    numel: int = 4096,
    dtype: torch.dtype = torch.float32,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run an isolated training-tensor binding canary without dispatch."""

    runtime_canary = dict(
        runtime_canary_report
        or build_kahan_adamw8bit_runtime_canary_scorecard(native_training_mode="canary")
    )
    live_probe = (
        _live_probe(numel=numel, dtype=dtype, workspace_root=workspace_root, arch=arch)
        if run_live_probe
        else _skipped_live_probe("live_probe_disabled")
    )
    validations = _validations(runtime_canary, live_probe)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_training_tensor_binding_canary_scorecard_v0",
        "gate": "kahan_adamw8bit_training_tensor_binding_canary",
        "ok": ready,
        "promotion_ready": False,
        "training_tensor_binding_canary_ready": ready,
        "training_tensor_binding_probe_ready": str(live_probe.get("status", "unknown")) in {"passed", "skipped"},
        "training_tensor_binding_parity_ready": bool(live_probe.get("training_tensor_binding_parity_passed", False)),
        "runtime_canary_e2e_no_regression_ready": bool(live_probe.get("e2e_no_regression_passed", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "entrypoint": ENTRYPOINT,
        "probe_kind": PROBE_KIND,
        "optimizer_kind": "kahan_adamw8bit",
        "optimizer_family": "adamw_quantized_kahan",
        "dtype": _dtype_name(dtype),
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "runtime_canary_summary": dict(runtime_canary.get("manifest_summary") or {}),
        "live_probe": live_probe,
        "validations": validations,
        "summary": {
            "live_probe_status": str(live_probe.get("status", "unknown")),
            "dtype": str(live_probe.get("dtype") or _dtype_name(dtype)),
            "kernel_executed": bool(live_probe.get("kernel_executed", False)),
            "training_tensor_binding_parity_passed": bool(
                live_probe.get("training_tensor_binding_parity_passed", False)
            ),
            "e2e_no_regression_passed": bool(live_probe.get("e2e_no_regression_passed", False)),
            "max_param_diff": live_probe.get("max_param_diff"),
            "max_absmax_diff": live_probe.get("max_absmax_diff"),
            "max_kahan_comp_diff": live_probe.get("max_kahan_comp_diff"),
            "quantized_state_mismatch_count": live_probe.get("quantized_state_mismatch_count"),
            "loss_diff": live_probe.get("loss_diff"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "kahan_adamw8bit_runtime_dispatch_disabled_pending_review",
                "kahan_adamw8bit_bf16_native_dtype_matrix_missing",
                "kahan_adamw8bit_checkpoint_resume_adapter_missing",
                "kahan_adamw8bit_real_training_matrix_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add KahanAdamW8bit bf16 native dtype matrix before real-training matrix"
            if ready
            else "fix KahanAdamW8bit training tensor binding canary blockers"
        ),
        "notes": [
            "This canary uses a synthetic isolated training graph and real autograd gradients.",
            "It binds a Parameter, grad, quantized states, absmax tensors, and Kahan compensation to the native kernel.",
            "It never touches user training runs and keeps native optimizer dispatch disabled.",
        ],
    }


def _live_probe(
    *,
    numel: int,
    dtype: torch.dtype,
    workspace_root: str | Path | None,
    arch: str | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _skipped_live_probe("cuda_unavailable")
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _failed_live_probe(
            "lulynx_native_entrypoint_missing",
            ["kahan_adamw8bit_training_tensor_binding_entrypoint_missing"],
        )
    try:
        dtype = dtype if dtype in {torch.float32, torch.bfloat16} else torch.float32
        value = torch.linspace(-0.5, 0.5, steps=max(int(numel), 1), device="cuda", dtype=torch.float32).to(dtype)
        prime_param = torch.nn.Parameter(value.clone())
        prime_optimizer = KahanAdamW8bit([prime_param], lr=1e-3, weight_decay=0.01)
        prime_loss = _backward_loss(prime_param, scale=0.35)
        prime_optimizer.step()
        prime_optimizer.zero_grad(set_to_none=True)
        checkpoint = copy.deepcopy(prime_optimizer.state_dict())
        saved_param = prime_param.detach().clone()

        reference_param = torch.nn.Parameter(saved_param.clone())
        reference_optimizer = KahanAdamW8bit([reference_param], lr=1e-3, weight_decay=0.01)
        reference_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        _ensure_kahan_comp_float32(reference_optimizer, reference_param)
        reference_loss = _backward_loss(reference_param, scale=0.7)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

        candidate_param = torch.nn.Parameter(saved_param.clone())
        candidate_optimizer = KahanAdamW8bit([candidate_param], lr=1e-3, weight_decay=0.01)
        candidate_optimizer.load_state_dict(copy.deepcopy(checkpoint))
        _ensure_kahan_comp_float32(candidate_optimizer, candidate_param)
        candidate_loss = _backward_loss(candidate_param, scale=0.7)
        state = candidate_optimizer.state[candidate_param]
        group = candidate_optimizer.param_groups[0]
        beta1, beta2 = group["betas"][:2]
        step_index = int(state.get("step") or 0)
        exp_avg_q = state["exp_avg_q"]
        exp_avg_sq_q = state["exp_avg_sq_q"]
        launch_config = {
            "lr": float(group["lr"]),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "step_index": step_index,
            "max_numel": max(int(numel), 1_048_576),
            "canary_probe_only": True,
            "training_tensor_binding": True,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        launch = dict(
            getattr(native, ENTRYPOINT)(
                candidate_param,
                candidate_param.grad.detach().contiguous(),
                exp_avg_q.data,
                exp_avg_sq_q.data,
                exp_avg_q.absmax,
                exp_avg_sq_q.absmax,
                state["kahan_comp"],
                json.dumps(launch_config),
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or "compute_89",
            )
        )
        if not bool(launch.get("ok", False)):
            return _failed_live_probe(
                str(launch.get("reason") or "training_tensor_binding_launch_failed"),
                [f"kahan_adamw8bit_training_tensor_binding_launch_failed:{launch.get('reason', 'unknown')}"],
                native_launch=launch,
            )
        state["step"] = step_index + 1
        candidate_optimizer.zero_grad(set_to_none=True)

        reference_state = reference_optimizer.state[reference_param]
        param_compare = _compare_tensor(reference_param.detach(), candidate_param.detach())
        state_compare = _compare_kahan_state(reference_state, state)
        step_match = int(reference_state.get("step", 0)) == int(state.get("step", 0))
        loss_diff = abs(float(reference_loss) - float(candidate_loss))
        finite = all(torch.isfinite(torch.tensor(v)).item() for v in [prime_loss, reference_loss, candidate_loss])
        ok = bool(param_compare["ok"]) and bool(state_compare["ok"]) and step_match and finite
        return {
            "schema_version": 1,
            "status": "passed" if ok else "failed",
            "device": str(candidate_param.device),
            "dtype": _dtype_name(dtype),
            "probe_kind": PROBE_KIND,
            "kernel_executed": bool(launch.get("kernel_executed", False)),
            "training_tensor_binding_parity_passed": ok,
            "e2e_no_regression_passed": ok,
            "native_launch": launch,
            "isolated_training_graph": True,
            "training_dispatch": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "prime_loss": float(prime_loss),
            "reference_loss": float(reference_loss),
            "candidate_loss": float(candidate_loss),
            "loss_diff": loss_diff,
            "losses_finite": finite,
            "reference_step": int(reference_state.get("step", 0)),
            "candidate_step": int(state.get("step", 0)),
            "step_match": step_match,
            "param_compare": param_compare,
            "state_compare": state_compare,
            "max_param_diff": param_compare.get("max_diff"),
            "max_absmax_diff": state_compare.get("max_absmax_diff"),
            "max_kahan_comp_diff": state_compare.get("max_kahan_comp_diff"),
            "quantized_state_mismatch_count": state_compare.get("quantized_state_mismatch_count"),
            "blocked_reasons": [] if ok else _live_blockers(param_compare, state_compare, step_match, finite),
        }
    except Exception as exc:
        return _failed_live_probe(
            f"{type(exc).__name__}: {exc}",
            [f"kahan_adamw8bit_training_tensor_binding_probe_failed:{type(exc).__name__}"],
        )


def _backward_loss(param: torch.nn.Parameter, *, scale: float) -> float:
    if param.grad is not None:
        param.grad = None
    values = param.float()
    loss = ((values * float(scale)).pow(2).mean() + values.mean() * 0.013)
    loss.backward()
    return float(loss.detach().cpu())


def _ensure_kahan_comp_float32(optimizer: KahanAdamW8bit, param: torch.nn.Parameter) -> None:
    state = optimizer.state[param]
    comp = state.get("kahan_comp")
    if isinstance(comp, torch.Tensor) and comp.dtype != torch.float32:
        state["kahan_comp"] = comp.float()


def _validations(runtime_canary: Mapping[str, Any], live_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    live_status = str(live_probe.get("status", "unknown"))
    live_ready = live_status == "skipped" or bool(live_probe.get("training_tensor_binding_parity_passed", False))
    return [
        _validation(
            "p8u_runtime_canary_manifest_ready",
            bool(runtime_canary.get("runtime_canary_manifest_ready", False)),
            "kahan_adamw8bit_runtime_canary_manifest_missing",
        ),
        _validation(
            "training_tensor_binding_probe_or_skip",
            live_ready,
            "kahan_adamw8bit_training_tensor_binding_canary_failed",
        ),
        _validation(
            "e2e_no_regression_or_skip",
            live_status == "skipped" or bool(live_probe.get("e2e_no_regression_passed", False)),
            "kahan_adamw8bit_training_tensor_binding_e2e_regression",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(live_probe.get("training_dispatch", True))
            and not bool(live_probe.get("training_path_enabled", True))
            and not bool(live_probe.get("native_dispatch_allowed", True)),
            "kahan_adamw8bit_training_tensor_binding_enabled_dispatch",
        ),
    ]


def _compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    tolerance = _param_tolerance(left.dtype)
    max_diff = _max_abs(left, right) if same_shape else float("inf")
    return {
        "schema_version": 1,
        "ok": same_shape and same_dtype and max_diff <= tolerance,
        "shape_match": same_shape,
        "dtype_match": same_dtype,
        "left_dtype": str(left.dtype).replace("torch.", ""),
        "right_dtype": str(right.dtype).replace("torch.", ""),
        "max_diff": max_diff,
        "tolerance": tolerance,
    }


def _compare_kahan_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    quant_mismatch = 0
    max_absmax_diff = 0.0
    for key in ("exp_avg_q", "exp_avg_sq_q"):
        left_q = left[key]
        right_q = right[key]
        quant_mismatch += int((left_q.data != right_q.data).sum().item())
        max_absmax_diff = max(max_absmax_diff, _max_abs(left_q.absmax, right_q.absmax))
    max_kahan = _max_abs(left["kahan_comp"], right["kahan_comp"])
    ok = quant_mismatch == 0 and max_absmax_diff <= FLOAT_TOLERANCE and max_kahan <= FLOAT_TOLERANCE
    return {
        "schema_version": 1,
        "ok": ok,
        "quantized_state_mismatch_count": quant_mismatch,
        "max_absmax_diff": max_absmax_diff,
        "max_kahan_comp_diff": max_kahan,
        "tolerance": FLOAT_TOLERANCE,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _param_tolerance(dtype: torch.dtype) -> float:
    return BF16_PARAM_TOLERANCE if dtype == torch.bfloat16 else FLOAT_TOLERANCE


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
    finite: bool,
) -> list[str]:
    blockers = []
    if not bool(param_compare.get("ok", False)):
        blockers.append("kahan_adamw8bit_training_tensor_binding_param_parity_failed")
    if not bool(state_compare.get("ok", False)):
        blockers.append("kahan_adamw8bit_training_tensor_binding_state_parity_failed")
    if not step_match:
        blockers.append("kahan_adamw8bit_training_tensor_binding_step_mismatch")
    if not finite:
        blockers.append("kahan_adamw8bit_training_tensor_binding_non_finite_loss")
    return blockers


def _failed_live_probe(reason: str, blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "kernel_executed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": blockers,
        **extra,
    }


def _skipped_live_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "reason": reason,
        "training_tensor_binding_parity_passed": False,
        "e2e_no_regression_passed": False,
        "kernel_executed": False,
        "training_dispatch": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "ENTRYPOINT",
    "BF16_PARAM_TOLERANCE",
    "PROBE_KIND",
    "build_kahan_adamw8bit_training_tensor_binding_canary_scorecard",
]
