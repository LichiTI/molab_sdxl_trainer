"""Scratch formula canary for AdamWScheduleFree state-machine updates."""

from __future__ import annotations

import importlib.util
from typing import Any, Mapping

import torch

from core.configs import OptimizerType
from core.turbocore_adamw_schedule_free_native_abi_scorecard import (
    build_adamw_schedule_free_native_abi_scorecard,
)


TARGET_OPTIMIZER = OptimizerType.ADAMW_SCHEDULE_FREE
TOLERANCE = 1.0e-6


def build_adamw_schedule_free_scratch_canary_scorecard(
    *,
    native_abi_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a scratch state-machine update against schedulefree.AdamWScheduleFree."""

    abi = dict(native_abi_report or build_adamw_schedule_free_native_abi_scorecard())
    if importlib.util.find_spec("schedulefree") is None:
        return _blocked("schedulefree_unavailable", abi)
    cases = [_case("warmup_step", steps=1, warmup_steps=4), _case("post_warmup_two_step", steps=2, warmup_steps=1)]
    failed = [case for case in cases if case.get("ok") is not True]
    unsafe = _unsafe_claims(abi)
    ready = abi.get("abi_contract_ready") is True and not failed and not unsafe
    blockers = _dedupe(
        unsafe + [reason for case in failed for reason in case.get("blocked_reasons", []) or []]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_scratch_canary_scorecard_v0",
        "gate": "adamw_schedule_free_scratch_formula_canary",
        "ok": ready,
        "promotion_ready": False,
        "scratch_formula_canary_ready": ready,
        "native_ready": False,
        "native_kernel_ready": False,
        "runtime_canary_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "optimizer_family": "adamw_schedule_free",
        "abi_scorecard": _compact_abi(abi),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "max_param_diff": max(float(case.get("param_max_abs_diff", 0.0)) for case in cases),
            "max_z_diff": max(float(case.get("z_max_abs_diff", 0.0)) for case in cases),
            "max_exp_avg_sq_diff": max(float(case.get("exp_avg_sq_max_abs_diff", 0.0)) for case in cases),
            "native_ready_count": 0,
            "runtime_canary_ready_count": 0,
            "training_path_enabled_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamw_schedule_free_native_kernel_missing",
                "adamw_schedule_free_runtime_canary_missing",
                "native_dispatch_not_allowed",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "implement AdamWScheduleFree native scratch kernel against this canary"
            if ready
            else "fix AdamWScheduleFree scratch canary blockers"
        ),
        "notes": [
            "This canary proves the state-machine formula only.",
            "It does not call Rust/CUDA, consume training tensors, or enable dispatch.",
            "AdamWScheduleFree still needs a native kernel and runtime canary.",
        ],
    }


def _case(name: str, *, steps: int, warmup_steps: int) -> dict[str, Any]:
    import schedulefree

    initial = torch.linspace(-0.25, 0.35, steps=8, dtype=torch.float32)
    grads = [
        torch.linspace(0.01, 0.08, steps=8, dtype=torch.float32),
        torch.linspace(-0.04, 0.03, steps=8, dtype=torch.float32),
    ][:steps]
    kwargs = {
        "lr": 1.0e-3,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "warmup_steps": warmup_steps,
        "r": 0.0,
        "weight_lr_power": 2.0,
        "foreach": False,
    }
    param = torch.nn.Parameter(initial.clone())
    optimizer = schedulefree.AdamWScheduleFree([param], **kwargs)
    optimizer.train()
    scratch = _ScratchState(param=initial.clone(), z=initial.clone(), exp_avg_sq=torch.zeros_like(initial))
    group = {
        "k": 0,
        "lr": kwargs["lr"],
        "betas": kwargs["betas"],
        "eps": kwargs["eps"],
        "weight_decay": kwargs["weight_decay"],
        "warmup_steps": warmup_steps,
        "r": kwargs["r"],
        "weight_lr_power": kwargs["weight_lr_power"],
        "weight_sum": 0.0,
        "lr_max": 0.0,
        "scheduled_lr": 0.0,
        "train_mode": True,
    }
    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _scratch_step(scratch, grad.clone(), group)
    state = optimizer.state[param]
    opt_group = optimizer.state_dict()["param_groups"][0]
    diffs = {
        "param_max_abs_diff": _max_abs(param.detach(), scratch.param),
        "z_max_abs_diff": _max_abs(state["z"], scratch.z),
        "exp_avg_sq_max_abs_diff": _max_abs(state["exp_avg_sq"], scratch.exp_avg_sq),
        "weight_sum_abs_diff": abs(float(opt_group["weight_sum"]) - float(group["weight_sum"])),
        "lr_max_abs_diff": abs(float(opt_group["lr_max"]) - float(group["lr_max"])),
        "scheduled_lr_abs_diff": abs(float(opt_group["scheduled_lr"]) - float(group["scheduled_lr"])),
    }
    ok = (
        max(diffs.values()) <= TOLERANCE
        and int(opt_group["k"]) == int(group["k"]) == steps
        and bool(opt_group["train_mode"]) is True
    )
    return {
        "schema_version": 1,
        "case": name,
        "ok": ok,
        "steps": steps,
        "warmup_steps": warmup_steps,
        **diffs,
        "k": int(opt_group["k"]),
        "scratch_k": int(group["k"]),
        "tolerance": TOLERANCE,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "blocked_reasons": [] if ok else [f"adamw_schedule_free_scratch_canary_failed:{name}"],
    }


class _ScratchState:
    def __init__(self, *, param: torch.Tensor, z: torch.Tensor, exp_avg_sq: torch.Tensor) -> None:
        self.param = param
        self.z = z
        self.exp_avg_sq = exp_avg_sq


def _scratch_step(state: _ScratchState, grad: torch.Tensor, group: dict[str, Any]) -> None:
    beta1, beta2 = group["betas"]
    k = int(group["k"])
    warmup_steps = int(group["warmup_steps"])
    sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
    bias_correction2 = 1.0 - beta2 ** (k + 1)
    lr = float(group["lr"]) * sched
    group["scheduled_lr"] = lr
    group["lr_max"] = max(lr, float(group["lr_max"]))
    weight = ((k + 1) ** float(group["r"])) * (float(group["lr_max"]) ** float(group["weight_lr_power"]))
    group["weight_sum"] = float(group["weight_sum"]) + weight
    ckp1 = weight / float(group["weight_sum"]) if float(group["weight_sum"]) else 0.0
    state.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    denom = state.exp_avg_sq.div(bias_correction2).sqrt().add(float(group["eps"]))
    grad_normalized = grad.div(denom)
    if float(group["weight_decay"]) != 0.0:
        grad_normalized.add_(state.param, alpha=float(group["weight_decay"]))
    state.param.lerp_(end=state.z, weight=ckp1)
    state.param.add_(grad_normalized, alpha=lr * (beta1 * (1.0 - ckp1) - 1.0))
    state.z.sub_(grad_normalized, alpha=lr)
    group["k"] = k + 1


def _compact_abi(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok") is True,
        "abi_contract_ready": report.get("abi_contract_ready") is True,
        "state_machine_reference_ready": report.get("state_machine_reference_ready") is True,
        "native_ready": report.get("native_ready") is True,
        "native_kernel_ready": report.get("native_kernel_ready") is True,
    }


def _unsafe_claims(report: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for field in ("training_path_enabled", "default_behavior_changed", "native_dispatch_allowed", "native_kernel_ready"):
        if report.get(field) is True:
            out.append(f"adamw_schedule_free_native_abi:{field}")
    return out


def _blocked(reason: str, abi: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_scratch_canary_scorecard_v0",
        "gate": "adamw_schedule_free_scratch_formula_canary",
        "ok": False,
        "promotion_ready": False,
        "scratch_formula_canary_ready": False,
        "native_ready": False,
        "native_kernel_ready": False,
        "runtime_canary_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_type": TARGET_OPTIMIZER.value,
        "abi_scorecard": _compact_abi(abi),
        "cases": [],
        "summary": {"case_count": 0, "passed_case_count": 0},
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "install schedulefree before AdamWScheduleFree scratch canary work",
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["build_adamw_schedule_free_scratch_canary_scorecard"]
