"""Native registry dry-run scorecard for V2-P7 simple optimizer kernels."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINTS = (
    "get_simple_optimizer_kernel_contracts",
    "dry_run_simple_optimizer_launch",
    "simple_optimizer_cpu_reference_guard",
)


def build_simple_optimizer_registry_scorecard() -> dict[str, Any]:
    """Validate native dry-run launch and CPU guard entrypoints."""

    native = native_with_entrypoints(*ENTRYPOINTS)
    if native is None:
        return _blocked("simple_optimizer_native_entrypoints_missing")
    try:
        capability = native.get_simple_optimizer_kernel_contracts()
        cases = [_run_case(native, "lion"), _run_case(native, "sgd_nesterov")]
    except Exception as exc:
        return _blocked(f"simple_optimizer_registry_probe_failed:{type(exc).__name__}: {exc}")
    failed = [case for case in cases if not bool(case.get("ok", False))]
    stage_ready = not failed and bool(_capability_ok(capability))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_registry_scorecard_v0",
        "gate": "simple_formula_kernel_registry_dry_run",
        "ok": stage_ready,
        "promotion_ready": False,
        "registry_stage_ready": stage_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "capability": capability,
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": len(cases) - len(failed),
            "dry_run_ready_count": sum(1 for case in cases if case.get("dry_run_ready", False)),
            "cpu_reference_guard_ready_count": sum(1 for case in cases if case.get("cpu_reference_guard_ready", False)),
        },
        "promotion_blockers": [
            "lion_native_kernel_parity_missing",
            "sgd_nesterov_native_kernel_parity_missing",
            "runtime_canary_hit_missing",
            "e2e_no_regression_missing",
        ],
        "blocked_reasons": [str(reason) for case in failed for reason in case.get("blocked_reasons", [])],
        "recommended_next_step": "implement Lion flat fp32 CUDA parity kernel, then SGDNesterov",
    }


def _run_case(native: Any, optimizer_kind: str) -> dict[str, Any]:
    plan = _plan(optimizer_kind)
    buffers = _role_buffers(optimizer_kind)
    config = _config(optimizer_kind)
    dry_run = native.dry_run_simple_optimizer_launch(json.dumps(plan))
    guard = native.simple_optimizer_cpu_reference_guard(
        json.dumps(plan),
        json.dumps(buffers),
        json.dumps(config),
    )
    expected = _expected_preview(optimizer_kind, buffers, config, int(config["preview_limit"]))
    actual = [float(item) for item in guard.get("param_preview", [])]
    max_diff = _max_abs_diff(actual, expected)
    tolerance = 1e-9
    blocked: list[str] = []
    if not bool(dry_run.get("ok", False)):
        blocked.append(f"{optimizer_kind}_dry_run_failed")
    if not bool(guard.get("ok", False)):
        blocked.append(f"{optimizer_kind}_cpu_reference_guard_failed")
    if bool(guard.get("parameters_mutated", True)):
        blocked.append(f"{optimizer_kind}_cpu_reference_guard_mutated")
    if max_diff > tolerance:
        blocked.append(f"{optimizer_kind}_cpu_reference_preview_mismatch")
    return {
        "schema_version": 1,
        "ok": not blocked,
        "optimizer_kind": optimizer_kind,
        "dry_run_ready": bool(dry_run.get("ok", False) and dry_run.get("would_launch_kernel", False)),
        "cpu_reference_guard_ready": bool(guard.get("ok", False) and not guard.get("parameters_mutated", True)),
        "native_kernel_present": bool(dry_run.get("native_kernel_present", False)),
        "training_path_enabled": bool(dry_run.get("training_path_enabled", False)),
        "kernel_executed": bool(guard.get("kernel_executed", False)),
        "max_param_preview_diff": max_diff,
        "tolerance": tolerance,
        "dry_run": dry_run,
        "cpu_reference_guard": guard,
        "expected_param_preview": expected,
        "blocked_reasons": blocked,
    }


def _capability_ok(capability: Mapping[str, Any]) -> bool:
    if not isinstance(capability, Mapping):
        return False
    if capability.get("training_path_enabled", True):
        return False
    if capability.get("native_kernel_present", True):
        return False
    plans = {str(item) for item in capability.get("supported_plans", [])}
    return {
        "lion_flat_fp32_launch_plan_v0",
        "sgd_nesterov_flat_fp32_launch_plan_v0",
    }.issubset(plans)


def _plan(optimizer_kind: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "optimizer_kind": optimizer_kind,
        "plan_kind": f"{optimizer_kind}_flat_fp32_launch_plan_v0",
        "numel": 4,
        "block_size": 256,
        "grid_blocks": 1,
        "training_path_enabled": False,
        "native_kernel_present": False,
    }


def _role_buffers(optimizer_kind: str) -> dict[str, list[float]]:
    if optimizer_kind == "lion":
        return {
            "param_flat": [0.25, -0.5, 0.125, -0.75],
            "grad_flat": [0.1, -0.2, 0.05, 0.3],
            "exp_avg": [0.0, 0.0, 0.0, 0.0],
        }
    return {
        "param_flat": [0.4, -0.2, 0.15, -0.35],
        "grad_flat": [0.03, -0.07, 0.11, -0.05],
        "momentum_buffer": [0.0, 0.0, 0.0, 0.0],
    }


def _config(optimizer_kind: str) -> dict[str, Any]:
    if optimizer_kind == "lion":
        return {
            "lr": 1e-3,
            "betas": [0.9, 0.99],
            "weight_decay": 0.01,
            "max_grad_norm": 0.0,
            "finite_check": True,
            "preview_limit": 4,
        }
    return {
        "lr": 1e-2,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "max_grad_norm": 0.0,
        "finite_check": True,
        "preview_limit": 4,
    }


def _expected_preview(
    optimizer_kind: str,
    buffers: Mapping[str, Sequence[float]],
    config: Mapping[str, Any],
    limit: int,
) -> list[float]:
    param = [float(item) for item in buffers["param_flat"]]
    grad = [float(item) for item in buffers["grad_flat"]]
    if optimizer_kind == "lion":
        exp_avg = [float(item) for item in buffers["exp_avg"]]
        beta1, beta2 = [float(item) for item in config["betas"]]
        lr = float(config["lr"])
        weight_decay = float(config["weight_decay"])
        out: list[float] = []
        for p_value, g_value, m_value in zip(param, grad, exp_avg):
            decayed = p_value * (1.0 - lr * weight_decay) if weight_decay else p_value
            update = m_value * beta1 + g_value * (1.0 - beta1)
            out.append(decayed - lr * _sign(update))
            _ = m_value * beta2 + g_value * (1.0 - beta2)
        return out[:limit]
    momentum_buffer = [float(item) for item in buffers["momentum_buffer"]]
    lr = float(config["lr"])
    momentum = float(config["momentum"])
    weight_decay = float(config["weight_decay"])
    out = []
    for p_value, g_value, buf_value in zip(param, grad, momentum_buffer):
        d_p = g_value + p_value * weight_decay if weight_decay else g_value
        next_buffer = buf_value * momentum + d_p
        update = d_p + momentum * next_buffer
        out.append(p_value - lr * update)
    return out[:limit]


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _max_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_registry_scorecard_v0",
        "gate": "simple_formula_kernel_registry_dry_run",
        "ok": False,
        "promotion_ready": False,
        "registry_stage_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "cases": [],
        "summary": {
            "case_count": 0,
            "passed_case_count": 0,
            "dry_run_ready_count": 0,
            "cpu_reference_guard_ready_count": 0,
        },
        "promotion_blockers": [reason],
        "blocked_reasons": [reason],
        "recommended_next_step": "build or load lulynx_native with simple optimizer registry entrypoints",
    }


__all__ = ["build_simple_optimizer_registry_scorecard"]
