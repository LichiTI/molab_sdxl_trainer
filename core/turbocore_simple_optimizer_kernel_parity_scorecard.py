"""CUDA scratch-kernel parity scorecard for V2-P7 simple optimizers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINTS = (
    "probe_lion_cuda_scratch_launch_py",
    "probe_sgd_nesterov_cuda_scratch_launch_py",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_simple_optimizer_kernel_parity_scorecard(
    *,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run real scratch-buffer CUDA parity probes without training dispatch."""

    native = native_with_entrypoints(*ENTRYPOINTS)
    if native is None:
        return _blocked("simple_optimizer_kernel_probe_entrypoints_missing")
    root = str(Path(workspace_root or REPO_ROOT).resolve())
    cuda_arch = str(arch or os.environ.get("LULYNX_NATIVE_CUDA_ARCH") or "compute_89")
    lion_case = _run_lion_case(native, root, cuda_arch)
    sgd_case = _run_sgd_nesterov_case(native, root, cuda_arch)
    lion_ready = bool(lion_case.get("native_kernel_parity_ready", False))
    sgd_ready = bool(sgd_case.get("native_kernel_parity_ready", False))
    hard_failures = [case for case in (lion_case, sgd_case) if bool(case.get("hard_failure", False))]
    blockers = _promotion_blockers(lion_ready=lion_ready, sgd_ready=sgd_ready)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_kernel_parity_scorecard_v0",
        "gate": "simple_formula_native_kernel_parity",
        "ok": not hard_failures,
        "promotion_ready": False,
        "kernel_parity_stage_ready": lion_ready and sgd_ready,
        "lion_native_kernel_parity": lion_ready,
        "sgd_nesterov_native_kernel_parity": sgd_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": lion_ready and sgd_ready,
        "workspace_root": root,
        "arch": cuda_arch,
        "cases": [lion_case, sgd_case],
        "summary": {
            "case_count": 2,
            "passed_case_count": sum(1 for case in (lion_case, sgd_case) if case.get("ok") is True),
            "kernel_executed_count": sum(1 for case in (lion_case, sgd_case) if case.get("kernel_executed") is True),
            "native_kernel_parity_ready_count": sum(
                1 for case in (lion_case, sgd_case) if case.get("native_kernel_parity_ready") is True
            ),
        },
        "promotion_blockers": blockers,
        "blocked_reasons": _case_blockers(lion_case) + _case_blockers(sgd_case),
        "recommended_next_step": _recommended_next_step(lion_ready=lion_ready, sgd_ready=sgd_ready),
    }


def _run_lion_case(native: Any, workspace_root: str, arch: str) -> dict[str, Any]:
    try:
        probe = native.probe_lion_cuda_scratch_launch_py(workspace_root, arch)
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "optimizer_kind": "lion",
            "hard_failure": True,
            "reason": f"lion_cuda_probe_failed:{type(exc).__name__}: {exc}",
            "kernel_executed": False,
            "native_kernel_parity_ready": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": ["lion_native_kernel_parity_missing"],
        }
    parity_ready = (
        bool(probe.get("ok", False))
        and bool(probe.get("kernel_executed", False))
        and bool(probe.get("parity_ok", False))
        and bool(probe.get("parameters_mutated", False))
        and not bool(probe.get("training_path_enabled", True))
    )
    return {
        "schema_version": 1,
        "ok": parity_ready,
        "optimizer_kind": "lion",
        "hard_failure": not parity_ready,
        "kernel_name": str(probe.get("kernel_name", "lion_flat_fp32_cuda_v0")),
        "kernel_executed": bool(probe.get("kernel_executed", False)),
        "parameters_mutated": bool(probe.get("parameters_mutated", False)),
        "training_path_enabled": bool(probe.get("training_path_enabled", True)),
        "native_dispatch_allowed": False,
        "native_kernel_parity_ready": parity_ready,
        "max_abs_diff": float(probe.get("max_abs_diff", 0.0) or 0.0),
        "param_max_abs_diff": float(probe.get("param_max_abs_diff", 0.0) or 0.0),
        "state_max_abs_diff": float(probe.get("exp_avg_max_abs_diff", 0.0) or 0.0),
        "tolerance": float(probe.get("tolerance", 5e-6) or 5e-6),
        "probe": probe,
        "blocked_reasons": [] if parity_ready else ["lion_native_kernel_parity_missing"],
    }


def _run_sgd_nesterov_case(native: Any, workspace_root: str, arch: str) -> dict[str, Any]:
    try:
        probe = native.probe_sgd_nesterov_cuda_scratch_launch_py(workspace_root, arch)
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "optimizer_kind": "sgd_nesterov",
            "hard_failure": True,
            "reason": f"sgd_nesterov_cuda_probe_failed:{type(exc).__name__}: {exc}",
            "kernel_executed": False,
            "native_kernel_parity_ready": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": ["sgd_nesterov_native_kernel_parity_missing"],
        }
    parity_ready = (
        bool(probe.get("ok", False))
        and bool(probe.get("kernel_executed", False))
        and bool(probe.get("parity_ok", False))
        and bool(probe.get("parameters_mutated", False))
        and not bool(probe.get("training_path_enabled", True))
    )
    return {
        "schema_version": 1,
        "ok": parity_ready,
        "optimizer_kind": "sgd_nesterov",
        "hard_failure": not parity_ready,
        "kernel_name": str(probe.get("kernel_name", "sgd_nesterov_flat_fp32_cuda_v0")),
        "kernel_executed": bool(probe.get("kernel_executed", False)),
        "parameters_mutated": bool(probe.get("parameters_mutated", False)),
        "training_path_enabled": bool(probe.get("training_path_enabled", True)),
        "native_dispatch_allowed": False,
        "native_kernel_parity_ready": parity_ready,
        "max_abs_diff": float(probe.get("max_abs_diff", 0.0) or 0.0),
        "param_max_abs_diff": float(probe.get("param_max_abs_diff", 0.0) or 0.0),
        "state_max_abs_diff": float(probe.get("momentum_buffer_max_abs_diff", 0.0) or 0.0),
        "tolerance": float(probe.get("tolerance", 5e-6) or 5e-6),
        "probe": probe,
        "arch": arch,
        "blocked_reasons": [] if parity_ready else ["sgd_nesterov_native_kernel_parity_missing"],
    }


def _promotion_blockers(*, lion_ready: bool, sgd_ready: bool) -> list[str]:
    blockers = []
    if not lion_ready:
        blockers.append("lion_native_kernel_parity_missing")
    if not sgd_ready:
        blockers.append("sgd_nesterov_native_kernel_parity_missing")
    blockers.extend(["runtime_canary_hit_missing", "e2e_no_regression_missing"])
    return blockers


def _case_blockers(case: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in case.get("blocked_reasons", []) or []]


def _recommended_next_step(*, lion_ready: bool, sgd_ready: bool) -> str:
    if not lion_ready:
        return "implement Lion flat fp32 CUDA parity kernel"
    if not sgd_ready:
        return "implement SGDNesterov flat fp32 CUDA parity kernel"
    return "wire simple formula optimizer runtime canary"


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_kernel_parity_scorecard_v0",
        "gate": "simple_formula_native_kernel_parity",
        "ok": False,
        "promotion_ready": False,
        "kernel_parity_stage_ready": False,
        "lion_native_kernel_parity": False,
        "sgd_nesterov_native_kernel_parity": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "cases": [],
        "summary": {
            "case_count": 0,
            "passed_case_count": 0,
            "kernel_executed_count": 0,
            "native_kernel_parity_ready_count": 0,
        },
        "promotion_blockers": [
            reason,
            "lion_native_kernel_parity_missing",
            "sgd_nesterov_native_kernel_parity_missing",
            "runtime_canary_hit_missing",
            "e2e_no_regression_missing",
        ],
        "blocked_reasons": [reason],
        "recommended_next_step": "build or load lulynx_native with simple optimizer CUDA scratch probe entrypoints",
    }


__all__ = ["build_simple_optimizer_kernel_parity_scorecard"]
