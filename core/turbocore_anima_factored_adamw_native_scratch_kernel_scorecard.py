"""CUDA scratch-kernel parity scorecard for AnimaFactoredAdamW."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from core.services.native_module_loader import native_with_entrypoints


ENTRYPOINT = "probe_anima_factored_adamw_cuda_scratch_launch_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_anima_factored_adamw_native_scratch_kernel_scorecard(
    *,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return _blocked("anima_factored_adamw_scratch_probe_entrypoint_missing")
    root = str(Path(workspace_root or REPO_ROOT).resolve())
    cuda_arch = str(arch or os.environ.get("LULYNX_NATIVE_CUDA_ARCH") or "compute_89")
    case = _run_case(native, root, cuda_arch)
    ready = bool(case.get("native_kernel_parity_ready", False))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_anima_factored_adamw_native_scratch_kernel_scorecard_v0",
        "gate": "anima_factored_adamw_native_scratch_kernel_parity",
        "ok": ready,
        "promotion_ready": False,
        "kernel_parity_stage_ready": ready,
        "anima_factored_adamw_native_kernel_parity": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": ready,
        "workspace_root": root,
        "arch": cuda_arch,
        "case": case,
        "summary": {
            "case_count": int(case.get("case_count", 0) or 0),
            "passed_case_count": int(case.get("passed_case_count", 0) or 0),
            "kernel_executed_count": 1 if case.get("kernel_executed") is True else 0,
            "native_kernel_parity_ready_count": 1 if ready else 0,
        },
        "promotion_blockers": [] if ready else ["anima_factored_adamw_native_kernel_parity_missing"],
        "blocked_reasons": _case_blockers(case),
        "recommended_next_step": (
            "extend AnimaFactoredAdamW native kernel to live tensor binding canary"
            if ready
            else "fix AnimaFactoredAdamW scratch CUDA kernel parity"
        ),
    }


def _run_case(native: Any, workspace_root: str, arch: str) -> dict[str, Any]:
    try:
        probe = native.probe_anima_factored_adamw_cuda_scratch_launch_py(workspace_root, arch)
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "optimizer_kind": "anima_factored_adamw",
            "hard_failure": True,
            "reason": f"anima_factored_adamw_cuda_probe_failed:{type(exc).__name__}: {exc}",
            "kernel_executed": False,
            "native_kernel_parity_ready": False,
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": ["anima_factored_adamw_native_kernel_parity_missing"],
        }
    cases = [item for item in probe.get("cases", []) or [] if isinstance(item, Mapping)]
    case_count = int(probe.get("case_count", len(cases)) or len(cases))
    passed_case_count = sum(1 for item in cases if bool(item.get("ok", False)))
    parity_ready = (
        bool(probe.get("ok", False))
        and bool(probe.get("kernel_executed", False))
        and bool(probe.get("parity_ok", False))
        and bool(probe.get("parameters_mutated", False))
        and case_count >= 2
        and passed_case_count == case_count
        and _has_case(cases, "factored_256x256")
        and _has_case(cases, "unfactored_4x4")
        and not bool(probe.get("training_path_enabled", True))
    )
    return {
        "schema_version": 1,
        "ok": parity_ready,
        "optimizer_kind": "anima_factored_adamw",
        "hard_failure": not parity_ready,
        "kernel_name": str(probe.get("kernel_name", "anima_factored_adamw_flat_fp32_cuda_v0")),
        "kernel_executed": bool(probe.get("kernel_executed", False)),
        "parameters_mutated": bool(probe.get("parameters_mutated", False)),
        "training_path_enabled": bool(probe.get("training_path_enabled", True)),
        "native_dispatch_allowed": False,
        "native_kernel_parity_ready": parity_ready,
        "case_count": case_count,
        "passed_case_count": passed_case_count,
        "max_abs_diff": float(probe.get("max_abs_diff", 0.0) or 0.0),
        "tolerance": float(probe.get("tolerance", 5e-6) or 5e-6),
        "cases": cases,
        "probe": probe,
        "blocked_reasons": [] if parity_ready else ["anima_factored_adamw_native_kernel_parity_missing"],
    }


def _has_case(cases: list[Mapping[str, Any]], name: str) -> bool:
    return any(str(item.get("case", "")) == name and bool(item.get("ok", False)) for item in cases)


def _case_blockers(case: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in case.get("blocked_reasons", []) or []]


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "turbocore_anima_factored_adamw_native_scratch_kernel_scorecard_v0",
        "gate": "anima_factored_adamw_native_scratch_kernel_parity",
        "ok": False,
        "promotion_ready": False,
        "kernel_parity_stage_ready": False,
        "anima_factored_adamw_native_kernel_parity": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "case": {},
        "summary": {
            "case_count": 0,
            "passed_case_count": 0,
            "kernel_executed_count": 0,
            "native_kernel_parity_ready_count": 0,
        },
        "promotion_blockers": [reason, "anima_factored_adamw_native_kernel_parity_missing"],
        "blocked_reasons": [reason],
        "recommended_next_step": "build or load lulynx_native with AnimaFactoredAdamW scratch probe entrypoint",
    }


__all__ = ["build_anima_factored_adamw_native_scratch_kernel_scorecard"]
