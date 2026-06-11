"""Native scratch-kernel parity for AdamWScheduleFree."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_adamw_schedule_free_scratch_canary_scorecard import (
    build_adamw_schedule_free_scratch_canary_scorecard,
)


ENTRYPOINT = "probe_adamw_schedule_free_cuda_scratch_launch_py"
KERNEL_NAME = "adamw_schedule_free_flat_fp32_cuda_v0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adamw_schedule_free_native_scratch_kernel_scorecard(
    *,
    scratch_canary_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run the native AdamWScheduleFree scratch probe without dispatch."""

    scratch = dict(scratch_canary_report or build_adamw_schedule_free_scratch_canary_scorecard())
    probe = _run_native_probe(workspace_root=workspace_root, arch=arch)
    parity_ready = _probe_parity_ready(probe)
    validations = _validations(scratch, probe, parity_ready)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe([reason for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_schedule_free_native_scratch_kernel_scorecard_v0",
        "gate": "adamw_schedule_free_native_scratch_kernel_parity",
        "ok": ready,
        "promotion_ready": False,
        "native_scratch_kernel_parity_ready": parity_ready,
        "native_kernel_ready": parity_ready,
        "runtime_canary_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "adamw_schedule_free",
        "optimizer_family": "adamw_schedule_free",
        "kernel_name": KERNEL_NAME,
        "entrypoint": ENTRYPOINT,
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "probe": probe,
        "validations": validations,
        "scratch_canary_summary": dict(scratch.get("summary") or {}),
        "summary": {
            "kernel_executed": probe.get("kernel_executed") is True,
            "parity_ok": probe.get("parity_ok") is True,
            "case_count": int(probe.get("case_count", 0) or 0),
            "passed_case_count": int(probe.get("passed_case_count", 0) or 0),
            "max_abs_diff": probe.get("max_abs_diff"),
            "training_path_enabled": False,
            "runtime_canary_ready": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamw_schedule_free_training_tensor_binding_missing",
                "adamw_schedule_free_runtime_canary_missing",
                "native_dispatch_not_allowed",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add AdamWScheduleFree runtime canary manifest with dispatch still disabled"
            if ready
            else "fix AdamWScheduleFree native scratch kernel parity blockers"
        ),
        "notes": [
            "This probe launches a native scratch CUDA kernel on synthetic fp32 buffers only.",
            "It validates param, z, exp_avg_sq, and param-group scalar updates.",
            "It does not consume training tensors and does not change optimizer dispatch.",
        ],
    }


def _run_native_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "adamw_schedule_free_cuda_scratch_launch_probe",
            "reason": "lulynx_native_entrypoint_missing",
            "entrypoint": ENTRYPOINT,
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "blocked_reasons": ["adamw_schedule_free_native_scratch_entrypoint_missing"],
        }
    try:
        return dict(getattr(native, ENTRYPOINT)(str(Path(workspace_root or REPO_ROOT).resolve()), arch or "compute_89"))
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "adamw_schedule_free_cuda_scratch_launch_probe",
            "reason": f"native_probe_failed:{type(exc).__name__}",
            "error": str(exc),
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "blocked_reasons": [f"adamw_schedule_free_native_scratch_probe_failed:{type(exc).__name__}"],
        }


def _probe_parity_ready(probe: Mapping[str, Any]) -> bool:
    return (
        probe.get("ok") is True
        and probe.get("kernel_executed") is True
        and probe.get("parity_ok") is True
        and probe.get("parameters_mutated") is True
        and not bool(probe.get("training_path_enabled", True))
    )


def _validations(
    scratch: Mapping[str, Any],
    probe: Mapping[str, Any],
    parity_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "scratch_formula_canary_ready",
            scratch.get("scratch_formula_canary_ready") is True,
            "adamw_schedule_free_scratch_formula_canary_missing",
        ),
        _validation(
            "native_scratch_kernel_parity",
            parity_ready,
            "adamw_schedule_free_native_scratch_kernel_parity_missing",
        ),
        _validation(
            "scratch_probe_not_training_dispatch",
            not bool(probe.get("training_dispatch", True))
            and not bool(probe.get("training_tensor_binding", True))
            and not bool(probe.get("training_path_enabled", True)),
            "adamw_schedule_free_native_scratch_probe_touched_training_path",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ENTRYPOINT", "KERNEL_NAME", "build_adamw_schedule_free_native_scratch_kernel_scorecard"]
