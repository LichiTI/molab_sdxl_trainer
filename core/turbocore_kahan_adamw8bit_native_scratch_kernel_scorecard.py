"""Report-only native scratch-kernel parity for KahanAdamW8bit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_kahan_adamw8bit_scratch_update_scorecard import (
    build_kahan_adamw8bit_scratch_update_scorecard,
)


ENTRYPOINT = "probe_kahan_adamw8bit_scratch_launch_py"
KERNEL_NAME = "kahan_adamw8bit_update_cuda_v0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_kahan_adamw8bit_native_scratch_kernel_scorecard(
    *,
    scratch_update_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run the native Kahan scratch kernel probe without training dispatch."""

    scratch = dict(scratch_update_report or build_kahan_adamw8bit_scratch_update_scorecard())
    probe = _run_native_probe(workspace_root=workspace_root, arch=arch)
    parity_ready = _probe_parity_ready(probe)
    validations = _validations(scratch, probe, parity_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_native_scratch_kernel_scorecard_v0",
        "gate": "kahan_adamw8bit_native_scratch_kernel_parity",
        "ok": ready,
        "promotion_ready": False,
        "native_scratch_kernel_parity_ready": parity_ready,
        "native_kernel_ready": parity_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "kahan_adamw8bit",
        "optimizer_family": "adamw_quantized_kahan",
        "kernel_name": KERNEL_NAME,
        "entrypoint": ENTRYPOINT,
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "probe": probe,
        "validations": validations,
        "scratch_update_summary": dict(scratch.get("summary") or {}),
        "summary": {
            "kernel_executed": bool(probe.get("kernel_executed", False)),
            "parity_ok": bool(probe.get("parity_ok", False)),
            "param_max_abs_diff": probe.get("param_max_abs_diff"),
            "kahan_comp_max_abs_diff": probe.get("kahan_comp_max_abs_diff"),
            "max_float_diff": probe.get("max_float_diff"),
            "quantized_state_mismatch_count": probe.get("quantized_state_mismatch_count"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "kahan_adamw8bit_training_tensor_binding_missing",
                "kahan_adamw8bit_runtime_canary_missing",
                "kahan_adamw8bit_bf16_native_dtype_matrix_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add KahanAdamW8bit runtime canary manifest with dispatch still disabled"
            if ready
            else "fix KahanAdamW8bit native scratch kernel parity blockers"
        ),
        "notes": [
            "This probe launches a native scratch CUDA kernel on synthetic fp32 buffers only.",
            "It validates quantized moment update plus Kahan compensation against a Rust reference.",
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
            "probe": "kahan_adamw8bit_scratch_launch_probe",
            "reason": "lulynx_native_entrypoint_missing",
            "entrypoint": ENTRYPOINT,
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "blocked_reasons": ["kahan_adamw8bit_native_scratch_entrypoint_missing"],
        }
    try:
        return dict(
            getattr(native, ENTRYPOINT)(
                str(Path(workspace_root or REPO_ROOT).resolve()),
                arch or "compute_89",
            )
        )
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "kahan_adamw8bit_scratch_launch_probe",
            "reason": f"native_probe_failed:{type(exc).__name__}",
            "error": str(exc),
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "blocked_reasons": [f"kahan_adamw8bit_native_scratch_probe_failed:{type(exc).__name__}"],
        }


def _probe_parity_ready(probe: Mapping[str, Any]) -> bool:
    return (
        bool(probe.get("ok", False))
        and bool(probe.get("kernel_executed", False))
        and bool(probe.get("parity_ok", False))
        and bool(probe.get("parameters_mutated", False))
        and not bool(probe.get("training_path_enabled", True))
    )


def _validations(
    scratch: Mapping[str, Any],
    probe: Mapping[str, Any],
    parity_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8s_scratch_update_parity_ready",
            bool(scratch.get("scratch_update_parity_ready", False)),
            "kahan_adamw8bit_scratch_update_parity_missing",
        ),
        _validation(
            "native_scratch_kernel_parity",
            parity_ready,
            "kahan_adamw8bit_native_scratch_kernel_parity_missing",
        ),
        _validation(
            "scratch_probe_not_training_dispatch",
            not bool(probe.get("training_dispatch", True))
            and not bool(probe.get("training_tensor_binding", True))
            and not bool(probe.get("training_path_enabled", True)),
            "kahan_adamw8bit_native_scratch_probe_touched_training_path",
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


__all__ = [
    "ENTRYPOINT",
    "KERNEL_NAME",
    "build_kahan_adamw8bit_native_scratch_kernel_scorecard",
]
