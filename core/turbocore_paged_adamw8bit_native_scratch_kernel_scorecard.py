"""Report-only native scratch-kernel parity for PagedAdamW8bit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_paged_adamw8bit_quantized_update_scorecard import (
    build_paged_adamw8bit_quantized_update_scorecard,
)


ENTRYPOINT = "probe_paged_adamw8bit_quantized_scratch_launch_py"
KERNEL_NAME = "paged_adamw8bit_dequant_update_requant_cuda_v0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_paged_adamw8bit_native_scratch_kernel_scorecard(
    *,
    quantized_update_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run the native scratch kernel probe without enabling training dispatch."""

    update = dict(
        quantized_update_report
        or build_paged_adamw8bit_quantized_update_scorecard(run_live_probe=False)
    )
    probe = _run_native_probe(workspace_root=workspace_root, arch=arch)
    parity_ready = _probe_parity_ready(probe)
    validations = _validations(update, probe, parity_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_native_scratch_kernel_scorecard_v0",
        "gate": "paged_adamw8bit_native_scratch_kernel_parity",
        "ok": ready,
        "promotion_ready": False,
        "native_scratch_kernel_parity_ready": parity_ready,
        "native_kernel_ready": parity_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "optimizer_kind": "paged_adamw8bit",
        "optimizer_family": "adamw_quantized_paged",
        "kernel_name": KERNEL_NAME,
        "entrypoint": ENTRYPOINT,
        "workspace_root": str(Path(workspace_root or REPO_ROOT).resolve()),
        "arch": str(arch or "compute_89"),
        "probe": probe,
        "validations": validations,
        "summary": {
            "kernel_executed": bool(probe.get("kernel_executed", False)),
            "parity_ok": bool(probe.get("parity_ok", False)),
            "param_max_abs_diff": probe.get("param_max_abs_diff"),
            "max_float_diff": probe.get("max_float_diff"),
            "state_uint8_mismatch_count": probe.get("state_uint8_mismatch_count"),
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_training_tensor_binding_missing",
                "paged_adamw8bit_runtime_canary_missing",
                "paged_adamw8bit_checkpoint_adapter_runtime_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add report-only runtime canary manifest for PagedAdamW8bit native scratch kernel"
            if ready
            else "fix PagedAdamW8bit native scratch kernel parity blockers"
        ),
        "notes": [
            "This probe launches a native scratch CUDA kernel on synthetic buffers only.",
            "It validates a qmap-compatible dequant/update/requant candidate against a Rust reference.",
            "It does not consume training tensors, does not implement bnb parity, and does not change dispatch.",
        ],
    }


def _run_native_probe(*, workspace_root: str | Path | None, arch: str | None) -> dict[str, Any]:
    clear_lulynx_native_cache()
    native = native_with_entrypoints(ENTRYPOINT)
    if native is None:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "paged_adamw8bit_quantized_scratch_launch_probe",
            "reason": "lulynx_native_entrypoint_missing",
            "entrypoint": ENTRYPOINT,
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "blocked_reasons": ["paged_adamw8bit_native_scratch_entrypoint_missing"],
        }
    try:
        return dict(getattr(native, ENTRYPOINT)(str(Path(workspace_root or REPO_ROOT).resolve()), arch or "compute_89"))
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "paged_adamw8bit_quantized_scratch_launch_probe",
            "reason": f"native_probe_failed:{type(exc).__name__}",
            "error": str(exc),
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
            "blocked_reasons": [f"paged_adamw8bit_native_scratch_probe_failed:{type(exc).__name__}"],
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
    update: Mapping[str, Any],
    probe: Mapping[str, Any],
    parity_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8e_quantized_update_contract_ready",
            bool(update.get("quantized_update_contract_ready", False)),
            "paged_adamw8bit_quantized_update_contract_missing",
        ),
        _validation(
            "native_scratch_kernel_parity",
            parity_ready,
            "paged_adamw8bit_native_scratch_kernel_parity_missing",
        ),
        _validation(
            "scratch_probe_not_training_dispatch",
            not bool(probe.get("training_dispatch", True))
            and not bool(probe.get("training_tensor_binding", True))
            and not bool(probe.get("training_path_enabled", True)),
            "paged_adamw8bit_native_scratch_probe_touched_training_path",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "ENTRYPOINT",
    "KERNEL_NAME",
    "build_paged_adamw8bit_native_scratch_kernel_scorecard",
]
