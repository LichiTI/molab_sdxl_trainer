"""Report-only native scratch-kernel parity for quantized simple variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_simple_optimizer_quantized_variant_parity_scorecard import (
    build_simple_optimizer_quantized_variant_parity_scorecard,
)


ENTRYPOINT = "probe_simple_quantized_optimizer_scratch_launch_py"
KERNEL_BY_OPTIMIZER = {
    OptimizerType.LION_8BIT: "lion8bit_dequant_update_requant_cuda_v0",
    OptimizerType.PAGED_LION_8BIT: "lion8bit_dequant_update_requant_cuda_v0",
    OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov8bit_dequant_update_requant_cuda_v0",
}
NATIVE_KIND_BY_OPTIMIZER = {
    OptimizerType.LION_8BIT: "lion8bit",
    OptimizerType.PAGED_LION_8BIT: "paged_lion8bit",
    OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov8bit",
}
TARGET_OPTIMIZERS = tuple(KERNEL_BY_OPTIMIZER)
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_simple_optimizer_quantized_native_scratch_scorecard(
    *,
    quantized_parity_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run synthetic native scratch probes while keeping dispatch disabled."""

    parity = dict(quantized_parity_report or build_simple_optimizer_quantized_variant_parity_scorecard())
    root = str(Path(workspace_root or REPO_ROOT).resolve())
    cuda_arch = str(arch or "compute_89")
    native = _load_native()
    rows = [
        _row(optimizer, parity, native=native, workspace_root=root, arch=cuda_arch)
        for optimizer in TARGET_OPTIMIZERS
    ]
    failed = [row for row in rows if row["native_scratch_kernel_parity_ready"] is not True]
    blockers = _dedupe(reason for row in failed for reason in _strings(row.get("blocked_reasons")))
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_quantized_native_scratch_scorecard_v0",
        "gate": "simple_formula_quantized_native_scratch_kernel_parity",
        "ok": ready,
        "promotion_ready": False,
        "native_scratch_kernel_parity_ready": ready,
        "native_kernel_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "entrypoint": ENTRYPOINT,
        "workspace_root": root,
        "arch": cuda_arch,
        "target_optimizer_types": [optimizer.value for optimizer in TARGET_OPTIMIZERS],
        "rows": rows,
        "quantized_parity_summary": dict(parity.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(TARGET_OPTIMIZERS),
            "native_scratch_kernel_ready_count": sum(
                1 for row in rows if row["native_scratch_kernel_parity_ready"] is True
            ),
            "kernel_executed_count": sum(1 for row in rows if row["kernel_executed"] is True),
            "parity_ready_count": sum(1 for row in rows if row["quantized_formula_parity_ready"] is True),
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "simple_quantized_variant_runtime_canary_missing",
                "simple_quantized_variant_training_loop_canary_missing",
                "simple_quantized_variant_product_rollout_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add report-only runtime canary manifests for quantized simple variant kernels"
            if ready
            else "build or fix lulynx_native simple quantized scratch probe parity"
        ),
        "notes": [
            "This scorecard launches native CUDA scratch kernels on synthetic buffers only.",
            "It validates dequantize/update/requantize parity against the Rust probe reference.",
            "It does not consume training tensors and does not enable runtime dispatch.",
        ],
    }


def _load_native() -> Any | None:
    clear_lulynx_native_cache()
    return native_with_entrypoints(ENTRYPOINT)


def _row(
    optimizer: OptimizerType,
    parity_report: Mapping[str, Any],
    *,
    native: Any | None,
    workspace_root: str,
    arch: str,
) -> dict[str, Any]:
    parity_ready = _parity_ready_for(optimizer, parity_report)
    probe = _run_probe(optimizer, native=native, workspace_root=workspace_root, arch=arch)
    native_ready = parity_ready and _probe_ready(probe)
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "optimizer_kind": NATIVE_KIND_BY_OPTIMIZER[optimizer],
        "optimizer_family": "simple_formula_quantized",
        "variant_status": "quantized_native_scratch_kernel_ready" if native_ready else "quantized_native_scratch_kernel_blocked",
        "quantized_formula_parity_ready": parity_ready,
        "native_scratch_kernel_parity_ready": native_ready,
        "native_kernel_ready": native_ready,
        "runtime_canary_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "kernel_name": KERNEL_BY_OPTIMIZER[optimizer],
        "entrypoint": ENTRYPOINT,
        "kernel_executed": bool(probe.get("kernel_executed", False)),
        "parameters_mutated": bool(probe.get("parameters_mutated", False)),
        "state_uint8_mismatch_count": int(probe.get("state_uint8_mismatch_count", 0) or 0),
        "max_float_diff": probe.get("max_float_diff"),
        "tolerance": probe.get("tolerance"),
        "probe": probe,
        "blocked_reasons": [] if native_ready else _row_blockers(parity_ready, probe, optimizer),
    }


def _run_probe(
    optimizer: OptimizerType,
    *,
    native: Any | None,
    workspace_root: str,
    arch: str,
) -> dict[str, Any]:
    if native is None:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "simple_quantized_optimizer_scratch_launch_probe",
            "reason": "lulynx_native_entrypoint_missing",
            "entrypoint": ENTRYPOINT,
            "optimizer_kind": NATIVE_KIND_BY_OPTIMIZER[optimizer],
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
        }
    try:
        return dict(getattr(native, ENTRYPOINT)(NATIVE_KIND_BY_OPTIMIZER[optimizer], workspace_root, arch))
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "simple_quantized_optimizer_scratch_launch_probe",
            "reason": f"native_probe_failed:{type(exc).__name__}",
            "error": str(exc),
            "optimizer_kind": NATIVE_KIND_BY_OPTIMIZER[optimizer],
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
        }


def _probe_ready(probe: Mapping[str, Any]) -> bool:
    return (
        bool(probe.get("ok", False))
        and bool(probe.get("kernel_executed", False))
        and bool(probe.get("parity_ok", False))
        and bool(probe.get("parameters_mutated", False))
        and not bool(probe.get("training_path_enabled", True))
        and not bool(probe.get("training_dispatch", True))
        and not bool(probe.get("training_tensor_binding", True))
    )


def _parity_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("formula_parity_ready") is True)


def _row_blockers(parity_ready: bool, probe: Mapping[str, Any], optimizer: OptimizerType) -> list[str]:
    blockers: list[str] = []
    if not parity_ready:
        blockers.append(f"{optimizer.value}_quantized_formula_parity_missing")
    if not _probe_ready(probe):
        blockers.append(f"{optimizer.value}_native_scratch_kernel_parity_missing")
    return blockers


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "ENTRYPOINT",
    "KERNEL_BY_OPTIMIZER",
    "build_simple_optimizer_quantized_native_scratch_scorecard",
]
