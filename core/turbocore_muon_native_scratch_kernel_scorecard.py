"""Default-off CUDA scratch-kernel parity scorecard for built-in Muon."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_muon_model_shape_aware_family_batch_scorecard import (
    build_muon_model_shape_aware_family_batch_scorecard,
)


ENTRYPOINT = "probe_muon_cuda_scratch_launch_py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_muon_native_scratch_kernel_scorecard(
    *,
    muon_model_shape_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Run the synthetic Muon CUDA probe without enabling dispatch."""

    model_shape = _as_dict(muon_model_shape_report or build_muon_model_shape_aware_family_batch_scorecard())
    root = str(Path(workspace_root or REPO_ROOT).resolve())
    cuda_arch = str(arch or os.environ.get("LULYNX_NATIVE_CUDA_ARCH") or "compute_89")
    native = _load_native()
    probe = _run_probe(native=native, workspace_root=root, arch=cuda_arch)
    case = _case(model_shape, probe)
    ready = case["native_scratch_kernel_ready"] is True
    blockers = _case_blockers(case)
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_native_scratch_kernel_scorecard_v0",
        "gate": "muon_model_shape_aware_native_scratch_kernel",
        "ok": ready,
        "promotion_ready": False,
        "native_scratch_kernel_ready": ready,
        "native_kernel_ready": ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_ready": False,
        "product_native_ready_count": 0,
        "entrypoint": ENTRYPOINT,
        "workspace_root": root,
        "arch": cuda_arch,
        "case": case,
        "model_shape_summary": _as_dict(model_shape.get("summary")),
        "summary": {
            "optimizer_count": 1,
            "native_scratch_kernel_ready_count": 1 if ready else 0,
            "native_kernel_ready_count": 1 if ready else 0,
            "kernel_executed_count": 1 if case["kernel_executed"] is True else 0,
            "parameters_mutated_count": 1 if case["parameters_mutated"] is True else 0,
            "model_shape_precondition_ready_count": 1
            if case["model_shape_precondition_ready"] is True
            else 0,
            "runtime_canary_ready_count": 0,
            "runtime_canary_hit_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "muon_runtime_tensor_binding_missing",
                "muon_training_loop_canary_missing",
                "muon_owner_release_approval_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Muon runtime tensor binding and training-loop canary with dispatch still default-off"
            if ready
            else "build or fix lulynx_native Muon CUDA scratch probe parity"
        ),
        "notes": [
            "This scorecard launches a Muon Newton-Schulz CUDA kernel on synthetic buffers only.",
            "It validates built-in Muon 2D momentum orthogonalization parity against the Rust reference.",
            "It does not consume training tensors and does not enable runtime/native/product dispatch.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _load_native() -> Any | None:
    clear_lulynx_native_cache()
    return native_with_entrypoints(ENTRYPOINT)


def _run_probe(*, native: Any | None, workspace_root: str, arch: str) -> dict[str, Any]:
    if native is None:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "muon_cuda_scratch_launch_probe",
            "reason": "lulynx_native_entrypoint_missing",
            "entrypoint": ENTRYPOINT,
            "optimizer_kind": "muon",
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
        }
    try:
        return dict(getattr(native, ENTRYPOINT)(workspace_root, arch))
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "muon_cuda_scratch_launch_probe",
            "reason": f"native_probe_failed:{type(exc).__name__}",
            "error": str(exc),
            "optimizer_kind": "muon",
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
        }


def _case(model_shape: Mapping[str, Any], probe: Mapping[str, Any]) -> dict[str, Any]:
    model_shape_ready = model_shape.get("muon_model_shape_aware_family_batch_ready") is True
    probe_ready = _probe_ready(probe)
    ready = model_shape_ready and probe_ready
    return {
        "schema_version": 1,
        "optimizer_type": "Muon",
        "native_route_family": "model_or_shape_aware",
        "native_kernel_name": "muon_flat_fp32_cuda_v0",
        "model_shape_precondition_ready": model_shape_ready,
        "native_scratch_kernel_ready": ready,
        "native_kernel_ready": ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "entrypoint": ENTRYPOINT,
        "kernel_executed": bool(probe.get("kernel_executed", False)),
        "parameters_mutated": bool(probe.get("parameters_mutated", False)),
        "scratch_buffers_only": bool(probe.get("scratch_buffers_only", False)),
        "training_tensor_binding": bool(probe.get("training_tensor_binding", True)),
        "training_dispatch": bool(probe.get("training_dispatch", True)),
        "parity_ok": bool(probe.get("parity_ok", False)),
        "max_abs_diff": probe.get("max_abs_diff"),
        "tolerance": probe.get("tolerance"),
        "probe": dict(probe),
        "next_gate": "muon_runtime_tensor_binding_canary",
        "blocked_reasons": [] if ready else _row_blockers(model_shape_ready, probe),
    }


def _probe_ready(probe: Mapping[str, Any]) -> bool:
    return (
        bool(probe.get("ok", False))
        and bool(probe.get("kernel_executed", False))
        and bool(probe.get("parity_ok", False))
        and bool(probe.get("parameters_mutated", False))
        and bool(probe.get("scratch_buffers_only", False))
        and not bool(probe.get("training_path_enabled", True))
        and not bool(probe.get("training_dispatch", True))
        and not bool(probe.get("training_tensor_binding", True))
    )


def _row_blockers(model_shape_ready: bool, probe: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not model_shape_ready:
        blockers.append("muon_model_shape_preconditions_missing")
    if not _probe_ready(probe):
        reason = str(probe.get("reason") or "muon_native_scratch_kernel_probe_missing")
        blockers.append(reason)
        blockers.append("muon_native_scratch_kernel_parity_missing")
    return _dedupe(blockers)


def _case_blockers(case: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in case.get("blocked_reasons", []) or [] if str(item)]


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_native_scratch_kernel_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out

__all__ = ["ENTRYPOINT", "build_muon_native_scratch_kernel_scorecard"]
