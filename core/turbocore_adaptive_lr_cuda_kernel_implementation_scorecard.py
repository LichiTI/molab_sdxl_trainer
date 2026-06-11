"""Default-off CUDA scratch-kernel implementation evidence for adaptive-LR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.services.native_module_loader import clear_lulynx_native_cache, native_with_entrypoints
from core.turbocore_adaptive_lr_cuda_kernel_contract_plan_scorecard import (
    build_adaptive_lr_cuda_kernel_contract_plan_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES


ENTRYPOINT = "probe_adaptive_lr_cuda_scratch_launch_py"
KERNEL_KIND_BY_FAMILY = {
    "adaptive_lr_prodigy": "prodigy",
    "adaptive_lr_dadapt": "dadapt",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_adaptive_lr_cuda_kernel_implementation_scorecard(
    *,
    contract_plan_report: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Run synthetic native CUDA probes while keeping dispatch disabled."""

    contract = _as_dict(contract_plan_report or build_adaptive_lr_cuda_kernel_contract_plan_scorecard())
    root = str(Path(workspace_root or REPO_ROOT).resolve())
    cuda_arch = str(arch or "compute_89")
    native = _load_native()
    family_probes = {
        family: _run_probe(kind, native=native, workspace_root=root, arch=cuda_arch)
        for family, kind in KERNEL_KIND_BY_FAMILY.items()
    }
    rows = [_row(case.optimizer.value, contract, family_probes) for case in TARGET_CASES]
    failed = [row for row in rows if row["cuda_kernel_implementation_ready"] is not True]
    blockers = _dedupe(reason for row in failed for reason in _strings(row.get("blocked_reasons")))
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_cuda_kernel_implementation_scorecard_v0",
        "gate": "adaptive_lr_cuda_kernel_implementation",
        "ok": ready,
        "promotion_ready": False,
        "cuda_kernel_implementation_ready": ready,
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
        "family_probes": family_probes,
        "rows": rows,
        "contract_plan_summary": _as_dict(contract.get("summary")),
        "summary": {
            "target_count": len(rows),
            "cuda_kernel_implementation_ready_count": sum(
                1 for row in rows if row["cuda_kernel_implementation_ready"] is True
            ),
            "state_machine_abi_implementation_ready_count": sum(
                1 for row in rows if row["state_machine_abi_implementation_ready"] is True
            ),
            "native_kernel_preconditions_implementation_ready_count": sum(
                1 for row in rows if row["native_kernel_preconditions_implementation_ready"] is True
            ),
            "kernel_executed_count": sum(1 for row in rows if row["kernel_executed"] is True),
            "contract_plan_ready_count": sum(1 for row in rows if row["cuda_kernel_contract_plan_ready"] is True),
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
                "adaptive_lr_runtime_tensor_binding_missing",
                "adaptive_lr_training_loop_canary_missing",
                "adaptive_lr_product_rollout_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add adaptive-LR runtime tensor binding and training-loop canary with dispatch still default-off"
            if ready
            else "build or fix lulynx_native adaptive-LR CUDA scratch probe parity"
        ),
        "notes": [
            "This scorecard launches native CUDA kernels on synthetic buffers only.",
            "It validates Prodigy/DAdapt family parity against the Rust probe reference.",
            "It does not consume training tensors and does not enable runtime dispatch.",
        ],
    }


def _load_native() -> Any | None:
    clear_lulynx_native_cache()
    return native_with_entrypoints(ENTRYPOINT)


def _row(
    optimizer_type: str,
    contract_report: Mapping[str, Any],
    family_probes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _contract_rows(contract_report).get(optimizer_type, {})
    family = str(contract.get("family") or _family(optimizer_type))
    probe = _as_dict(family_probes.get(family))
    contract_ready = contract.get("cuda_kernel_contract_plan_ready") is True
    probe_ready = _probe_ready(probe)
    ready = contract_ready and probe_ready
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": family,
        "optimizer_kind": KERNEL_KIND_BY_FAMILY[family],
        "state_machine_status": "cuda_kernel_implementation_ready" if ready else "cuda_kernel_implementation_blocked",
        "cuda_kernel_contract_plan_ready": contract_ready,
        "cuda_kernel_implementation_ready": ready,
        "state_machine_abi_implementation_ready": ready,
        "native_kernel_preconditions_implementation_ready": ready,
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
        "reduce_kernel_executed": bool(probe.get("reduce_kernel_executed", False)),
        "apply_kernel_executed": bool(probe.get("apply_kernel_executed", False)),
        "parameters_mutated": bool(probe.get("parameters_mutated", False)),
        "max_abs_diff": probe.get("max_abs_diff"),
        "tolerance": probe.get("tolerance"),
        "probe": probe,
        "next_gate": "adaptive_lr_runtime_tensor_binding_canary",
        "blocked_reasons": [] if ready else _row_blockers(optimizer_type, contract_ready, probe),
    }


def _run_probe(
    optimizer_kind: str,
    *,
    native: Any | None,
    workspace_root: str,
    arch: str,
) -> dict[str, Any]:
    if native is None:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "adaptive_lr_cuda_scratch_launch_probe",
            "reason": "lulynx_native_entrypoint_missing",
            "entrypoint": ENTRYPOINT,
            "optimizer_kind": optimizer_kind,
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
        }
    try:
        return dict(getattr(native, ENTRYPOINT)(optimizer_kind, workspace_root, arch))
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe": "adaptive_lr_cuda_scratch_launch_probe",
            "reason": f"native_probe_failed:{type(exc).__name__}",
            "error": str(exc),
            "optimizer_kind": optimizer_kind,
            "kernel_executed": False,
            "parameters_mutated": False,
            "training_path_enabled": False,
            "native_kernel_present": False,
        }


def _probe_ready(probe: Mapping[str, Any]) -> bool:
    return (
        bool(probe.get("ok", False))
        and bool(probe.get("kernel_executed", False))
        and bool(probe.get("reduce_kernel_executed", False))
        and bool(probe.get("apply_kernel_executed", False))
        and bool(probe.get("parity_ok", False))
        and bool(probe.get("parameters_mutated", False))
        and not bool(probe.get("training_path_enabled", True))
        and not bool(probe.get("training_dispatch", True))
        and not bool(probe.get("training_tensor_binding", True))
    )


def _row_blockers(optimizer_type: str, contract_ready: bool, probe: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not contract_ready:
        blockers.append(f"{optimizer_type}_adaptive_lr_cuda_contract_plan_missing")
    if not _probe_ready(probe):
        blockers.append(f"{optimizer_type}_adaptive_lr_cuda_kernel_probe_missing")
    return blockers


def _contract_rows(contract_report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type") or ""): row
        for row in contract_report.get("rows", [])
        if isinstance(row, Mapping)
    }


def _family(optimizer_type: str) -> str:
    if optimizer_type in {"AutoProdigy", "prodigy", "prodigyplus.ProdigyPlusScheduleFree"}:
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ENTRYPOINT", "build_adaptive_lr_cuda_kernel_implementation_scorecard"]
