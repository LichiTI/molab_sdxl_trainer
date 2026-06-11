"""Static CUDA kernel/probe inventory for TurboCore optimizer work.

This report intentionally does not execute CUDA and does not enable dispatch.
It ties selected plugin optimizer families to native source/probe assets so the
roadmap can distinguish kernel assets from product training support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.turbocore_plugin_optimizer_selector_scorecard import (
    build_plugin_optimizer_selector_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_SRC = REPO_ROOT / "backend" / "native" / "src"
CUDA_SRC = NATIVE_SRC / "cuda"
ARTIFACT = (
    REPO_ROOT
    / "temp"
    / "turbocore_optimizer"
    / "turbocore_optimizer_native_kernel_inventory_scorecard.json"
)

FAMILY_SHARED_KERNELS = {
    "adaptive_lr_state_machine": ("adaptive_lr_state_machine_cuda_v0",),
    "closure_or_second_order": ("optimizer_family_precondition_contract_cuda_v0",),
    "custom_formula": ("optimizer_family_precondition_contract_cuda_v0",),
    "factored_memory_layout": ("optimizer_family_precondition_contract_cuda_v0",),
    "fused_backward": ("optimizer_family_precondition_contract_cuda_v0",),
    "model_or_shape_aware": ("optimizer_family_precondition_contract_cuda_v0",),
    "state_adapter_special": ("optimizer_family_precondition_contract_cuda_v0",),
}

FAMILY_SHARED_PROBES = {
    "adam_like_formula": ("cuda_adamw_runtime", "turbocore_adamw_kernel_contract"),
    "adaptive_lr_state_machine": ("cuda_adaptive_lr_live_launch_probe", "cuda_adaptive_lr_scratch_probe"),
    "closure_or_second_order": ("turbocore_optimizer_family_kernel_contract",),
    "custom_formula": ("turbocore_optimizer_family_kernel_contract",),
    "factored_memory_layout": ("turbocore_optimizer_family_kernel_contract",),
    "fused_backward": ("turbocore_optimizer_family_kernel_contract",),
    "model_or_shape_aware": ("turbocore_optimizer_family_kernel_contract",),
    "simple_formula": ("cuda_simple_optimizer_runtime", "turbocore_simple_optimizer_kernel_contract"),
    "state_adapter_special": ("turbocore_optimizer_family_kernel_contract",),
}

SOURCE_ALIASES = {
    "schedulefreeadamw": ("adamw_schedule_free_flat_fp32_cuda_v0",),
    "schedulefreeradam": ("schedulefree_radam_flat_fp32_cuda_v0",),
    "schedulefreesgd": ("schedulefree_sgd_flat_fp32_cuda_v0",),
    "signsgd": ("sign_momentum_flat_fp32_cuda_v0",),
    "sgdw": ("sign_momentum_flat_fp32_cuda_v0",),
    "tiger": ("sign_momentum_flat_fp32_cuda_v0",),
}

PROBE_ALIASES = {
    "schedulefreeadamw": ("cuda_adamw_schedule_free_live_launch_probe",),
    "schedulefreeradam": ("cuda_schedulefree_radam_live_launch_probe",),
    "schedulefreesgd": ("cuda_schedulefree_sgd_live_launch_probe",),
    "signsgd": ("cuda_simple_optimizer_scratch_probe",),
    "sgdw": ("cuda_simple_optimizer_scratch_probe",),
    "tiger": ("cuda_simple_optimizer_scratch_probe",),
    "lion": ("cuda_simple_optimizer_scratch_probe",),
    "sgd": ("cuda_simple_optimizer_runtime",),
    "sgd_nesterov": ("cuda_sgd_nesterov_scratch_probe",),
}


def build_optimizer_native_kernel_inventory_scorecard(
    *,
    workspace_root: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    root = Path(workspace_root or REPO_ROOT).resolve()
    selector = build_plugin_optimizer_selector_scorecard()
    source_stems = _file_stems(root / "backend" / "native" / "src" / "cuda", "*.cu")
    rust_stems = _file_stems(root / "backend" / "native" / "src", "*.rs")
    rows = [
        _optimizer_row(dict(row), source_stems=source_stems, rust_stems=rust_stems)
        for row in selector.get("rows", [])
        if isinstance(row, Mapping)
    ]
    family_rows = _family_rows(rows)
    source_ready = [row for row in rows if row["kernel_source_present"]]
    probe_ready = [row for row in rows if row["rust_probe_present"]]
    dispatch_closed = all(not row["training_path_enabled"] and not row["native_dispatch_allowed"] for row in rows)
    payload = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_native_kernel_inventory_scorecard_v0",
        "gate": "optimizer_native_kernel_inventory_static",
        "ok": bool(rows) and dispatch_closed,
        "promotion_ready": False,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "workspace_root": str(root),
        "kernel_inventory_ready": bool(rows),
        "static_inventory_only": True,
        "cuda_executed": False,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "default_behavior_changed": False,
        "summary": {
            "plugin_optimizer_count": len(rows),
            "family_count": len(family_rows),
            "kernel_source_present_count": len(source_ready),
            "rust_probe_present_count": len(probe_ready),
            "product_native_ready_count": 0,
            "training_path_enabled_count": 0,
            "native_dispatch_allowed_count": 0,
        },
        "family_rows": family_rows,
        "rows": rows,
        "promotion_blockers": [
            "kernel_inventory_is_static_not_execution_proof",
            "product_dispatch_review_missing",
            "training_path_dispatch_not_enabled",
        ],
        "recommended_next_step": (
            "batch selected optimizer runtime dispatch rehearsals from families with kernel/probe assets"
        ),
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _optimizer_row(
    selector_row: Mapping[str, Any],
    *,
    source_stems: set[str],
    rust_stems: set[str],
) -> dict[str, Any]:
    name = str(selector_row.get("optimizer_name", "")).strip().lower()
    family = str(selector_row.get("native_route_family", "")).strip()
    source_candidates = _source_candidates(name, family)
    probe_candidates = _probe_candidates(name, family)
    matched_sources = sorted(candidate for candidate in source_candidates if candidate in source_stems)
    matched_probes = sorted(candidate for candidate in probe_candidates if candidate in rust_stems)
    return {
        "schema_version": 1,
        "optimizer_name": name,
        "native_route_family": family,
        "kernel_source_present": bool(matched_sources),
        "rust_probe_present": bool(matched_probes),
        "matched_kernel_sources": matched_sources,
        "matched_rust_probes": matched_probes,
        "source_candidates": sorted(source_candidates),
        "probe_candidates": sorted(probe_candidates),
        "static_inventory_only": True,
        "cuda_executed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_dispatch_ready": False,
        "product_native_ready": False,
    }


def _family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row["native_route_family"]) for row in rows})
    output = []
    for family in families:
        family_rows = [row for row in rows if row["native_route_family"] == family]
        source_count = sum(1 for row in family_rows if row["kernel_source_present"])
        probe_count = sum(1 for row in family_rows if row["rust_probe_present"])
        output.append(
            {
                "schema_version": 1,
                "native_route_family": family,
                "optimizer_count": len(family_rows),
                "kernel_source_present_count": source_count,
                "kernel_source_missing_count": len(family_rows) - source_count,
                "rust_probe_present_count": probe_count,
                "rust_probe_missing_count": len(family_rows) - probe_count,
                "product_native_ready_count": 0,
                "training_path_enabled": False,
                "native_dispatch_allowed": False,
            }
        )
    return output


def _source_candidates(name: str, family: str) -> set[str]:
    candidates = {
        f"{name}_flat_fp32_cuda_v0",
        f"{name}_update_cuda_v0",
        f"{name}_state_machine_cuda_v0",
    }
    candidates.update(SOURCE_ALIASES.get(name, ()))
    candidates.update(FAMILY_SHARED_KERNELS.get(family, ()))
    return {item for item in candidates if item}


def _probe_candidates(name: str, family: str) -> set[str]:
    base = name.replace("schedulefree", "schedulefree_")
    candidates = {
        f"cuda_{name}_live_launch_probe",
        f"cuda_{name}_scratch_probe",
        f"cuda_{base}_live_launch_probe",
        f"cuda_{base}_scratch_probe",
    }
    candidates.update(PROBE_ALIASES.get(name, ()))
    candidates.update(FAMILY_SHARED_PROBES.get(family, ()))
    return {item for item in candidates if item}


def _file_stems(path: Path, pattern: str) -> set[str]:
    if not path.exists():
        return set()
    return {item.stem for item in path.glob(pattern) if item.is_file()}


__all__ = ["build_optimizer_native_kernel_inventory_scorecard"]
