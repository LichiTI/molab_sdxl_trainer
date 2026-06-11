"""TurboCore evidence and validation-gate helpers.

These helpers keep TurboCore rollout decisions tied to observable project
state.  They deliberately do not enable TurboCore features; they only report
what evidence exists and what is still blocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _listish(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    normalized = str(value).strip().lower()
    return [normalized] if normalized else []


def _file_exists(path: Any) -> bool:
    raw = str(path or "").strip()
    return bool(raw) and Path(raw).is_file()


def _candidate_registry_snapshot() -> Dict[str, Any]:
    try:
        from core.turbocore_candidates import list_turbocore_candidates

        candidates = list_turbocore_candidates()
        return {
            "available": True,
            "features": candidates,
            "native_available": any(
                bool(item.get("native")) and bool(item.get("available"))
                for rows in candidates.values()
                for item in rows
            ),
            "reserved_native_candidates": [
                item.get("name")
                for rows in candidates.values()
                for item in rows
                if bool(item.get("native")) and not bool(item.get("available"))
            ],
        }
    except Exception as exc:  # pragma: no cover - import layout dependent
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_turbocore_evidence(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return read-only evidence that can inform future TurboCore gates."""

    model_type = str(config.get("model_type", config.get("model_arch", "unknown")) or "unknown").strip().lower()
    training_type = str(config.get("training_type", "lora") or "lora").strip().lower()
    pcie_format = str(config.get("pcie_transfer_format", "off") or "off").strip().lower()
    pcie_benchmark_path = (
        config.get("pcie_transfer_benchmark_path")
        or config.get("pcie_transfer_format_benchmark_path")
        or config.get("transfer_format_benchmark_path")
    )
    cache_mode = str(config.get("pcie_delta_cache_mode", "observe") or "observe").strip().lower()

    tensorcore_status: Dict[str, Any] = {
        "status": "unknown",
        "research_only": True,
        "reason": "tensorcore_kernel_roadmap_unavailable",
    }
    try:
        from core.lulynx_trainer.tensorcore_transfer_kernel import tensorcore_kernel_roadmap

        roadmap = tensorcore_kernel_roadmap()
        tensorcore_status = {
            "status": str(roadmap.get("status", "design_skeleton") or "design_skeleton"),
            "research_only": True,
            "selected_spec": (roadmap.get("selected_spec") or {}).get("name", ""),
            "guardrails": list(roadmap.get("guardrails") or []),
            "reason": "research_only_not_training_path",
        }
    except Exception as exc:  # pragma: no cover - depends on import layout
        tensorcore_status["error"] = f"{type(exc).__name__}: {exc}"

    anima_staged_enabled = (
        model_type == "anima"
        and _boolish(config.get("enable_mixed_resolution_training", False))
    )
    requested_features = _listish(config.get("turbocore_features"))

    workspace_pipeline_prototype: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "workspace_pipeline_prototype_unavailable",
        "training_path_enabled": False,
    }
    workspace_pipeline_native_schema: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "workspace_pipeline_native_schema_unavailable",
        "training_path_enabled": False,
    }
    try:
        from core.turbocore_workspace_pipeline import (
            build_workspace_pipeline_native_capability_stub,
            build_workspace_pipeline_prototype_report,
        )

        workspace_pipeline_prototype = build_workspace_pipeline_prototype_report(
            prefetch_depth=max(int(config.get("turbocore_prefetch_depth", 2) or 2), 1),
            workspace_mb=max(int(config.get("turbocore_workspace_mb", 0) or 0), 0),
        )
        workspace_pipeline_native_schema = build_workspace_pipeline_native_capability_stub()
    except Exception as exc:  # pragma: no cover - defensive reporting only
        workspace_pipeline_prototype["error"] = f"{type(exc).__name__}: {exc}"
        workspace_pipeline_native_schema["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "schema_version": 1,
        "route": {
            "model_type": model_type,
            "training_type": training_type,
            "is_lora_route": training_type == "lora",
            "cache_first_candidate": model_type in {"anima", "newbie"},
            "anima_staged_resolution_enabled": anima_staged_enabled,
        },
        "pcie_transfer": {
            "format_requested": pcie_format,
            "experimental_low_precision_requested": pcie_format in {"fp8_e4m3", "int8_rowwise", "uint4_rowwise", "nf4"},
            "benchmark_path": str(pcie_benchmark_path or ""),
            "benchmark_present": _file_exists(pcie_benchmark_path),
            "recommendation_only": True,
        },
        "offload_sensing": {
            "smart_sensing_enabled": _boolish(config.get("vram_smart_sensing_enabled", True), default=True),
            "streaming_allowed": _boolish(config.get("vram_smart_sensing_streaming_enabled", True), default=True),
            "sparse_swap_allowed": _boolish(config.get("vram_smart_sensing_sparse_swap_enabled", True), default=True),
            "delta_cache_auto_allowed": _boolish(config.get("vram_smart_sensing_delta_cache_enabled", False)),
            "enhanced_protection_mode": _boolish(config.get("enhanced_protection_mode", False)),
        },
        "cache_v0": {
            "delta_cache_enabled": _boolish(config.get("pcie_delta_cache_enabled", False)),
            "mode": cache_mode,
            "recommendation_only": cache_mode != "cache_v0",
            "auto_enable_default": False,
        },
        "tensorcore_transfer": tensorcore_status,
        "candidate_registry": _candidate_registry_snapshot(),
        "native_data_pipeline": {
            "requested": "data_pipeline" in requested_features,
            "status": "prototype",
            "reason": "python_abi_prototype_only_native_bridge_not_implemented",
            "developer_only": True,
            "prototype": workspace_pipeline_prototype,
            "native_capability_schema": workspace_pipeline_native_schema["features"]["data_pipeline"]
            if "features" in workspace_pipeline_native_schema
            else workspace_pipeline_native_schema,
        },
        "workspace_pool": {
            "requested": "workspace_pool" in requested_features,
            "requested_mb": max(int(config.get("turbocore_workspace_mb", 0) or 0), 0),
            "status": "prototype",
            "reason": "python_workspace_pool_model_only_native_pool_not_implemented",
            "developer_only": True,
            "prototype": workspace_pipeline_prototype,
            "native_capability_schema": workspace_pipeline_native_schema["features"]["workspace_pool"]
            if "features" in workspace_pipeline_native_schema
            else workspace_pipeline_native_schema,
        },
        "turbocore_readiness_notes": [
            "native training kernels are still required before TurboCore can activate",
            "LoRA fused delta benchmark evidence should precede Rust/CUDA kernel work",
            "native optimizer benchmark evidence should precede Rust/CUDA AdamW/update work",
            "parity anchors must pass before any native candidate can be considered active",
            "TensorCore-friendly transfer remains research-only until repeated benchmark wins are proven",
            "Triton/Rust-CUDA candidates may be registered for discovery while remaining unavailable to training",
        ],
    }


def build_turbocore_validation_status(
    config: Dict[str, Any],
    *,
    capability_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Summarize TurboCore validation gates from current evidence."""

    evidence = build_turbocore_evidence(config)
    report = capability_report or {}
    native_bridge = ((report.get("native_bridge") or {}).get("training_bridge") or {}) if isinstance(report, dict) else {}
    bridge_available = bool(native_bridge.get("available", False))
    active_features = list(report.get("active_features") or []) if isinstance(report, dict) else []

    gates = [
        {
            "gate": "build_and_initialization",
            "status": "partial" if bridge_available else "blocked",
            "reason": "native bridge query is available" if bridge_available else "native TurboCore training bridge unavailable or query-only",
        },
        {
            "gate": "correctness",
            "status": "blocked",
            "reason": "no active native TurboCore feature has parity coverage yet",
        },
        {
            "gate": "stability",
            "status": "blocked",
            "reason": "no repeated-step native execution or teardown evidence yet",
        },
        {
            "gate": "numeric_safety",
            "status": "blocked",
            "reason": "loss stability and clipping/finite-check behavior remain untested for native paths",
        },
        {
            "gate": "performance_justification",
            "status": "partial" if evidence["pcie_transfer"]["benchmark_present"] else "blocked",
            "reason": "PCIe benchmark evidence exists; LoRA fused/native optimizer benchmark still required"
            if evidence["pcie_transfer"]["benchmark_present"]
            else "representative LoRA/native optimizer/TensorCore benchmarks are still required",
        },
        {
            "gate": "observability",
            "status": "partial",
            "reason": "resolver, capability report, and evidence report are available; hybrid feature telemetry waits for native features",
        },
        {
            "gate": "product_readiness",
            "status": "blocked",
            "reason": "TurboCore remains CLI/developer-only until native features pass gates",
        },
    ]

    ready_for_ui = all(gate["status"] == "pass" for gate in gates)
    return {
        "schema_version": 1,
        "summary": {
            "ready_for_ui": ready_for_ui,
            "active_native_features": active_features,
            "recommended_next_step": "run LoRA fused/native optimizer benchmarks and parity anchors before native kernel work",
        },
        "gates": gates,
        "evidence": evidence,
    }


__all__ = [
    "build_turbocore_evidence",
    "build_turbocore_validation_status",
]
