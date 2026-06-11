from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

from .family_adapters import (
    build_export_plan,
    get_family_adapter,
    inspect_family_checkpoint,
    path_record,
    resolve_family_components,
)
from .resource_budget import estimate_base_model_tensorrt_resource_budget


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _module_info(name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"available": _module_available(name)}
    if not info["available"]:
        return info
    try:
        module = __import__(name)
        info["version"] = str(getattr(module, "__version__", ""))
    except Exception as exc:
        info.update({"available": False, "error": str(exc)})
    return info


def _torch_info() -> dict[str, Any]:
    info = _module_info("torch")
    if not info.get("available"):
        return info
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        info.update({
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
        })
        if cuda_available:
            memory: dict[str, int] = {}
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                memory = {"gpu_free_bytes": int(free_bytes), "gpu_total_bytes": int(total_bytes)}
            except Exception:
                props = torch.cuda.get_device_properties(0)
                memory = {"gpu_total_bytes": int(getattr(props, "total_memory", 0) or 0)}
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
                **memory,
            })
    except Exception as exc:
        info.update({"available": False, "error": str(exc)})
    return info


def _dependency_report() -> dict[str, Any]:
    return {
        "torch": _torch_info(),
        "onnx": _module_info("onnx"),
        "tensorrt": _module_info("tensorrt"),
        "diffusers": _module_info("diffusers"),
        "transformers": _module_info("transformers"),
        "safetensors": _module_info("safetensors"),
        "onnxruntime": _module_info("onnxruntime"),
    }


def _dependency_notes(deps: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if not deps["torch"].get("available"):
        notes.append("torch_missing")
    if not deps["torch"].get("cuda_available"):
        notes.append("cuda_unavailable")
    for name in ("onnx", "tensorrt", "diffusers", "transformers", "safetensors"):
        if not deps[name].get("available"):
            notes.append(f"{name}_missing")
    return notes


def probe_base_model_tensorrt(
    *,
    model_path: str = "",
    model_family: str = "anima",
    model_root: str = "",
    output_dir: str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    adapter = get_family_adapter(model_family)
    deps = _dependency_report()
    notes = _dependency_notes(deps)
    resolved = resolve_family_components(adapter, model_root)
    if model_path:
        override = path_record(Path(model_path))
        resolved["components"][adapter.export_component] = {
            **resolved["components"].get(adapter.export_component, {}),
            **override,
            "override": True,
            "export_target": True,
        }
        resolved["export_target_path"] = str(Path(model_path))
        if not override["exists"] and adapter.export_component not in resolved["missing_required"]:
            resolved["missing_required"].append(adapter.export_component)
    if resolved["missing_required"]:
        notes.append("family_components_missing")

    plan = build_export_plan(adapter, resolved, output_dir=output_dir)
    checkpoint = inspect_family_checkpoint(adapter, resolved)
    budget = estimate_base_model_tensorrt_resource_budget(
        family=adapter.family,
        static_shape=plan["static_shape"],
        engine_precision=plan["engine_precision"],
        output_path=plan["onnx_path"],
        checkpoint=checkpoint,
        dependencies=deps,
    )
    plan["resource_budget"] = budget
    if _has_production_shape_resource_gate(budget) and "production_shape_resource_gate" not in plan["cautions"]:
        plan["cautions"].append("production_shape_resource_gate")
    ready_for_export_spike = not notes and not plan["blocking"]
    if adapter.family == "anima" and not notes and plan["blocking"] == []:
        ready_for_export_spike = True
    return {
        "schema_version": 2,
        "kind": "base_model_tensorrt_probe",
        "model_family": adapter.family,
        "family": {
            "label": adapter.label,
            "architecture": adapter.architecture,
            "export_component": adapter.export_component,
            "notes": list(adapter.notes),
        },
        "model": path_record(model_path) if model_path else path_record(resolved["export_target_path"]),
        "resolved": resolved,
        "checkpoint": checkpoint,
        "export_plan": plan,
        "dependencies": deps,
        "recommendation": {
            "ready_for_unet_export_spike": ready_for_export_spike,
            "ready_for_transformer_export_spike": ready_for_export_spike,
            "next_step": _next_step(adapter.family, ready_for_export_spike),
            "notes": notes,
        },
        "scope": {
            "official_feature": False,
            "launcher_role": "dependency/status orchestration only",
            "experiment_boundary": "backend/core/base_model_tensorrt + backend/core/tools/lulynx_lab",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _next_step(family: str, ready_for_export_spike: bool) -> str:
    if not ready_for_export_spike:
        return "resolve_family_or_dependency_blockers"
    if family == "newbie":
        return "newbie_fp32_static_transformer_runtime_gate_and_low_precision_followup"
    if family == "anima":
        return "anima_transformer_onnx_export_spike"
    return "transformer_onnx_export_spike"


def _has_production_shape_resource_gate(budget: dict[str, Any]) -> bool:
    gate = budget.get("gate", {}) if isinstance(budget, dict) else {}
    if not isinstance(gate, dict):
        return False
    blocking = set(gate.get("blocking") or [])
    warnings = set(gate.get("warnings") or [])
    shape_gate_items = {
        "output_disk_below_minimum",
        "output_disk_below_recommended_headroom",
        "gpu_total_below_runtime_estimate",
        "gpu_free_below_runtime_estimate",
        "estimate_requires_full_transformer_calibration",
    }
    return bool((blocking | warnings) & shape_gate_items)
