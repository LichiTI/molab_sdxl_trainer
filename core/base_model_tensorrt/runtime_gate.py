from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.contracts.base_model_tensorrt_runtime import (
    BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID,
    BASE_MODEL_TENSORRT_RUNTIME_RESULT_SCHEMA_ID,
    BaseModelTensorRtRuntimeResult,
)

from .newbie_export import NewbieStaticShape, default_newbie_checkpoint, parse_layer_indices
from .newbie_engine import default_newbie_engine_path
from .resource_budget import estimate_base_model_tensorrt_resource_budget


STATIC_TENSORRT_RUNTIME_ABI = "base_model_static_tensorrt_runtime_v1"


def build_static_transformer_runtime_gate(
    *,
    family: str = "newbie",
    engine_path: str | Path = "",
    output_dir: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | Mapping[str, Any] | None = None,
    precision: str = "fp32",
    opset: int = 18,
    dependencies: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    family_key = str(family or "newbie").strip().lower()
    normalized_precision = _normalize_precision(precision)
    normalized_shape = _normalize_newbie_shape(shape)
    layers = parse_layer_indices(layer_indices)
    engine = _resolve_engine_path(
        family=family_key,
        engine_path=engine_path,
        output_dir=output_dir,
        layer_indices=layers,
        shape=normalized_shape,
        opset=opset,
        precision=normalized_precision,
    )
    deps = _dependency_snapshot(dependencies)
    checkpoint_record = dict(checkpoint or _checkpoint_record(family_key))
    budget = estimate_base_model_tensorrt_resource_budget(
        family=family_key,
        static_shape=normalized_shape.to_dict(),
        engine_precision=normalized_precision,
        output_path=engine,
        checkpoint=checkpoint_record,
        dependencies=deps,
    )
    blockers, warnings = _runtime_issues(
        family=family_key,
        precision=normalized_precision,
        shape=normalized_shape,
        engine=engine,
        dependencies=deps,
        budget=budget,
    )
    loadable = not blockers
    return {
        "schema_version": 1,
        "kind": "base_model_static_tensorrt_runtime_gate",
        "success": True,
        "abi": STATIC_TENSORRT_RUNTIME_ABI,
        "request_schema_id": BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID,
        "result_schema_id": BASE_MODEL_TENSORRT_RUNTIME_RESULT_SCHEMA_ID,
        "family": family_key,
        "component": "transformer",
        "engine_path": str(engine),
        "engine": _engine_record(engine, budget),
        "precision": normalized_precision,
        "opset": int(opset),
        "layer_indices": list(layers),
        "shape": normalized_shape.to_dict(),
        "input_signature": normalized_shape.input_signature(),
        "dependencies": deps,
        "resource_budget": budget,
        "runtime_loadable": loadable,
        "lab_runtime_allowed": loadable,
        "generation_path_enabled": False,
        "training_path_enabled": False,
        "launcher_role": "dependency/status orchestration only",
        "loadability": "lab_runtime_loadable" if loadable else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "actions_required": _actions_required(blockers, warnings),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def build_static_transformer_runtime_gate_result(
    **kwargs: Any,
) -> BaseModelTensorRtRuntimeResult:
    return BaseModelTensorRtRuntimeResult.from_gate_report(build_static_transformer_runtime_gate(**kwargs))


def _resolve_engine_path(
    *,
    family: str,
    engine_path: str | Path,
    output_dir: str | Path,
    layer_indices: Sequence[int],
    shape: NewbieStaticShape,
    opset: int,
    precision: str,
) -> Path:
    if str(engine_path or "").strip():
        return Path(engine_path)
    if family == "newbie":
        return default_newbie_engine_path(
            output_dir=output_dir,
            shape=shape,
            layer_indices=layer_indices,
            opset=opset,
            precision=precision,
        )
    root = Path(output_dir) if str(output_dir or "").strip() else Path(".")
    return root / f"{family}_transformer_static_{precision}.engine"


def _runtime_issues(
    *,
    family: str,
    precision: str,
    shape: NewbieStaticShape,
    engine: Path,
    dependencies: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if family != "newbie":
        blockers.append("static_tensorrt_runtime_gate_only_validated_for_newbie")
    if precision != "fp32":
        blockers.append("newbie_low_precision_static_transformer_not_release_safe")
    if not _is_validated_newbie_production_shape(shape):
        blockers.append("newbie_static_shape_not_validated_for_runtime_gate")
    if not engine.is_file():
        blockers.append("engine_file_missing")
    elif engine.stat().st_size <= 0:
        blockers.append("engine_file_empty")
    elif _engine_size_below_expected(engine, budget):
        blockers.append("engine_file_smaller_than_measured_anchor")

    if not _dep_available(dependencies, "tensorrt"):
        blockers.append("tensorrt_missing")
    torch_info = dependencies.get("torch", {}) if isinstance(dependencies, Mapping) else {}
    if not isinstance(torch_info, Mapping) or not torch_info.get("available"):
        blockers.append("torch_missing")
    elif not torch_info.get("cuda_available"):
        blockers.append("torch_cuda_unavailable")

    gate = budget.get("gate", {}) if isinstance(budget, Mapping) else {}
    gate_blocking = set(gate.get("blocking") or []) if isinstance(gate, Mapping) else set()
    gate_warnings = set(gate.get("warnings") or []) if isinstance(gate, Mapping) else set()
    if "gpu_total_below_runtime_estimate" in gate_blocking:
        blockers.append("gpu_total_below_runtime_estimate")
    if "gpu_free_below_runtime_estimate" in gate_warnings:
        blockers.append("gpu_free_below_runtime_estimate")
    for item in sorted(gate_warnings - {"gpu_free_below_runtime_estimate"}):
        warnings.append(item)
    return _dedupe(blockers), _dedupe(warnings)


def _normalize_newbie_shape(shape: NewbieStaticShape | Mapping[str, Any] | None) -> NewbieStaticShape:
    if isinstance(shape, NewbieStaticShape):
        result = shape
    else:
        data = dict(shape or {})
        result = NewbieStaticShape(
            batch=int(data.get("batch") or 1),
            latent_channels=int(data.get("latent_channels") or 16),
            latent_height=int(data.get("latent_height") or 64),
            latent_width=int(data.get("latent_width") or 64),
            tokens=int(data.get("tokens") or 512),
            hidden_dim=int(data.get("hidden_dim") or 2304),
            pooled_dim=int(data.get("pooled_dim") or 1024),
            patch_size=int(data.get("patch_size") or 2),
        )
    result.validate()
    return result


def _is_validated_newbie_production_shape(shape: NewbieStaticShape) -> bool:
    return shape.to_dict() == NewbieStaticShape(latent_height=64, latent_width=64, tokens=512).to_dict()


def _engine_record(engine: Path, budget: Mapping[str, Any]) -> dict[str, Any]:
    expected = int((budget.get("estimates") or {}).get("engine_bytes") or 0) if isinstance(budget, Mapping) else 0
    exists = engine.is_file()
    size = engine.stat().st_size if exists else 0
    return {
        "path": str(engine),
        "exists": exists,
        "bytes": size,
        "expected_bytes": expected,
        "expected_tolerance": 0.2,
    }


def _engine_size_below_expected(engine: Path, budget: Mapping[str, Any]) -> bool:
    if not engine.is_file():
        return False
    expected = int((budget.get("estimates") or {}).get("engine_bytes") or 0) if isinstance(budget, Mapping) else 0
    return expected > 0 and engine.stat().st_size < int(expected * 0.8)


def _checkpoint_record(family: str) -> dict[str, Any]:
    if family != "newbie":
        return {}
    path = default_newbie_checkpoint()
    return {"path": str(path), "bytes": path.stat().st_size if path.is_file() else 0}


def _dependency_snapshot(dependencies: Mapping[str, Any] | None) -> dict[str, Any]:
    if dependencies is not None:
        return {str(key): dict(value) if isinstance(value, Mapping) else value for key, value in dependencies.items()}
    torch_info: dict[str, Any] = {"available": importlib.util.find_spec("torch") is not None}
    if torch_info["available"]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_info.update({"version": getattr(torch, "__version__", ""), "cuda_available": cuda_available})
            if cuda_available:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                torch_info.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_free_bytes": int(free_bytes),
                    "gpu_total_bytes": int(total_bytes),
                })
        except Exception as exc:
            torch_info.update({"available": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "torch": torch_info,
        "tensorrt": {"available": importlib.util.find_spec("tensorrt") is not None},
    }


def _dep_available(dependencies: Mapping[str, Any], key: str) -> bool:
    value = dependencies.get(key, {}) if isinstance(dependencies, Mapping) else {}
    return bool(isinstance(value, Mapping) and value.get("available"))


def _normalize_precision(value: str | None) -> str:
    key = str(value or "fp32").strip().lower().replace("-", "_")
    return {"float32": "fp32", "float16": "fp16", "half": "fp16", "bfloat16": "bf16"}.get(key, key)


def _actions_required(blockers: Sequence[str], warnings: Sequence[str]) -> list[str]:
    actions: list[str] = []
    if "engine_file_missing" in blockers:
        actions.append("build_or_select_static_tensorrt_engine")
    if "tensorrt_missing" in blockers:
        actions.append("install_optional_tensorrt_runtime")
    if "torch_cuda_unavailable" in blockers or "gpu_total_below_runtime_estimate" in blockers:
        actions.append("use_cuda_runtime_with_sufficient_vram")
    if "newbie_static_shape_not_validated_for_runtime_gate" in blockers:
        actions.append("export_and_validate_requested_static_shape_before_runtime_gate")
    if "gpu_total_below_same_process_parity_estimate" in warnings:
        actions.append("use_split_offline_parity_for_validation")
    return _dedupe(actions)


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item)))
