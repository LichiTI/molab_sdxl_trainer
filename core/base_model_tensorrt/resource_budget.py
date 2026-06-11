from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Mapping


GIB = 1024 ** 3


FAMILY_RESOURCE_ANCHORS: dict[str, dict[str, Any]] = {
    "newbie": {
        "confidence": "measured_full36_production_shape",
        "tiny_shape": {"batch": 1, "latent_channels": 16, "latent_height": 4, "latent_width": 4, "tokens": 4, "patch_size": 2},
        "production_shape": {"batch": 1, "latent_channels": 16, "latent_height": 64, "latent_width": 64, "tokens": 512, "patch_size": 2},
        "layer_count": 36,
        "hidden_dim": 2304,
        "attention_heads": 24,
        "ffn_dim": 9216,
        "checkpoint_bytes": 6_973_329_400,
        "onnx_artifact_bytes": 12_602_123_294,
        "engine_bytes_by_precision": {
            "fp32": 12_608_380_596,
            "fp16": 6_310_000_000,
            "bf16": 6_310_000_000,
        },
    },
    "anima": {
        "confidence": "block0_tiny_shape_extrapolated",
        "tiny_shape": {"batch": 1, "latent_channels": 16, "latent_height": 4, "latent_width": 4, "tokens": 4, "patch_size": 2},
        "production_shape": {"batch": 1, "latent_channels": 16, "latent_height": 64, "latent_width": 64, "tokens": 512, "patch_size": 2},
        "layer_count": 28,
        "hidden_dim": 2048,
        "attention_heads": 16,
        "ffn_dim": 8192,
        "onnx_artifact_bytes": int(10.6 * GIB),
        "engine_bytes_by_precision": {
            "fp32": int(10.6 * GIB),
            "fp16": int(5.4 * GIB),
            "bf16": int(5.4 * GIB),
        },
    },
}


def estimate_base_model_tensorrt_resource_budget(
    *,
    family: str,
    static_shape: Mapping[str, Any],
    engine_precision: str,
    output_path: str | Path,
    checkpoint: Mapping[str, Any] | None = None,
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    family_key = str(family or "").strip().lower()
    anchor = FAMILY_RESOURCE_ANCHORS.get(family_key, {})
    precision = _normalize_precision(engine_precision)
    shape = _shape_with_defaults(static_shape, anchor.get("production_shape", {}))
    checkpoint_bytes = _checkpoint_bytes(checkpoint, anchor)
    onnx_bytes = int(anchor.get("onnx_artifact_bytes") or max(checkpoint_bytes, 0))
    engine_bytes = _engine_bytes(anchor, precision, onnx_bytes)
    activation = _activation_estimate_bytes(shape, anchor, precision)
    workspace_bytes = 4 * GIB
    build_scratch_bytes = max(onnx_bytes, engine_bytes, workspace_bytes)
    minimum_output_disk_bytes = onnx_bytes + engine_bytes + workspace_bytes
    recommended_output_disk_bytes = _ceil_gib(int((onnx_bytes + engine_bytes) * 2.25 + build_scratch_bytes + workspace_bytes))
    runtime_gpu_bytes = _ceil_gib(engine_bytes + activation["runtime_activation_bytes"] + GIB)
    torch_reference_gpu_bytes = _ceil_gib(checkpoint_bytes + activation["parity_activation_bytes"] + 2 * GIB)
    same_process_parity_gpu_bytes = _ceil_gib(engine_bytes + checkpoint_bytes + activation["parity_activation_bytes"] + 2 * GIB)
    offline_parity_gpu_bytes = max(runtime_gpu_bytes, torch_reference_gpu_bytes)
    build_gpu_bytes = _ceil_gib(max(engine_bytes // 3, workspace_bytes) + activation["runtime_activation_bytes"] + 2 * GIB)
    disk = _disk_record(Path(output_path))
    gpu = _gpu_record(dependencies)
    gate = _gate(
        disk_free_bytes=disk.get("free_bytes"),
        gpu_total_bytes=gpu.get("total_bytes"),
        gpu_free_bytes=gpu.get("free_bytes"),
        minimum_output_disk_bytes=minimum_output_disk_bytes,
        recommended_output_disk_bytes=recommended_output_disk_bytes,
        runtime_gpu_bytes=runtime_gpu_bytes,
        parity_gpu_bytes=same_process_parity_gpu_bytes,
        confidence=str(anchor.get("confidence") or "unknown"),
    )
    return {
        "schema_version": 1,
        "kind": "base_model_tensorrt_resource_budget",
        "family": family_key,
        "shape": shape,
        "engine_precision": precision,
        "estimate_confidence": str(anchor.get("confidence") or "unknown"),
        "anchor": {
            "tiny_shape": anchor.get("tiny_shape", {}),
            "onnx_artifact_bytes": onnx_bytes,
            "engine_bytes": engine_bytes,
            "checkpoint_bytes": checkpoint_bytes,
        },
        "estimates": {
            "onnx_artifact_bytes": onnx_bytes,
            "engine_bytes": engine_bytes,
            "minimum_output_disk_bytes": minimum_output_disk_bytes,
            "recommended_output_disk_bytes": recommended_output_disk_bytes,
            "build_scratch_bytes": build_scratch_bytes,
            "runtime_gpu_bytes": runtime_gpu_bytes,
            "torch_reference_gpu_bytes": torch_reference_gpu_bytes,
            "offline_parity_gpu_bytes": offline_parity_gpu_bytes,
            "parity_gpu_bytes": same_process_parity_gpu_bytes,
            "same_process_parity_gpu_bytes": same_process_parity_gpu_bytes,
            "build_gpu_bytes": build_gpu_bytes,
            **activation,
        },
        "environment": {
            "output_disk": disk,
            "gpu": gpu,
        },
        "gate": gate,
        "assumptions": _assumptions(family_key, precision, anchor),
    }


def _ceil_gib(value: int) -> int:
    return int(math.ceil(max(0, int(value)) / GIB) * GIB)


def _normalize_precision(value: str | None) -> str:
    key = str(value or "fp16").strip().lower().replace("-", "_")
    return {"float32": "fp32", "float16": "fp16", "half": "fp16", "bfloat16": "bf16"}.get(key, key)


def _shape_with_defaults(shape: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, int]:
    result = {str(key): int(value) for key, value in defaults.items() if value is not None}
    for key, value in shape.items():
        result[str(key)] = int(value)
    result.setdefault("batch", 1)
    result.setdefault("latent_channels", 16)
    result.setdefault("latent_height", 64)
    result.setdefault("latent_width", 64)
    result.setdefault("tokens", 512)
    result.setdefault("patch_size", 2)
    return result


def _checkpoint_bytes(checkpoint: Mapping[str, Any] | None, anchor: Mapping[str, Any]) -> int:
    if checkpoint:
        value = checkpoint.get("bytes")
        if isinstance(value, int) and value > 0:
            return value
    return int(anchor.get("checkpoint_bytes") or 0)


def _engine_bytes(anchor: Mapping[str, Any], precision: str, fallback: int) -> int:
    engines = anchor.get("engine_bytes_by_precision")
    if isinstance(engines, Mapping):
        value = engines.get(precision) or engines.get("fp16") or engines.get("fp32")
        if isinstance(value, int) and value > 0:
            return value
    return int(fallback)


def _activation_estimate_bytes(shape: Mapping[str, int], anchor: Mapping[str, Any], precision: str) -> dict[str, int]:
    dtype_bytes = 4 if precision == "fp32" else 2
    batch = int(shape.get("batch", 1))
    patch_size = max(1, int(shape.get("patch_size", 2)))
    visual_tokens = max(1, (int(shape.get("latent_height", 64)) // patch_size) * (int(shape.get("latent_width", 64)) // patch_size))
    hidden_dim = max(1, int(anchor.get("hidden_dim") or 2048))
    heads = max(1, int(anchor.get("attention_heads") or 16))
    ffn_dim = max(hidden_dim, int(anchor.get("ffn_dim") or hidden_dim * 4))
    hidden_bytes = batch * visual_tokens * hidden_dim * dtype_bytes
    ffn_bytes = batch * visual_tokens * ffn_dim * dtype_bytes
    attention_score_bytes = batch * heads * visual_tokens * visual_tokens * dtype_bytes
    per_block_workspace = hidden_bytes * 8 + ffn_bytes * 2 + attention_score_bytes * 2
    runtime_activation = _ceil_gib(per_block_workspace * 2)
    parity_activation = _ceil_gib(per_block_workspace * 5)
    return {
        "visual_tokens": visual_tokens,
        "hidden_state_bytes": hidden_bytes,
        "ffn_hidden_bytes": ffn_bytes,
        "attention_score_bytes": attention_score_bytes,
        "estimated_per_block_workspace_bytes": per_block_workspace,
        "runtime_activation_bytes": runtime_activation,
        "parity_activation_bytes": parity_activation,
    }


def _disk_record(path: Path) -> dict[str, Any]:
    target = path if path.suffix == "" else path.parent
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        return {
            "path": str(target),
            "probe_path": str(probe),
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
        }
    except Exception as exc:
        return {"path": str(target), "available": False, "error": str(exc)}


def _gpu_record(dependencies: Mapping[str, Any] | None) -> dict[str, Any]:
    torch_info = (dependencies or {}).get("torch", {}) if isinstance(dependencies, Mapping) else {}
    if not isinstance(torch_info, Mapping) or not torch_info.get("cuda_available"):
        return {"available": False, "reason": "cuda_unavailable_or_not_reported"}
    return {
        "available": True,
        "name": torch_info.get("gpu_name", ""),
        "total_bytes": torch_info.get("gpu_total_bytes"),
        "free_bytes": torch_info.get("gpu_free_bytes"),
        "bf16_supported": torch_info.get("bf16_supported"),
    }


def _gate(
    *,
    disk_free_bytes: Any,
    gpu_total_bytes: Any,
    gpu_free_bytes: Any,
    minimum_output_disk_bytes: int,
    recommended_output_disk_bytes: int,
    runtime_gpu_bytes: int,
    parity_gpu_bytes: int,
    confidence: str,
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    if not isinstance(disk_free_bytes, int):
        warnings.append("output_disk_free_unknown")
    elif disk_free_bytes < minimum_output_disk_bytes:
        blocking.append("output_disk_below_minimum")
    elif disk_free_bytes < recommended_output_disk_bytes:
        warnings.append("output_disk_below_recommended_headroom")
    if not isinstance(gpu_total_bytes, int):
        warnings.append("gpu_total_memory_unknown")
    elif gpu_total_bytes < runtime_gpu_bytes:
        blocking.append("gpu_total_below_runtime_estimate")
    elif gpu_total_bytes < parity_gpu_bytes:
        warnings.append("gpu_total_below_same_process_parity_estimate")
    if isinstance(gpu_free_bytes, int) and gpu_free_bytes < runtime_gpu_bytes:
        warnings.append("gpu_free_below_runtime_estimate")
    if confidence.endswith("extrapolated"):
        warnings.append("estimate_requires_full_transformer_calibration")
    status = "fail" if blocking else "warn" if warnings else "pass"
    return {
        "status": status,
        "blocking": blocking,
        "warnings": warnings,
    }


def _assumptions(family: str, precision: str, anchor: Mapping[str, Any]) -> list[str]:
    confidence = str(anchor.get("confidence") or "")
    artifact_basis = (
        "ONNX and TensorRT engine bytes are anchored to measured production static export/build artifacts."
        if confidence == "measured_full36_production_shape"
        else "ONNX and TensorRT engine bytes are anchored to tiny-shape static export measurements because weights dominate artifact size."
    )
    items = [
        "Transformer-only budget; prompt encoders, VAE, scheduler, LoRA hooks, and sampling are outside this engine.",
        artifact_basis,
        "Runtime GPU estimate includes engine bytes, a conservative activation/workspace buffer, and 1 GiB overhead.",
        "Offline parity can run PyTorch reference and TensorRT runtime in separate processes; same-process parity keeps both resident together.",
    ]
    if family == "newbie":
        items.append("Newbie release validation should keep FP32 until low-precision hidden-state drift is solved.")
    if str(anchor.get("confidence") or "").endswith("extrapolated"):
        items.append("This family still needs a full-transformer tiny-shape artifact anchor to replace block extrapolation.")
    if precision != "fp32":
        items.append("Low-precision artifact/runtime estimates do not imply parity acceptance.")
    return items
