from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .newbie_export import NewbieStaticShape, create_newbie_synthetic_inputs, parse_layer_indices
from .runtime_gate import build_static_transformer_runtime_gate
from .runtime_adapter import StaticTransformerRuntimeSpec, StaticTransformerTensorRtRuntime
from .tensor_artifacts import save_tensor_output_artifact, summarize_tensor_artifact_value


def run_newbie_static_tensorrt_runtime_smoke(
    *,
    engine_path: str | Path = "",
    output_dir: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    device: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 1337,
    precision: str = "fp32",
    opset: int = 18,
    artifact_path: str | Path = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or NewbieStaticShape(latent_height=64, latent_width=64, tokens=512)
    layers = parse_layer_indices(layer_indices)
    gate = build_static_transformer_runtime_gate(
        family="newbie",
        engine_path=engine_path,
        output_dir=output_dir,
        layer_indices=layers,
        shape=shape,
        precision=precision,
        opset=opset,
    )
    if not gate.get("runtime_loadable"):
        return {
            "schema_version": 1,
            "kind": "newbie_static_tensorrt_runtime_smoke",
            "success": False,
            "blocked": True,
            "gate": gate,
            "blockers": list(gate.get("blockers") or []),
            "warnings": list(gate.get("warnings") or []),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }

    inputs = create_newbie_synthetic_inputs(shape=shape, device=device, dtype_name=dtype_name, seed=seed)
    runtime = StaticTransformerTensorRtRuntime(StaticTransformerRuntimeSpec.from_gate_report(gate))

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_started = time.perf_counter()
    output = runtime.infer({
        "sample": inputs["sample"],
        "timestep": inputs["timestep"],
        "encoder_hidden_states": inputs["encoder_hidden_states"],
        "text_embeds": inputs["text_embeds"],
    })
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_started
    metadata = {
        "model_family": "newbie",
        "source": "tensorrt_runtime_smoke",
        "engine_path": gate["engine_path"],
        "layer_indices": list(layers),
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "device": str(inputs["sample"].device),
        "dtype": _normalize_dtype_name(dtype_name),
        "seed": int(seed),
        "precision": _normalize_precision(precision),
        "opset": int(opset),
        "output_name": "sample_out",
    }
    artifact = None
    if str(artifact_path or "").strip():
        artifact = save_tensor_output_artifact(artifact_path, output, metadata=metadata, output_name="sample_out")
    return {
        "schema_version": 1,
        "kind": "newbie_static_tensorrt_runtime_smoke",
        "success": True,
        "blocked": False,
        "gate": gate,
        "engine_path": gate["engine_path"],
        "layer_indices": list(layers),
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "device": str(inputs["sample"].device),
        "dtype": _normalize_dtype_name(dtype_name),
        "seed": int(seed),
        "precision": _normalize_precision(precision),
        "output": summarize_tensor_artifact_value(output),
        "artifact": artifact,
        "generation_path_enabled": False,
        "training_path_enabled": False,
        "infer_seconds": round(infer_seconds, 4),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _normalize_dtype_name(value: str | None) -> str:
    key = str(value or "float32").strip().lower()
    return {"fp32": "float32", "fp16": "float16", "half": "float16", "bf16": "bfloat16"}.get(key, key)


def _normalize_precision(value: str | None) -> str:
    key = str(value or "fp32").strip().lower().replace("-", "_")
    return {"float32": "fp32", "float16": "fp16", "half": "fp16", "bfloat16": "bf16"}.get(key, key)
