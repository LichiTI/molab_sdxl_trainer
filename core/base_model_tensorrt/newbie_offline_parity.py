from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .newbie_engine import create_newbie_static_export_wrapper, default_newbie_engine_path
from .newbie_export import (
    NewbieStaticShape,
    create_newbie_synthetic_inputs,
    default_newbie_checkpoint,
    default_newbie_config_path,
    load_newbie_selective_wrapper,
    parse_layer_indices,
)
from .static_engine import StaticTensorRtEngine
from .tensor_artifacts import compare_tensor_output_artifacts, save_tensor_output_artifact


def write_newbie_torch_output_artifact(
    *,
    artifact_path: str | Path,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    config_path: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    device: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 1337,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or NewbieStaticShape()
    layers = parse_layer_indices(layer_indices)
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_newbie_checkpoint(model_root)
    cfg_path = Path(config_path) if str(config_path or "").strip() else default_newbie_config_path(model_root)
    model, selected_keys, target_device, normalized_dtype = load_newbie_selective_wrapper(
        checkpoint,
        config_path=cfg_path,
        layer_indices=layers,
        device=device,
        dtype_name=dtype_name,
    )
    wrapper = create_newbie_static_export_wrapper(model, shape).eval()
    inputs = create_newbie_synthetic_inputs(shape=shape, device=target_device, dtype_name=normalized_dtype, seed=seed)
    args = (inputs["sample"], inputs["timestep"], inputs["encoder_hidden_states"], inputs["text_embeds"])

    import torch

    with torch.no_grad():
        output = wrapper(*args)

    metadata = _metadata(
        source="torch",
        checkpoint_path=checkpoint,
        config_path=cfg_path,
        layer_indices=layers,
        shape=shape,
        device=str(target_device),
        dtype=normalized_dtype,
        seed=seed,
        selected_key_count=len(selected_keys),
    )
    artifact = save_tensor_output_artifact(artifact_path, output, metadata=metadata, output_name="sample_out")
    return _result(
        kind="newbie_tensorrt_torch_output_artifact",
        artifact=artifact,
        layer_indices=layers,
        shape=shape,
        device=str(target_device),
        dtype=normalized_dtype,
        seed=seed,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        extra={"checkpoint_path": str(checkpoint), "config_path": str(cfg_path), "selected_key_count": len(selected_keys)},
    )


def write_newbie_tensorrt_output_artifact(
    *,
    artifact_path: str | Path,
    engine_path: str | Path = "",
    output_dir: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    device: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 1337,
    opset: int = 18,
    precision: str = "fp32",
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or NewbieStaticShape()
    layers = parse_layer_indices(layer_indices)
    engine = Path(engine_path) if str(engine_path or "").strip() else default_newbie_engine_path(
        output_dir=output_dir,
        shape=shape,
        layer_indices=layers,
        opset=opset,
        precision=precision,
    )
    inputs = create_newbie_synthetic_inputs(shape=shape, device=device, dtype_name=dtype_name, seed=seed)
    input_device = str(inputs["sample"].device)
    runtime = StaticTensorRtEngine(engine)
    outputs = runtime.infer({
        "sample": inputs["sample"],
        "timestep": inputs["timestep"],
        "encoder_hidden_states": inputs["encoder_hidden_states"],
        "text_embeds": inputs["text_embeds"],
    })
    if "sample_out" not in outputs:
        raise RuntimeError(f"TensorRT engine did not return sample_out, got {list(outputs)}")
    metadata = _metadata(
        source="tensorrt",
        engine_path=engine,
        layer_indices=layers,
        shape=shape,
        device=input_device,
        dtype=_normalize_dtype_name(dtype_name),
        seed=seed,
        opset=opset,
        precision=precision,
        runtime_device="cuda",
    )
    artifact = save_tensor_output_artifact(artifact_path, outputs["sample_out"], metadata=metadata, output_name="sample_out")
    return _result(
        kind="newbie_tensorrt_runtime_output_artifact",
        artifact=artifact,
        layer_indices=layers,
        shape=shape,
        device=input_device,
        dtype=_normalize_dtype_name(dtype_name),
        seed=seed,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        extra={"engine_path": str(engine), "precision": precision, "opset": int(opset)},
    )


def compare_newbie_output_artifacts(
    *,
    torch_artifact_path: str | Path,
    tensorrt_artifact_path: str | Path,
) -> dict[str, Any]:
    result = compare_tensor_output_artifacts(
        torch_artifact_path,
        tensorrt_artifact_path,
        reference_label="torch",
        candidate_label="tensorrt",
    )
    result.update({"kind": "newbie_tensorrt_offline_output_parity"})
    return result


def _metadata(
    *,
    source: str,
    layer_indices: Sequence[int],
    shape: NewbieStaticShape,
    device: str,
    dtype: str,
    seed: int,
    output_name: str = "sample_out",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "model_family": "newbie",
        "source": source,
        "layer_indices": list(layer_indices),
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "device": device,
        "dtype": _normalize_dtype_name(dtype),
        "seed": int(seed),
        "output_name": output_name,
        **{key: str(value) if isinstance(value, Path) else value for key, value in extra.items()},
    }


def _result(
    *,
    kind: str,
    artifact: dict[str, Any],
    layer_indices: Sequence[int],
    shape: NewbieStaticShape,
    device: str,
    dtype: str,
    seed: int,
    elapsed_seconds: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "success": True,
        **extra,
        "artifact_path": artifact.get("path", ""),
        "artifact": artifact,
        "layer_indices": list(layer_indices),
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "device": device,
        "dtype": _normalize_dtype_name(dtype),
        "seed": int(seed),
        "elapsed_seconds": elapsed_seconds,
    }


def _normalize_dtype_name(value: str | None) -> str:
    key = str(value or "float32").strip().lower()
    aliases = {"fp32": "float32", "fp16": "float16", "half": "float16", "bf16": "bfloat16"}
    return aliases.get(key, key)
