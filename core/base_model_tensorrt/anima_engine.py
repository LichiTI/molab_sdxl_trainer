from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .anima_export import (
    AnimaStaticShape,
    create_anima_static_export_wrapper,
    create_anima_synthetic_inputs,
    default_anima_checkpoint,
    default_anima_engine_path,
    default_anima_onnx_path,
    load_anima_export_subset,
    parse_block_indices,
    summarize_tensor,
)
from .static_engine import StaticTensorRtEngine, build_static_tensorrt_engine, compare_tensor_outputs


def build_anima_tensorrt_engine(
    *,
    onnx_path: str | Path = "",
    output_path: str | Path = "",
    output_dir: str | Path = "",
    block_indices: str | Sequence[int] = (0,),
    shape: AnimaStaticShape | None = None,
    opset: int = 18,
    precision: str = "fp16",
    workspace_mb: int = 4096,
) -> dict[str, Any]:
    shape = shape or AnimaStaticShape()
    blocks = parse_block_indices(block_indices)
    src = Path(onnx_path) if str(onnx_path or "").strip() else default_anima_onnx_path(
        output_dir=output_dir,
        shape=shape,
        block_indices=blocks,
        opset=opset,
    )
    dst = Path(output_path) if str(output_path or "").strip() else default_anima_engine_path(
        output_dir=output_dir,
        shape=shape,
        block_indices=blocks,
        opset=opset,
        precision=precision,
    )
    result = build_static_tensorrt_engine(
        onnx_path=src,
        output_path=dst,
        precision=precision,
        workspace_mb=workspace_mb,
    )
    result.update({
        "model_family": "anima",
        "block_indices": list(blocks),
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
    })
    return result


def compare_anima_tensorrt_parity(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    engine_path: str | Path = "",
    output_dir: str | Path = "",
    block_indices: str | Sequence[int] = (0,),
    shape: AnimaStaticShape | None = None,
    device: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 1337,
    opset: int = 18,
    precision: str = "fp16",
    disable_mmap: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or AnimaStaticShape()
    blocks = parse_block_indices(block_indices)
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_anima_checkpoint(model_root)
    engine = Path(engine_path) if str(engine_path or "").strip() else default_anima_engine_path(
        output_dir=output_dir,
        shape=shape,
        block_indices=blocks,
        opset=opset,
        precision=precision,
    )
    model, weight_report, target_device, normalized_dtype = load_anima_export_subset(
        checkpoint,
        block_indices=blocks,
        device=device,
        dtype_name=dtype_name,
        disable_mmap=disable_mmap,
    )
    wrapper = create_anima_static_export_wrapper(model, shape).eval()
    inputs = create_anima_synthetic_inputs(
        shape=shape,
        device=target_device,
        dtype_name=normalized_dtype,
        seed=seed,
    )

    import torch

    with torch.no_grad():
        torch_output = wrapper(*inputs)

    runtime = StaticTensorRtEngine(engine)
    trt_outputs = runtime.infer({
        "sample": inputs[0],
        "timestep": inputs[1],
        "encoder_hidden_states": inputs[2],
        "padding_mask": inputs[3],
    })
    if "sample_out" not in trt_outputs:
        raise RuntimeError(f"TensorRT engine did not return sample_out, got {list(trt_outputs)}")
    trt_output = trt_outputs["sample_out"]
    comparison = compare_tensor_outputs(torch_output, trt_output)
    return {
        "schema_version": 1,
        "kind": "anima_tensorrt_parity",
        "success": bool(comparison.get("same_shape")) and bool(comparison.get("all_finite", True)),
        "checkpoint_path": str(checkpoint),
        "engine_path": str(engine),
        "block_indices": list(blocks),
        "device": str(target_device),
        "dtype": normalized_dtype,
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "weight_load_report": _report_to_dict(weight_report),
        "torch_output": summarize_tensor(torch_output),
        "tensorrt_output": summarize_tensor(trt_output),
        "comparison": comparison,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _report_to_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        return dict(report.to_dict())
    if hasattr(report, "__dict__"):
        return dict(report.__dict__)
    return {"repr": repr(report)}
