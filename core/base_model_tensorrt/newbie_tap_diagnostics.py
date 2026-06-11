from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .newbie_engine import (
    NEWBIE_TAP_OUTPUT_NAMES,
    create_newbie_static_export_wrapper,
    default_newbie_engine_path,
)
from .newbie_export import (
    NewbieStaticShape,
    create_newbie_synthetic_inputs,
    default_newbie_checkpoint,
    default_newbie_config_path,
    load_newbie_selective_wrapper,
    parse_layer_indices,
    summarize_tensor,
)
from .static_engine import StaticTensorRtEngine, compare_tensor_outputs


def compare_newbie_tensorrt_tap_parity(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    config_path: str | Path = "",
    engine_path: str | Path = "",
    output_dir: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    tap_layer_index: int = 0,
    device: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 1337,
    opset: int = 18,
    precision: str = "fp32",
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or NewbieStaticShape()
    layers = parse_layer_indices(layer_indices)
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_newbie_checkpoint(model_root)
    cfg_path = Path(config_path) if str(config_path or "").strip() else default_newbie_config_path(model_root)
    engine = Path(engine_path) if str(engine_path or "").strip() else default_newbie_engine_path(output_dir=output_dir, shape=shape, layer_indices=layers, opset=opset, precision=precision)
    model, selected_keys, target_device, normalized_dtype = load_newbie_selective_wrapper(
        checkpoint,
        config_path=cfg_path,
        layer_indices=layers,
        device=device,
        dtype_name=dtype_name,
    )
    wrapper = create_newbie_static_export_wrapper(model, shape, tap_layer_index=int(tap_layer_index)).eval()
    inputs = create_newbie_synthetic_inputs(shape=shape, device=target_device, dtype_name=normalized_dtype, seed=seed)
    args = (inputs["sample"], inputs["timestep"], inputs["encoder_hidden_states"], inputs["text_embeds"])

    import torch

    with torch.no_grad():
        torch_outputs = _named_outputs(wrapper(*args))

    runtime = StaticTensorRtEngine(engine)
    trt_outputs = runtime.infer({
        "sample": inputs["sample"],
        "timestep": inputs["timestep"],
        "encoder_hidden_states": inputs["encoder_hidden_states"],
        "text_embeds": inputs["text_embeds"],
    })
    missing = [name for name in NEWBIE_TAP_OUTPUT_NAMES if name not in trt_outputs]
    if missing:
        raise RuntimeError(f"TensorRT engine missing tap outputs: {missing}; got {list(trt_outputs)}")
    comparisons = {
        name: compare_tensor_outputs(torch_outputs[name], trt_outputs[name])
        for name in NEWBIE_TAP_OUTPUT_NAMES
    }
    first_unacceptable = next((name for name in NEWBIE_TAP_OUTPUT_NAMES if not comparisons[name].get("parity_acceptable", False)), "")
    first_unacceptable_tap = next((name for name in NEWBIE_TAP_OUTPUT_NAMES[1:] if not comparisons[name].get("parity_acceptable", False)), "")
    return {
        "schema_version": 1,
        "kind": "newbie_tensorrt_tap_parity",
        "success": all(bool(item.get("same_shape")) and bool(item.get("all_finite", True)) for item in comparisons.values()),
        "checkpoint_path": str(checkpoint),
        "config_path": str(cfg_path),
        "engine_path": str(engine),
        "layer_indices": list(layers),
        "tap_layer_index": int(tap_layer_index),
        "selected_key_count": len(selected_keys),
        "device": str(target_device),
        "dtype": normalized_dtype,
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "output_names": list(NEWBIE_TAP_OUTPUT_NAMES),
        "first_unacceptable_output": first_unacceptable,
        "first_unacceptable_tap": first_unacceptable_tap,
        "torch_outputs": {name: summarize_tensor(torch_outputs[name]) for name in NEWBIE_TAP_OUTPUT_NAMES},
        "tensorrt_outputs": {name: summarize_tensor(trt_outputs[name]) for name in NEWBIE_TAP_OUTPUT_NAMES},
        "comparisons": comparisons,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _named_outputs(output: Any) -> dict[str, Any]:
    values = output if isinstance(output, tuple) else (output,)
    return {name: values[index] for index, name in enumerate(NEWBIE_TAP_OUTPUT_NAMES[: len(values)])}
