from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_static_tensorrt_engine(
    *,
    onnx_path: str | Path,
    output_path: str | Path,
    precision: str = "fp16",
    workspace_mb: int = 4096,
    fp32_layer_policy: str = "none",
    fp32_layer_name_filters: Sequence[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    src = Path(onnx_path)
    if not src.is_file():
        raise FileNotFoundError(f"ONNX model not found: {src}")
    dst = Path(output_path)
    if dst.suffix.lower() not in {".engine", ".plan"}:
        dst = dst.with_suffix(".engine")
    dst.parent.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt  # type: ignore

    requested_precision = _normalize_precision_request(precision)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(src)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(errors))

    config = builder.create_builder_config()
    workspace_bytes = max(256, int(workspace_mb or 4096)) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_bytes

    actual_precision = "fp32"
    notes: list[str] = []
    if requested_precision == "fp16":
        if bool(getattr(builder, "platform_has_fast_fp16", False)):
            config.set_flag(trt.BuilderFlag.FP16)
            actual_precision = "fp16"
            if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        else:
            notes.append("fp16_unavailable")
    elif requested_precision == "bf16":
        if hasattr(trt.BuilderFlag, "BF16"):
            config.set_flag(trt.BuilderFlag.BF16)
            actual_precision = "bf16"
            if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        else:
            notes.append("bf16_unavailable")
    elif requested_precision != "fp32":
        notes.append(f"unsupported_precision:{requested_precision}")

    constraints = _apply_fp32_layer_policy(trt, network, fp32_layer_policy, fp32_layer_name_filters) if actual_precision in {"fp16", "bf16"} else _empty_fp32_constraints()
    if constraints["constrained_layers"]:
        actual_precision = f"{actual_precision}_mixed"
        notes.append(f"fp32_layer_policy:{constraints['policy']}")
        if constraints["name_filters"]:
            notes.append("fp32_layer_name_filters:" + ",".join(constraints["name_filters"]))

    input_shapes = _network_io_shapes(network, "input")
    output_shapes = _network_io_shapes(network, "output")
    dynamic_inputs = [name for name, shape in input_shapes.items() if any(dim <= 0 for dim in shape)]
    if dynamic_inputs:
        raise RuntimeError(f"Static TensorRT spike cannot build dynamic inputs yet: {dynamic_inputs}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned no engine")

    payload = bytes(serialized)
    dst.write_bytes(payload)
    return {
        "schema_version": 1,
        "kind": "base_model_static_tensorrt_engine_build",
        "success": True,
        "onnx_path": str(src),
        "onnx_sha256": _file_sha256(src),
        "onnx_artifact_bytes": _onnx_artifact_bytes(src),
        "engine_path": str(dst),
        "requested_precision": requested_precision,
        "precision": actual_precision,
        "fp32_layer_policy": constraints["policy"],
        "fp32_precision_constraints": constraints,
        "workspace_mb": max(256, int(workspace_mb or 4096)),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "tensorrt_version": getattr(trt, "__version__", ""),
        "bytes": len(payload),
        "notes": notes,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


class StaticTensorRtEngine:
    def __init__(self, engine_path: str | Path) -> None:
        import tensorrt as trt  # type: ignore
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires CUDA, but torch.cuda is not available")
        src = Path(engine_path)
        if not src.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {src}")

        self.trt = trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(src.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {src}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_names, self.output_names = self._io_names()
        self.stream = torch.cuda.Stream()

    def infer(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        cuda_inputs: dict[str, Any] = {}
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT input: {name}")
            expected_dtype = _torch_dtype_for_trt(self.trt, self.engine.get_tensor_dtype(name))
            tensor = inputs[name].contiguous().to(device="cuda", dtype=expected_dtype)
            self.context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))
            cuda_inputs[name] = tensor

        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"TensorRT could not infer shapes for: {unresolved}")

        outputs: dict[str, Any] = {}
        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise RuntimeError(f"TensorRT output has unresolved shape for {name}: {shape}")
            dtype = _torch_dtype_for_trt(self.trt, self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)

        for name, tensor in cuda_inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(int(self.stream.cuda_stream))
            if not ok:
                raise RuntimeError("TensorRT execution failed")
        current_stream.wait_stream(self.stream)
        return outputs

    def _io_names(self) -> tuple[list[str], list[str]]:
        inputs: list[str] = []
        outputs: list[str] = []
        for index in range(int(self.engine.num_io_tensors)):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == self.trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if not inputs or not outputs:
            raise RuntimeError(f"Invalid TensorRT engine IO: {inputs} -> {outputs}")
        return inputs, outputs


def compare_tensor_outputs(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    ref = reference.detach().float().cpu()
    got = candidate.detach().float().cpu()
    if tuple(ref.shape) != tuple(got.shape):
        return {
            "same_shape": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(got.shape),
        }
    diff = (ref - got).abs()
    denom = ref.abs().clamp_min(1e-6)
    rel = diff / denom
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    max_rel = float(rel.max().item())
    mean_rel = float(rel.mean().item())
    acceptable = mean_abs <= 0.05 and max_abs <= 0.2
    return {
        "same_shape": True,
        "shape": list(ref.shape),
        "all_finite": bool(torch.isfinite(got).all().item()),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "parity_acceptable": acceptable,
        "parity_thresholds": {"mean_abs_max": 0.05, "max_abs_max": 0.2},
    }


def _network_io_shapes(network: Any, kind: str) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    is_input = str(kind).lower() == "input"
    count = int(network.num_inputs if is_input else network.num_outputs)
    for index in range(count):
        tensor = network.get_input(index) if is_input else network.get_output(index)
        shapes[tensor.name] = [int(dim) for dim in tuple(tensor.shape)]
    return shapes


def _normalize_precision_request(value: str | None) -> str:
    key = str(value or "fp16").strip().lower().replace("-", "_")
    aliases = {
        "float": "fp32",
        "float32": "fp32",
        "half": "fp16",
        "float16": "fp16",
        "bfloat16": "bf16",
    }
    return aliases.get(key, key)


def _empty_fp32_constraints(policy: str = "none", name_filters: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "policy": policy,
        "name_filters": list(name_filters or ()),
        "constrained_layers": 0,
        "constrained_outputs": 0,
        "constrained_types": {},
        "constrained_name_filter_layers": 0,
        "sample_constrained_layer_names": [],
        "sample_name_filter_layer_names": [],
    }


def _apply_fp32_layer_policy(trt: Any, network: Any, policy: str, fp32_layer_name_filters: Sequence[str] | None = None) -> dict[str, Any]:
    normalized = str(policy or "none").strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "false", "0"}:
        normalized = "none"
    if normalized not in {"none", "all", "sensitive", "sensitive_projections", "sensitive_block_matmul", "norm", "non_matmul"}:
        raise ValueError(f"Unsupported TensorRT FP32 layer policy: {policy}")
    name_filters = _normalize_layer_name_filters(fp32_layer_name_filters)
    constrained_layers = 0
    constrained_outputs = 0
    constrained_name_filter_layers = 0
    constrained_types: dict[str, int] = {}
    sample_names: list[str] = []
    sample_filter_names: list[str] = []
    if normalized == "none" and not name_filters:
        return _empty_fp32_constraints(normalized, name_filters)
    for index in range(int(network.num_layers)):
        layer = network.get_layer(index)
        layer_type = _layer_type_name(layer)
        layer_name = str(layer.name)
        name_filter_matched = _matches_layer_name_filter(layer_name, name_filters)
        if not (_should_force_fp32(layer_type, normalized, layer_name) or name_filter_matched):
            continue
        did_set = False
        if _layer_has_float_output(trt, layer):
            try:
                layer.precision = trt.float32
                did_set = True
            except Exception:
                pass
        for output_index in range(int(layer.num_outputs)):
            output = layer.get_output(output_index)
            if not _is_float_tensor(trt, output):
                continue
            try:
                layer.set_output_type(output_index, trt.float32)
                constrained_outputs += 1
                did_set = True
            except Exception:
                pass
        if did_set:
            constrained_layers += 1
            if name_filter_matched:
                constrained_name_filter_layers += 1
                if len(sample_filter_names) < 32:
                    sample_filter_names.append(layer_name)
            if len(sample_names) < 32:
                sample_names.append(layer_name)
            constrained_types[layer_type] = constrained_types.get(layer_type, 0) + 1
    return {
        "policy": normalized,
        "name_filters": list(name_filters),
        "constrained_layers": constrained_layers,
        "constrained_outputs": constrained_outputs,
        "constrained_types": constrained_types,
        "constrained_name_filter_layers": constrained_name_filter_layers,
        "sample_constrained_layer_names": sample_names,
        "sample_name_filter_layer_names": sample_filter_names,
    }


def _normalize_layer_name_filters(values: Sequence[str] | None) -> tuple[str, ...]:
    filters: list[str] = []
    for value in values or ():
        item = str(value or "").strip().lower().replace("\\", "/")
        if item and item not in filters:
            filters.append(item)
    return tuple(filters)


def _matches_layer_name_filter(layer_name: str, filters: Sequence[str]) -> bool:
    if not filters:
        return False
    name = str(layer_name or "").lower().replace("\\", "/")
    return any(item in name for item in filters)


def _layer_type_name(layer: Any) -> str:
    return str(layer.type).split(".")[-1].upper()


def _layer_has_float_output(trt: Any, layer: Any) -> bool:
    for output_index in range(int(layer.num_outputs)):
        if _is_float_tensor(trt, layer.get_output(output_index)):
            return True
    return False


def _is_float_tensor(trt: Any, tensor: Any) -> bool:
    dtype = getattr(tensor, "dtype", None)
    float_types = {trt.float32, trt.float16, trt.DataType.FLOAT, trt.DataType.HALF}
    if hasattr(trt, "bfloat16"):
        float_types.add(trt.bfloat16)
    if hasattr(trt.DataType, "BF16"):
        float_types.add(trt.DataType.BF16)
    return dtype in float_types


def _should_force_fp32(layer_type: str, policy: str, layer_name: str = "") -> bool:
    if policy == "all":
        return True
    if policy == "norm":
        return layer_type in {"NORMALIZATION", "REDUCE"}
    if policy == "non_matmul":
        return layer_type not in {"MATRIX_MULTIPLY", "CONSTANT", "SHAPE", "GATHER", "SHUFFLE", "SLICE", "UNSQUEEZE"}
    if policy in {"sensitive", "sensitive_projections"}:
        if policy == "sensitive_projections" and _is_projection_matmul(layer_type, layer_name):
            return True
        return layer_type in {
            "ACTIVATION",
            "ELEMENTWISE",
            "NORMALIZATION",
            "REDUCE",
            "SELECT",
            "SOFTMAX",
            "UNARY",
        }
    if policy == "sensitive_block_matmul":
        if _is_block_matmul(layer_type, layer_name):
            return True
        return _should_force_fp32(layer_type, "sensitive", layer_name)
    return False


def _is_projection_matmul(layer_type: str, layer_name: str) -> bool:
    if layer_type != "MATRIX_MULTIPLY":
        return False
    name = layer_name.lower().replace("\\", "/")
    return any(
        marker in name
        for marker in (
            "x_embedder",
            "clip_text_pooled_proj",
            "time_text_embed",
            "adaln_modulation",
            "/mlp.0/",
            "/mlp.2/",
            "/out/",
            "/w2/",
            "/linear/",
        )
    )


def _is_block_matmul(layer_type: str, layer_name: str) -> bool:
    if layer_type != "MATRIX_MULTIPLY":
        return False
    name = layer_name.lower().replace("\\", "/")
    return any(marker in name for marker in ("/qkv/", "/matmul", "/out/", "/w1/", "/w2/", "/w3/"))


def _torch_dtype_for_trt(trt: Any, dtype: Any) -> Any:
    import torch

    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT8:
        return torch.int8
    if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
        return torch.bfloat16
    if hasattr(trt.DataType, "BOOL") and dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype}")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _onnx_artifact_bytes(path: Path) -> int:
    total = path.stat().st_size if path.exists() else 0
    try:
        import onnx  # type: ignore

        model = onnx.load(str(path), load_external_data=False)
        locations: set[str] = set()
        for tensor in model.graph.initializer:
            for item in tensor.external_data:
                if item.key == "location" and item.value:
                    locations.add(str(item.value))
        for location in locations:
            data_path = path.parent / location
            if data_path.is_file():
                total += data_path.stat().st_size
    except Exception:
        pass
    return total
