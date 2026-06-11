# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""TensorRT preparation helpers for the image upscaler.

The module covers safe preparation steps that can be run outside the launcher
process: environment probing, RRDBNet-to-ONNX export, and ONNX-to-TensorRT engine
building.  TensorRT inference is kept separate because engines are GPU/version
and shape specific.
"""

from __future__ import annotations

import importlib.util
import json
import hashlib
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models" / "upscaler"
DEFAULT_TILE_PRESETS = (128, 256)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _clamp_tile(value: Any, fallback: int) -> int:
    try:
        tile = int(value)
    except Exception:
        tile = fallback
    return max(16, min(tile, 4096))


def _shape_to_list(shape: Any) -> list[int]:
    return [int(dim) for dim in tuple(shape)]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(path: Path) -> Path:
    return Path(str(path) + ".json")


def _read_metadata(path: Path) -> dict[str, Any] | None:
    meta_path = _metadata_path(path)
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_metadata(path: Path, data: dict[str, Any]) -> None:
    meta_path = _metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _artifact_record(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "metadata": _read_metadata(path) if exists else None,
    }


def _model_path_from_name(model_name: str) -> Path | None:
    if not model_name:
        return None
    candidate = MODELS_DIR / f"{model_name}.pth"
    if candidate.is_file():
        return candidate
    matches = sorted(MODELS_DIR.glob(f"{model_name}*.pth"))
    return matches[0] if matches else None


def default_onnx_path(model_path: str | Path, *, tile_size: int) -> Path:
    return MODELS_DIR / "onnx" / f"{Path(model_path).stem}_tile{int(tile_size)}.onnx"


def default_engine_path(onnx_path: str | Path, *, precision: str = "fp16", opt_tile_size: int | None = None) -> Path:
    stem = Path(onnx_path).stem
    tile_suffix = f"_{int(opt_tile_size)}" if opt_tile_size and f"tile{int(opt_tile_size)}" not in stem else ""
    return MODELS_DIR / "tensorrt" / f"{stem}{tile_suffix}_{str(precision or 'fp16').lower()}.engine"


def _matching_onnx_metadata(
    metadata: dict[str, Any] | None,
    *,
    model_sha256: str,
    scale: int,
    tile_size: int,
    opset: int,
    dynamic_axes: bool,
) -> bool:
    if not metadata:
        return False
    return (
        metadata.get("kind") == "upscaler_onnx_export"
        and metadata.get("model_sha256") == model_sha256
        and int(metadata.get("scale", 0)) == int(scale)
        and int(metadata.get("tile_size", 0)) == int(tile_size)
        and int(metadata.get("opset", 0)) == int(opset)
        and bool(metadata.get("dynamic_axes", False)) == bool(dynamic_axes)
    )


def _matching_engine_metadata(
    metadata: dict[str, Any] | None,
    *,
    onnx_sha256: str,
    precision: str,
    min_tile_size: int,
    opt_tile_size: int,
    max_tile_size: int,
    workspace_mb: int,
) -> bool:
    if not metadata:
        return False
    profile = metadata.get("profile") if isinstance(metadata.get("profile"), dict) else {}
    return (
        metadata.get("kind") == "upscaler_tensorrt_engine_build"
        and metadata.get("onnx_sha256") == onnx_sha256
        and str(metadata.get("requested_precision", "")).lower() == str(precision).lower()
        and int(profile.get("min_tile_size", 0)) == int(min_tile_size)
        and int(profile.get("opt_tile_size", 0)) == int(opt_tile_size)
        and int(profile.get("max_tile_size", 0)) == int(max_tile_size)
        and int(metadata.get("workspace_mb", 0)) == int(workspace_mb)
    )


def _profile_shape(base_shape: list[int], *, tile_size: int) -> tuple[int, ...]:
    shape = list(base_shape)
    if len(shape) == 4:
        shape[0] = 1 if shape[0] <= 0 else shape[0]
        shape[1] = 3 if shape[1] <= 0 else shape[1]
        shape[2] = tile_size
        shape[3] = tile_size
        return tuple(shape)
    return tuple(1 if dim <= 0 else dim for dim in shape)


def rrdb_state_from_checkpoint(loadnet: Any) -> dict[str, Any]:
    """Return RRDBNet-compatible state dict from common ESRGAN checkpoints."""
    if isinstance(loadnet, dict) and "params_ema" in loadnet:
        state = loadnet["params_ema"]
    elif isinstance(loadnet, dict) and "params" in loadnet:
        state = loadnet["params"]
    else:
        state = loadnet
    if not isinstance(state, dict):
        return state
    if "conv_first.weight" in state:
        return state
    if "model.0.weight" not in state:
        return state

    converted: dict[str, Any] = {}
    for key, value in state.items():
        new_key = _convert_esrgan_rrdb_key(str(key))
        converted[new_key] = value
    return converted


def _convert_esrgan_rrdb_key(key: str) -> str:
    if key.startswith("model.0."):
        return key.replace("model.0.", "conv_first.", 1)
    if key.startswith("model.1.sub.23."):
        return key.replace("model.1.sub.23.", "trunk_conv.", 1)
    if key.startswith("model.3."):
        return key.replace("model.3.", "upconv1.", 1)
    if key.startswith("model.6."):
        return key.replace("model.6.", "upconv2.", 1)
    if key.startswith("model.8."):
        return key.replace("model.8.", "HRconv.", 1)
    if key.startswith("model.10."):
        return key.replace("model.10.", "conv_last.", 1)
    if not key.startswith("model.1.sub."):
        return key
    parts = key.split(".")
    if len(parts) < 8:
        return key
    block_index = parts[3]
    rdb = {"RDB1": "rdb1", "RDB2": "rdb2", "RDB3": "rdb3"}.get(parts[4], parts[4].lower())
    conv = parts[5]
    param = parts[-1]
    return f"RRDB_trunk.{block_index}.{rdb}.{conv}.{param}"


def probe_acceleration() -> dict[str, Any]:
    """Return local readiness for ONNX export and future TensorRT builds."""
    started = time.perf_counter()
    torch_info: dict[str, Any] = {"available": False}
    onnx_info: dict[str, Any] = {"available": _module_available("onnx")}
    trt_info: dict[str, Any] = {"available": _module_available("tensorrt")}

    if onnx_info["available"]:
        try:
            import onnx  # type: ignore

            onnx_info["version"] = getattr(onnx, "__version__", "")
        except Exception as exc:
            onnx_info.update({"available": False, "error": str(exc)})

    if trt_info["available"]:
        try:
            import tensorrt as trt  # type: ignore

            trt_info["version"] = getattr(trt, "__version__", "")
        except Exception as exc:
            trt_info.update({"available": False, "error": str(exc)})

    if _module_available("torch"):
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_info.update({
                "available": True,
                "version": str(torch.__version__),
                "cuda_available": cuda_available,
                "cuda_version": getattr(torch.version, "cuda", None),
            })
            if cuda_available:
                torch_info.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
                })
        except Exception as exc:
            torch_info.update({"available": False, "error": str(exc)})

    ready_for_onnx = bool(torch_info.get("available") and onnx_info.get("available"))
    ready_for_trt = bool(ready_for_onnx and torch_info.get("cuda_available") and trt_info.get("available"))
    notes: list[str] = []
    if not torch_info.get("available"):
        notes.append("torch_missing")
    if not onnx_info.get("available"):
        notes.append("onnx_missing")
    if not torch_info.get("cuda_available"):
        notes.append("cuda_unavailable")
    if not trt_info.get("available"):
        notes.append("tensorrt_missing")

    return {
        "schema_version": 1,
        "kind": "upscaler_tensorrt_probe",
        "torch": torch_info,
        "onnx": onnx_info,
        "tensorrt": trt_info,
        "recommendation": {
            "ready_for_onnx_export": ready_for_onnx,
            "ready_for_tensorrt_build": ready_for_trt,
            "next_step": "export_onnx" if ready_for_onnx else "install_missing_dependencies",
            "notes": notes,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def inspect_artifacts(
    *,
    model_name: str = "",
    model_path: str = "",
    scale: int = 4,
    tile_presets: list[int] | tuple[int, ...] | None = None,
    precision: str = "fp16",
    dynamic_axes: bool = False,
    opset: int = 18,
    workspace_mb: int = 2048,
) -> dict[str, Any]:
    """Return known ONNX/TensorRT artifacts for the selected upscaler model."""
    started = time.perf_counter()
    resolved_model = Path(model_path) if model_path else _model_path_from_name(model_name)
    tiles = [_clamp_tile(tile, 128) for tile in (tile_presets or DEFAULT_TILE_PRESETS)]
    tiles = sorted(set(tiles))
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "upscaler_artifact_status",
        "model_name": model_name,
        "model_path": str(resolved_model) if resolved_model else "",
        "scale": int(scale or 4),
        "precision": str(precision or "fp16").lower(),
        "tiles": [],
        "elapsed_seconds": 0.0,
    }
    if not resolved_model or not resolved_model.is_file():
        result["missing_model"] = True
        result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return result

    model_sha256 = _file_sha256(resolved_model)
    result["model_sha256"] = model_sha256
    for tile in tiles:
        onnx_path = default_onnx_path(resolved_model, tile_size=tile)
        engine_path = default_engine_path(onnx_path, precision=str(precision or "fp16"), opt_tile_size=tile)
        onnx_record = _artifact_record(onnx_path)
        engine_record = _artifact_record(engine_path)
        onnx_record["matches"] = _matching_onnx_metadata(
            onnx_record.get("metadata"),
            model_sha256=model_sha256,
            scale=int(scale or 4),
            tile_size=tile,
            opset=int(opset or 18),
            dynamic_axes=bool(dynamic_axes),
        )
        if onnx_record["exists"]:
            try:
                onnx_sha256 = _file_sha256(Path(onnx_record["path"]))
            except Exception:
                onnx_sha256 = ""
            engine_record["matches"] = _matching_engine_metadata(
                engine_record.get("metadata"),
                onnx_sha256=onnx_sha256,
                precision=str(precision or "fp16"),
                min_tile_size=tile,
                opt_tile_size=tile,
                max_tile_size=tile,
                workspace_mb=max(256, int(workspace_mb or 2048)),
            )
        else:
            engine_record["matches"] = False
        result["tiles"].append({"tile_size": tile, "onnx": onnx_record, "engine": engine_record})
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return result


def _load_rrdb_state(model_path: Path, *, scale: int, device: str) -> Any:
    import torch

    from core.upscaler.architecture import RRDBNet

    model = RRDBNet(in_nc=3, out_nc=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    loadnet = torch.load(str(model_path), map_location=torch.device("cpu"), weights_only=True)
    state = rrdb_state_from_checkpoint(loadnet)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def export_rrdb_to_onnx(
    *,
    model_path: str,
    output_path: str,
    scale: int = 4,
    tile_size: int = 256,
    opset: int = 18,
    dynamic_axes: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export an RRDBNet upscaler checkpoint to ONNX for future TensorRT build."""
    started = time.perf_counter()
    src = Path(model_path)
    if not src.is_file():
        raise FileNotFoundError(f"Upscaler model not found: {model_path}")
    if src.suffix.lower() != ".pth":
        raise ValueError("Only .pth RRDBNet upscaler models can be exported to ONNX")

    import torch

    target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    tile_size = max(32, min(int(tile_size), 2048))
    scale = int(scale or 4)
    opset = max(13, int(opset or 18))
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    model_sha256 = _file_sha256(src)
    cached_metadata = _read_metadata(dst)
    if dst.is_file() and _matching_onnx_metadata(
        cached_metadata,
        model_sha256=model_sha256,
        scale=scale,
        tile_size=tile_size,
        opset=opset,
        dynamic_axes=bool(dynamic_axes),
    ):
        cached = dict(cached_metadata or {})
        cached.update({
            "success": True,
            "cached": True,
            "output_path": str(dst),
            "bytes": dst.stat().st_size,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        })
        return cached

    model = _load_rrdb_state(src, scale=scale, device=target_device)
    dummy = torch.randn(1, 3, tile_size, tile_size, device=target_device)
    axes = None
    if dynamic_axes:
        axes = {
            "input": {2: "height", 3: "width"},
            "output": {2: "height_out", 3: "width_out"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(dst),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            dynamic_axes=axes,
        )

    check: dict[str, Any] = {"available": False, "ok": None}
    if _module_available("onnx"):
        try:
            import onnx  # type: ignore

            exported = onnx.load(str(dst))
            onnx.checker.check_model(exported)
            check.update({"available": True, "ok": True})
        except Exception as exc:
            check.update({"available": True, "ok": False, "error": str(exc)})

    output_shape = [1, 3, tile_size * scale, tile_size * scale]
    result = {
        "schema_version": 1,
        "kind": "upscaler_onnx_export",
        "success": True,
        "cached": False,
        "model_path": str(src),
        "model_sha256": model_sha256,
        "output_path": str(dst),
        "scale": scale,
        "tile_size": tile_size,
        "opset": opset,
        "dynamic_axes": bool(dynamic_axes),
        "device": target_device,
        "input_shape": [1, 3, tile_size, tile_size],
        "output_shape": output_shape,
        "onnx_check": check,
        "bytes": dst.stat().st_size if dst.exists() else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    _write_metadata(dst, result)
    return result


def build_tensorrt_engine(
    *,
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    min_tile_size: int = 128,
    opt_tile_size: int = 256,
    max_tile_size: int = 512,
    workspace_mb: int = 2048,
) -> dict[str, Any]:
    """Build a TensorRT engine from an ONNX upscaler model."""
    started = time.perf_counter()
    src = Path(onnx_path)
    if not src.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if src.suffix.lower() != ".onnx":
        raise ValueError("TensorRT engine build requires a .onnx model")

    dst = Path(output_path)
    if dst.suffix.lower() not in {".engine", ".plan"}:
        dst = dst.with_suffix(".engine")
    dst.parent.mkdir(parents=True, exist_ok=True)

    requested_precision = str(precision or "fp16").strip().lower()
    min_tile = _clamp_tile(min_tile_size, 128)
    opt_tile = _clamp_tile(opt_tile_size, max(min_tile, 256))
    max_tile = _clamp_tile(max_tile_size, max(opt_tile, 512))
    min_tile = min(min_tile, opt_tile)
    max_tile = max(max_tile, opt_tile)
    workspace_mb = max(256, int(workspace_mb or 2048))
    onnx_sha256 = _file_sha256(src)
    cached_metadata = _read_metadata(dst)
    if dst.is_file() and _matching_engine_metadata(
        cached_metadata,
        onnx_sha256=onnx_sha256,
        precision=requested_precision,
        min_tile_size=min_tile,
        opt_tile_size=opt_tile,
        max_tile_size=max_tile,
        workspace_mb=workspace_mb,
    ):
        cached = dict(cached_metadata or {})
        cached.update({
            "success": True,
            "cached": True,
            "engine_path": str(dst),
            "bytes": dst.stat().st_size,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        })
        return cached

    if not _module_available("tensorrt"):
        raise RuntimeError("TensorRT Python package is not installed")

    import tensorrt as trt  # type: ignore

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(src)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(errors))

    config = builder.create_builder_config()
    workspace_bytes = workspace_mb * 1024 * 1024
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
        else:
            notes.append("fp16_unavailable")
    elif requested_precision not in {"fp32", "float32"}:
        notes.append(f"unsupported_precision:{requested_precision}")

    input_shapes: dict[str, list[int]] = {}
    dynamic_inputs: list[tuple[str, list[int]]] = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = _shape_to_list(tensor.shape)
        input_shapes[tensor.name] = shape
        if any(dim <= 0 for dim in shape):
            dynamic_inputs.append((tensor.name, shape))

    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        for name, shape in dynamic_inputs:
            profile.set_shape(
                name,
                _profile_shape(shape, tile_size=min_tile),
                _profile_shape(shape, tile_size=opt_tile),
                _profile_shape(shape, tile_size=max_tile),
            )
        config.add_optimization_profile(profile)

    output_shapes: dict[str, list[int]] = {}
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        output_shapes[tensor.name] = _shape_to_list(tensor.shape)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned no engine")

    payload = bytes(serialized)
    dst.write_bytes(payload)
    result = {
        "schema_version": 1,
        "kind": "upscaler_tensorrt_engine_build",
        "success": True,
        "cached": False,
        "onnx_path": str(src),
        "onnx_sha256": onnx_sha256,
        "engine_path": str(dst),
        "requested_precision": requested_precision,
        "precision": actual_precision,
        "workspace_mb": workspace_mb,
        "dynamic_inputs": [name for name, _shape in dynamic_inputs],
        "profile": {
            "min_tile_size": min_tile,
            "opt_tile_size": opt_tile,
            "max_tile_size": max_tile,
        },
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "tensorrt_version": getattr(trt, "__version__", ""),
        "bytes": len(payload),
        "notes": notes,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    _write_metadata(dst, result)
    return result
