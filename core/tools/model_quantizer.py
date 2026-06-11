"""Model quantization helpers for Launcher toolbox actions."""

from __future__ import annotations

import os
import sys
import gc
from pathlib import Path
from typing import Any

import torch

try:
    from core.lulynx_trainer.lulynx_quantized_safetensors import (
        PLAIN_FORMATS,
        ROWWISE_FORMATS,
        SCHEMA_KEY,
        SUPPORTED_QUANT_FORMATS,
        normalize_decode_dtype,
        normalize_quant_format,
        quantize_plain_state_dict,
        quantize_rowwise_state_dict,
    )
except ImportError:
    from backend.core.lulynx_trainer.lulynx_quantized_safetensors import (
        PLAIN_FORMATS,
        ROWWISE_FORMATS,
        SCHEMA_KEY,
        SUPPORTED_QUANT_FORMATS,
        normalize_decode_dtype,
        normalize_quant_format,
        quantize_plain_state_dict,
        quantize_rowwise_state_dict,
    )

try:
    from core.gguf_quantizer_engine import GGUF_EXTERNAL_QUANT_FORMATS, run_external_gguf_quantizer
except ImportError:
    from backend.core.gguf_quantizer_engine import GGUF_EXTERNAL_QUANT_FORMATS, run_external_gguf_quantizer

try:
    from core.gguf_arch_adapter import plan_gguf_export
except ImportError:
    from backend.core.gguf_arch_adapter import plan_gguf_export

try:
    from core.tools.model_quantization_validator import validate_quantized_model_file
except ImportError:
    from backend.core.tools.model_quantization_validator import validate_quantized_model_file

GGUF_CONTAINER_FORMATS = {"gguf_f16", "gguf_f32"}
GGUF_PYTHON_QUANT_FORMATS = {"gguf_q8_0"}
GGUF_FORMATS = GGUF_CONTAINER_FORMATS | GGUF_PYTHON_QUANT_FORMATS | GGUF_EXTERNAL_QUANT_FORMATS
SUPPORTED_MODEL_QUANT_FORMATS = SUPPORTED_QUANT_FORMATS | GGUF_FORMATS

_GGUF_ALIASES = {
    "gguf": "gguf_f16",
    "gguf_fp16": "gguf_f16",
    "gguf_float16": "gguf_f16",
    "gguf_half": "gguf_f16",
    "gguf_fp32": "gguf_f32",
    "gguf_float32": "gguf_f32",
    "gguf_q8": "gguf_q8_0",
    "q8_0": "gguf_q8_0",
    "q8": "gguf_q8_0",
    "gguf_q4_k": "gguf_q4_k_m",
    "q4_k": "gguf_q4_k_m",
    "q4_k_m": "gguf_q4_k_m",
    "gguf_q5_k": "gguf_q5_k_m",
    "q5_k": "gguf_q5_k_m",
    "q5_k_m": "gguf_q5_k_m",
}


def quantize_model_file(
    input_path: str,
    output_path: str,
    quant_format: str = "fp16",
    *,
    decode_dtype: str = "fp16",
    preserve_metadata: bool = True,
    overwrite: bool = False,
    gguf_arch: str = "generic",
    gguf_name: str = "",
    gguf_metadata: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Quantize a tensor checkpoint into one of the supported Lulynx formats."""

    src = Path(input_path)
    dst = Path(output_path)
    fmt = normalize_model_quant_format(quant_format)
    decode_dtype = normalize_decode_dtype(decode_dtype)
    _validate_paths(src, dst, overwrite=overwrite, output_format="gguf" if fmt in GGUF_FORMATS else "safetensors")

    dst.parent.mkdir(parents=True, exist_ok=True)
    input_size = os.path.getsize(src)
    state: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    if fmt in GGUF_EXTERNAL_QUANT_FORMATS:
        if src.suffix.lower() != ".gguf":
            raise ValueError(
                "GGUF K quantization requires an existing llama.cpp-compatible .gguf input. "
                "Use gguf_f16/gguf_f32/gguf_q8_0 for direct tensor export, or convert the model to GGUF first."
            )
        stats = run_external_gguf_quantizer(src, dst, fmt, search_roots=_gguf_quantizer_search_roots(src, dst))
        tensor_count = _count_gguf_tensors(dst)
    else:
        state, metadata = _load_state_and_metadata(src)
        tensor_count = len(state)

    if fmt in GGUF_EXTERNAL_QUANT_FORMATS:
        pass
    elif fmt in GGUF_CONTAINER_FORMATS or fmt in GGUF_PYTHON_QUANT_FORMATS:
        stats = _export_gguf_state_dict(
            state,
            dst,
            fmt,
            arch=gguf_arch,
            name=gguf_name or src.stem,
            source_path=src,
            metadata=_coerce_gguf_metadata(gguf_metadata),
        )
    elif fmt in PLAIN_FORMATS:
        output_state, stats = quantize_plain_state_dict(state, fmt)
        output_metadata = dict(metadata) if preserve_metadata else {}
        output_metadata = _without_lulynx_metadata(output_metadata)
    elif fmt in ROWWISE_FORMATS:
        output_state, quant_metadata, stats = quantize_rowwise_state_dict(
            state,
            fmt,
            decode_dtype=decode_dtype,
        )
        output_metadata = dict(metadata) if preserve_metadata else {}
        output_metadata = _without_lulynx_metadata(output_metadata)
        output_metadata.update(quant_metadata)
    else:
        raise ValueError(f"unsupported quantization format: {fmt}")

    if fmt not in GGUF_FORMATS:
        from safetensors.torch import save_file

        save_file(output_state, str(dst), metadata=_stringify_metadata(output_metadata))
        del output_state
        del output_metadata
        state.clear()
        metadata.clear()
        gc.collect()

    output_size = os.path.getsize(dst)
    converted = int(stats.get("converted_tensors") or 0)
    skipped = int(stats.get("skipped_tensors") or 0)
    result = {
        "success": True,
        "input_path": str(src),
        "output_path": str(dst),
        "quant_format": fmt,
        "output_format": "gguf" if fmt in GGUF_FORMATS else "safetensors",
        "decode_dtype": decode_dtype,
        "supported_formats": sorted(SUPPORTED_MODEL_QUANT_FORMATS),
        "tensor_count": tensor_count,
        "converted_tensors": converted,
        "skipped_tensors": skipped,
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "compression_ratio": round(output_size / input_size, 6) if input_size else 0,
    }
    if fmt in ROWWISE_FORMATS:
        result["native_converted_tensors"] = int(stats.get("native_converted_tensors") or 0)
        result["rowwise_provider"] = str(stats.get("rowwise_provider") or "torch")
    if fmt in SUPPORTED_QUANT_FORMATS:
        result["validation"] = validate_quantized_model_file(src, dst, fmt, decode_dtype=decode_dtype)
    if fmt in GGUF_FORMATS:
        result["gguf_provider"] = str(stats.get("gguf_provider") or "python")
        if stats.get("gguf_arch"):
            result["gguf_arch"] = str(stats["gguf_arch"])
        if stats.get("gguf_quant_type"):
            result["gguf_quant_type"] = str(stats["gguf_quant_type"])
        if stats.get("gguf_quantizer"):
            result["gguf_quantizer"] = str(stats["gguf_quantizer"])
        if stats.get("warnings"):
            result["warnings"] = list(stats["warnings"])
    return result


def normalize_model_quant_format(value: Any) -> str:
    text = str(value or "fp16").strip().lower().replace("-", "_")
    text = _GGUF_ALIASES.get(text, text)
    if text in GGUF_FORMATS:
        return text
    return normalize_quant_format(text)


def _validate_paths(src: Path, dst: Path, *, overwrite: bool, output_format: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")
    expected_suffix = ".gguf" if output_format == "gguf" else ".safetensors"
    if dst.suffix.lower() != expected_suffix:
        raise ValueError(f"output_path must end with {expected_suffix}")
    try:
        same_file = src.resolve() == dst.resolve()
    except OSError:
        same_file = False
    if same_file:
        raise ValueError("output_path must be different from input_path")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst}")


def _load_state_and_metadata(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors import safe_open

        try:
            from core.lulynx_trainer.safetensors_loader import load_safetensors
        except ImportError:
            from backend.core.lulynx_trainer.safetensors_loader import load_safetensors

        with safe_open(str(path), framework="pt", device="cpu") as probe:
            metadata = dict(probe.metadata() or {})
        if metadata.get(SCHEMA_KEY):
            return load_safetensors(str(path), device="cpu", disable_mmap=False), _without_lulynx_metadata(metadata)

        state: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state[str(key)] = handle.get_tensor(key).detach().cpu().contiguous()
        return state, metadata

    if suffix in {".pt", ".pth", ".ckpt"}:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "state_dict" in data and isinstance(data["state_dict"], dict):
            data = data["state_dict"]
        if not isinstance(data, dict):
            raise ValueError("PyTorch file does not contain a state dict")
        return {
            str(key): value.detach().cpu().contiguous()
            for key, value in data.items()
            if torch.is_tensor(value)
        }, {}

    raise ValueError("input_path must be a .safetensors, .pt, .pth, or .ckpt file")


def _export_gguf_state_dict(
    state: dict[str, torch.Tensor],
    dst: Path,
    fmt: str,
    *,
    arch: str,
    name: str,
    source_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        import numpy as np
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "GGUF export requires the optional gguf module in Launcher support dependency. "
            "Install or repair Launcher 支持依赖 with gguf included."
        ) from exc

    writer_cls = getattr(gguf, "GGUFWriter", None)
    if writer_cls is None:
        raise RuntimeError("Installed gguf module does not expose GGUFWriter")

    qtype = _gguf_quant_type(gguf, fmt)
    target_dtype = np.float16 if fmt == "gguf_f16" else np.float32
    plan = plan_gguf_export(state, arch=arch, name=name, source_path=source_path, metadata=metadata)
    writer = writer_cls(str(dst), plan.arch)
    converted = 0
    skipped = int(plan.skipped_tensors or 0)
    try:
        _apply_gguf_metadata(writer, plan.metadata)
        _apply_gguf_file_type(writer, gguf, fmt, qtype)
        for key, tensor in plan.tensors.items():
            if not torch.is_tensor(tensor):
                skipped += 1
                continue
            value = tensor.detach().cpu().contiguous()
            if qtype is not None and value.is_floating_point():
                array = value.to(torch.float32).numpy().astype(np.float32, copy=False)
                if _can_quantize_gguf_array(gguf, array, qtype):
                    quantized = gguf.quantize(array, qtype)
                    writer.add_tensor(str(key), quantized, raw_dtype=qtype)
                    converted += 1
                    continue
                writer.add_tensor(str(key), array)
                skipped += 1
                continue
            if value.is_floating_point():
                value = value.to(torch.float16 if fmt == "gguf_f16" else torch.float32)
                converted += 1
            else:
                skipped += 1
            writer.add_tensor(str(key), value.numpy().astype(target_dtype, copy=False) if value.is_floating_point() else value.numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        close = getattr(writer, "close", None)
        if callable(close):
            close()
    if not dst.is_file():
        raise RuntimeError("GGUF writer completed without creating the output file")
    result = {
        "converted_tensors": converted,
        "skipped_tensors": skipped,
        "format": fmt,
        "gguf_provider": "python",
        "gguf_arch": plan.arch,
    }
    if qtype is not None:
        result["gguf_quant_type"] = str(qtype.name)
    if plan.warnings:
        result["warnings"] = list(plan.warnings)
    return result


def _apply_gguf_metadata(writer: Any, metadata: dict[str, Any]) -> None:
    method_by_key = {
        "name": "add_name",
        "context_length": "add_context_length",
        "embedding_length": "add_embedding_length",
        "block_count": "add_block_count",
        "feed_forward_length": "add_feed_forward_length",
        "head_count": "add_head_count",
        "head_count_kv": "add_head_count_kv",
        "rope_dimension_count": "add_rope_dimension_count",
        "layer_norm_rms_eps": "add_layer_norm_rms_eps",
        "rope_freq_base": "add_rope_freq_base",
        "vocab_size": "add_vocab_size",
    }
    for key, value in metadata.items():
        method = getattr(writer, method_by_key.get(str(key), ""), None)
        if callable(method):
            method(value)


def _apply_gguf_file_type(writer: Any, gguf: Any, fmt: str, qtype: Any | None) -> None:
    method = getattr(writer, "add_file_type", None)
    if not callable(method):
        return
    if qtype is not None:
        method(qtype)
        return
    quant_type = gguf.GGMLQuantizationType.F16 if fmt == "gguf_f16" else gguf.GGMLQuantizationType.F32
    method(quant_type)


def _gguf_quant_type(gguf: Any, fmt: str) -> Any | None:
    if fmt == "gguf_q8_0":
        return gguf.GGMLQuantizationType.Q8_0
    return None


def _can_quantize_gguf_array(gguf: Any, array: Any, qtype: Any) -> bool:
    try:
        block_size = int(gguf.GGML_QUANT_SIZES[qtype][0])
    except Exception:
        return False
    shape = getattr(array, "shape", ())
    return bool(shape) and int(shape[-1]) > 0 and int(shape[-1]) % block_size == 0


def _gguf_quantizer_search_roots(src: Path, dst: Path) -> list[Path]:
    roots = [Path(sys.executable).resolve().parent, src.parent.resolve(), dst.parent.resolve(), Path.cwd().resolve()]
    return list(dict.fromkeys(roots))


def _count_gguf_tensors(path: Path) -> int:
    try:
        import gguf

        return len(gguf.GGUFReader(str(path)).tensors)
    except Exception:
        return 0


def _coerce_gguf_metadata(value: dict[str, Any] | str | None) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError("gguf_metadata must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("gguf_metadata must be a JSON object")
        return parsed
    raise ValueError("gguf_metadata must be a JSON object")


def _stringify_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        output[str(key)] = str(value)
    return output


def _without_lulynx_metadata(metadata: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in metadata.items()
        if not str(key).startswith("lulynx.quantization.")
    }


__all__ = ["GGUF_FORMATS", "SUPPORTED_MODEL_QUANT_FORMATS", "normalize_model_quant_format", "quantize_model_file"]
