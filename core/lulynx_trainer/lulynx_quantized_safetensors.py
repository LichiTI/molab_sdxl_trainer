"""Lulynx safetensors quantization contract.

The fp16/bf16/fp8 variants remain plain safetensors dtype conversions.  The
rowwise formats store a compact payload plus metadata and are decoded by the
trainer-owned safetensors loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

try:
    from .native_rowwise_quant import try_quantize_rowwise_tensor_native
except ImportError:  # pragma: no cover - supports direct file loading in smoke scripts
    try:
        from core.lulynx_trainer.native_rowwise_quant import try_quantize_rowwise_tensor_native
    except ImportError:
        try_quantize_rowwise_tensor_native = None  # type: ignore[assignment]

SCHEMA_KEY = "lulynx.quantization.schema"
FORMAT_KEY = "lulynx.quantization.format"
TENSORS_KEY = "lulynx.quantization.tensors"
SCHEMA_VERSION = "1"
QUANT_PREFIX = "__lulynx_quantized__"

PLAIN_FORMATS = {"fp16", "bf16", "fp8_e4m3fn"}
ROWWISE_FORMATS = {"lulynx_int8_rowwise", "lulynx_uint4_rowwise"}
SUPPORTED_QUANT_FORMATS = PLAIN_FORMATS | ROWWISE_FORMATS

_FORMAT_ALIASES = {
    "float16": "fp16",
    "raw_fp16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "raw_bf16": "bf16",
    "fp8": "fp8_e4m3fn",
    "fp8_e4m3": "fp8_e4m3fn",
    "float8": "fp8_e4m3fn",
    "float8_e4m3": "fp8_e4m3fn",
    "float8_e4m3fn": "fp8_e4m3fn",
    "int8": "lulynx_int8_rowwise",
    "int8_rowwise": "lulynx_int8_rowwise",
    "lulynx_int8": "lulynx_int8_rowwise",
    "uint4": "lulynx_uint4_rowwise",
    "int4": "lulynx_uint4_rowwise",
    "uint4_rowwise": "lulynx_uint4_rowwise",
    "lulynx_uint4": "lulynx_uint4_rowwise",
}

_DECODE_DTYPE_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class QuantizedTensorEntry:
    key: str
    format: str
    shape: list[int]
    original_dtype: str
    q_key: str
    scale_key: str
    offset_key: str = ""
    decode_dtype: str = "fp16"
    original_cols: int = 0
    quantization_variant: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "key": self.key,
            "format": self.format,
            "shape": list(self.shape),
            "original_dtype": self.original_dtype,
            "q_key": self.q_key,
            "scale_key": self.scale_key,
            "decode_dtype": self.decode_dtype,
            "original_cols": int(self.original_cols),
        }
        if self.offset_key:
            payload["offset_key"] = self.offset_key
        if self.quantization_variant:
            payload["quantization_variant"] = self.quantization_variant
        return payload


def normalize_quant_format(value: Any) -> str:
    fmt = str(value or "fp16").strip().lower().replace("-", "_")
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in SUPPORTED_QUANT_FORMATS:
        raise ValueError(f"unsupported quantization format: {value!r}")
    return fmt


def normalize_decode_dtype(value: Any) -> str:
    text = str(value or "fp16").strip().lower().replace("-", "_")
    if text not in _DECODE_DTYPE_ALIASES:
        raise ValueError(f"unsupported decode dtype: {value!r}")
    return "bf16" if text == "bfloat16" else "fp32" if text == "float32" else "fp16" if text in {"float16", "half"} else text


def torch_dtype_for_plain_format(fmt: str) -> torch.dtype:
    fmt = normalize_quant_format(fmt)
    if fmt == "fp16":
        return torch.float16
    if fmt == "bf16":
        return torch.bfloat16
    if fmt == "fp8_e4m3fn":
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")
        return dtype
    raise ValueError(f"not a plain dtype format: {fmt}")


def decode_dtype_to_torch(value: Any) -> torch.dtype:
    return _DECODE_DTYPE_ALIASES[normalize_decode_dtype(value)]


def is_lulynx_quantized_metadata(metadata: dict[str, str] | None) -> bool:
    return bool(metadata and metadata.get(SCHEMA_KEY) == SCHEMA_VERSION and metadata.get(TENSORS_KEY))


def parse_quantized_tensor_entries(metadata: dict[str, str] | None) -> list[QuantizedTensorEntry]:
    if not is_lulynx_quantized_metadata(metadata):
        return []
    raw = str((metadata or {}).get(TENSORS_KEY) or "[]")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("Lulynx quantized safetensors metadata must contain a tensor list")
    entries: list[QuantizedTensorEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        entries.append(
            QuantizedTensorEntry(
                key=str(item.get("key") or ""),
                format=normalize_quant_format(item.get("format") or ""),
                shape=[int(dim) for dim in item.get("shape") or []],
                original_dtype=str(item.get("original_dtype") or ""),
                q_key=str(item.get("q_key") or ""),
                scale_key=str(item.get("scale_key") or ""),
                offset_key=str(item.get("offset_key") or item.get("zero_key") or ""),
                decode_dtype=normalize_decode_dtype(item.get("decode_dtype") or "fp16"),
                original_cols=int(item.get("original_cols") or 0),
                quantization_variant=str(item.get("quantization_variant") or ""),
            )
        )
    return [entry for entry in entries if entry.key and entry.q_key and entry.scale_key]


def can_rowwise_quantize(tensor: torch.Tensor) -> bool:
    return torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dim() >= 2 and tensor.numel() > 0


def quantize_plain_state_dict(
    state: dict[str, torch.Tensor],
    fmt: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    dtype = torch_dtype_for_plain_format(fmt)
    converted: dict[str, torch.Tensor] = {}
    stats = {"converted_tensors": 0, "skipped_tensors": 0, "format": normalize_quant_format(fmt)}
    for key, tensor in state.items():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            converted[key] = tensor.detach().cpu().to(dtype=dtype).contiguous()
            stats["converted_tensors"] += 1
        else:
            converted[key] = tensor.detach().cpu().contiguous() if torch.is_tensor(tensor) else tensor
            stats["skipped_tensors"] += 1
    return converted, stats


def quantize_rowwise_state_dict(
    state: dict[str, torch.Tensor],
    fmt: str,
    *,
    decode_dtype: str = "fp16",
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    fmt = normalize_quant_format(fmt)
    if fmt not in ROWWISE_FORMATS:
        raise ValueError(f"not a rowwise Lulynx format: {fmt}")
    decode_dtype = normalize_decode_dtype(decode_dtype)
    _ensure_no_reserved_keys(state)
    output: dict[str, torch.Tensor] = {}
    entries: list[QuantizedTensorEntry] = []
    stats = {
        "converted_tensors": 0,
        "skipped_tensors": 0,
        "format": fmt,
        "native_converted_tensors": 0,
        "rowwise_provider": "torch",
    }
    for index, (key, tensor) in enumerate(state.items()):
        if not can_rowwise_quantize(tensor):
            output[key] = tensor.detach().cpu().contiguous() if torch.is_tensor(tensor) else tensor
            stats["skipped_tensors"] += 1
            continue
        q_key = f"{QUANT_PREFIX}.{index}.q"
        scale_key = f"{QUANT_PREFIX}.{index}.scale"
        q, scale, offset, original_cols, provider, variant = _quantize_rowwise_tensor(tensor.detach().cpu(), fmt)
        output[q_key] = q
        output[scale_key] = scale
        offset_key = ""
        if offset is not None:
            offset_key = f"{QUANT_PREFIX}.{index}.offset"
            output[offset_key] = offset
        entries.append(
            QuantizedTensorEntry(
                key=key,
                format=fmt,
                shape=[int(dim) for dim in tensor.shape],
                original_dtype=str(tensor.dtype).replace("torch.", ""),
                q_key=q_key,
                scale_key=scale_key,
                offset_key=offset_key,
                decode_dtype=decode_dtype,
                original_cols=original_cols,
                quantization_variant=variant,
            )
        )
        stats["converted_tensors"] += 1
        if provider != "torch":
            stats["native_converted_tensors"] += 1
            stats["rowwise_provider"] = provider
    metadata = {
        SCHEMA_KEY: SCHEMA_VERSION,
        FORMAT_KEY: fmt,
        TENSORS_KEY: json.dumps([entry.to_dict() for entry in entries], ensure_ascii=False, separators=(",", ":")),
    }
    return output, metadata, stats


def dequantize_state_dict(
    state: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
) -> dict[str, torch.Tensor]:
    entries = parse_quantized_tensor_entries(metadata)
    if not entries:
        return state
    reserved_keys = {entry.q_key for entry in entries} | {entry.scale_key for entry in entries}
    reserved_keys.update(entry.offset_key for entry in entries if entry.offset_key)
    decoded = {key: value for key, value in state.items() if key not in reserved_keys and not key.startswith(f"{QUANT_PREFIX}.")}
    for entry in entries:
        if entry.q_key not in state or entry.scale_key not in state:
            raise KeyError(f"quantized tensor payload missing for {entry.key}")
        offset = state[entry.offset_key] if entry.offset_key else None
        decoded[entry.key] = decode_quantized_tensor(entry, state[entry.q_key], state[entry.scale_key], offset)
    return decoded


def decode_quantized_tensor(
    entry: QuantizedTensorEntry,
    q: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    dtype = decode_dtype_to_torch(entry.decode_dtype)
    if entry.format == "lulynx_int8_rowwise":
        return _decode_int8_rowwise(q, scale, entry.shape, dtype)
    if entry.format == "lulynx_uint4_rowwise":
        return _decode_uint4_rowwise(q, scale, offset, entry.shape, entry.original_cols, dtype)
    raise ValueError(f"unsupported Lulynx quantized tensor format: {entry.format}")


def _quantize_rowwise_tensor(tensor: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, str, str]:
    if try_quantize_rowwise_tensor_native is not None:
        native_result = try_quantize_rowwise_tensor_native(tensor, fmt)
        if native_result is not None:
            q, scale, original_cols, metadata = native_result
            return q, scale, None, original_cols, str(metadata.get("provider") or "lulynx_native.rowwise_quant_v1"), str(metadata.get("variant") or "symmetric_absmax_v1")

    shape = tuple(int(dim) for dim in tensor.shape)
    rows = shape[0]
    flat = tensor.float().contiguous().view(rows, -1)
    if not torch.isfinite(flat).all():
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    if fmt == "lulynx_int8_rowwise":
        scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(flat / scale).clamp(-127, 127).to(torch.int8).contiguous()
        return q, scale.to(torch.float16).contiguous(), None, flat.shape[1], "torch", "symmetric_absmax_v1"
    return _quantize_uint4_affine_blocks(tensor)


def _quantize_uint4_affine_blocks(tensor: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]:
    flat = tensor.float().contiguous().view(-1)
    if not torch.isfinite(flat).all():
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    rows = max((int(flat.numel()) + block_size - 1) // block_size, 1)
    padded = rows * block_size
    if padded != int(flat.numel()):
        pad_value = flat[-1] if flat.numel() else flat.new_tensor(0.0)
        flat = torch.cat([flat, pad_value.expand(padded - int(flat.numel()))])
    blocks = flat.view(rows, block_size)
    offset = blocks.amin(dim=1, keepdim=True)
    maximum = blocks.amax(dim=1, keepdim=True)
    scale = (maximum - offset).clamp_min(1e-8) / 15.0
    q = torch.round((blocks - offset) / scale).clamp(0, 15).to(torch.uint8)
    packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).contiguous()
    return (
        packed,
        scale.to(torch.float16).contiguous(),
        offset.to(torch.float16).contiguous(),
        block_size,
        "torch",
        "affine_uint4_blockwise_v2",
    )


def _decode_int8_rowwise(q: torch.Tensor, scale: torch.Tensor, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    result = q.to(dtype=dtype) * scale.to(dtype=dtype)
    return result.contiguous().view(*shape)


def _decode_uint4_rowwise(
    packed: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    shape: list[int],
    original_cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.uint8)
    q[:, 0::2] = low
    q[:, 1::2] = high
    q = q[:, : int(original_cols)]
    if offset is not None:
        result = q.to(dtype=dtype) * scale.to(dtype=dtype) + offset.to(dtype=dtype)
    else:
        result = (q.to(dtype=dtype) - 8.0) * scale.to(dtype=dtype)
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return result.contiguous().view(-1)[:numel].view(*shape)


def _ensure_no_reserved_keys(state: dict[str, torch.Tensor]) -> None:
    reserved = f"{QUANT_PREFIX}."
    collisions = [key for key in state if str(key).startswith(reserved)]
    if collisions:
        raise ValueError(f"input already contains reserved Lulynx quantization keys: {collisions[:3]}")


__all__ = [
    "FORMAT_KEY",
    "PLAIN_FORMATS",
    "ROWWISE_FORMATS",
    "SCHEMA_KEY",
    "SCHEMA_VERSION",
    "SUPPORTED_QUANT_FORMATS",
    "TENSORS_KEY",
    "can_rowwise_quantize",
    "decode_quantized_tensor",
    "dequantize_state_dict",
    "is_lulynx_quantized_metadata",
    "normalize_decode_dtype",
    "normalize_quant_format",
    "parse_quantized_tensor_entries",
    "quantize_plain_state_dict",
    "quantize_rowwise_state_dict",
]
