"""Experimental tensor-friendly PCIe transfer formats.

This module is a safe staging layer for Streaming Offload experiments.  It
does not hook into training by itself; callers opt in by packing frozen CPU
weights with a :class:`TransferFormatPolicy` and decoding them on the target
device before compute.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import statistics
from pathlib import Path
from typing import Any

import torch


VALID_TRANSFER_FORMATS = {
    "raw_fp16",
    "raw_bf16",
    "fp8_e4m3",
    "int8_rowwise",
    "uint4_rowwise",
}

STATIC_TRANSFER_FORMAT_ORDER = (
    "fp8_e4m3",
    "raw_bf16",
    "raw_fp16",
    "int8_rowwise",
    "uint4_rowwise",
)

_ALIASES = {
    "off": "raw_fp16",
    "none": "raw_fp16",
    "fp16": "raw_fp16",
    "float16": "raw_fp16",
    "half": "raw_fp16",
    "bf16": "raw_bf16",
    "bfloat16": "raw_bf16",
    "fp8": "fp8_e4m3",
    "float8": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "int8": "int8_rowwise",
    "i8": "int8_rowwise",
    "uint4": "uint4_rowwise",
    "int4": "uint4_rowwise",
    "nf4": "uint4_rowwise",
    "u4": "uint4_rowwise",
}


def normalize_transfer_format(value: Any, *, default: str = "raw_fp16") -> str:
    fmt = str(value or default).strip().lower().replace("-", "_")
    fmt = _ALIASES.get(fmt, fmt)
    if fmt not in VALID_TRANSFER_FORMATS:
        raise ValueError(f"unsupported transfer format: {value!r}")
    return fmt


def available_transfer_formats() -> dict[str, dict[str, Any]]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    return {
        "raw_fp16": {
            "available": True,
            "experimental": False,
            "bits_per_value": 16,
            "tensor_core_friendly": True,
            "decode_path": "h2d_cast",
            "training_format_score": 0.72,
        },
        "raw_bf16": {
            "available": True,
            "experimental": False,
            "bits_per_value": 16,
            "tensor_core_friendly": True,
            "decode_path": "h2d_cast",
            "training_format_score": 0.74,
        },
        "fp8_e4m3": {
            "available": fp8_dtype is not None,
            "experimental": True,
            "bits_per_value": 8,
            "torch_dtype": None if fp8_dtype is None else str(fp8_dtype).replace("torch.", ""),
            "tensor_core_friendly": True,
            "decode_path": "h2d_fp8_to_compute_dtype",
            "training_format_score": 0.86,
        },
        "int8_rowwise": {
            "available": True,
            "experimental": True,
            "bits_per_value": 8,
            "tensor_core_friendly": False,
            "decode_path": "rowwise_scale_multiply",
            "training_format_score": 0.68,
        },
        "uint4_rowwise": {
            "available": True,
            "experimental": True,
            "bits_per_value": 4,
            "tensor_core_friendly": False,
            "decode_path": "unpack_nibbles_then_scale",
            "training_format_score": 0.52,
        },
    }


def _static_transfer_format_ranking() -> list[dict[str, Any]]:
    formats = available_transfer_formats()
    order_index = {fmt: idx for idx, fmt in enumerate(STATIC_TRANSFER_FORMAT_ORDER)}
    rows = [
        {
            "format": fmt,
            "available": bool(meta.get("available", False)),
            "bits_per_value": int(meta.get("bits_per_value", 0) or 0),
            "tensor_core_friendly": bool(meta.get("tensor_core_friendly", False)),
            "decode_path": str(meta.get("decode_path", "")),
            "training_format_score": float(meta.get("training_format_score", 0.0) or 0.0),
            "recommendation_score": round(1.0 - float(meta.get("training_format_score", 0.0) or 0.0), 6),
            "recommendation_source": "static",
        }
        for fmt, meta in formats.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["available"]),
            order_index.get(str(row["format"]), 999),
            -float(row["training_format_score"]),
        ),
    )


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _metric(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row:
            value = _as_float(row.get(name))
            if value is not None:
                return value
    return None


def _load_benchmark_payload(benchmark: Any = None, benchmark_path: str | Path | None = None) -> Any:
    if benchmark_path is not None:
        return json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    if isinstance(benchmark, (str, Path)):
        path = Path(benchmark)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return benchmark


def _iter_benchmark_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if payload is None:
        return rows
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                rows.extend(_iter_benchmark_rows(item))
        return rows
    if not isinstance(payload, dict):
        return rows
    if "format" in payload:
        rows.append(dict(payload))
    for key in ("results", "formats", "ranked_formats", "recommendation", "recommendations"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.extend(_iter_benchmark_rows(item))
    cases = payload.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            shape = case.get("shape")
            for row in _iter_benchmark_rows(case):
                if shape is not None and "shape" not in row:
                    row["shape"] = shape
                rows.append(row)
    return rows


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _rank_from_benchmark(
    rows: list[dict[str, Any]],
    *,
    pack_weight: float,
    reuse_factor: float,
    error_weight: float,
    max_error_mae: float | None,
) -> list[dict[str, Any]]:
    availability = available_transfer_formats()
    by_format: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        try:
            fmt = normalize_transfer_format(row.get("format"))
        except ValueError:
            continue
        metrics = by_format.setdefault(
            fmt,
            {"pack_ms": [], "h2d_ms": [], "decode_ms": [], "total_ms": [], "error_mae": [], "transfer_mb": []},
        )
        aliases = {
            "pack_ms": ("pack_ms", "cpu_pack_ms"),
            "h2d_ms": ("h2d_ms", "copy_ms", "transfer_ms"),
            "decode_ms": ("decode_ms", "decode_h2d_ms", "unpack_ms"),
            "total_ms": ("total_ms", "latency_ms", "decode_h2d_matmul_ms", "decode_h2d_ms"),
            "error_mae": ("error_mae", "mae", "mean_abs_error"),
            "transfer_mb": ("transfer_mb", "payload_mb"),
        }
        for metric_name, names in aliases.items():
            value = _metric(row, *names)
            if value is not None:
                metrics[metric_name].append(value)

    ranked: list[dict[str, Any]] = []
    for fmt, values in by_format.items():
        meta = availability.get(fmt, {})
        pack_ms = _median(values["pack_ms"])
        h2d_ms = _median(values["h2d_ms"])
        decode_ms = _median(values["decode_ms"])
        total_ms = _median(values["total_ms"])
        if total_ms is None:
            parts = [value for value in (h2d_ms, decode_ms) if value is not None]
            total_ms = sum(parts) if parts else None
        if total_ms is None:
            continue
        error_mae = _median(values["error_mae"]) or 0.0
        error_penalty = max(error_mae - float(max_error_mae), 0.0) * float(error_weight) if max_error_mae is not None else error_mae * float(error_weight)
        amortized_pack_ms = float(pack_ms or 0.0) / max(float(reuse_factor or 1.0), 1.0)
        score = float(total_ms) + (amortized_pack_ms * float(pack_weight)) + error_penalty
        ranked.append(
            {
                "format": fmt,
                "available": bool(meta.get("available", False)),
                "bits_per_value": int(meta.get("bits_per_value", 0) or 0),
                "tensor_core_friendly": bool(meta.get("tensor_core_friendly", False)),
                "decode_path": str(meta.get("decode_path", "")),
                "pack_ms": None if pack_ms is None else round(float(pack_ms), 4),
                "amortized_pack_ms": round(float(amortized_pack_ms), 4),
                "reuse_factor": round(max(float(reuse_factor or 1.0), 1.0), 4),
                "h2d_ms": None if h2d_ms is None else round(float(h2d_ms), 4),
                "decode_ms": None if decode_ms is None else round(float(decode_ms), 4),
                "total_ms": round(float(total_ms), 4),
                "error_mae": round(float(error_mae), 8),
                "transfer_mb": None if not values["transfer_mb"] else round(float(_median(values["transfer_mb"]) or 0.0), 4),
                "recommendation_score": round(float(score), 6),
                "recommendation_source": "benchmark",
            }
        )
    ranked.sort(key=lambda row: (not bool(row["available"]), float(row["recommendation_score"]), int(row["bits_per_value"]), str(row["format"])))
    return ranked


def recommend_transfer_formats(
    benchmark: Any = None,
    *,
    benchmark_path: str | Path | None = None,
    pack_weight: float = 0.15,
    reuse_factor: float = 1.0,
    error_weight: float = 1000.0,
    max_error_mae: float | None = 0.02,
) -> list[dict[str, Any]]:
    """Rank PCIe transfer formats from benchmark rows, falling back to static guidance.

    Accepted benchmark rows may use either the standalone benchmark names
    (``cpu_pack_ms``, ``decode_h2d_ms``) or compact names such as
    ``pack_ms``, ``h2d_ms``, ``decode_ms`` and ``error_mae``. ``reuse_factor``
    amortizes one-time CPU packing cost for repeated frozen-weight transfers.
    """

    payload = _load_benchmark_payload(benchmark, benchmark_path)
    rows = _iter_benchmark_rows(payload)
    ranked = _rank_from_benchmark(
        rows,
        pack_weight=pack_weight,
        reuse_factor=max(float(reuse_factor or 1.0), 1.0),
        error_weight=error_weight,
        max_error_mae=max_error_mae,
    )
    if not ranked:
        return _static_transfer_format_ranking()

    seen = {str(row["format"]) for row in ranked}
    for fallback in _static_transfer_format_ranking():
        if str(fallback["format"]) not in seen:
            clone = dict(fallback)
            clone["recommendation_source"] = "static_fallback"
            ranked.append(clone)
    return ranked


def transfer_format_experiment_plan(
    benchmark: Any = None,
    *,
    benchmark_path: str | Path | None = None,
    reuse_factor: float = 1.0,
) -> dict[str, Any]:
    """Guidance for the TensorCore-friendly transfer-format route."""

    ranked = recommend_transfer_formats(benchmark, benchmark_path=benchmark_path, reuse_factor=reuse_factor)
    recommended_first = next(
        (str(row["format"]) for row in ranked if bool(row.get("available", False))),
        "raw_bf16",
    )
    return {
        "enabled": True,
        "route": "tensorcore_friendly_transfer_format",
        "recommended_first": recommended_first,
        "ranked_formats": ranked,
        "reuse_factor": round(max(float(reuse_factor or 1.0), 1.0), 4),
        "guardrails": [
            "only apply to CPU-pinned frozen Linear weights",
            "keep LoRA/adapter trainable parameters in their normal optimizer dtype",
            "amortize CPU pack cost only when the same frozen weights are reused across many steps",
            "benchmark H2D+decode latency separately from model quality before making it automatic",
        ],
    }


def _pin(tensor: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
    if not pin_memory or not torch.cuda.is_available():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


@dataclass(frozen=True)
class TransferFormatPolicy:
    """Opt-in policy for frozen tensor CPU storage and H2D decode."""

    format: str = "raw_fp16"
    pin_memory: bool = True
    experimental: bool = False

    def __post_init__(self) -> None:
        fmt = normalize_transfer_format(self.format)
        object.__setattr__(self, "format", fmt)
        if available_transfer_formats()[fmt]["experimental"] and not self.experimental:
            raise ValueError(f"transfer format {fmt!r} is experimental; pass experimental=True")


@dataclass
class PackedTensor:
    format: str
    payload: Any
    shape: tuple[int, ...]
    transfer_bytes: int
    metadata: dict[str, Any]

    @property
    def transfer_mb(self) -> float:
        return float(self.transfer_bytes) / (1024.0 * 1024.0)

    def to(self, device: torch.device | str, *, compute_dtype: torch.dtype | None = None) -> torch.Tensor:
        return decode_transfer_tensor(self, device=device, compute_dtype=compute_dtype)


def pack_tensor_for_transfer(tensor: torch.Tensor, policy: TransferFormatPolicy | str) -> PackedTensor:
    if isinstance(policy, str):
        policy = TransferFormatPolicy(format=policy, experimental=True)
    fmt = policy.format
    if fmt == "raw_fp16":
        payload = _pin(tensor.to(dtype=torch.float16).contiguous(), pin_memory=policy.pin_memory)
        return PackedTensor(fmt, payload, tuple(tensor.shape), _tensor_bytes(payload), {"dtype": "float16"})
    if fmt == "raw_bf16":
        payload = _pin(tensor.to(dtype=torch.bfloat16).contiguous(), pin_memory=policy.pin_memory)
        return PackedTensor(fmt, payload, tuple(tensor.shape), _tensor_bytes(payload), {"dtype": "bfloat16"})
    if fmt == "fp8_e4m3":
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")
        payload = _pin(tensor.to(dtype=fp8_dtype).contiguous(), pin_memory=policy.pin_memory)
        return PackedTensor(fmt, payload, tuple(tensor.shape), _tensor_bytes(payload), {"dtype": "float8_e4m3fn"})
    if fmt == "int8_rowwise":
        src = tensor.float().contiguous()
        scale = src.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(src / scale).clamp(-127, 127).to(torch.int8).contiguous()
        q = _pin(q, pin_memory=policy.pin_memory)
        scale = _pin(scale.to(torch.float16).contiguous(), pin_memory=policy.pin_memory)
        return PackedTensor(fmt, (q, scale), tuple(tensor.shape), _tensor_bytes(q) + _tensor_bytes(scale), {"scale": "rowwise_fp16"})
    if fmt == "uint4_rowwise":
        src = tensor.float().contiguous()
        rows, cols = src.shape
        scale = src.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
        q = torch.round(src / scale).clamp(-7, 7).to(torch.int16) + 8
        if cols % 2:
            q = torch.cat([q, torch.full((rows, 1), 8, dtype=torch.int16)], dim=1)
        q = q.to(torch.uint8)
        packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).contiguous()
        packed = _pin(packed, pin_memory=policy.pin_memory)
        scale = _pin(scale.to(torch.float16).contiguous(), pin_memory=policy.pin_memory)
        return PackedTensor(
            fmt,
            (packed, scale, int(cols)),
            tuple(tensor.shape),
            _tensor_bytes(packed) + _tensor_bytes(scale),
            {"scale": "rowwise_fp16", "packed": "two_signed_uint4_per_byte"},
        )
    raise ValueError(f"unsupported transfer format: {fmt}")


def decode_transfer_tensor(
    packed: PackedTensor,
    *,
    device: torch.device | str,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    device = torch.device(device)
    compute_dtype = compute_dtype or torch.float16
    fmt = normalize_transfer_format(packed.format)
    if fmt in {"raw_fp16", "raw_bf16"}:
        return packed.payload.to(device=device, dtype=compute_dtype, non_blocking=True)
    if fmt == "fp8_e4m3":
        return packed.payload.to(device=device, non_blocking=True).to(dtype=compute_dtype)
    if fmt == "int8_rowwise":
        q_cpu, scale_cpu = packed.payload
        q_gpu = q_cpu.to(device=device, non_blocking=True)
        scale_gpu = scale_cpu.to(device=device, dtype=compute_dtype, non_blocking=True)
        return q_gpu.to(dtype=compute_dtype) * scale_gpu
    if fmt == "uint4_rowwise":
        packed_cpu, scale_cpu, original_cols = packed.payload
        packed_gpu = packed_cpu.to(device=device, non_blocking=True)
        low = packed_gpu & 0x0F
        high = (packed_gpu >> 4) & 0x0F
        q = torch.empty((packed_gpu.shape[0], packed_gpu.shape[1] * 2), device=device, dtype=torch.uint8)
        q[:, 0::2] = low
        q[:, 1::2] = high
        q = q[:, : int(original_cols)]
        scale_gpu = scale_cpu.to(device=device, dtype=compute_dtype, non_blocking=True)
        return (q.to(dtype=compute_dtype) - 8.0) * scale_gpu
    raise ValueError(f"unsupported transfer format: {packed.format}")


def estimate_transfer_bytes(shape: tuple[int, ...] | torch.Size, fmt: str) -> int:
    fmt = normalize_transfer_format(fmt)
    if len(shape) != 2:
        raise ValueError("experimental transfer formats currently expect 2D Linear weights")
    rows, cols = int(shape[0]), int(shape[1])
    if fmt in {"raw_fp16", "raw_bf16"}:
        return rows * cols * 2
    if fmt in {"fp8_e4m3", "int8_rowwise"}:
        return rows * cols + rows * 2 if fmt == "int8_rowwise" else rows * cols
    if fmt == "uint4_rowwise":
        return rows * ((cols + 1) // 2) + rows * 2
    raise ValueError(f"unsupported transfer format: {fmt}")
