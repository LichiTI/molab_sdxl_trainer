"""Decode-only benchmark prototype for TensorCore-friendly transfer tiles.

This script stays out of the training path on purpose. It benchmarks a
reference ``tc_fp8_tile_v1`` CPU-pack + H2D + decode flow and compares it
against the existing plain ``fp8_e4m3`` transfer format.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional experiment dependency
    triton = None
    tl = None

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.tensorcore_transfer_kernel import (  # type: ignore[no-redef]
        TC_FP8_TILE_V1,
        TensorCoreTransferFormatSpec,
        tensorcore_kernel_roadmap,
    )
    from lulynx_trainer.transfer_format import (  # type: ignore[no-redef]
        TransferFormatPolicy,
        decode_transfer_tensor,
        pack_tensor_for_transfer,
    )
else:
    from .tensorcore_transfer_kernel import (
        TC_FP8_TILE_V1,
        TensorCoreTransferFormatSpec,
        tensorcore_kernel_roadmap,
    )
    from .transfer_format import (
        TransferFormatPolicy,
        decode_transfer_tensor,
        pack_tensor_for_transfer,
    )


FP8_E4M3_MAX = 448.0

SHAPE_PRESETS: dict[str, list[tuple[int, int]]] = {
    "single": [],
    "real_linear_short": [
        (768, 768),
        (1024, 1024),
        (2048, 2048),
        (3072, 1024),
        (1024, 3072),
    ],
    "sd_attention_short": [
        (320, 320),
        (640, 640),
        (1280, 1280),
        (2560, 1280),
        (1280, 2560),
    ],
    "dit_linear_short": [
        (1152, 1152),
        (3072, 3072),
        (4608, 1536),
        (1536, 4608),
    ],
}


if triton is not None and tl is not None:
    @triton.jit
    def _tc_fp8_tile_decode_kernel(
        payload,
        scales,
        output,
        rows: tl.constexpr,
        cols: tl.constexpr,
        col_tiles: tl.constexpr,
        tile_m: tl.constexpr,
        tile_k: tl.constexpr,
        block_size: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        tile_row = tile_id // col_tiles
        tile_col = tile_id - tile_row * col_tiles
        offsets = tl.arange(0, block_size)
        row_in_tile = offsets // tile_k
        col_in_tile = offsets - row_in_tile * tile_k
        out_row = tile_row * tile_m + row_in_tile
        out_col = tile_col * tile_k + col_in_tile
        mask = (row_in_tile < tile_m) & (col_in_tile < tile_k) & (out_row < rows) & (out_col < cols)
        values = tl.load(payload + tile_id * block_size + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scales + tile_id)
        decoded = values * scale
        tl.store(output + out_row * cols + out_col, decoded, mask=mask)

    @triton.jit
    def _tc_fp8_tile_fused_matmul_kernel(
        activation,
        payload,
        scales,
        output,
        batch: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        col_tiles: tl.constexpr,
        tile_m: tl.constexpr,
        tile_k: tl.constexpr,
        block_b: tl.constexpr,
    ):
        batch_block = tl.program_id(0)
        row_tile = tl.program_id(1)
        offs_b = batch_block * block_b + tl.arange(0, block_b)
        offs_m = tl.arange(0, tile_m)
        offs_k = tl.arange(0, tile_k)
        out_rows = row_tile * tile_m + offs_m
        acc = tl.zeros((block_b, tile_m), dtype=tl.float32)

        for col_tile in range(0, col_tiles):
            global_k = col_tile * tile_k + offs_k
            act = tl.load(
                activation + offs_b[:, None] * cols + global_k[None, :],
                mask=(offs_b[:, None] < batch) & (global_k[None, :] < cols),
                other=0.0,
            )
            tile_id = row_tile * col_tiles + col_tile
            raw = tl.load(
                payload + tile_id * tile_m * tile_k + offs_m[:, None] * tile_k + offs_k[None, :],
                mask=(out_rows[:, None] < rows) & (global_k[None, :] < cols),
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(scales + tile_id).to(tl.float32)
            weight = (raw * scale).to(tl.float16)
            acc += tl.dot(act, tl.trans(weight))

        tl.store(
            output + offs_b[:, None] * rows + out_rows[None, :],
            acc,
            mask=(offs_b[:, None] < batch) & (out_rows[None, :] < rows),
        )

    @triton.jit
    def _tc_fp8_tile_fused_matmul_rowgroup_kernel(
        activation,
        payload,
        scales,
        output,
        batch: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        row_tiles: tl.constexpr,
        col_tiles: tl.constexpr,
        tile_m: tl.constexpr,
        tile_k: tl.constexpr,
        block_b: tl.constexpr,
        row_group: tl.constexpr,
        block_m: tl.constexpr,
    ):
        batch_block = tl.program_id(0)
        row_group_id = tl.program_id(1)
        row_tile_base = row_group_id * row_group
        offs_b = batch_block * block_b + tl.arange(0, block_b)
        offs_m = tl.arange(0, block_m)
        offs_k = tl.arange(0, tile_k)
        local_row_tile = offs_m // tile_m
        local_m = offs_m - local_row_tile * tile_m
        source_row_tile = row_tile_base + local_row_tile
        out_rows = source_row_tile * tile_m + local_m
        acc = tl.zeros((block_b, block_m), dtype=tl.float32)

        for col_tile in range(0, col_tiles):
            global_k = col_tile * tile_k + offs_k
            act = tl.load(
                activation + offs_b[:, None] * cols + global_k[None, :],
                mask=(offs_b[:, None] < batch) & (global_k[None, :] < cols),
                other=0.0,
            )
            tile_id = source_row_tile[:, None] * col_tiles + col_tile
            raw = tl.load(
                payload + tile_id * tile_m * tile_k + local_m[:, None] * tile_k + offs_k[None, :],
                mask=(source_row_tile[:, None] < row_tiles) & (out_rows[:, None] < rows) & (global_k[None, :] < cols),
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(scales + source_row_tile * col_tiles + col_tile, mask=source_row_tile < row_tiles, other=0.0).to(tl.float32)
            weight = (raw * scale[:, None]).to(tl.float16)
            acc += tl.dot(act, tl.trans(weight))

        tl.store(
            output + offs_b[:, None] * rows + out_rows[None, :],
            acc,
            mask=(offs_b[:, None] < batch) & (source_row_tile[None, :] < row_tiles) & (out_rows[None, :] < rows),
        )


@dataclass
class PackedTensorCoreTileTensor:
    spec_name: str
    payload: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    tile_grid: tuple[int, int]
    payload_bytes: int
    scale_bytes: int
    aligned_bytes: int
    metadata: dict[str, Any]

    @property
    def transfer_mb(self) -> float:
        return float(self.payload_bytes + self.scale_bytes) / (1024.0 * 1024.0)

    @property
    def aligned_transfer_mb(self) -> float:
        return float(self.aligned_bytes) / (1024.0 * 1024.0)

    def as_summary(self) -> dict[str, Any]:
        return {
            "spec": self.spec_name,
            "original_shape": [int(self.original_shape[0]), int(self.original_shape[1])],
            "padded_shape": [int(self.padded_shape[0]), int(self.padded_shape[1])],
            "tile_shape": [int(self.tile_shape[0]), int(self.tile_shape[1])],
            "tile_grid": [int(self.tile_grid[0]), int(self.tile_grid[1])],
            "payload_bytes": int(self.payload_bytes),
            "scale_bytes": int(self.scale_bytes),
            "actual_transfer_bytes": int(self.payload_bytes + self.scale_bytes),
            "aligned_bytes": int(self.aligned_bytes),
            "transfer_mb": round(float(self.transfer_mb), 6),
            "aligned_transfer_mb": round(float(self.aligned_transfer_mb), 6),
            "metadata": dict(self.metadata),
        }


def _dtype(name: str) -> torch.dtype:
    value = str(name or "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_cpu_ms(fn: Callable[[], object], *, iters: int) -> tuple[float, object]:
    samples: list[float] = []
    result = None
    for _ in range(max(int(iters), 1)):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), result


def _time_cuda_ms(fn: Callable[[], object], *, device: torch.device, iters: int, warmup: int) -> tuple[float, object]:
    result = None
    for _ in range(max(int(warmup), 0)):
        result = fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(max(int(iters), 1)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples), result


def pack_tc_fp8_tile_reference(
    tensor: torch.Tensor,
    *,
    spec: TensorCoreTransferFormatSpec | None = None,
) -> PackedTensorCoreTileTensor:
    spec = spec or TC_FP8_TILE_V1
    if str(spec.name) != "tc_fp8_tile_v1":
        raise ValueError(f"reference packer currently supports only tc_fp8_tile_v1, got {spec.name!r}")
    if tensor.ndim != 2:
        raise ValueError("tc_fp8_tile_v1 expects a 2D Linear weight tensor")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")

    rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
    tile_m = int(spec.tile_m)
    tile_k = int(spec.tile_k)
    row_tiles = (rows + tile_m - 1) // tile_m
    col_tiles = (cols + tile_k - 1) // tile_k
    padded_rows = row_tiles * tile_m
    padded_cols = col_tiles * tile_k

    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.float32)
    padded[:rows, :cols] = tensor.float()
    tiles = padded.view(row_tiles, tile_m, col_tiles, tile_k).permute(0, 2, 1, 3).contiguous()
    scale = tiles.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8) / FP8_E4M3_MAX
    normalized = tiles / scale
    payload = normalized.to(dtype=fp8_dtype).contiguous()
    scale_fp16 = scale.to(dtype=torch.float16).contiguous()
    tile_count = int(row_tiles * col_tiles)
    payload_bytes = int(payload.numel() * payload.element_size())
    scale_bytes = int(scale_fp16.numel() * scale_fp16.element_size())
    aligned_bytes = tile_count * int(spec.tile_bytes)
    padding_ratio = 0.0
    if padded_rows and padded_cols:
        padding_ratio = 1.0 - ((rows * cols) / float(padded_rows * padded_cols))
    return PackedTensorCoreTileTensor(
        spec_name=str(spec.name),
        payload=payload,
        scales=scale_fp16,
        original_shape=(rows, cols),
        padded_shape=(padded_rows, padded_cols),
        tile_shape=(tile_m, tile_k),
        tile_grid=(row_tiles, col_tiles),
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        aligned_bytes=aligned_bytes,
        metadata={
            "payload_dtype": str(spec.payload_dtype),
            "scale_dtype": str(spec.scale_dtype),
            "tile_count": tile_count,
            "padding_ratio": round(float(padding_ratio), 8),
            "layout": "row_tile_major",
            "decode_target": str(spec.fused_decode_target),
            "reference_only": True,
        },
    )


def decode_tc_fp8_tile_reference(
    packed: PackedTensorCoreTileTensor,
    *,
    device: torch.device | str,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    device = torch.device(device)
    payload_gpu = packed.payload.to(device=device, non_blocking=True)
    scales_gpu = packed.scales.to(device=device, dtype=compute_dtype, non_blocking=True)
    decoded_tiles = payload_gpu.to(dtype=compute_dtype) * scales_gpu
    row_tiles, col_tiles = packed.tile_grid
    tile_m, tile_k = packed.tile_shape
    decoded = decoded_tiles.permute(0, 2, 1, 3).contiguous().view(row_tiles * tile_m, col_tiles * tile_k)
    rows, cols = packed.original_shape
    return decoded[:rows, :cols].contiguous()


def triton_tc_fp8_tile_decode_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


def decode_tc_fp8_tile_triton(
    packed: PackedTensorCoreTileTensor,
    *,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    implementation: str = "triton_decode_v0",
) -> torch.Tensor:
    if not triton_tc_fp8_tile_decode_available():
        raise RuntimeError("Triton is not available for tc_fp8_tile_v1 decode")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Triton tc_fp8_tile_v1 decode requires a CUDA device")
    if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported compute dtype for Triton decode: {compute_dtype}")

    rows, cols = packed.original_shape
    row_tiles, col_tiles = packed.tile_grid
    tile_m, tile_k = packed.tile_shape
    block_size = int(tile_m) * int(tile_k)
    payload_gpu = packed.payload.contiguous().view(-1).to(device=device, non_blocking=True)
    if implementation == "triton_decode_v1":
        scales_gpu = packed.scales.contiguous().view(-1).to(device=device, dtype=compute_dtype, non_blocking=True)
        num_warps = 1
    else:
        scales_gpu = packed.scales.contiguous().view(-1).to(device=device, dtype=torch.float32, non_blocking=True)
        num_warps = 4
    output = torch.empty((int(rows), int(cols)), device=device, dtype=compute_dtype)
    grid = (int(row_tiles) * int(col_tiles),)
    _tc_fp8_tile_decode_kernel[grid](
        payload_gpu,
        scales_gpu,
        output,
        int(rows),
        int(cols),
        int(col_tiles),
        int(tile_m),
        int(tile_k),
        block_size,
        num_warps=num_warps,
    )
    return output


def fused_matmul_tc_fp8_tile_triton(
    activation: torch.Tensor,
    packed: PackedTensorCoreTileTensor,
    *,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    block_b: int = 16,
    num_warps: int = 4,
) -> torch.Tensor:
    if not triton_tc_fp8_tile_decode_available():
        raise RuntimeError("Triton is not available for tc_fp8_tile_v1 fused matmul")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Triton tc_fp8_tile_v1 fused matmul requires a CUDA device")
    if activation.ndim != 2:
        raise ValueError("activation must be a 2D tensor")
    if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported compute dtype for Triton fused matmul: {compute_dtype}")

    rows, cols = packed.original_shape
    row_tiles, col_tiles = packed.tile_grid
    tile_m, tile_k = packed.tile_shape
    batch = int(activation.shape[0])
    if int(activation.shape[1]) != int(cols):
        raise ValueError(f"activation cols {activation.shape[1]} != packed cols {cols}")

    act_gpu = activation.to(device=device, dtype=compute_dtype, non_blocking=True).contiguous()
    payload_gpu = packed.payload.contiguous().view(-1).to(device=device, non_blocking=True)
    scales_gpu = packed.scales.contiguous().view(-1).to(device=device, dtype=compute_dtype, non_blocking=True)
    output = torch.empty((batch, int(rows)), device=device, dtype=compute_dtype)
    grid = (triton.cdiv(batch, int(block_b)), int(row_tiles))
    _tc_fp8_tile_fused_matmul_kernel[grid](
        act_gpu,
        payload_gpu,
        scales_gpu,
        output,
        batch,
        int(rows),
        int(cols),
        int(col_tiles),
        int(tile_m),
        int(tile_k),
        int(block_b),
        num_warps=int(num_warps),
    )
    return output


def fused_matmul_tc_fp8_tile_rowgroup_triton(
    activation: torch.Tensor,
    packed: PackedTensorCoreTileTensor,
    *,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    block_b: int = 16,
    row_group: int = 2,
    num_warps: int = 4,
) -> torch.Tensor:
    if not triton_tc_fp8_tile_decode_available():
        raise RuntimeError("Triton is not available for tc_fp8_tile_v1 fused matmul rowgroup")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Triton tc_fp8_tile_v1 fused matmul rowgroup requires a CUDA device")
    if activation.ndim != 2:
        raise ValueError("activation must be a 2D tensor")
    if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported compute dtype for Triton fused matmul rowgroup: {compute_dtype}")

    rows, cols = packed.original_shape
    row_tiles, col_tiles = packed.tile_grid
    tile_m, tile_k = packed.tile_shape
    row_group = max(int(row_group), 1)
    block_m = int(tile_m) * row_group
    batch = int(activation.shape[0])
    if int(activation.shape[1]) != int(cols):
        raise ValueError(f"activation cols {activation.shape[1]} != packed cols {cols}")

    act_gpu = activation.to(device=device, dtype=compute_dtype, non_blocking=True).contiguous()
    payload_gpu = packed.payload.contiguous().view(-1).to(device=device, non_blocking=True)
    scales_gpu = packed.scales.contiguous().view(-1).to(device=device, dtype=compute_dtype, non_blocking=True)
    output = torch.empty((batch, int(rows)), device=device, dtype=compute_dtype)
    grid = (triton.cdiv(batch, int(block_b)), triton.cdiv(int(row_tiles), row_group))
    _tc_fp8_tile_fused_matmul_rowgroup_kernel[grid](
        act_gpu,
        payload_gpu,
        scales_gpu,
        output,
        batch,
        int(rows),
        int(cols),
        int(row_tiles),
        int(col_tiles),
        int(tile_m),
        int(tile_k),
        int(block_b),
        int(row_group),
        int(block_m),
        num_warps=int(num_warps),
    )
    return output


def describe_tc_fp8_tile_case(
    rows: int,
    cols: int,
    *,
    spec: TensorCoreTransferFormatSpec | None = None,
) -> dict[str, Any]:
    spec = spec or TC_FP8_TILE_V1
    tile_m = int(spec.tile_m)
    tile_k = int(spec.tile_k)
    row_tiles = (int(rows) + tile_m - 1) // tile_m
    col_tiles = (int(cols) + tile_k - 1) // tile_k
    padded_rows = row_tiles * tile_m
    padded_cols = col_tiles * tile_k
    tile_count = row_tiles * col_tiles
    aligned_bytes = tile_count * int(spec.tile_bytes)
    actual_tc_bytes = int(padded_rows) * int(padded_cols) + tile_count * 2
    dense_fp16_bytes = int(rows) * int(cols) * 2
    dense_fp8_bytes = int(rows) * int(cols)
    return {
        "spec": spec.as_dict(),
        "shape": {"rows": int(rows), "cols": int(cols)},
        "tile_grid": {"row_tiles": row_tiles, "col_tiles": col_tiles, "tile_count": tile_count},
        "padded_shape": {"rows": padded_rows, "cols": padded_cols},
        "estimated_transfer_bytes": {
            "tc_fp8_tile_v1_payload_plus_scale": actual_tc_bytes,
            "tc_fp8_tile_v1_aligned": aligned_bytes,
            "dense_raw_fp16": dense_fp16_bytes,
            "dense_fp8_e4m3": dense_fp8_bytes,
        },
        "padding_ratio": round(1.0 - ((int(rows) * int(cols)) / float(max(padded_rows * padded_cols, 1))), 8),
        "reference_only": True,
    }


def _run_case(
    *,
    rows: int,
    cols: int,
    compute_dtype: torch.dtype,
    iters: int,
    warmup: int,
    pack_iters: int,
    device: torch.device,
    batch: int,
) -> dict[str, Any]:
    weight = torch.randn((int(rows), int(cols)), dtype=torch.float32) * 0.02
    activation = torch.randn((int(batch), int(cols)), dtype=torch.float32) * 0.02
    baseline_pack_ms, baseline_packed = _time_cpu_ms(
        lambda: pack_tensor_for_transfer(weight, TransferFormatPolicy(format="fp8_e4m3", experimental=True)),
        iters=pack_iters,
    )
    tc_pack_ms, tc_packed = _time_cpu_ms(
        lambda: pack_tc_fp8_tile_reference(weight),
        iters=pack_iters,
    )

    baseline_decode_ms, baseline_decoded = _time_cuda_ms(
        lambda: decode_transfer_tensor(baseline_packed, device=device, compute_dtype=compute_dtype),
        device=device,
        iters=iters,
        warmup=warmup,
    )
    tc_decode_ms, tc_decoded = _time_cuda_ms(
        lambda: decode_tc_fp8_tile_reference(tc_packed, device=device, compute_dtype=compute_dtype),
        device=device,
        iters=iters,
        warmup=warmup,
    )
    triton_rows: list[dict[str, Any]] = []
    triton_comparison: dict[str, Any] = {}
    if triton_tc_fp8_tile_decode_available():
        for implementation in ("triton_decode_v0", "triton_decode_v1"):
            try:
                triton_decode_ms, triton_decoded = _time_cuda_ms(
                    lambda impl=implementation: decode_tc_fp8_tile_triton(
                        tc_packed,
                        device=device,
                        compute_dtype=compute_dtype,
                        implementation=impl,
                    ),
                    device=device,
                    iters=iters,
                    warmup=warmup,
                )
                triton_error_mae = float((triton_decoded.float() - weight.to(device=device, dtype=torch.float32)).abs().mean().item())
                triton_speedup = round(float(baseline_decode_ms / max(triton_decode_ms, 1e-8)), 4) if baseline_decode_ms > 0 else None
                triton_rows.append(
                    {
                        "format": "tc_fp8_tile_v1",
                        "implementation": implementation,
                        "cpu_pack_ms": round(float(tc_pack_ms), 4),
                        "transfer_mb": round(float(tc_packed.transfer_mb), 6),
                        "decode_h2d_ms": round(float(triton_decode_ms), 4),
                        "error_mae": round(float(triton_error_mae), 8),
                        "metadata": {
                            **tc_packed.as_summary(),
                            "triton_decode": True,
                            "num_warps": 1 if implementation == "triton_decode_v1" else 4,
                            "scale_h2d_dtype": str(compute_dtype).replace("torch.", "") if implementation == "triton_decode_v1" else "float32",
                        },
                    }
                )
                triton_comparison[f"{implementation}_speedup_vs_fp8_e4m3"] = triton_speedup
                triton_comparison[f"{implementation}_decode_ms"] = round(float(triton_decode_ms), 4)
                triton_comparison[f"{implementation}_error"] = ""
            except Exception as exc:
                triton_comparison[f"{implementation}_speedup_vs_fp8_e4m3"] = None
                triton_comparison[f"{implementation}_decode_ms"] = None
                triton_comparison[f"{implementation}_error"] = f"{type(exc).__name__}: {exc}"
    error_mae = float((tc_decoded.float() - weight.to(device=device, dtype=torch.float32)).abs().mean().item())
    speedup = None
    if baseline_decode_ms > 0:
        speedup = round(float(baseline_decode_ms / max(tc_decode_ms, 1e-8)), 4)
    act_gpu = activation.to(device=device, dtype=compute_dtype, non_blocking=True).contiguous()
    baseline_decode_matmul_ms, baseline_decode_matmul_out = _time_cuda_ms(
        lambda: F.linear(act_gpu, decode_transfer_tensor(baseline_packed, device=device, compute_dtype=compute_dtype)),
        device=device,
        iters=iters,
        warmup=warmup,
    )
    reference_decode_matmul_ms, reference_decode_matmul_out = _time_cuda_ms(
        lambda: F.linear(act_gpu, decode_tc_fp8_tile_reference(tc_packed, device=device, compute_dtype=compute_dtype)),
        device=device,
        iters=iters,
        warmup=warmup,
    )
    fused_matmul_rows: list[dict[str, Any]] = []
    fused_matmul_errors: dict[str, str] = {}
    if triton_tc_fp8_tile_decode_available():
        for implementation, block_b, fused_num_warps, row_group in (
            ("triton_fused_decode_matmul_v0", 16, 4, 1),
            ("triton_fused_decode_matmul_v1_b8", 8, 4, 1),
            ("triton_fused_decode_matmul_v1_b32", 32, 4, 1),
            ("triton_fused_decode_matmul_v2_rg2_b16", 16, 4, 2),
            ("triton_fused_decode_matmul_v2_rg2_b32", 32, 4, 2),
            ("triton_fused_decode_matmul_v2_rg2_b32_w8", 32, 8, 2),
        ):
            try:
                fused_matmul_ms, fused_matmul_out = _time_cuda_ms(
                    lambda block_b=block_b, fused_num_warps=fused_num_warps, row_group=row_group: (
                        fused_matmul_tc_fp8_tile_triton(
                            activation,
                            tc_packed,
                            device=device,
                            compute_dtype=compute_dtype,
                            block_b=block_b,
                            num_warps=fused_num_warps,
                        )
                        if row_group == 1
                        else fused_matmul_tc_fp8_tile_rowgroup_triton(
                            activation,
                            tc_packed,
                            device=device,
                            compute_dtype=compute_dtype,
                            block_b=block_b,
                            row_group=row_group,
                            num_warps=fused_num_warps,
                        )
                    ),
                    device=device,
                    iters=iters,
                    warmup=warmup,
                )
                fused_matmul_mae = float((fused_matmul_out.float() - reference_decode_matmul_out.float()).abs().mean().item())
                fused_matmul_rows.append(
                    {
                        "format": "tc_fp8_tile_v1",
                        "implementation": implementation,
                        "decode_h2d_matmul_ms": round(float(fused_matmul_ms), 4),
                        "error_mae_vs_reference": round(float(fused_matmul_mae), 8),
                        "metadata": {
                            "activation_batch": int(batch),
                            "block_b": int(block_b),
                            "num_warps": int(fused_num_warps),
                            "row_group": int(row_group),
                        },
                    }
                )
                fused_matmul_errors[implementation] = ""
            except Exception as exc:
                fused_matmul_errors[implementation] = f"{type(exc).__name__}: {exc}"
    results = [
        {
            "format": "fp8_e4m3",
            "cpu_pack_ms": round(float(baseline_pack_ms), 4),
            "transfer_mb": round(float(baseline_packed.transfer_mb), 6),
            "decode_h2d_ms": round(float(baseline_decode_ms), 4),
            "error_mae": None,
            "metadata": dict(baseline_packed.metadata),
        },
        {
            "format": "tc_fp8_tile_v1",
            "implementation": "reference_torch",
            "cpu_pack_ms": round(float(tc_pack_ms), 4),
            "transfer_mb": round(float(tc_packed.transfer_mb), 6),
            "decode_h2d_ms": round(float(tc_decode_ms), 4),
            "error_mae": round(error_mae, 8),
            "metadata": tc_packed.as_summary(),
        },
    ]
    results.extend(triton_rows)
    matmul_results = [
        {
            "format": "fp8_e4m3",
            "implementation": "decode_then_matmul",
            "decode_h2d_matmul_ms": round(float(baseline_decode_matmul_ms), 4),
            "metadata": {"activation_batch": int(batch)},
        },
        {
            "format": "tc_fp8_tile_v1",
            "implementation": "reference_decode_then_matmul",
            "decode_h2d_matmul_ms": round(float(reference_decode_matmul_ms), 4),
            "metadata": {"activation_batch": int(batch)},
        },
    ]
    matmul_results.extend(fused_matmul_rows)
    best_fused_matmul = min(fused_matmul_rows, key=lambda row: float(row["decode_h2d_matmul_ms"])) if fused_matmul_rows else None
    best_triton_decode = min(
        triton_rows,
        key=lambda row: float(row["decode_h2d_ms"]),
    ) if triton_rows else None
    return {
        "shape": {"rows": int(rows), "cols": int(cols)},
        "activation": {"batch": int(batch), "cols": int(cols)},
        "describe": describe_tc_fp8_tile_case(rows, cols),
        "results": results,
        "matmul_results": matmul_results,
        "comparison": {
            "reference_decode_speedup_vs_fp8_e4m3": speedup,
            "triton_decode_speedup_vs_fp8_e4m3": None
            if best_triton_decode is None
            else round(float(baseline_decode_ms / max(float(best_triton_decode["decode_h2d_ms"]), 1e-8)), 4),
            "triton_decode_best": None if best_triton_decode is None else best_triton_decode["implementation"],
            "transfer_mb_delta_vs_fp8_e4m3": round(float(tc_packed.transfer_mb - baseline_packed.transfer_mb), 6),
            "cpu_pack_ms_delta_vs_fp8_e4m3": round(float(tc_pack_ms - baseline_pack_ms), 4),
            "triton_error": triton_comparison.get("triton_decode_v1_error") or triton_comparison.get("triton_decode_v0_error") or "",
            "triton_variants": triton_comparison,
            "reference_decode_matmul_speedup_vs_fp8_e4m3": round(
                float(baseline_decode_matmul_ms / max(reference_decode_matmul_ms, 1e-8)),
                4,
            ),
            "triton_fused_matmul_speedup_vs_fp8_e4m3": None
            if best_fused_matmul is None
            else round(float(baseline_decode_matmul_ms / max(float(best_fused_matmul["decode_h2d_matmul_ms"]), 1e-8)), 4),
            "triton_fused_matmul_best": None if best_fused_matmul is None else best_fused_matmul["implementation"],
            "triton_fused_matmul_error": "; ".join(
                f"{name}: {message}" for name, message in fused_matmul_errors.items() if message
            ),
            "triton_fused_matmul_variants": fused_matmul_errors,
        },
    }


def run_tensorcore_decode_benchmark(
    *,
    shapes: list[tuple[int, int]],
    compute_dtype: torch.dtype,
    iters: int,
    warmup: int,
    pack_iters: int,
    seed: int,
    batch: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TensorCore decode-only benchmark prototype")
    if getattr(torch, "float8_e4m3fn", None) is None:
        raise RuntimeError("PyTorch float8_e4m3 support is required for tc_fp8_tile_v1 prototype")
    torch.manual_seed(int(seed))
    device = torch.device("cuda")
    cases = [
        _run_case(
            rows=rows,
            cols=cols,
            compute_dtype=compute_dtype,
            iters=iters,
            warmup=warmup,
            pack_iters=pack_iters,
            device=device,
            batch=batch,
        )
        for rows, cols in shapes
    ]
    return {
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "prototype": "decode_only_and_fused_matmul_triton_v0",
        "triton_available": triton_tc_fp8_tile_decode_available(),
        "spec": TC_FP8_TILE_V1.as_dict(),
        "roadmap": tensorcore_kernel_roadmap(),
        "cases": cases,
        "summary": summarize_tensorcore_cases(cases),
    }


def summarize_tensorcore_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        shape = case.get("shape") if isinstance(case, dict) else {}
        comparison = case.get("comparison") if isinstance(case, dict) else {}
        if not isinstance(shape, dict) or not isinstance(comparison, dict):
            continue
        decode_speedup = comparison.get("triton_decode_speedup_vs_fp8_e4m3")
        fused_speedup = comparison.get("triton_fused_matmul_speedup_vs_fp8_e4m3")
        rows.append(
            {
                "shape": f"{int(shape.get('rows', 0))}x{int(shape.get('cols', 0))}",
                "rows": int(shape.get("rows", 0)),
                "cols": int(shape.get("cols", 0)),
                "triton_decode_speedup_vs_fp8_e4m3": decode_speedup,
                "triton_decode_best": comparison.get("triton_decode_best"),
                "triton_fused_matmul_speedup_vs_fp8_e4m3": fused_speedup,
                "triton_fused_matmul_best": comparison.get("triton_fused_matmul_best"),
                "transfer_mb_delta_vs_fp8_e4m3": comparison.get("transfer_mb_delta_vs_fp8_e4m3"),
            }
        )

    best_decode = max(
        (row for row in rows if isinstance(row.get("triton_decode_speedup_vs_fp8_e4m3"), (int, float))),
        key=lambda row: float(row["triton_decode_speedup_vs_fp8_e4m3"]),
        default=None,
    )
    best_fused = max(
        (row for row in rows if isinstance(row.get("triton_fused_matmul_speedup_vs_fp8_e4m3"), (int, float))),
        key=lambda row: float(row["triton_fused_matmul_speedup_vs_fp8_e4m3"]),
        default=None,
    )
    promising = [
        row for row in rows
        if float(row.get("triton_decode_speedup_vs_fp8_e4m3") or 0.0) >= 1.05
        or float(row.get("triton_fused_matmul_speedup_vs_fp8_e4m3") or 0.0) >= 1.05
    ]
    return {
        "case_count": len(rows),
        "best_decode": best_decode,
        "best_fused_matmul": best_fused,
        "promising_cases": promising,
        "decision": "keep_research_only" if not promising else "benchmark_more_before_any_training_path",
        "note": "speedup is vs PyTorch fp8_e4m3 decode/decode+matmul baseline; values below 1.0 are slower",
    }


def _preset_shapes(name: str) -> list[tuple[int, int]]:
    key = str(name or "").strip().lower()
    if not key or key == "single":
        return []
    if key not in SHAPE_PRESETS:
        raise ValueError(f"unknown shape preset {name!r}; choices: {', '.join(sorted(SHAPE_PRESETS))}")
    return list(SHAPE_PRESETS[key])


def _parse_shapes(value: str, *, rows: int, cols: int) -> list[tuple[int, int]]:
    raw = str(value or "").strip()
    if not raw:
        return [(int(rows), int(cols))]
    shapes: list[tuple[int, int]] = []
    for item in raw.split(","):
        token = item.strip().lower().replace("*", "x")
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"shape must look like ROWSxCOLS: {item!r}")
        left, right = token.split("x", 1)
        shapes.append((int(left), int(right)))
    return shapes or [(int(rows), int(cols))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--shapes", default="", help="Comma-separated ROWSxCOLS list, e.g. 4096x4096,8192x4096")
    parser.add_argument("--shape-preset", default="", choices=sorted(SHAPE_PRESETS), help="Use a built-in real-model Linear shape preset")
    parser.add_argument("--compute-dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pack-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    payload = run_tensorcore_decode_benchmark(
        shapes=_parse_shapes(args.shapes, rows=int(args.rows), cols=int(args.cols)) if str(args.shapes or "").strip() else (_preset_shapes(args.shape_preset) or [(int(args.rows), int(args.cols))]),
        compute_dtype=_dtype(args.compute_dtype),
        iters=max(int(args.iters), 1),
        warmup=max(int(args.warmup), 0),
        pack_iters=max(int(args.pack_iters), 1),
        seed=int(args.seed),
        batch=max(int(args.batch), 1),
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
