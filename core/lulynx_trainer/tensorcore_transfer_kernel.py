"""TensorCore-friendly transfer-format kernel planning.

This module intentionally does not JIT or launch a custom kernel yet.  It
captures the format contract and launch constraints for the future fused
H2D/decode/matmul path so experiments can share one conservative spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorCoreTransferFormatSpec:
    name: str
    payload_dtype: str
    compute_dtype: str
    tile_m: int
    tile_k: int
    scale_dtype: str
    scale_granularity: str
    alignment_bytes: int
    bits_per_value: int
    fused_decode_target: str
    experimental: bool = True

    @property
    def values_per_tile(self) -> int:
        return int(self.tile_m) * int(self.tile_k)

    @property
    def payload_bytes_per_tile(self) -> int:
        return (self.values_per_tile * int(self.bits_per_value) + 7) // 8

    @property
    def scale_bytes_per_tile(self) -> int:
        return 2 if self.scale_dtype in {"fp16", "bf16"} else 4

    @property
    def tile_bytes(self) -> int:
        raw = self.payload_bytes_per_tile + self.scale_bytes_per_tile
        align = max(int(self.alignment_bytes), 1)
        return ((raw + align - 1) // align) * align

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "payload_dtype": self.payload_dtype,
            "compute_dtype": self.compute_dtype,
            "tile_m": int(self.tile_m),
            "tile_k": int(self.tile_k),
            "scale_dtype": self.scale_dtype,
            "scale_granularity": self.scale_granularity,
            "alignment_bytes": int(self.alignment_bytes),
            "bits_per_value": int(self.bits_per_value),
            "values_per_tile": self.values_per_tile,
            "payload_bytes_per_tile": self.payload_bytes_per_tile,
            "scale_bytes_per_tile": self.scale_bytes_per_tile,
            "tile_bytes": self.tile_bytes,
            "fused_decode_target": self.fused_decode_target,
            "experimental": bool(self.experimental),
        }


TC_FP8_TILE_V1 = TensorCoreTransferFormatSpec(
    name="tc_fp8_tile_v1",
    payload_dtype="fp8_e4m3",
    compute_dtype="bf16_or_fp16",
    tile_m=16,
    tile_k=64,
    scale_dtype="fp16",
    scale_granularity="tile",
    alignment_bytes=128,
    bits_per_value=8,
    fused_decode_target="decode_to_shared_then_wgmma_or_mma",
)

TC_INT8_TILE_V1 = TensorCoreTransferFormatSpec(
    name="tc_int8_tile_v1",
    payload_dtype="int8",
    compute_dtype="fp16",
    tile_m=16,
    tile_k=64,
    scale_dtype="fp16",
    scale_granularity="tile",
    alignment_bytes=128,
    bits_per_value=8,
    fused_decode_target="dequant_to_shared_then_mma",
)

TC_UINT4_TILE_V1 = TensorCoreTransferFormatSpec(
    name="tc_uint4_tile_v1",
    payload_dtype="uint4_packed",
    compute_dtype="fp16",
    tile_m=16,
    tile_k=64,
    scale_dtype="fp16",
    scale_granularity="tile",
    alignment_bytes=128,
    bits_per_value=4,
    fused_decode_target="unpack_dequant_to_shared_then_mma",
)


_SPECS = {
    TC_FP8_TILE_V1.name: TC_FP8_TILE_V1,
    TC_INT8_TILE_V1.name: TC_INT8_TILE_V1,
    TC_UINT4_TILE_V1.name: TC_UINT4_TILE_V1,
}


def available_tensorcore_transfer_specs() -> dict[str, dict[str, Any]]:
    return {name: spec.as_dict() for name, spec in _SPECS.items()}


def choose_tensorcore_transfer_spec(*, prefer_size: bool = False) -> TensorCoreTransferFormatSpec:
    """Pick the first implementation target for a future fused kernel.

    FP8 is the default because it keeps the payload TensorCore-aligned while
    avoiding the unpack overhead of 4-bit paths.  UINT4 is only preferred for
    size-first research runs.
    """

    if prefer_size:
        return TC_UINT4_TILE_V1
    return TC_FP8_TILE_V1


def tensorcore_kernel_roadmap() -> dict[str, Any]:
    spec = choose_tensorcore_transfer_spec()
    return {
        "enabled": True,
        "status": "design_skeleton",
        "selected_spec": spec.as_dict(),
        "milestones": [
            "benchmark current pack+h2d+decode formats per GPU",
            "prototype Triton decode-only kernel for tc_fp8_tile_v1",
            "compare decode-only against torch dtype cast path",
            "prototype fused decode+matmul for frozen Linear inference path",
            "gate training use behind explicit experimental flag and parity smoke",
        ],
        "guardrails": [
            "do not apply to trainable LoRA/adapter parameters",
            "do not auto-enable from smart sensing or enhanced protection",
            "fallback to existing transfer_format.py decode path on any error",
            "require quality-error and speed benchmarks before UI exposure",
        ],
    }


def triton_decode_kernel_stub(spec: TensorCoreTransferFormatSpec | None = None) -> str:
    """Return a non-executable Triton sketch for design review.

    Keeping this as text avoids importing Triton or compiling kernels during
    normal training startup.
    """

    spec = spec or choose_tensorcore_transfer_spec()
    return f"""
# Sketch only: {spec.name}
# payload: {spec.payload_dtype}, scale: {spec.scale_dtype}/{spec.scale_granularity}
# tile: {spec.tile_m}x{spec.tile_k}, aligned={spec.alignment_bytes} bytes
# 1. Load one compressed tile from CPU-pinned/GPU staging buffer.
# 2. Load tile scale.
# 3. Decode into shared/register tile as {spec.compute_dtype}.
# 4. Feed the tile into TensorCore matmul or write decoded tile for fallback.
"""
