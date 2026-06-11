"""Smoke test for the decode-only TensorCore transfer benchmark prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorcore_transfer_kernel import TC_FP8_TILE_V1  # noqa: E402
from tensorcore_transfer_kernel_benchmark import (  # noqa: E402
    describe_tc_fp8_tile_case,
    pack_tc_fp8_tile_reference,
)


def main() -> None:
    description = describe_tc_fp8_tile_case(33, 70)
    assert description["spec"]["name"] == "tc_fp8_tile_v1"
    assert description["tile_grid"]["row_tiles"] == 3
    assert description["tile_grid"]["col_tiles"] == 2
    assert description["estimated_transfer_bytes"]["tc_fp8_tile_v1_aligned"] == 6 * TC_FP8_TILE_V1.tile_bytes
    assert description["estimated_transfer_bytes"]["dense_raw_fp16"] == 33 * 70 * 2
    if getattr(torch, "float8_e4m3fn", None) is None:
        print("PASS: tensorcore transfer benchmark smoke (metadata only; float8 unavailable)")
        return

    weight = torch.linspace(-1.0, 1.0, steps=33 * 70, dtype=torch.float32).view(33, 70)
    packed = pack_tc_fp8_tile_reference(weight)
    assert packed.spec_name == "tc_fp8_tile_v1"
    assert packed.original_shape == (33, 70)
    assert packed.padded_shape == (48, 128)
    assert packed.tile_grid == (3, 2)
    assert packed.payload.shape == (3, 2, 16, 64)
    assert packed.payload_bytes == 3 * 2 * 16 * 64
    assert packed.scale_bytes == 3 * 2 * 2
    assert packed.aligned_bytes == 6 * TC_FP8_TILE_V1.tile_bytes
    print("PASS: tensorcore transfer benchmark smoke")


if __name__ == "__main__":
    main()
