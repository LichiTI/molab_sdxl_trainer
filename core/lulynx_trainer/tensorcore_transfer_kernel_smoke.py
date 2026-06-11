"""Smoke test for TensorCore-friendly transfer kernel planning."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorcore_transfer_kernel import (
    available_tensorcore_transfer_specs,
    choose_tensorcore_transfer_spec,
    tensorcore_kernel_roadmap,
    triton_decode_kernel_stub,
)


def main() -> None:
    specs = available_tensorcore_transfer_specs()
    assert "tc_fp8_tile_v1" in specs
    assert specs["tc_fp8_tile_v1"]["tile_m"] == 16
    assert specs["tc_fp8_tile_v1"]["tile_k"] == 64
    assert specs["tc_fp8_tile_v1"]["tile_bytes"] % 128 == 0
    assert choose_tensorcore_transfer_spec().name == "tc_fp8_tile_v1"
    assert choose_tensorcore_transfer_spec(prefer_size=True).name == "tc_uint4_tile_v1"
    roadmap = tensorcore_kernel_roadmap()
    assert roadmap["status"] == "design_skeleton"
    assert roadmap["selected_spec"]["name"] == "tc_fp8_tile_v1"
    stub = triton_decode_kernel_stub()
    assert "Sketch only" in stub and "tc_fp8_tile_v1" in stub
    print("PASS: tensorcore transfer kernel planning smoke")


if __name__ == "__main__":
    main()
