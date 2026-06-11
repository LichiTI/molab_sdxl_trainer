# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for lbw_parser: weight parsing, block-ID detection, and TE mapping."""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))


def _import_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name!r} from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_lbw = _import_module("lbw_parser", os.path.join(_HERE, "lbw_parser.py"))
parse_lbw_weights = _lbw.parse_lbw_weights
apply_lbw_to_layer_key = _lbw.apply_lbw_to_layer_key
SDXL_BLOCKS = _lbw.SDXL_BLOCKS
SD15_BLOCKS = _lbw.SD15_BLOCKS


def test_parse_sdxl_26() -> None:
    """26 comma-separated values produce a dict with 26 entries keyed to SDXL block IDs."""
    weights_str = "1,0.5,0.5,0.5,1,1,1,1,1,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    result = parse_lbw_weights(weights_str, "sdxl")
    assert len(result) == 26, f"Expected 26 entries, got {len(result)}"
    for block_id in SDXL_BLOCKS:
        assert block_id in result, f"Block ID {block_id!r} missing from result"
    print(f"  PASS: test_parse_sdxl_26 ({len(result)} entries)")


def test_parse_sd15_17() -> None:
    """17 comma-separated values produce a dict with 17 entries keyed to SD15 block IDs."""
    values = ",".join(["1.0"] * 17)
    result = parse_lbw_weights(values, "sd15")
    assert len(result) == 17, f"Expected 17 entries, got {len(result)}"
    for block_id in SD15_BLOCKS:
        assert block_id in result, f"Block ID {block_id!r} missing from SD15 result"
    print(f"  PASS: test_parse_sd15_17 ({len(result)} entries)")


def test_wrong_count_error() -> None:
    """Passing 10 values for SDXL (expects 26) raises ValueError."""
    values = ",".join(["1.0"] * 10)
    try:
        parse_lbw_weights(values, "sdxl")
        assert False, "Should have raised ValueError for wrong count"
    except ValueError as e:
        assert "26" in str(e) or "10" in str(e) or "mismatch" in str(e).lower() or "count" in str(e).lower(), (
            f"ValueError message did not mention count issue: {e}"
        )
    print("  PASS: test_wrong_count_error")


def test_block_id_detection() -> None:
    """apply_lbw_to_layer_key resolves a down-block attention key to its correct weight."""
    weights_str = "1,0.5,0.5,0.5,1,1,1,1,1,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    lbw_weights = parse_lbw_weights(weights_str, "sdxl")
    key = "lora_unet_down_blocks_3_attentions_1_proj_in.lora_down.weight"
    weight = apply_lbw_to_layer_key(key, lbw_weights)
    assert isinstance(weight, float), f"Expected float weight, got {type(weight)}"
    assert 0.0 <= weight <= 1.0, f"Weight {weight} outside [0, 1]"
    print(f"  PASS: test_block_id_detection (weight={weight})")


def test_te_maps_to_base() -> None:
    """A key containing 'text_encoder' maps to the BASE block weight."""
    weights_str = "1,0.5,0.5,0.5,1,1,1,1,1,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    lbw_weights = parse_lbw_weights(weights_str, "sdxl")
    key = "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"
    weight = apply_lbw_to_layer_key(key, lbw_weights)
    # BASE block is conventionally the first entry in SDXL_BLOCKS
    base_block = SDXL_BLOCKS[0]
    expected = lbw_weights[base_block]
    assert weight == expected, (
        f"Expected TE key to map to BASE weight {expected}, got {weight}"
    )
    print(f"  PASS: test_te_maps_to_base (BASE weight={expected})")


def main() -> int:
    print("LBW Parser Smoke Tests")
    print("=" * 40)
    test_parse_sdxl_26()
    test_parse_sd15_17()
    test_wrong_count_error()
    test_block_id_detection()
    test_te_maps_to_base()
    print("=" * 40)
    print("All LBW parser smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
