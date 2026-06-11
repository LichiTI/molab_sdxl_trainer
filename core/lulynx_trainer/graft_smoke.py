# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for graft.py (Phase 8.11 / #119)."""

from __future__ import annotations

import os
import sys
import importlib.util
import tempfile
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.graft",
    os.path.join(_HERE, "graft.py"),
)
_g = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.graft"] = _g
_spec.loader.exec_module(_g)


def _make_lora_sd(out_dim: int = 16, in_dim: int = 16, rank: int = 4, alpha: float = 4.0,
                  base: str = "model.layer1") -> dict:
    return {
        f"{base}.lora_down.weight": torch.randn(rank, in_dim),
        f"{base}.lora_up.weight": torch.randn(out_dim, rank),
        f"{base}.alpha": torch.tensor(alpha),
    }


def test_find_lora_pairs_recognises_down_up():
    sd = _make_lora_sd()
    pairs = _g._find_lora_pairs(sd)
    assert len(pairs) == 1
    base, d, u = pairs[0]
    assert base == "model.layer1"
    assert d.endswith(".lora_down.weight")
    assert u.endswith(".lora_up.weight")
    print("PASS: _find_lora_pairs recognises down/up convention")


def test_svd_truncate_preserves_dominant_directions():
    torch.manual_seed(0)
    # Construct a known low-rank delta
    out, in_, rank = 16, 16, 4
    base_up = torch.randn(out, rank)
    base_down = torch.randn(rank, in_)
    delta = base_up @ base_down

    up, down = _g._svd_truncate(delta, target_rank=rank)
    reconstructed = up @ down
    err = (reconstructed - delta).abs().max().item()
    assert err < 1e-3, f"reconstruction error too large: {err}"
    print(f"PASS: SVD truncation reconstructs rank-{rank} delta within {err:.2e}")


def test_svd_truncate_pads_when_input_is_lower_rank():
    delta = torch.zeros(8, 8)
    up, down = _g._svd_truncate(delta, target_rank=4)
    assert up.shape == (8, 4)
    assert down.shape == (4, 8)
    print("PASS: SVD truncation pads to target_rank when input is degenerate")


def test_graft_loras_single_input_round_trips():
    torch.manual_seed(0)
    sd = _make_lora_sd(rank=8, alpha=8.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.pt"
        _g.save_lora_state_dict(sd, str(path))

        cfg = _g.GRAFTConfig(target_rank=8)
        merged = _g.graft_loras([str(path)], cfg)

        # Output should contain the same base key with new down/up
        assert "model.layer1.lora_down.weight" in merged
        assert "model.layer1.lora_up.weight" in merged
        assert merged["model.layer1.lora_down.weight"].shape == (8, 16)
        assert merged["model.layer1.lora_up.weight"].shape == (16, 8)
    print("PASS: graft_loras single-input round-trips")


def test_graft_loras_two_inputs_combines_deltas():
    torch.manual_seed(0)
    sd_a = _make_lora_sd(rank=4, alpha=4.0, base="model.layer1")
    sd_b = _make_lora_sd(rank=4, alpha=4.0, base="model.layer1")

    with tempfile.TemporaryDirectory() as tmp:
        pa = Path(tmp) / "a.pt"
        pb = Path(tmp) / "b.pt"
        _g.save_lora_state_dict(sd_a, str(pa))
        _g.save_lora_state_dict(sd_b, str(pb))

        # Use rank=8 so the merge can losslessly represent the sum of two rank-4 deltas
        cfg = _g.GRAFTConfig(target_rank=8)
        merged = _g.graft_loras([str(pa), str(pb)], cfg, weights=[1.0, 1.0])

        merged_up = merged["model.layer1.lora_up.weight"]
        merged_down = merged["model.layer1.lora_down.weight"]
        merged_alpha = merged["model.layer1.alpha"].item()
        scale = merged_alpha / 8
        merged_delta = (merged_up @ merged_down) * scale

        delta_a = (sd_a["model.layer1.lora_up.weight"] @ sd_a["model.layer1.lora_down.weight"]) * (4.0 / 4)
        delta_b = (sd_b["model.layer1.lora_up.weight"] @ sd_b["model.layer1.lora_down.weight"]) * (4.0 / 4)
        expected = delta_a + delta_b

        err = (merged_delta - expected).abs().max().item()
        assert err < 1e-3, f"merged delta error too large: {err}"
    print(f"PASS: graft_loras combines two deltas (max err {err:.2e})")


def test_graft_loras_weighted_combination():
    torch.manual_seed(0)
    sd_a = _make_lora_sd(rank=4, alpha=4.0)
    sd_b = _make_lora_sd(rank=4, alpha=4.0)

    with tempfile.TemporaryDirectory() as tmp:
        pa = Path(tmp) / "a.pt"
        pb = Path(tmp) / "b.pt"
        _g.save_lora_state_dict(sd_a, str(pa))
        _g.save_lora_state_dict(sd_b, str(pb))

        cfg = _g.GRAFTConfig(target_rank=4)
        merged_eq = _g.graft_loras([str(pa), str(pb)], cfg, weights=[1.0, 1.0])
        merged_skew = _g.graft_loras([str(pa), str(pb)], cfg, weights=[2.0, 0.5])

        # Skewed merge must differ from equal-weight merge
        diff = (merged_eq["model.layer1.lora_up.weight"] -
                merged_skew["model.layer1.lora_up.weight"]).abs().sum().item()
        assert diff > 0
    print("PASS: weight argument changes merged adapter")


def test_save_load_round_trip():
    sd = _make_lora_sd()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.pt"
        _g.save_lora_state_dict(sd, str(path))
        loaded = _g.load_lora_state_dict(str(path))
        for k in sd:
            assert k in loaded
            assert torch.allclose(sd[k], loaded[k])
    print("PASS: save/load round-trip preserves tensors")


def test_skip_missing_keys_when_only_in_one_checkpoint():
    sd_a = _make_lora_sd(base="layer.A")
    sd_b = _make_lora_sd(base="layer.B")  # disjoint key

    with tempfile.TemporaryDirectory() as tmp:
        pa = Path(tmp) / "a.pt"
        pb = Path(tmp) / "b.pt"
        _g.save_lora_state_dict(sd_a, str(pa))
        _g.save_lora_state_dict(sd_b, str(pb))

        cfg = _g.GRAFTConfig(target_rank=4, skip_missing=True)
        merged = _g.graft_loras([str(pa), str(pb)], cfg)
        # Only canonical (first checkpoint) keys appear
        assert any(k.startswith("layer.A") for k in merged)
        assert not any(k.startswith("layer.B") for k in merged)
    print("PASS: skip_missing=True drops keys not in first checkpoint")


def test_weight_normalize_scales_to_one():
    torch.manual_seed(0)
    sd_a = _make_lora_sd(rank=4, alpha=4.0)
    sd_b = _make_lora_sd(rank=4, alpha=4.0)

    with tempfile.TemporaryDirectory() as tmp:
        pa = Path(tmp) / "a.pt"
        pb = Path(tmp) / "b.pt"
        _g.save_lora_state_dict(sd_a, str(pa))
        _g.save_lora_state_dict(sd_b, str(pb))

        # rank=8 to fit the sum of two rank-4 deltas losslessly
        cfg = _g.GRAFTConfig(target_rank=8, weight_normalize=True, alpha=8.0)
        merged = _g.graft_loras([str(pa), str(pb)], cfg, weights=[1.0, 1.0])

        # alpha=8, target_rank=8 -> scale=1
        merged_delta = (merged["model.layer1.lora_up.weight"] @
                       merged["model.layer1.lora_down.weight"])
        # weights normalised to 0.5 each
        delta_a = sd_a["model.layer1.lora_up.weight"] @ sd_a["model.layer1.lora_down.weight"]
        delta_b = sd_b["model.layer1.lora_up.weight"] @ sd_b["model.layer1.lora_down.weight"]
        expected_avg = 0.5 * delta_a + 0.5 * delta_b
        err = (merged_delta - expected_avg).abs().max().item()
        assert err < 1e-3, f"normalised merge error: {err}"
    print(f"PASS: weight_normalize produces averaged delta (err={err:.2e})")


if __name__ == "__main__":
    test_find_lora_pairs_recognises_down_up()
    test_svd_truncate_preserves_dominant_directions()
    test_svd_truncate_pads_when_input_is_lower_rank()
    test_graft_loras_single_input_round_trips()
    test_graft_loras_two_inputs_combines_deltas()
    test_graft_loras_weighted_combination()
    test_save_load_round_trip()
    test_skip_missing_keys_when_only_in_one_checkpoint()
    test_weight_normalize_scales_to_one()
    print("\nAll GRAFT smoke tests passed!")
