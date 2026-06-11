# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for lora_rank_resize: down/up shape correctness and alpha rescaling."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

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


_rr = _import_module("lora_rank_resize", os.path.join(_HERE, "lora_rank_resize.py"))
resize_lora_rank = _rr.resize_lora_rank

import torch
from safetensors.torch import save_file, load_file


def _make_rank8_safetensors(path: str, in_dim: int = 64, out_dim: int = 64) -> None:
    """Write a minimal rank-8 LoRA safetensors to *path*."""
    rank = 8
    tensors = {
        "layer0.lora_down.weight": torch.randn(rank, in_dim),
        "layer0.lora_up.weight": torch.randn(out_dim, rank),
        "layer0.alpha": torch.tensor(8.0),
    }
    save_file(tensors, path)


def test_resize_down() -> None:
    """Resizing rank 8 → 4 produces lora_down (4, 64) and lora_up (64, 4)."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        src = f.name
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dst = f.name
    try:
        _make_rank8_safetensors(src)
        resize_lora_rank(src, dst, target_rank=4)
        tensors = load_file(dst)
        assert tensors["layer0.lora_down.weight"].shape == (4, 64), (
            f"Expected lora_down (4, 64), got {tensors['layer0.lora_down.weight'].shape}"
        )
        assert tensors["layer0.lora_up.weight"].shape == (64, 4), (
            f"Expected lora_up (64, 4), got {tensors['layer0.lora_up.weight'].shape}"
        )
    finally:
        for p in (src, dst):
            try:
                os.unlink(p)
            except OSError:
                pass
    print("  PASS: test_resize_down")


def test_resize_up() -> None:
    """Resizing rank 4 → 8 is clamped to available singular values (min(target, rank) = 4)."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        src = f.name
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dst = f.name
    try:
        rank = 4
        save_file(
            {
                "layer0.lora_down.weight": torch.randn(rank, 64),
                "layer0.lora_up.weight": torch.randn(64, rank),
                "layer0.alpha": torch.tensor(4.0),
            },
            src,
        )
        resize_lora_rank(src, dst, target_rank=8)
        tensors = load_file(dst)
        actual_r = tensors["layer0.lora_down.weight"].shape[0]
        assert actual_r <= 8, (
            f"Expected rank <= 8, got {actual_r}"
        )
    finally:
        for p in (src, dst):
            try:
                os.unlink(p)
            except OSError:
                pass
    print(f"  PASS: test_resize_up (actual rank={actual_r})")


def test_alpha_rescaled() -> None:
    """After rank 8 → 4, alpha is rescaled to approximately 8.0 * (4/8) = 4.0."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        src = f.name
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dst = f.name
    try:
        _make_rank8_safetensors(src)
        resize_lora_rank(src, dst, target_rank=4)
        tensors = load_file(dst)
        alpha_val = float(tensors["layer0.alpha"].item())
        expected = 8.0 * (4 / 8)
        assert abs(alpha_val - expected) < 0.01, (
            f"Expected alpha ~{expected}, got {alpha_val}"
        )
    finally:
        for p in (src, dst):
            try:
                os.unlink(p)
            except OSError:
                pass
    print("  PASS: test_alpha_rescaled")


def test_reconstruction_bounded() -> None:
    """Reconstruction error norm of the resized LoRA is small relative to the original."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        src = f.name
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        dst = f.name
    try:
        torch.manual_seed(42)
        _make_rank8_safetensors(src)
        orig = load_file(src)
        W_orig = orig["layer0.lora_up.weight"] @ orig["layer0.lora_down.weight"]

        resize_lora_rank(src, dst, target_rank=4)
        resized = load_file(dst)
        W_resized = resized["layer0.lora_up.weight"] @ resized["layer0.lora_down.weight"]

        orig_norm = W_orig.norm().item()
        err_norm = (W_orig - W_resized).norm().item()
        ratio = err_norm / (orig_norm + 1e-8)
        assert ratio < 1.5, (
            f"Reconstruction error ratio {ratio:.4f} too large (expected < 1.5)"
        )
    finally:
        for p in (src, dst):
            try:
                os.unlink(p)
            except OSError:
                pass
    print(f"  PASS: test_reconstruction_bounded (err/orig ratio ~ {ratio:.4f})")


def main() -> int:
    print("LoRA Rank Resize Smoke Tests")
    print("=" * 40)
    test_resize_down()
    test_resize_up()
    test_alpha_rescaled()
    test_reconstruction_bounded()
    print("=" * 40)
    print("All LoRA rank resize smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
