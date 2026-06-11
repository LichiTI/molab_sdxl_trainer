# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for MicroDecoder VAE: forward shape, param count, PIL output, pixel range."""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name!r} from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mv = _import_module("micro_vae", os.path.join(_HERE, "micro_vae.py"))
MicroDecoder = _mv.MicroDecoder
micro_decode = _mv.micro_decode

import torch
import torch.nn as nn


def test_forward_shape() -> None:
    """MicroDecoder(4, 3) with input (1, 4, 8, 8) produces output (1, 3, 64, 64) — 8x upscale."""
    decoder = MicroDecoder(latent_channels=4, out_channels=3)
    decoder.eval()
    with torch.no_grad():
        x = torch.randn(1, 4, 8, 8)
        out = decoder(x)
    assert out.shape == (1, 3, 64, 64), f"Expected (1, 3, 64, 64), got {out.shape}"
    print("  PASS: test_forward_shape")


def test_param_count() -> None:
    """MicroDecoder parameter count is under 5 million."""
    decoder = MicroDecoder(latent_channels=4, out_channels=3)
    total = sum(p.numel() for p in decoder.parameters())
    assert total < 5_000_000, f"Parameter count {total} exceeds 5_000_000"
    print(f"  PASS: test_param_count ({total:,} params)")


def test_micro_decode_returns_pil() -> None:
    """micro_decode with random latents returns a PIL.Image.Image."""
    import PIL.Image
    decoder = MicroDecoder(latent_channels=4, out_channels=3)
    decoder.eval()
    latents = torch.randn(1, 4, 8, 8)
    result = micro_decode(latents, decoder)
    assert isinstance(result, PIL.Image.Image), (
        f"Expected PIL.Image.Image, got {type(result)}"
    )
    print("  PASS: test_micro_decode_returns_pil")


def test_output_pixel_range() -> None:
    """Pixels in the PIL image decoded by micro_decode are within [0, 255]."""
    import numpy as np
    decoder = MicroDecoder(latent_channels=4, out_channels=3)
    decoder.eval()
    latents = torch.randn(1, 4, 8, 8)
    img = micro_decode(latents, decoder)
    arr = np.array(img)
    assert arr.min() >= 0, f"Pixel min {arr.min()} below 0"
    assert arr.max() <= 255, f"Pixel max {arr.max()} above 255"
    print("  PASS: test_output_pixel_range")


def test_eval_mode() -> None:
    """MicroDecoder can be switched to eval(); all parameters frozen via requires_grad=False."""
    decoder = MicroDecoder(latent_channels=4, out_channels=3)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    trainable = [n for n, p in decoder.named_parameters() if p.requires_grad]
    assert trainable == [], f"Expected no trainable params after freeze, got {trainable}"
    print("  PASS: test_eval_mode")


def main() -> int:
    print("MicroVAE Smoke Tests")
    print("=" * 40)
    test_forward_shape()
    test_param_count()
    test_micro_decode_returns_pil()
    test_output_pixel_range()
    test_eval_mode()
    print("=" * 40)
    print("All MicroVAE smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
