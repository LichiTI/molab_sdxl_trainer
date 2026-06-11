# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Perlin noise generation utilities."""

from __future__ import annotations
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pn = _import_module("perlin_noise", os.path.join(_HERE, "perlin_noise.py"))
generate_perlin_2d = _pn.generate_perlin_2d
apply_perlin_noise_offset = _pn.apply_perlin_noise_offset

import torch


def test_output_shape():
    shape = (2, 4, 64, 64)
    result = generate_perlin_2d(shape, scale=8.0)
    assert result.shape == torch.Size(shape), (
        f"Expected shape {shape}, got {result.shape}"
    )


def test_spatial_correlation():
    shape = (1, 1, 64, 64)
    noise = generate_perlin_2d(shape, scale=8.0)
    # Adjacent pixel differences (horizontal neighbors)
    neighbor_diff = (noise[:, :, :, 1:] - noise[:, :, :, :-1]).abs().mean().item()
    # Random-pair differences: shuffle spatial positions
    flat = noise.view(-1)
    idx_a = torch.randperm(flat.numel())
    idx_b = torch.randperm(flat.numel())
    random_diff = (flat[idx_a] - flat[idx_b]).abs().mean().item()
    assert neighbor_diff < random_diff, (
        f"Adjacent pixels should be more similar than random pairs. "
        f"neighbor_diff={neighbor_diff:.4f}, random_diff={random_diff:.4f}"
    )


def test_strength_zero_noop():
    shape = (1, 4, 32, 32)
    noise = generate_perlin_2d(shape, scale=4.0)
    result = apply_perlin_noise_offset(noise, strength=0)
    assert torch.allclose(result, noise), (
        "apply_perlin_noise_offset with strength=0 should return noise unchanged"
    )


def test_different_scales():
    shape = (1, 1, 64, 64)
    low_scale = generate_perlin_2d(shape, scale=2.0)
    high_scale = generate_perlin_2d(shape, scale=16.0)
    # Different scales should produce meaningfully different patterns
    diff = (low_scale - high_scale).abs().mean().item()
    assert diff > 1e-4, (
        f"scale=2.0 and scale=16.0 should produce different patterns, mean diff={diff}"
    )


def test_device_dtype():
    shape = (1, 2, 32, 32)
    result = generate_perlin_2d(shape, scale=4.0)
    assert result.device.type == "cpu", f"Expected CPU tensor, got {result.device}"
    assert result.dtype == torch.float32, (
        f"Expected float32, got {result.dtype}"
    )


if __name__ == "__main__":
    test_output_shape()
    test_spatial_correlation()
    test_strength_zero_noop()
    test_different_scales()
    test_device_dtype()
    print("perlin_noise_smoke: all tests passed")
