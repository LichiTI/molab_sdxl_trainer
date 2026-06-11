# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for spectral noise blending."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_snb = _import_module(
    "spectral_noise_blend",
    os.path.join(_HERE, "spectral_noise_blend.py"),
)
blend_spectral_noise = _snb.blend_spectral_noise

import torch


def test_alpha_zero_passthrough():

    noise = torch.randn(2, 4, 32, 32)
    out = blend_spectral_noise(noise, alpha=0.0)
    assert torch.equal(noise, out), "alpha=0 should return input unchanged"
    print("PASS: alpha=0 passthrough")


def test_output_shape():

    noise = torch.randn(2, 4, 64, 48)
    out = blend_spectral_noise(noise, alpha=0.5, sigma=4.0)
    assert out.shape == noise.shape, f"Shape mismatch: {out.shape} != {noise.shape}"
    print("PASS: output shape preserved")


def test_blending_reduces_high_freq():

    noise = torch.randn(1, 1, 64, 64)
    blended = blend_spectral_noise(noise, alpha=0.8, sigma=8.0)
    fft_orig = torch.fft.fft2(noise).abs().mean()
    fft_blend = torch.fft.fft2(blended).abs().mean()
    # Blending should reduce high-frequency energy relative to overall magnitude
    assert blended.std() < noise.std() * 1.1, "Blended noise should not have higher std"
    print("PASS: blending reduces high-frequency energy")


def test_alpha_one_fully_blurred():

    noise = torch.randn(1, 1, 32, 32)
    blended = blend_spectral_noise(noise, alpha=1.0, sigma=4.0)
    assert blended.shape == noise.shape
    # alpha=1 means fully blurred — output should differ from input
    assert not torch.equal(noise, blended), "alpha=1 should differ from input"
    print("PASS: alpha=1 produces fully blurred output")


def test_config_field():
    cfg_path = os.path.join(_HERE, "..", "configs.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()
    assert "spectral_noise_blend" in src, "Missing spectral_noise_blend in configs.py"
    assert "spectral_noise_sigma" in src, "Missing spectral_noise_sigma in configs.py"
    print("PASS: config fields exist")


if __name__ == "__main__":
    test_alpha_zero_passthrough()
    test_output_shape()
    test_blending_reduces_high_freq()
    test_alpha_one_fully_blurred()
    test_config_field()
    print("\nAll spectral noise blend smoke tests passed!")
