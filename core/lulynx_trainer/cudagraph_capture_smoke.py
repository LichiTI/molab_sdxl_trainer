"""Smoke tests for CUDAGraph capture.

Tests run on CPU (no CUDA required) and verify:
  1. cudagraph_available() detection
  2. CUDAGraphCapture construction and warmup (CPU-safe)
  3. Static tensor allocation shape/dtype matching
  4. _copy_to_static copies data correctly
  5. replay raises RuntimeError when not captured
"""

from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

_cg = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.cudagraph_capture",
    os.path.join(_HERE, "cudagraph_capture.py"),
)
_cg_mod = importlib.util.module_from_spec(_cg)
sys.modules["core.lulynx_trainer.cudagraph_capture"] = _cg_mod
_cg.loader.exec_module(_cg_mod)

import torch
import torch.nn as nn


class _SimpleModel(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x) + x


class _DictModel(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, *, scale=1.0):
        return self.linear(x) * scale


def test_cudagraph_available():
    """cudagraph_available() should return bool."""
    result = _cg_mod.cudagraph_available()
    assert isinstance(result, bool)
    print(f"PASS: cudagraph_available() = {result}")


def test_capture_construction():
    """CUDAGraphCapture should be constructable on CPU."""
    model = _SimpleModel()
    sample = torch.randn(2, 16)
    capture = _cg_mod.CUDAGraphCapture(model, sample, device="cpu")
    assert not capture.is_captured
    print("PASS: CUDAGraphCapture construction")


def test_warmup_cpu():
    """Warmup should run without errors on CPU."""
    model = _SimpleModel()
    sample = torch.randn(2, 16)
    capture = _cg_mod.CUDAGraphCapture(model, sample, device="cpu")
    capture.warmup(num_steps=2)
    assert not capture.is_captured
    print("PASS: Warmup on CPU")


def test_static_tensor_allocation():
    """_make_static should create tensors with matching shape/dtype."""
    model = _SimpleModel()
    sample = torch.randn(2, 16, dtype=torch.float32)
    capture = _cg_mod.CUDAGraphCapture(model, sample, device="cpu")
    static = capture._make_static(sample)
    assert static.shape == sample.shape, f"Shape mismatch: {static.shape} vs {sample.shape}"
    assert static.dtype == sample.dtype
    assert torch.allclose(static, torch.zeros_like(sample))
    print("PASS: Static tensor allocation shape/dtype")


def test_static_dict_allocation():
    """_make_static should handle dict inputs."""
    model = _DictModel()
    sample_dict = {"x": torch.randn(2, 16), "scale": 1.0}
    capture = _cg_mod.CUDAGraphCapture(model, sample_dict, device="cpu")
    static = capture._make_static(sample_dict)
    assert isinstance(static, dict)
    assert static["x"].shape == (2, 16)
    assert static["scale"] == 1.0
    print("PASS: Static dict allocation")


def test_copy_to_static():
    """_copy_to_static should copy data into static tensors."""
    model = _SimpleModel()
    sample = torch.randn(2, 16)
    capture = _cg_mod.CUDAGraphCapture(model, sample, device="cpu")
    capture._static_inputs = capture._make_static(sample)
    new_data = torch.ones(2, 16) * 5.0
    capture._copy_to_static(new_data)
    assert torch.allclose(capture._static_inputs, new_data)
    print("PASS: _copy_to_static copies data")


def test_replay_without_capture_raises():
    """replay() should raise RuntimeError when not captured."""
    model = _SimpleModel()
    sample = torch.randn(2, 16)
    capture = _cg_mod.CUDAGraphCapture(model, sample, device="cpu")
    try:
        capture.replay(sample)
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "not captured" in str(e).lower()
    print("PASS: replay without capture raises RuntimeError")


if __name__ == "__main__":
    test_cudagraph_available()
    test_capture_construction()
    test_warmup_cpu()
    test_static_tensor_allocation()
    test_static_dict_allocation()
    test_copy_to_static()
    test_replay_without_capture_raises()
    print("\nAll CUDAGraph capture smoke tests passed!")
