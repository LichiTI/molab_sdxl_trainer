"""Smoke test for DynamicResourceManager: VRAM usage, OOM handling, step, peak tracking."""
from __future__ import annotations

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load resource_manager via importlib from core/training_components/resource_manager.py
_rm_path = os.path.join(_HERE, "..", "..", "core", "training_components", "resource_manager.py")
_rm_spec = importlib.util.spec_from_file_location(
    "core.training_components.resource_manager",
    _rm_path,
)
_rm_mod = importlib.util.module_from_spec(_rm_spec)
sys.modules["core.training_components.resource_manager"] = _rm_mod
_rm_spec.loader.exec_module(_rm_mod)

DynamicResourceManager = _rm_mod.DynamicResourceManager
ResourceConfig = _rm_mod.ResourceConfig

import torch


def test_vram_usage_no_cuda():
    """VRAM usage reports ok when no CUDA is available."""
    rm = DynamicResourceManager(ResourceConfig())
    vram = rm.get_vram_usage()
    assert "used_gb" in vram
    assert "total_gb" in vram
    assert "usage_ratio" in vram
    # On non-CUDA machines, values should be 0
    if not torch.cuda.is_available():
        assert vram["used_gb"] == 0
        assert vram["total_gb"] == 0
        assert vram["usage_ratio"] == 0


def test_handle_oom_returns_gracefully():
    """handle_oom returns a boolean without crashing on CPU-only machines."""
    rm = DynamicResourceManager(ResourceConfig())
    rm.current_batch_size = 2
    rm.current_accumulation = 4
    result = rm.handle_oom()
    assert isinstance(result, bool), f"handle_oom should return bool, got {type(result)}"


def test_step_returns_valid_status():
    """step() returns a dict with all expected keys."""
    rm = DynamicResourceManager(ResourceConfig(enable_adaptive_batch=False))
    rm.current_batch_size = 2
    rm.current_accumulation = 4
    result = rm.step()
    assert "vram_status" in result
    assert "batch_adjusted" in result
    assert "new_batch_size" in result
    assert "new_accumulation" in result
    assert "checkpointing_enabled" in result
    assert "cpu_offload_enabled" in result
    # On CPU-only, status should be "ok"
    if not torch.cuda.is_available():
        assert result["vram_status"] == "ok"


def test_peak_tracking_works():
    """Peak VRAM tracking correctly records values."""
    rm = DynamicResourceManager(
        ResourceConfig(enable_peak_tracking=True, peak_tracking_window=5)
    )
    rm.current_batch_size = 1
    rm.current_accumulation = 1
    for _ in range(3):
        rm.step()
    peak = rm.get_peak_vram_gb()
    assert isinstance(peak, float), f"Peak should be float, got {type(peak)}"
    assert peak >= 0.0, f"Peak should be non-negative, got {peak}"


def test_check_vram_valid_status():
    """check_vram returns one of the valid status strings."""
    rm = DynamicResourceManager(ResourceConfig())
    status = rm.check_vram()
    assert status in ("ok", "warning", "critical", "emergency"), (
        f"Unexpected VRAM status: {status}"
    )


def test_handle_oom_reduces_batch():
    """handle_oom reduces batch_size when possible."""
    rm = DynamicResourceManager(
        ResourceConfig(min_batch_size=1, max_batch_size=8, enable_adaptive_batch=True)
    )
    rm.current_batch_size = 4
    rm.current_accumulation = 2
    recovered = rm.handle_oom()
    assert recovered is True, "Should recover from OOM"
    assert rm.current_batch_size < 4, f"Batch should have been reduced: {rm.current_batch_size}"


if __name__ == "__main__":
    print("ResourceManager Smoke Tests")
    print("=" * 40)
    test_vram_usage_no_cuda()
    print("PASS: vram_usage_no_cuda")
    test_handle_oom_returns_gracefully()
    print("PASS: handle_oom_returns_gracefully")
    test_step_returns_valid_status()
    print("PASS: step_returns_valid_status")
    test_peak_tracking_works()
    print("PASS: peak_tracking_works")
    test_check_vram_valid_status()
    print("PASS: check_vram_valid_status")
    test_handle_oom_reduces_batch()
    print("PASS: handle_oom_reduces_batch")
    print("=" * 40)
    print("All ResourceManager smoke tests passed!")
