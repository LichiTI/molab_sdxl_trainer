"""Smoke tests for VRAM swap and CPU offload functionality."""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_adapter_cpu_residency():
    """Test AdapterCPUResidency registers parameters and estimates savings."""
    from core.lulynx_trainer.memory_optimizations import AdapterCPUResidency

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residency = AdapterCPUResidency(device=device)

    # Create some trainable parameters
    params = [nn.Parameter(torch.randn(64, 64)) for _ in range(4)]
    count = residency.register_parameters(params)
    assert count == 4, f"Expected 4 registered, got {count}"

    savings = residency.estimate_vram_savings_mb()
    # 4 * 64 * 64 * 4 bytes (float32) = 65536 bytes ≈ 0.0625 MB
    expected_mb = 4 * 64 * 64 * 4 / (1024 ** 2)
    assert abs(savings - expected_mb) < 0.01, f"Expected ~{expected_mb:.4f} MB, got {savings:.4f}"
    print("  [PASS] adapter_cpu_residency")


def test_block_swap_offloader_construction():
    """Test BlockSwapOffloader construction and prepare_before_forward."""
    from core.lulynx_trainer.memory_optimizations import BlockSwapOffloader

    device = torch.device("cpu")  # Use CPU for testing; real usage is CUDA

    # Create a simple model with "blocks"
    blocks = nn.ModuleList([nn.Linear(32, 32) for _ in range(6)])
    offloader = BlockSwapOffloader(
        blocks=blocks,
        blocks_to_swap=2,
        device=device,
        enable_backward=False,
        should_swap=lambda mod, name: not getattr(mod, "_lora_leaf", False),
    )
    assert offloader.num_blocks == 6
    assert offloader.blocks_to_swap == 2
    print("  [PASS] block_swap_offloader_construction")


def test_block_swap_ensure_and_prefetch():
    """Test ensure_block_on_device and prefetch_next methods."""
    from core.lulynx_trainer.memory_optimizations import BlockSwapOffloader

    device = torch.device("cpu")
    blocks = nn.ModuleList([nn.Linear(32, 32) for _ in range(6)])
    offloader = BlockSwapOffloader(
        blocks=blocks,
        blocks_to_swap=2,
        device=device,
        enable_backward=False,
    )
    # These should not raise
    offloader.ensure_block_on_device(0)  # GPU resident block
    offloader.ensure_block_on_device(5)  # CPU swap block
    offloader.prefetch_next(3)  # prefetch next after block 3
    print("  [PASS] block_swap_ensure_and_prefetch")


def test_move_weights_to_device():
    """Test _move_weights_to_device utility."""
    from core.lulynx_trainer.memory_optimizations import _move_weights_to_device

    device = torch.device("cpu")
    model = nn.Linear(32, 32)
    # Should not raise
    _move_weights_to_device(model, device, should_swap=None)
    print("  [PASS] move_weights_to_device")


def test_resource_manager_vram_check():
    """Test DynamicResourceManager VRAM check returns valid status."""
    from core.training_components.resource_manager import DynamicResourceManager, ResourceConfig

    config = ResourceConfig(
        vram_warning_threshold=0.85,
        vram_critical_threshold=0.92,
        vram_emergency_threshold=0.97,
    )
    rm = DynamicResourceManager(config)
    status = rm.check_vram()
    assert status in ("ok", "warning", "critical", "emergency"), f"Unexpected status: {status}"

    vram = rm.get_vram_usage()
    assert "used_gb" in vram
    assert "total_gb" in vram
    assert "usage_ratio" in vram
    print("  [PASS] resource_manager_vram_check")


def test_resource_manager_step():
    """Test DynamicResourceManager.step() returns valid result."""
    from core.training_components.resource_manager import DynamicResourceManager, ResourceConfig

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
    print("  [PASS] resource_manager_step")


def test_resource_manager_peak_tracking():
    """Test peak VRAM tracking."""
    from core.training_components.resource_manager import DynamicResourceManager, ResourceConfig

    rm = DynamicResourceManager(ResourceConfig(enable_peak_tracking=True, peak_tracking_window=5))
    for _ in range(3):
        rm.step()
    peak = rm.get_peak_vram_gb()
    assert isinstance(peak, float)
    assert peak >= 0.0
    print("  [PASS] resource_manager_peak_tracking")


def test_sdxl_low_vram_config():
    """Test that sdxl_low_vram_optimization config field exists and defaults to False."""
    from core.lulynx_trainer.config import LulynxConfig
    cfg = LulynxConfig()
    assert hasattr(cfg, "sdxl_low_vram_optimization")
    assert cfg.sdxl_low_vram_optimization is False
    print("  [PASS] sdxl_low_vram_config")


if __name__ == "__main__":
    print("VRAM Swap / CPU Offload Smoke Tests")
    print("=" * 40)
    test_adapter_cpu_residency()
    test_block_swap_offloader_construction()
    test_block_swap_ensure_and_prefetch()
    test_move_weights_to_device()
    test_resource_manager_vram_check()
    test_resource_manager_step()
    test_resource_manager_peak_tracking()
    test_sdxl_low_vram_config()
    print("=" * 40)
    print("All VRAM smoke tests passed!")
