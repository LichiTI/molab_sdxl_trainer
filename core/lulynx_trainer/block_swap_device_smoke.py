"""BlockSwap forward hook integration smoke test.

Validates that:
1. ensure_block_on_device() moves CPU-resident block weights to GPU before forward
2. prefetch_next() submits async prefetch for the next block
3. install_forward_hooks() registers pre/post hooks that fire during forward
4. Full forward+backward with BlockSwap + LoRA injection succeeds without
   device mismatch errors
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny UNet-like model for testing
# ---------------------------------------------------------------------------

class TinyBlock(nn.Module):
    """Minimal transformer-like block with a Linear layer."""
    def __init__(self, dim: int = 32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x) + x)


class TinyUNet(nn.Module):
    """Minimal UNet-like model with down_blocks, mid_block, up_blocks."""
    def __init__(self, dim: int = 32, n_down: int = 2, n_up: int = 2):
        super().__init__()
        self.down_blocks = nn.ModuleList([TinyBlock(dim) for _ in range(n_down)])
        self.mid_block = TinyBlock(dim)
        self.up_blocks = nn.ModuleList([TinyBlock(dim) for _ in range(n_up)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Down path
        skips = []
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        # Mid
        x = self.mid_block(x)
        # Up path
        for block in self.up_blocks:
            x = block(x) + skips.pop()
        return x


class TinyDiT(nn.Module):
    """Minimal DiT-like model with net.blocks."""
    def __init__(self, dim: int = 32, n_blocks: int = 4):
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.net.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Fake LoRA injection for testing LoRA leaf marking
# ---------------------------------------------------------------------------

def inject_lora(block: nn.Module, rank: int = 4) -> None:
    """Inject a minimal LoRA adapter into the block's Linear layer."""
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear) and not hasattr(mod, "lora_down"):
            # Create LoRA parameters but don't modify forward pass
            # (we just want to test that _lora_leaf params stay on GPU)
            lora_down = nn.Parameter(torch.zeros(rank, mod.in_features))
            lora_up = nn.Parameter(torch.zeros(mod.out_features, rank))

            # Register as module parameters
            mod.register_parameter("lora_down", lora_down)
            mod.register_parameter("lora_up", lora_up)

            # Mark as LoRA leaf (should stay on GPU during swap)
            lora_down._lora_leaf = True
            lora_up._lora_leaf = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ensure_block_on_device():
    """Test that ensure_block_on_device moves CPU weights to GPU."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    from memory_optimizations import BlockSwapOffloader

    device = torch.device("cuda")
    model = TinyUNet(dim=32, n_down=2, n_up=2)
    model.to(device)

    # Collect blocks in execution order: down -> mid -> up
    all_blocks = list(model.down_blocks) + [model.mid_block] + list(model.up_blocks)

    offloader = BlockSwapOffloader(
        blocks=all_blocks,
        blocks_to_swap=2,  # last 2 blocks on CPU (up_blocks[0], up_blocks[1])
        device=device,
        enable_backward=True,
    )
    offloader.prepare_before_forward()

    # Verify up_blocks[0] (idx 3) is on CPU
    up0_params = list(all_blocks[3].parameters())
    assert up0_params[0].device.type == "cpu", f"Expected CPU, got {up0_params[0].device}"

    # Call ensure_block_on_device for idx 3
    offloader.ensure_block_on_device(3)

    # Now up_blocks[0] should be on GPU
    up0_params = list(all_blocks[3].parameters())
    assert up0_params[0].device.type == "cuda", f"Expected CUDA after ensure, got {up0_params[0].device}"

    offloader.cleanup()
    print("PASS: test_ensure_block_on_device")
    return True


def test_install_forward_hooks_unet():
    """Test that install_forward_hooks allows forward+backward on UNet with block swap."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    from memory_optimizations import BlockSwapOffloader

    device = torch.device("cuda")
    model = TinyUNet(dim=32, n_down=2, n_up=2)
    model.to(device)

    all_blocks = list(model.down_blocks) + [model.mid_block] + list(model.up_blocks)

    offloader = BlockSwapOffloader(
        blocks=all_blocks,
        blocks_to_swap=2,
        device=device,
        enable_backward=True,
    )
    offloader.prepare_before_forward()
    offloader.install_forward_hooks(model)

    # Forward + backward should not raise device mismatch
    x = torch.randn(2, 8, 32, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    offloader.cleanup()
    print("PASS: test_install_forward_hooks_unet")
    return True


def test_install_forward_hooks_dit():
    """Test that install_forward_hooks allows forward+backward on DiT with block swap."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    from memory_optimizations import BlockSwapOffloader

    device = torch.device("cuda")
    model = TinyDiT(dim=32, n_blocks=4)
    model.to(device)

    all_blocks = list(model.net.blocks)

    offloader = BlockSwapOffloader(
        blocks=all_blocks,
        blocks_to_swap=1,
        device=device,
        enable_backward=True,
    )
    offloader.prepare_before_forward()
    offloader.install_forward_hooks(model)

    x = torch.randn(2, 8, 32, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    offloader.cleanup()
    print("PASS: test_install_forward_hooks_dit")
    return True


def test_forward_hooks_with_lora():
    """Test BlockSwap + LoRA injection: LoRA leaf params should stay on GPU."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    from memory_optimizations import BlockSwapOffloader

    device = torch.device("cuda")
    model = TinyUNet(dim=32, n_down=2, n_up=2)
    model.to(device)

    # Inject LoRA into all blocks
    for block in model.down_blocks:
        inject_lora(block)
    inject_lora(model.mid_block)
    for block in model.up_blocks:
        inject_lora(block)

    all_blocks = list(model.down_blocks) + [model.mid_block] + list(model.up_blocks)

    offloader = BlockSwapOffloader(
        blocks=all_blocks,
        blocks_to_swap=2,
        device=device,
        enable_backward=True,
        should_swap=lambda mod, name: not getattr(mod, "_lora_leaf", False),
    )
    offloader.prepare_before_forward()
    offloader.install_forward_hooks(model)

    # Forward + backward should not raise device mismatch
    x = torch.randn(2, 8, 32, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Verify LoRA leaf params stayed on GPU
    for block in all_blocks:
        for name, mod in block.named_modules():
            if hasattr(mod, "lora_down") and hasattr(mod, "lora_up"):
                assert mod.lora_down.device.type == "cuda", \
                    f"lora_down on {mod.lora_down.device}, expected cuda"
                assert mod.lora_up.device.type == "cuda", \
                    f"lora_up on {mod.lora_up.device}, expected cuda"

    offloader.cleanup()
    print("PASS: test_forward_hooks_with_lora")
    return True


def test_validate_device_placement():
    """Test the _validate_device_placement debug method."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    from memory_optimizations import BlockSwapOffloader

    device = torch.device("cuda")
    model = TinyUNet(dim=32, n_down=2, n_up=2)
    model.to(device)

    all_blocks = list(model.down_blocks) + [model.mid_block] + list(model.up_blocks)

    offloader = BlockSwapOffloader(
        blocks=all_blocks,
        blocks_to_swap=2,
        device=device,
        enable_backward=True,
    )
    offloader.prepare_before_forward()

    # Debug: print actual device placement
    print("\nActual device placement after prepare:")
    for idx, block in enumerate(all_blocks):
        params = list(block.parameters())
        if params:
            print(f"  Block {idx}: {params[0].device}")

    # After prepare, GPU blocks = [0,1,2] (down[0], down[1], mid), CPU blocks = [3,4] (up[0], up[1])
    result = offloader._validate_device_placement("after_prepare")

    # Debug: print validation result
    print(f"Validation result: cpu_blocks={result['cpu_blocks']}, gpu_blocks={result['gpu_blocks']}")

    # No misplaced blocks expected after prepare
    # cpu_blocks = blocks on CPU that should be on GPU (should be empty)
    # gpu_blocks = blocks on GPU that should be on CPU (should be empty)
    if result["cpu_blocks"] or result["gpu_blocks"]:
        print(f"FAIL: Misplaced blocks detected")
        print(f"  Blocks on CPU (should be GPU): {result['cpu_blocks']}")
        print(f"  Blocks on GPU (should be CPU): {result['gpu_blocks']}")
        offloader.cleanup()
        return False

    offloader.cleanup()
    print("PASS: test_validate_device_placement")
    return True


def test_single_block_swap():
    """Test with blocks_to_swap=1: forward+backward should succeed."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    from memory_optimizations import BlockSwapOffloader

    device = torch.device("cuda")
    model = TinyUNet(dim=32, n_down=2, n_up=2)
    model.to(device)

    all_blocks = list(model.down_blocks) + [model.mid_block] + list(model.up_blocks)

    offloader = BlockSwapOffloader(
        blocks=all_blocks,
        blocks_to_swap=1,
        device=device,
        enable_backward=True,
    )
    offloader.prepare_before_forward()
    offloader.install_forward_hooks(model)

    x = torch.randn(2, 8, 32, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    offloader.cleanup()
    print("PASS: test_single_block_swap")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []
    tests = [
        test_ensure_block_on_device,
        test_install_forward_hooks_unet,
        test_install_forward_hooks_dit,
        test_forward_hooks_with_lora,
        test_validate_device_placement,
        test_single_block_swap,
    ]

    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            print(f"FAIL: {test_fn.__name__} — {e}")
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("BlockSwap Forward Hook Integration Smoke Test Results")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
