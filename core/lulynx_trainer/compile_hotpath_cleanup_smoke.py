"""Smoke test for compile hot-path cleanup.

Validates that:
1. CompileOptimizedOps methods produce correct results
2. Operations are torch.compile-safe (no graph breaks)
3. Optimized loss computation matches reference implementation
4. Graph break monitoring works correctly
"""
from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path for relative imports
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compile_hotpath_cleanup import (
    CompileOptimizedOps,
    optimize_loss_computation,
    create_compile_friendly_forward_wrapper,
    GraphBreakMonitor,
)


# ---------------------------------------------------------------------------
# Test 1: CompileOptimizedOps correctness
# ---------------------------------------------------------------------------

def test_compile_optimized_ops_correctness():
    """Verify that optimized ops produce correct results."""
    ops = CompileOptimizedOps()

    # Test safe_mean
    x = torch.randn(4, 8, 16)
    assert torch.allclose(ops.safe_mean(x), x.mean())
    assert torch.allclose(ops.safe_mean(x, dim=1), x.mean(dim=1))

    # Test safe_sum
    assert torch.allclose(ops.safe_sum(x), x.sum())
    assert torch.allclose(ops.safe_sum(x, dim=(1, 2)), x.sum(dim=(1, 2)))

    # Test masked_loss
    loss = torch.randn(4, 8, 16)
    mask = torch.randint(0, 2, (4, 8, 16)).bool()

    # With mask
    result = ops.masked_loss(loss, mask, reduction="mean")
    expected = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    assert torch.allclose(result, expected)

    # Without mask
    result_no_mask = ops.masked_loss(loss, None, reduction="mean")
    assert torch.allclose(result_no_mask, loss.mean())

    # Test safe_clamp
    x = torch.randn(10)
    assert torch.allclose(ops.safe_clamp(x, min_val=-1.0, max_val=1.0), x.clamp(-1.0, 1.0))
    assert torch.allclose(ops.safe_clamp(x, min_val=-1.0), x.clamp(min=-1.0))

    # Test safe_normalize
    x = torch.randn(4, 8)
    normalized = ops.safe_normalize(x, dim=-1)
    norms = torch.norm(normalized, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Test weighted_mean
    loss = torch.randn(4, 8)
    weights = torch.rand(4, 8)
    result = ops.weighted_mean(loss, weights)
    expected = (loss * weights).sum() / weights.sum()
    assert torch.allclose(result, expected)

    # Test reduce_loss_per_sample
    loss_4d = torch.randn(4, 3, 8, 8)
    reduced = ops.reduce_loss_per_sample(loss_4d)
    assert reduced.shape == (4,)
    expected = loss_4d.mean(dim=(1, 2, 3))
    assert torch.allclose(reduced, expected)

    print("PASS: test_compile_optimized_ops_correctness")
    return True


# ---------------------------------------------------------------------------
# Test 2: torch.compile compatibility
# ---------------------------------------------------------------------------

def test_torch_compile_compatibility():
    """Verify that operations are torch.compile-safe."""
    if not hasattr(torch, "compile"):
        print("SKIP: test_torch_compile_compatibility — torch.compile not available")
        return True

    ops = CompileOptimizedOps()

    # Create a function that uses optimized ops
    def loss_fn(pred, target, mask):
        loss = F.mse_loss(pred, target, reduction="none")
        return ops.masked_loss(loss, mask, reduction="mean")

    # Compile it
    try:
        compiled_fn = torch.compile(loss_fn, backend="eager", fullgraph=True)

        # Run it
        pred = torch.randn(4, 8, 16)
        target = torch.randn(4, 8, 16)
        mask = torch.randint(0, 2, (4, 8, 16)).bool()

        result = compiled_fn(pred, target, mask)
        assert result.shape == ()  # Scalar output

        print("PASS: test_torch_compile_compatibility")
        return True
    except Exception as e:
        print(f"FAIL: test_torch_compile_compatibility — {e}")
        return False


# ---------------------------------------------------------------------------
# Test 3: Optimized loss computation
# ---------------------------------------------------------------------------

def test_optimized_loss_computation():
    """Verify optimized loss computation matches reference."""
    pred = torch.randn(4, 3, 8, 8)
    target = torch.randn(4, 3, 8, 8)
    mask = torch.randint(0, 2, (4, 3, 8, 8)).bool()
    weights = torch.rand(4, 1, 1, 1)

    # Test with mask
    result = optimize_loss_computation(
        pred, target, F.mse_loss, mask=mask, reduction="mean"
    )
    loss_ref = F.mse_loss(pred, target, reduction="none")
    expected = (loss_ref * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    # Debug output
    if not torch.allclose(result, expected):
        print(f"DEBUG: result={result.item():.6f}, expected={expected.item():.6f}")
        print(f"DEBUG: diff={abs(result.item() - expected.item()):.6e}")

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-6)

    # Test with weights
    result_weighted = optimize_loss_computation(
        pred, target, F.mse_loss, weights=weights, reduction="mean"
    )
    loss_ref = F.mse_loss(pred, target, reduction="none")
    # weighted_mean computes (loss * weights).sum() / weights.sum(), not .mean()
    expected_weighted = (loss_ref * weights).sum() / weights.sum()

    # Debug output
    if not torch.allclose(result_weighted, expected_weighted):
        print(f"DEBUG weights: result={result_weighted.item():.6f}, expected={expected_weighted.item():.6f}")
        print(f"DEBUG weights: diff={abs(result_weighted.item() - expected_weighted.item()):.6e}")

    assert torch.allclose(result_weighted, expected_weighted)

    # Test without mask or weights
    result_plain = optimize_loss_computation(
        pred, target, F.mse_loss, reduction="mean"
    )
    expected_plain = F.mse_loss(pred, target, reduction="mean")
    assert torch.allclose(result_plain, expected_plain)

    print("PASS: test_optimized_loss_computation")
    return True


# ---------------------------------------------------------------------------
# Test 4: Forward wrapper
# ---------------------------------------------------------------------------

def test_forward_wrapper():
    """Verify forward wrapper works correctly."""
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
            return self.linear(sample)

    model = SimpleModel()
    wrapper = create_compile_friendly_forward_wrapper(model)

    # Test forward pass
    sample = torch.randn(4, 8)
    timestep = torch.tensor([100, 200, 300, 400])
    encoder_hidden_states = torch.randn(4, 8)

    result = wrapper(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )

    assert result.shape == (4, 8)

    print("PASS: test_forward_wrapper")
    return True


# ---------------------------------------------------------------------------
# Test 5: Graph break monitoring
# ---------------------------------------------------------------------------

def test_graph_break_monitoring():
    """Verify graph break monitoring works."""
    monitor = GraphBreakMonitor()

    # Log some breaks
    monitor.log_graph_break("dynamic control flow", "loss_computation:42")
    monitor.log_graph_break("tensor.item() call", "validation:15")

    summary = monitor.get_summary()
    assert summary["total_breaks"] == 2
    assert summary["recompile_count"] == 2
    assert len(summary["breaks"]) == 2

    # Reset
    monitor.reset()
    summary_after = monitor.get_summary()
    assert summary_after["total_breaks"] == 0

    print("PASS: test_graph_break_monitoring")
    return True


# ---------------------------------------------------------------------------
# Test 6: Compile-safe vs unsafe comparison
# ---------------------------------------------------------------------------

def test_compile_safe_vs_unsafe():
    """Compare compile-safe ops with unsafe patterns."""
    ops = CompileOptimizedOps()

    # Pattern 1: Conditional scaling
    x = torch.randn(4, 8)
    scale = 2.0
    condition = True

    # Unsafe pattern (causes graph break):
    # if condition:
    #     result_unsafe = x * scale
    # else:
    #     result_unsafe = x

    # Safe pattern:
    result_safe = ops.conditional_scale(x, scale, condition)
    expected = x * scale if condition else x
    assert torch.allclose(result_safe, expected)

    # Pattern 2: Masked loss with None check
    loss = torch.randn(4, 8, 16)
    mask = torch.randint(0, 2, (4, 8, 16)).bool()

    # Unsafe pattern (causes graph break):
    # if mask is not None:
    #     result_unsafe = (loss * mask).sum() / mask.sum()
    # else:
    #     result_unsafe = loss.mean()

    # Safe pattern:
    result_safe_with_mask = ops.masked_loss(loss, mask, reduction="mean")
    result_safe_no_mask = ops.masked_loss(loss, None, reduction="mean")

    expected_with_mask = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    expected_no_mask = loss.mean()

    assert torch.allclose(result_safe_with_mask, expected_with_mask)
    assert torch.allclose(result_safe_no_mask, expected_no_mask)

    print("PASS: test_compile_safe_vs_unsafe")
    return True


# ---------------------------------------------------------------------------
# Test 7: Expand mask to loss
# ---------------------------------------------------------------------------

def test_expand_mask_to_loss():
    """Verify mask expansion works correctly."""
    ops = CompileOptimizedOps()

    # Test different dimension differences
    loss_4d = torch.randn(4, 3, 8, 8)
    mask_1d = torch.randint(0, 2, (4,)).bool()
    mask_2d = torch.randint(0, 2, (4, 3)).bool()

    # Expand 1D mask to 4D
    expanded_1d = ops.expand_mask_to_loss(mask_1d, loss_4d)
    assert expanded_1d.shape == loss_4d.shape

    # Expand 2D mask to 4D
    expanded_2d = ops.expand_mask_to_loss(mask_2d, loss_4d)
    assert expanded_2d.shape == loss_4d.shape

    # Verify values are correct
    for i in range(4):
        if mask_1d[i]:
            assert expanded_1d[i].all()
        else:
            assert not expanded_1d[i].any()

    print("PASS: test_expand_mask_to_loss")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []
    tests = [
        test_compile_optimized_ops_correctness,
        test_torch_compile_compatibility,
        test_optimized_loss_computation,
        test_forward_wrapper,
        test_graph_break_monitoring,
        test_compile_safe_vs_unsafe,
        test_expand_mask_to_loss,
    ]

    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {e}")
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("Compile Hot-Path Cleanup Smoke Test Results")
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
