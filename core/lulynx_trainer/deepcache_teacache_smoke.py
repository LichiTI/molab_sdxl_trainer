# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for DeepCache / TeaCache driven through the unified cache seam.

The Spectrum/SmoothCache seam smoke lives in ``unified_cache_seam_smoke.py``;
this file proves the two *self-contained* block-level backends added alongside
them:

* **Default-off parity** -- ``backend="none"`` reproduces the plain block loop
  bitwise, and an enabled backend with no reuse opportunity still matches.
* **DeepCache** reuses deep-block outputs on non-key steps (compute count drops)
  while shallow blocks and key steps always recompute.
* **TeaCache** reuses a block's output while its input is changing slowly
  (identical inputs -> guaranteed reuse after warmup) and recomputes once the
  input moves or the consecutive-skip bound is hit.

Run directly:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/deepcache_teacache_smoke.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch
import torch.nn as nn

from core.lulynx_trainer.unified_cache_seam import build_cache_seam
from core.lulynx_trainer.deepcache import DeepCacheStore
from core.lulynx_trainer.teacache import TeaCacheStore


class _CountingBlock(nn.Module):
    """A deterministic DiT-ish block that records how often it actually runs."""

    def __init__(self, dim: int, scale: float) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(dim) * scale)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + 0.01 * torch.tanh(self.lin(x))


def _make_blocks(n: int, dim: int) -> "list[_CountingBlock]":
    return [_CountingBlock(dim, scale=1.0 + 0.1 * i) for i in range(n)]


def _reset_calls(blocks) -> None:
    for b in blocks:
        b.calls = 0


def _run_denoise(seam, blocks, x0, n_steps, step_delta):
    """Mimic ``_run_blocks`` inside a denoise loop driving the seam.

    Each step walks blocks ``0..N-1`` in order through ``seam.run_block`` (the
    self-contained stores detect the step boundary from that wrap-around), then
    perturbs ``x`` by ``step_delta`` to imitate the sampler's per-step update.
    """
    x = x0.clone()
    for _step in range(n_steps):
        for bi, block in enumerate(blocks):
            x = seam.run_block(block, bi, x)
        x = x + step_delta
    return x


def _plain_denoise(blocks, x0, n_steps, step_delta):
    x = x0.clone()
    for _step in range(n_steps):
        for block in blocks:
            x = block(x)
        x = x + step_delta
    return x


def test_default_off_is_bitwise_parity() -> None:
    torch.manual_seed(0)
    dim, n = 8, 6
    blocks = _make_blocks(n, dim)
    x0 = torch.randn(2, dim)
    delta = torch.full((2, dim), 0.05)

    with torch.no_grad():
        baseline = _plain_denoise(blocks, x0, n_steps=5, step_delta=delta)
        _reset_calls(blocks)
        seam_none = build_cache_seam(enabled=True, backend="none")
        out_none = _run_denoise(seam_none, blocks, x0, n_steps=5, step_delta=delta)

    assert torch.equal(baseline, out_none), "backend=none diverged from the plain loop"
    # Every block ran every step (no reuse) -> 5 calls each.
    assert all(b.calls == 5 for b in blocks), "backend=none skipped a block"
    print("PASS: default-off (backend=none) is bitwise parity, recomputes every block")


def test_deepcache_reuses_deep_blocks_on_non_key_steps() -> None:
    torch.manual_seed(1)
    dim, n = 8, 10
    blocks = _make_blocks(n, dim)
    x0 = torch.randn(1, dim)
    delta = torch.full((1, dim), 0.02)

    seam = build_cache_seam(
        enabled=True, backend="deepcache", deepcache_interval=3, deepcache_deep_fraction=0.4
    )
    with torch.no_grad():
        out = _run_denoise(seam, blocks, x0, n_steps=6, step_delta=delta)

    assert out.shape == x0.shape, "DeepCache changed the output shape"
    # deep_fraction 0.4 of 10 blocks -> blocks [0..3] shallow (always run),
    # blocks [4..9] deep (reuse on non-key steps).
    shallow_calls = [blocks[i].calls for i in range(4)]
    deep_calls = [blocks[i].calls for i in range(4, n)]
    assert all(c == 6 for c in shallow_calls), f"shallow blocks must always run, got {shallow_calls}"
    assert all(c < 6 for c in deep_calls), f"deep blocks must reuse on non-key steps, got {deep_calls}"
    stats = seam.stats()
    assert stats["reuse_block_calls"] > 0, "DeepCache never reused a deep block"
    print(
        f"PASS: DeepCache reuses deep blocks on non-key steps "
        f"(shallow={shallow_calls[0]}/6, deep={deep_calls[0]}/6, reuses={stats['reuse_block_calls']})"
    )


def test_teacache_reuses_on_slow_input_then_recomputes() -> None:
    torch.manual_seed(2)
    dim, n = 8, 4
    # Identity-ish blocks + zero step delta -> the per-block input repeats exactly
    # across steps, so rel-L1 == 0 and TeaCache must reuse after warmup.
    blocks = _make_blocks(n, dim)
    x0 = torch.randn(1, dim)
    zero_delta = torch.zeros(1, dim)

    seam = build_cache_seam(enabled=True, backend="teacache", teacache_rel_l1_threshold=0.05)
    with torch.no_grad():
        out = _run_denoise(seam, blocks, x0, n_steps=6, step_delta=zero_delta)

    assert out.shape == x0.shape, "TeaCache changed the output shape"
    stats = seam.stats()
    assert stats["reuse_block_calls"] > 0, "TeaCache never reused on a slow-moving input"
    # warmup_steps=2 default -> first 2 steps always compute; with N blocks the
    # minimum compute count is 2*N (warmup) but bounded reuse forces periodic
    # recompute, so total calls sit strictly between full (6*N) and warmup (2*N).
    total_calls = sum(b.calls for b in blocks)
    assert 2 * n <= total_calls < 6 * n, f"TeaCache compute count out of expected band: {total_calls}"
    print(
        f"PASS: TeaCache reuses on slow input after warmup "
        f"(reuses={stats['reuse_block_calls']}, total_calls={total_calls} of {6*n})"
    )


def test_teacache_fast_input_recomputes() -> None:
    torch.manual_seed(3)
    dim, n = 8, 4
    blocks = _make_blocks(n, dim)
    x0 = torch.randn(1, dim)
    # Large per-step delta -> rel-L1 exceeds threshold every step -> no reuse.
    big_delta = torch.full((1, dim), 5.0)

    seam = build_cache_seam(enabled=True, backend="teacache", teacache_rel_l1_threshold=0.001)
    with torch.no_grad():
        _run_denoise(seam, blocks, x0, n_steps=5, step_delta=big_delta)

    # No block should have been reused -> every block ran every step.
    assert all(b.calls == 5 for b in blocks), f"TeaCache wrongly reused on a fast-moving input: {[b.calls for b in blocks]}"
    stats = seam.stats()
    assert stats["reuse_block_calls"] == 0, f"expected zero reuse on fast input, got {stats['reuse_block_calls']}"
    print("PASS: TeaCache recomputes every block when the input moves fast (no premature skip)")


def test_store_clear_resets_state() -> None:
    dc = DeepCacheStore(interval=2, deep_fraction=0.5)
    dc.decide(0)
    dc.push(0, torch.zeros(2))
    dc.clear()
    assert dc.stats()["cached_blocks"] == 0 and dc.stats()["reuse_block_calls"] == 0

    tc = TeaCacheStore(rel_l1_threshold=0.1)
    tc.decide(0, torch.zeros(2))
    tc.push(0, torch.zeros(2))
    tc.clear()
    assert tc.stats()["cached_blocks"] == 0 and tc.stats()["reuse_block_calls"] == 0
    print("PASS: DeepCache/TeaCache clear() resets cache + counters")


def main() -> int:
    test_default_off_is_bitwise_parity()
    test_deepcache_reuses_deep_blocks_on_non_key_steps()
    test_teacache_reuses_on_slow_input_then_recomputes()
    test_teacache_fast_input_recomputes()
    test_store_clear_resets_state()
    print("\n[deepcache_teacache_smoke] 5/5 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
