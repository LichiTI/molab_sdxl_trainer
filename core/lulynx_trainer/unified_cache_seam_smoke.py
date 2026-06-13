# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke for the unified DiT block cache seam (CPU only).

Mirrors the ``_run_blocks`` block loop with a fake block list and verifies:
1. PARITY RED-LINE: backend="none" (disabled) == baseline, bitwise.
2. PARITY RED-LINE: enabled seam with no active step context == baseline
   (the probe never marks a block cacheable, so every block recomputes).
3. dispatch routing: spectrum -> SpectrumCache, smoothcache -> SmoothCacheStore,
   tgate/none -> disabled passthrough.
4. reuse: on a cacheable SmoothCache step with a stored output, the seam returns
   the cached tensor (so it really drives the execution primitive).
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import torch  # noqa: E402

from core.lulynx_trainer.unified_cache_seam import (  # noqa: E402
    UnifiedCacheSeam,
    UnifiedCacheSeamPolicy,
    build_cache_seam,
)
from core.lulynx_trainer.spectrum_probe import SpectrumCache  # noqa: E402
from core.lulynx_trainer.smoothcache import (  # noqa: E402
    SmoothCacheStore,
    SmoothCacheStepDecision,
    smoothcache_step_context,
)


def _blocks():
    torch.manual_seed(0)
    mats = [torch.randn(8, 8) * 0.1 for _ in range(4)]
    return [(lambda x, m=m: torch.tanh(x @ m)) for m in mats]


def _run_loop(x, blocks, seam):
    """Faithful mirror of the production ``_run_blocks`` dispatch."""
    for i, block in enumerate(blocks):
        if seam is not None and seam.enabled:
            x = seam.run_block(block, i, x)
        else:
            x = block(x)
    return x


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    x0 = torch.randn(2, 8)

    blocks = _blocks()
    baseline = _run_loop(x0.clone(), blocks, None)

    # 1. parity: disabled (backend=none) ----------------------------------
    seam_none = build_cache_seam(enabled=True, backend="none")
    out_none = _run_loop(x0.clone(), _blocks(), seam_none)
    checks.append(("parity_disabled", torch.equal(baseline, out_none), f"max={float((baseline-out_none).abs().max()):.2e}"))

    # 2. parity: enabled but no active step context -----------------------
    seam_sc = build_cache_seam(enabled=True, backend="smoothcache")
    out_noctx = _run_loop(x0.clone(), _blocks(), seam_sc)
    checks.append(("parity_no_context", torch.equal(baseline, out_noctx), f"max={float((baseline-out_noctx).abs().max()):.2e}"))

    seam_spec = build_cache_seam(enabled=True, backend="spectrum")
    out_spec = _run_loop(x0.clone(), _blocks(), seam_spec)
    checks.append(("parity_spectrum_no_context", torch.equal(baseline, out_spec), f"max={float((baseline-out_spec).abs().max()):.2e}"))

    # 3. dispatch routing --------------------------------------------------
    from core.lulynx_trainer.deepcache import DeepCacheStore  # noqa: E402
    from core.lulynx_trainer.teacache import TeaCacheStore  # noqa: E402

    routes_ok = (
        isinstance(build_cache_seam(enabled=True, backend="spectrum")._cache, SpectrumCache)
        and isinstance(build_cache_seam(enabled=True, backend="smoothcache")._cache, SmoothCacheStore)
        and isinstance(build_cache_seam(enabled=True, backend="deepcache")._cache, DeepCacheStore)
        and isinstance(build_cache_seam(enabled=True, backend="teacache")._cache, TeaCacheStore)
        and build_cache_seam(enabled=True, backend="tgate").enabled is False
        and build_cache_seam(enabled=True, backend="none").enabled is False
        and UnifiedCacheSeamPolicy(enabled=True, backend="spectrum").normalized().enabled is True
    )
    checks.append(("dispatch_routes", routes_ok, "spectrum/smoothcache/deepcache/teacache/tgate/none"))

    # 4. reuse on a cacheable SmoothCache step ----------------------------
    seam = UnifiedCacheSeam(UnifiedCacheSeamPolicy(enabled=True, backend="smoothcache"))
    const_block = lambda x: x + 1.0  # noqa: E731
    x = torch.zeros(2, 8)
    # warmup step: nothing cacheable -> compute + store
    with smoothcache_step_context(SmoothCacheStepDecision(0, 2, frozenset(), "warmup")):
        first = seam.run_block(const_block, 0, x)
    # cacheable step: block 0 marked cacheable -> reuse stored output, ignore new input
    with smoothcache_step_context(SmoothCacheStepDecision(1, 2, frozenset({0}), "cache")):
        reused = seam.run_block(const_block, 0, x + 999.0)
    reuse_ok = torch.equal(first, reused) and torch.equal(first, torch.ones(2, 8))
    checks.append(("reuse_on_cacheable_step", reuse_ok, f"reused==first={torch.equal(first, reused)}"))

    from core.lulynx_trainer.unified_cache_seam_scorecard import build_unified_cache_seam_scorecard

    card = build_unified_cache_seam_scorecard(
        parity_disabled=checks[0][1],
        parity_no_context=checks[1][1] and checks[2][1],
        dispatch_routes_correct=checks[3][1],
        reuse_on_cacheable_step=checks[4][1],
        backends_supported=["spectrum", "smoothcache", "deepcache", "teacache"],
    )
    checks.append(("scorecard_ok", card["ok"], f"default_changed={card['default_behavior_changed']}"))

    ok = all(passed for _, passed, _ in checks)
    print("=== unified_cache_seam smoke ===")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"scorecard ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
