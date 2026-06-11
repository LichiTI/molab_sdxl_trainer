# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for the Lulynx SmoothCache subsystem (v1). CPU is fine.

Run with the flashattention env:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/smoothcache_smoke.py

Checks: (1) the calibrator produces a schedule whose cacheable-step count grows
monotonically with the error threshold and is empty at alpha=0; (2) the observe-
only probe accounts cacheable block calls against a schedule; (3) the execution
layer is bit-identical to a plain block call at alpha=0/no-schedule and returns
the cached tensor on a cacheable step; (4) with the feature disabled nothing is
reused (regression red line).  Emits the subsystem scorecard.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch

from core.lulynx_trainer.smoothcache import (
    SmoothCacheCalibrator,
    SmoothCachePolicy,
    SmoothCacheState,
    SmoothCacheStore,
    observe_block_call,
    reset_smoothcache_probe_stats,
    run_with_smoothcache,
    smoothcache_step_context,
    snapshot_smoothcache_probe_stats,
)
from core.lulynx_trainer.smoothcache_scorecard import build_smoothcache_scorecard


def check_calibration() -> tuple[bool, list[int]]:
    print("== calibration: schedule monotonic in error threshold ==")
    cal = SmoothCacheCalibrator()
    # Strictly decreasing per-step relative-L1 change: [0.5, 0.2, 0.1, 0.05, 0.02].
    values = [100.0, 150.0, 180.0, 198.0, 207.9, 207.9 * 1.02]
    for v in values:
        cal.record(0, torch.full((4,), float(v)))
    alphas = [0.0, 0.03, 0.08, 0.15, 0.25]
    counts = [sum(len(s) for s in cal.build_schedule(a).values()) for a in alphas]
    empty_at_zero = counts[0] == 0
    monotonic = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    ok = empty_at_zero and monotonic and counts[-1] > 0
    print(f"  alphas={alphas} counts={counts} empty_at_0={empty_at_zero} monotonic={monotonic}  {'OK' if ok else 'FAIL'}")
    return ok, counts


def check_observe() -> tuple[bool, dict]:
    print("== observe-only probe accounts cacheable block calls ==")
    schedule = {0: frozenset({2, 3}), 1: frozenset({3})}
    state = SmoothCacheState(
        SmoothCachePolicy(enabled=True, schedule=schedule, warmup_steps=0),
        total_steps=4,
    )
    reset_smoothcache_probe_stats()
    reused = 0
    for i in range(4):
        with smoothcache_step_context(state.decide(i)):
            for b in (0, 1):
                if observe_block_call(block_index=b):
                    reused += 1
    stats = snapshot_smoothcache_probe_stats()
    # Expected reuse: (step2,b0), (step3,b0), (step3,b1) = 3 of 8 block calls.
    ok = reused == 3 and stats["would_reuse_block_calls"] == 3 and stats["block_calls_observed"] == 8
    print(f"  reused={reused} stats={stats}  {'OK' if ok else 'FAIL'}")
    return ok, stats


def check_execution() -> bool:
    print("== execution: alpha=0 parity + cacheable-step reuse ==")
    torch.manual_seed(0)

    def block_fn(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

    # No schedule: every step computes -> bit-identical to plain block_fn.
    store = SmoothCacheStore()
    state = SmoothCacheState(SmoothCachePolicy(enabled=True, schedule=None), total_steps=3)
    parity = True
    for i in range(3):
        x = torch.randn(4)
        with smoothcache_step_context(state.decide(i)):
            cached_out = run_with_smoothcache(block_fn, 0, store, x)
        parity = parity and torch.equal(cached_out, x * 2.0)

    # With a schedule: step 1 is cacheable -> returns the stored step-0 output.
    schedule = {0: frozenset({1})}
    store2 = SmoothCacheStore()
    state2 = SmoothCacheState(
        SmoothCachePolicy(enabled=True, schedule=schedule, warmup_steps=0),
        total_steps=2,
    )
    x0, x1 = torch.randn(4), torch.randn(4)
    with smoothcache_step_context(state2.decide(0)):
        o0 = run_with_smoothcache(block_fn, 0, store2, x0)
    with smoothcache_step_context(state2.decide(1)):
        o1 = run_with_smoothcache(block_fn, 0, store2, x1)
    reuse_ok = torch.equal(o1, o0) and not torch.equal(o1, x1 * 2.0)
    ok = parity and reuse_ok
    print(f"  alpha0_parity={parity} cacheable_reuse={reuse_ok}  {'OK' if ok else 'FAIL'}")
    return ok


def check_disabled() -> bool:
    print("== regression red line: disabled never reuses ==")
    schedule = {0: frozenset({1})}
    state = SmoothCacheState(SmoothCachePolicy(enabled=False, schedule=schedule), total_steps=2)
    reset_smoothcache_probe_stats()
    any_reuse = False
    for i in range(2):
        with smoothcache_step_context(state.decide(i)):
            if observe_block_call(block_index=0):
                any_reuse = True
    no_context = observe_block_call(block_index=0)  # no active step context
    ok = (not any_reuse) and (no_context is False)
    print(f"  any_reuse={any_reuse} no_context_returns_false={no_context is False}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    cal_ok, counts = check_calibration()
    observe_ok, observe_stats = check_observe()
    exec_ok = check_execution()
    disabled_ok = check_disabled()

    observed = max(observe_stats.get("block_calls_observed", 1), 1)
    reused = observe_stats.get("would_reuse_block_calls", 0)
    speedup = observed / max(observed - reused, 1)

    scorecard = build_smoothcache_scorecard(
        calibration_verified=cal_ok,
        schedule_monotonic=cal_ok,
        reuse_parity_at_alpha0=exec_ok and disabled_ok,
        observe_probe_wired=observe_ok,
        theoretical_speedup=speedup,
        total_steps=4,
        error_threshold=0.08,
    )
    print("\n== scorecard ==")
    for k, v in scorecard.items():
        print(f"  {k}: {v}")

    all_ok = cal_ok and observe_ok and exec_ok and disabled_ok
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT", f"| scorecard.ok={scorecard['ok']}")
    sys.exit(0 if all_ok else 1)
