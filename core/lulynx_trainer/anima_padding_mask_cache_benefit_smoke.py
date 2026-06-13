"""#171 — padding-mask cache benefit micro-benchmark (the only un-absorbed item
from the #170 ref/anima_lora-1.8.1 speedup borrow-list).

ref anima_lora-1.8.1 caches the per-batch padding mask to avoid rebuilding it in
the hot loop. This measures whether that borrow is worth wiring into our native
anima trainer, on representative 1024px latent shapes.

Finding it answers (objective, no GPU): our collator (`_collate_latents`) already
short-circuits the dominant **same-resolution** batch to ``padding_mask=None``
(no tensor built at all). A mask is only constructed for **mixed-resolution
bucket** batches, and that construction is a microsecond-scale CPU bool fill —
negligible against a ~5 s/step 4096-token DiT forward+backward. So caching it
buys effectively nothing; this records the number behind that verdict.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/anima_padding_mask_cache_benefit_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.dirname(_BACKEND)
for _p in (_BACKEND, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from core.lulynx_trainer.anima_cached_dataset import _collate_latents

# Representative anima 1024px latent: 16 channels, 128x128 (8x downsample).
_C, _H, _W = 16, 128, 128
_BATCH = 2
_ITERS = 2000
# Representative measured native-anima 1024 faithful step time (fp8 runs: ~5 s/it).
_STEP_TIME_S = 5.0


def _same_res_batch():
    return [torch.zeros(_C, _H, _W) for _ in range(_BATCH)]


def _mixed_res_batch():
    # Force the padding path: one full-size + one shorter latent (even dims).
    return [torch.zeros(_C, _H, _W), torch.zeros(_C, _H - 16, _W - 16)]


def _time(fn, iters: int) -> float:
    # warmup
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    print("== #171 padding-mask cache benefit (1024px latent, B=%d) ==" % _BATCH)

    # 1) same-resolution path: no mask is built at all.
    same = _same_res_batch()
    _, mask_same = _collate_latents(same)
    assert mask_same is None, "same-res batch unexpectedly built a padding_mask"
    t_same = _time(lambda: _collate_latents(_same_res_batch()), _ITERS)
    print(f"  same-res collate (returns mask=None): {t_same*1e6:8.2f} us/batch")

    # 2) mixed-resolution path: the only case that constructs a mask.
    mixed = _mixed_res_batch()
    _, mask_mixed = _collate_latents(mixed)
    assert mask_mixed is not None and mask_mixed.dtype == torch.bool
    assert mask_mixed.shape == (2, 1, _H, _W)
    t_mixed = _time(lambda: _collate_latents(_mixed_res_batch()), _ITERS)
    print(f"  mixed-res collate (builds mask):       {t_mixed*1e6:8.2f} us/batch")

    # 3) isolate just the mask build+fill cost (mixed minus same baseline).
    mask_cost_s = max(t_mixed - t_same, 0.0)
    frac_of_step = mask_cost_s / _STEP_TIME_S
    print(f"  isolated mask build cost:              {mask_cost_s*1e6:8.2f} us/batch")
    print(f"  fraction of a ~{_STEP_TIME_S:.0f}s/step:                 {frac_of_step*100:.5f}%")

    # Verdict: caching the mask can save at most `mask_cost_s` per step, and only
    # on mixed-resolution batches; the same-resolution path already builds nothing.
    worth_wiring = frac_of_step > 0.005  # >0.5% of step time would be worth it
    verdict = "WORTH WIRING" if worth_wiring else "negligible — do NOT wire a cache"
    print(f"  VERDICT: {verdict} (threshold 0.5% of step time)")
    assert not worth_wiring, (
        "padding-mask build unexpectedly exceeds 0.5% of step time — re-evaluate"
    )

    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
