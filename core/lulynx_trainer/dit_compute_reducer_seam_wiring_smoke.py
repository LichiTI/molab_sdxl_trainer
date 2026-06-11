# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Local wiring + parity smoke for the DiT block compute-reducer seam.

TREAD / DiffCR / BlockSkip are already-ready primitives whose disabled path is
an exact ``block(tokens)`` call. ``dit_compute_reducer_seam`` is the single
opt-in switch that drives ONE of them inside the live Anima/Newbie DiT
``_run_blocks`` loop (the same method that already hosts the unified cache
seam). This smoke proves the now-wired runtime path:

  * the seam is **default-off**: ``strategy="none"`` -> the seam is never
    enabled and ``get_active_compute_reducer_seam`` returns ``None``, so the
    block dispatch is **bitwise-identical** to legacy (the parity red-line);
  * each strategy, when explicitly selected, is **shape-stable** (output token
    shape unchanged) and **actually reduces compute** (output diverges from the
    plain block loop -- it would be a no-op bug otherwise);
  * the underlying primitives keep an **exact disabled-path parity** (the
    ``disabled_parity_ok`` evidence the trainer gate asks for);
  * the per-reducer scorecard's *local* blockers (shape, disabled-parity) clear
    while the real-model A/B blocker (``real_anima_newbie_ab_missing``) stays --
    loss-parity / quality-drift A/B is the operator's GPU job, not faked here.

The smoke mirrors the production ``_run_blocks`` dispatch with a fake block list
(exactly like ``unified_cache_seam_smoke``) so it exercises the real seam object
and the real primitives, no copies.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/dit_compute_reducer_seam_wiring_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    here = Path(__file__).resolve()
    backend_root = here.parents[2]          # .../backend  -> exposes `core`
    repo_root = here.parents[3]             # repo root     -> exposes `backend`
    for path in (str(repo_root), str(backend_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.dit_compute_reducer_seam import (
    build_compute_reducer_seam,
    compute_reducer_seam_context,
    get_active_compute_reducer_seam,
)
from core.lulynx_trainer.tread_token_routing import (
    TreadTokenRoutePolicy,
    build_tread_token_route_plan,
    build_tread_token_route_scorecard,
    run_tread_routed_block,
)
from core.lulynx_trainer.diffcr_token_compression import (
    DiffCRTokenCompressionPolicy,
    run_diffcr_compressed_block,
)
from core.lulynx_trainer.dit_blockskip_training_spike import (
    DiTBlockSkipDecision,
    apply_dit_blockskip_decision,
)

MODEL_SEED = 0
DIM = 16
BATCH = 2
TOKENS = 6
BLOCKS = 4


class _Block(nn.Module):
    """Cross-token block requiring conditioning args (mirrors a DiT block)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mix = nn.Linear(dim, dim, bias=False)

    def forward(self, x, emb, context, adaln_lora=None):
        # Raise if the seam fails to forward conditioning -> structural check.
        if emb is None or context is None:
            raise ValueError("conditioning args were not forwarded to the block")
        # The pooled mean makes the block cross-token, so any token routing /
        # compression / block skip changes the result (no silent no-op).
        pooled = x.mean(dim=1, keepdim=True)
        return self.proj(x) + self.mix(pooled)


def _build_blocks() -> nn.ModuleList:
    torch.manual_seed(MODEL_SEED)
    return nn.ModuleList([_Block(DIM) for _ in range(BLOCKS)])


def _inputs():
    torch.manual_seed(MODEL_SEED + 1)
    x = torch.randn(BATCH, TOKENS, DIM)
    emb = torch.randn(BATCH, DIM)
    context = torch.randn(BATCH, 3, DIM)
    return x, emb, context, None


def _plain_run(blocks, x, emb, context, adaln) -> torch.Tensor:
    out = x
    for block in blocks:
        out = block(out, emb, context, adaln)
    return out


def _seam_run(blocks, x, emb, context, adaln) -> torch.Tensor:
    """Faithful mirror of the production _run_blocks reducer dispatch."""
    reducer = get_active_compute_reducer_seam()
    out = x
    for index, block in enumerate(blocks):
        if reducer is not None:
            out = reducer.run_block(block, index, out, emb, context, adaln)
        else:
            out = block(out, emb, context, adaln)
    return out


def _config_field_defaults() -> dict:
    fields = getattr(UnifiedTrainingConfig, "model_fields", None)
    if fields is None:  # pydantic v1 fallback
        fields = UnifiedTrainingConfig.__fields__
    return {name: getattr(field, "default", None) for name, field in fields.items()}


def run() -> dict:
    results: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    blocks = _build_blocks()
    x, emb, context, adaln = _inputs()
    plain = _plain_run(blocks, x, emb, context, adaln)

    # --- 1. CONFIG defaults --------------------------------------------------
    defaults = _config_field_defaults()
    default_off = (
        defaults.get("dit_compute_reducer_strategy") == "none"
        and defaults.get("dit_compute_reducer_keep_ratio") == 1.0
        and defaults.get("dit_compute_reducer_compression_ratio") == 1.0
        and defaults.get("dit_compute_reducer_skip_ratio") == 0.0
        and defaults.get("dit_compute_reducer_skip_every") == 0
    )
    check("config_default_off", default_off, f"strategy={defaults.get('dit_compute_reducer_strategy')}")

    # --- 2. "none" strategy never activates ----------------------------------
    none_seam = build_compute_reducer_seam(enabled=True, strategy="none", total_blocks=BLOCKS)
    with compute_reducer_seam_context(none_seam):
        active_none = get_active_compute_reducer_seam()
    check("none_strategy_inactive", (not none_seam.enabled) and active_none is None)

    # --- 3. PARITY: no seam published ----------------------------------------
    no_ctx = _seam_run(blocks, x, emb, context, adaln)
    check("parity_no_seam_bitwise", torch.equal(no_ctx, plain))

    # --- 4. PARITY: "none" seam published ------------------------------------
    with compute_reducer_seam_context(none_seam):
        none_run = _seam_run(blocks, x, emb, context, adaln)
    check("parity_none_published_bitwise", torch.equal(none_run, plain))

    # --- 5. TREAD: shape-stable + changes compute ----------------------------
    tread_seam = build_compute_reducer_seam(
        enabled=True, strategy="tread", total_blocks=BLOCKS, keep_ratio=0.5,
    )
    with compute_reducer_seam_context(tread_seam):
        tread_out = _seam_run(blocks, x, emb, context, adaln)
    check(
        "tread_shape_stable_and_changes",
        tread_out.shape == plain.shape and not torch.equal(tread_out, plain),
        f"shape={tuple(tread_out.shape)}",
    )

    # --- 6. DIFFCR: shape-stable + changes compute ---------------------------
    diffcr_seam = build_compute_reducer_seam(
        enabled=True, strategy="diffcr", total_blocks=BLOCKS, compression_ratio=0.5,
    )
    with compute_reducer_seam_context(diffcr_seam):
        diffcr_out = _seam_run(blocks, x, emb, context, adaln)
    check(
        "diffcr_shape_stable_and_changes",
        diffcr_out.shape == plain.shape and not torch.equal(diffcr_out, plain),
        f"shape={tuple(diffcr_out.shape)}",
    )

    # --- 7. BLOCKSKIP: shape-stable + skips + changes compute ----------------
    blockskip_seam = build_compute_reducer_seam(
        enabled=True, strategy="blockskip", total_blocks=BLOCKS,
        skip_every=2, min_block=1,
    )
    blockskip_seam.set_step(8, 100)
    plan = blockskip_seam._ensure_blockskip_plan()
    with compute_reducer_seam_context(blockskip_seam):
        blockskip_out = _seam_run(blocks, x, emb, context, adaln)
    check(
        "blockskip_shape_stable_skips_and_changes",
        (
            blockskip_out.shape == plain.shape
            and plan is not None
            and plan.skipped_blocks >= 1
            and not torch.equal(blockskip_out, plain)
        ),
        f"skipped={None if plan is None else plan.skipped_blocks}",
    )

    # --- 8/9/10. Primitive disabled-path bitwise parity ----------------------
    one = blocks[0]
    def inner(toks):
        return one(toks, emb, context, adaln)

    tread_disabled, _ = run_tread_routed_block(x, inner, TreadTokenRoutePolicy(enabled=False))
    check("tread_primitive_disabled_parity", torch.equal(tread_disabled, inner(x)))

    diffcr_disabled, _ = run_diffcr_compressed_block(x, inner, DiffCRTokenCompressionPolicy(enabled=False))
    check("diffcr_primitive_disabled_parity", torch.equal(diffcr_disabled, inner(x)))

    no_skip = DiTBlockSkipDecision(
        block_index=0, step_index=0, total_blocks=BLOCKS, total_steps=0,
        skip=False, reuse_residual=True, reason="scheduled_forward",
        estimated_block_compute_fraction=1.0,
    )
    blockskip_disabled = apply_dit_blockskip_decision(x, inner, no_skip)
    check("blockskip_primitive_disabled_parity", torch.equal(blockskip_disabled, inner(x)))

    # --- 11. Scorecard: local blockers clear, GPU A/B blocker stays ----------
    live_plan = build_tread_token_route_plan(x, TreadTokenRoutePolicy(enabled=True, keep_ratio=0.5))
    scorecard = build_tread_token_route_scorecard(
        live_plan, shape_stable=True, disabled_parity_ok=True,
    )
    blockers = scorecard["blocked_reasons"]
    check(
        "scorecard_local_cleared_gpu_flagged",
        "shape_stability_evidence_missing" not in blockers
        and "disabled_parity_evidence_missing" not in blockers
        and "real_anima_newbie_ab_missing" in blockers,
        f"blockers={blockers}",
    )

    passed = sum(1 for r in results if r["ok"])
    return {
        "smoke": "dit_compute_reducer_seam_wiring_smoke",
        "passed": passed,
        "total": len(results),
        "ok": passed == len(results),
        "results": results,
    }


def main() -> int:
    report = run()
    for r in report["results"]:
        status = "PASS" if r["ok"] else "FAIL"
        line = f"  [{status}] {r['check']}"
        if r["detail"]:
            line += f"  ({r['detail']})"
        print(line)
    print(f"\n[dit_compute_reducer_seam_wiring_smoke] {report['passed']}/{report['total']} checks passed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
