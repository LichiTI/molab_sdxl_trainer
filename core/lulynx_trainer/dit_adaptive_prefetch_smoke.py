"""CPU smoke for the opt-in adaptive stream-offload prefetch policy.

The prefetch controller short-circuits to ``enabled=False`` without CUDA, so the
runtime H2D path cannot run here.  Instead we drive the *pure* decision helpers
that the adaptive policy is built from -- ``_prefetch_targets`` (which blocks to
stage) and ``_adapt_depth_from`` (online depth growth) -- plus the transit-chain
default, which together are the whole behavioural surface of the new mode:

  1. ``original`` selection is fixed-depth ``[i .. i+depth]`` and ignores a live
     blockskip seam (the parity red-line: today's behaviour is unchanged).
  2. ``adaptive`` + a published blockskip seam drops exactly the blocks the seam
     will identity-skip from the prefetch window and counts them.
  3. ``adaptive`` depth grows by one per high-miss-rate forward and caps at
     ``_max_depth`` (never unbounded).
  4. The default (mode unset) is byte-for-byte the explicit ``original``, and the
     request->config->controller transit chain carries ``policy`` through.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/dit_adaptive_prefetch_smoke.py
"""

from __future__ import annotations

import torch.nn as nn
from types import SimpleNamespace

try:
    from . import anima_block_residency as anima_residency
    from .anima_block_residency import apply_anima_block_residency
    from .dit_block_prefetch_controller import DitBlockPrefetchController
    from .dit_compute_reducer_seam import (
        DiTComputeReducerSeam,
        DiTComputeReducerSeamPolicy,
        compute_reducer_seam_context,
    )
    from .dit_residency_planner import build_dit_residency_plan
    from .native_unet.weight_residency import LulynxManagedLinear
    from .sampler import create_sampler_from_trainer
except ImportError:  # pragma: no cover - direct-file smoke fallback
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.lulynx_trainer import anima_block_residency as anima_residency
    from core.lulynx_trainer.anima_block_residency import apply_anima_block_residency
    from core.lulynx_trainer.dit_block_prefetch_controller import DitBlockPrefetchController
    from core.lulynx_trainer.dit_compute_reducer_seam import (
        DiTComputeReducerSeam,
        DiTComputeReducerSeamPolicy,
        compute_reducer_seam_context,
    )
    from core.lulynx_trainer.dit_residency_planner import build_dit_residency_plan
    from core.lulynx_trainer.native_unet.weight_residency import LulynxManagedLinear
    from core.lulynx_trainer.sampler import create_sampler_from_trainer


def _managed_linear(in_features: int, out_features: int) -> LulynxManagedLinear:
    layer = LulynxManagedLinear(in_features, out_features, bias=False)
    layer.weight.requires_grad_(False)
    return layer


class _SyntheticBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _managed_linear(1024, 1024)
        self.self_attn = _managed_linear(1024, 1024)


class _SyntheticAnimaModel(nn.Module):
    def __init__(self, n: int = 6) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([_SyntheticBlock() for _ in range(n)])


def _controller(mode: str, depth: int, n: int = 6) -> DitBlockPrefetchController:
    blocks = list(_SyntheticAnimaModel(n).net.blocks)
    plan = build_dit_residency_plan(
        blocks, family="anima", mode="streaming_offload", requested_min_parameter_count=0,
    )
    # device="cpu" => controller reports enabled=False, but the pure helpers
    # under test never consult enabled/CUDA.
    return DitBlockPrefetchController(
        blocks, plan, device="cpu", depth=depth, prefetch_mode=mode, install_hooks=False,
    )


def _blockskip_seam(total_blocks: int) -> DiTComputeReducerSeam:
    policy = DiTComputeReducerSeamPolicy(
        enabled=True, strategy="blockskip", skip_every=2, min_block=1, warmup_steps=0,
    )
    return DiTComputeReducerSeam(policy, total_blocks=total_blocks, step_index=0, total_steps=0)


def test_original_mode_is_fixed_depth_and_ignores_blockskip() -> None:
    ctl = _controller("original", depth=2, n=6)
    assert ctl.policy == "original" and ctl.adaptive is False
    # original: [i .. i+depth] regardless of any published seam.
    assert ctl._prefetch_targets(0) == [0, 1, 2]
    assert ctl._prefetch_targets(3) == [3, 4, 5]
    with compute_reducer_seam_context(_blockskip_seam(6)):
        assert ctl._prefetch_targets(0) == [0, 1, 2]  # seam ignored under original
    assert ctl._skipped_blockskip == 0
    print("PASS: test_original_mode_is_fixed_depth_and_ignores_blockskip")


def test_adaptive_blockskip_skips_predicted_blocks() -> None:
    ctl = _controller("adaptive", depth=5, n=6)  # window [0..5] covers all blocks
    assert ctl.policy == "adaptive" and ctl.adaptive is True
    # No seam published -> adaptive stages the full window (skip_fn is None).
    assert ctl._prefetch_targets(0) == [0, 1, 2, 3, 4, 5]
    assert ctl._skipped_blockskip == 0

    seam = _blockskip_seam(6)
    with compute_reducer_seam_context(seam):
        ground_truth_skipped = [i for i in range(6) if seam.should_skip_block(i)]
        assert ground_truth_skipped == [1, 3, 5], ground_truth_skipped  # determinism guard
        targets = ctl._prefetch_targets(0)
    assert targets == [0, 2, 4], targets  # skipped blocks pruned from staging
    assert ctl._skipped_blockskip == 3
    print("PASS: test_adaptive_blockskip_skips_predicted_blocks")


def test_adaptive_depth_grows_on_miss_and_caps() -> None:
    ctl = _controller("adaptive", depth=1, n=6)
    assert ctl._adaptive_depth == 1 and ctl._max_depth == 4
    # (cumulative consumed, cumulative missed) -> expected adaptive_depth after step
    steps = [
        ((10, 0), 1),   # 0% miss -> no growth
        ((15, 5), 2),   # 50% miss -> grow
        ((20, 10), 3),  # 50% miss -> grow
        ((25, 15), 4),  # 50% miss -> grow to cap
        ((30, 20), 4),  # still high miss but already at _max_depth -> hold
    ]
    seen = []
    for (consumed, missed), expected in steps:
        ctl._adapt_depth_from(consumed, missed)
        seen.append(ctl._adaptive_depth)
        assert ctl._adaptive_depth == expected, (consumed, missed, ctl._adaptive_depth, expected)
        assert ctl._adaptive_depth <= ctl._max_depth
    assert seen == [1, 2, 3, 4, 4], seen
    assert ctl._depth_grew == 3, ctl._depth_grew
    print("PASS: test_adaptive_depth_grows_on_miss_and_caps")


def test_default_matches_explicit_original_through_chain() -> None:
    # Controller-level: unset mode == explicit "original", byte-for-byte selection.
    default = _controller("original", depth=2, n=6)  # _controller passes mode explicitly
    bare_blocks = list(_SyntheticAnimaModel(6).net.blocks)
    bare_plan = build_dit_residency_plan(
        bare_blocks, family="anima", mode="streaming_offload", requested_min_parameter_count=0,
    )
    bare = DitBlockPrefetchController(bare_blocks, bare_plan, device="cpu", depth=2, install_hooks=False)
    assert bare.policy == "original" and bare.adaptive is False
    assert bare._prefetch_targets(1) == default._prefetch_targets(1) == [1, 2, 3]
    assert bare.as_dict()["policy"] == "original"

    # Transit chain: apply_anima_block_residency carries policy into report.prefetch
    # (enabled=False on CPU, but policy is set before the CUDA gate).
    rep_default = apply_anima_block_residency(
        _SyntheticAnimaModel(6), mode="streaming_offload", min_parameter_count=0,
        prefetch_enabled=True, prefetch_depth=1,
    ).as_dict()
    rep_adaptive = apply_anima_block_residency(
        _SyntheticAnimaModel(6), mode="streaming_offload", min_parameter_count=0,
        prefetch_enabled=True, prefetch_depth=1, prefetch_mode="adaptive",
    ).as_dict()
    assert rep_default["prefetch"].get("policy") == "original", rep_default["prefetch"]
    assert rep_adaptive["prefetch"].get("policy") == "adaptive", rep_adaptive["prefetch"]
    print("PASS: test_default_matches_explicit_original_through_chain")


def test_sampler_restores_adaptive_prefetch_mode() -> None:
    captured: dict[str, object] = {}

    class _FakeReport:
        def as_dict(self) -> dict:
            return {"prefetch": {"policy": "adaptive"}}

    def fake_apply(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeReport()

    trainer = SimpleNamespace(
        model=SimpleNamespace(
            model_arch="anima",
            unet=object(),
            vae=object(),
            text_encoder_1=object(),
            text_encoder_2=None,
            tokenizer_1=object(),
            tokenizer_2=None,
        ),
        config=SimpleNamespace(
            model_arch="anima",
            preview_device="gpu",
            ephemeral_preview_pipeline=True,
            lulynx_weight_residency="resident",
            lulynx_weight_residency_min_params=0,
            anima_block_residency="streaming_offload",
            anima_block_residency_min_params=0,
            anima_block_prefetch=True,
            anima_block_prefetch_depth=2,
            anima_block_prefetch_mode="adaptive",
            sample_width=0,
            sample_height=0,
            sample_seed=0,
            sample_algorithm="sde",
            sample_sde_eta=1.0,
        ),
        noise_scheduler=None,
        device="cpu",
        dtype=None,
        lora_injector=None,
    )

    sampler = create_sampler_from_trainer(trainer)
    assert sampler is not None
    assert sampler.dit_block_prefetch_mode == "adaptive"

    original_apply = anima_residency.apply_anima_block_residency
    anima_residency.apply_anima_block_residency = fake_apply
    try:
        sampler._restore_dit_block_residency_after_preview()
    finally:
        anima_residency.apply_anima_block_residency = original_apply

    assert captured["model"] is trainer.model.unet
    kwargs = captured["kwargs"]
    assert kwargs["prefetch_enabled"] is True
    assert kwargs["prefetch_depth"] == 2
    assert kwargs["prefetch_mode"] == "adaptive"
    print("PASS: test_sampler_restores_adaptive_prefetch_mode")


if __name__ == "__main__":
    test_original_mode_is_fixed_depth_and_ignores_blockskip()
    test_adaptive_blockskip_skips_predicted_blocks()
    test_adaptive_depth_grows_on_miss_and_caps()
    test_default_matches_explicit_original_through_chain()
    test_sampler_restores_adaptive_prefetch_mode()
    print("PASS: dit_adaptive_prefetch_smoke")
