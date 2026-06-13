"""Parity + engagement smoke: BlockSkip now runs UNDER the faithful native-Anima
forward (3D RoPE) and coexists with block checkpointing.

Historically the compute-reducer seam was force-disabled under faithful mode
because its ``run_block`` carried no ``rope_emb`` (see ``anima_native_dit``).
TREAD/DiffCR genuinely cannot run there (they route/merge tokens, mis-aligning
per-position RoPE), but BlockSkip skips whole blocks via *token-preserving
identity passthrough*, so it threads cleanly. ``_run_blocks`` now drives
BlockSkip from its own loop under faithful (``faithful_skip``), keeping rope_emb
and block-checkpointing intact.

This smoke locks the three invariants on real weights (CPU, float32, small block
subset — RoPE / skip behaviour is per-block and block-count independent). Skips
clean when no local checkpoint is present.

  A. PARITY red-line: a published BlockSkip seam that skips *nothing*
     (skip_ratio=0) leaves the faithful forward bit-for-bit identical to the
     no-seam baseline.
  B. ENGAGEMENT: a BlockSkip seam that skips some blocks under faithful actually
     skips (>=1 block via ``should_skip_block``), the forward stays finite, and
     the output differs from the baseline (the skip changed the computation).
  C. THREE-ON coexistence + trainable: faithful + BlockSkip + block
     checkpointing all on -> finite forward, >=1 block skipped, and an injected
     LoRA still receives finite, non-zero gradients.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/anima_faithful_blockskip_compat_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset
from core.lulynx_trainer.anima_targets import get_anima_dit_targets
from core.lulynx_trainer.dit_compute_reducer_seam import (
    build_compute_reducer_seam,
    compute_reducer_seam_context,
)
from core.lulynx_trainer.lora_injector import LoRAInjector

BLOCKS = 6


def _resolve_checkpoint() -> Path | None:
    base = Path(__file__).resolve().parents[3] / "models" / "anima" / "diffusion_models"
    for name in ("anima-base-v1.0.safetensors", "anima-preview2.safetensors"):
        candidate = base / name
        if candidate.exists():
            return candidate
    matches = sorted(base.glob("*.safetensors")) if base.exists() else []
    return matches[0] if matches else None


def _build(checkpoint: Path):
    model, _report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(BLOCKS)),
        device="cpu",
        dtype=torch.float32,
        faithful=True,
    )
    assert getattr(model, "anima_faithful", False), "expected faithful subset"
    return model


def _blockskip_seam(skip_ratio: float):
    seam = build_compute_reducer_seam(
        enabled=True,
        strategy="blockskip",
        total_blocks=BLOCKS,
        total_steps=10,
        skip_ratio=skip_ratio,
        warmup_steps=0,
        min_block=1,
    )
    seam.set_total_blocks(BLOCKS)
    seam.set_step(0, 10)
    return seam


def main() -> int:
    checkpoint = _resolve_checkpoint()
    if checkpoint is None:
        print("SKIP: no local Anima checkpoint under models/anima/diffusion_models")
        return 0

    assert get_anima_dit_targets  # keep import meaningful even on skip paths

    torch.manual_seed(0)
    latents = torch.randn(1, 16, 8, 8)
    timesteps = torch.tensor([0.5])
    context = torch.randn(1, 8, 1024) * 0.04

    # Baseline: faithful forward, no reducer context at all.
    model = _build(checkpoint)
    model.train()  # train mode so the (later) checkpoint gate can engage
    model.set_anima_block_checkpointing(False)
    with torch.no_grad():
        baseline = model(latents.clone(), timesteps, context).sample
    assert baseline.shape == latents.shape and torch.isfinite(baseline).all()

    # --- A. PARITY: seam present but skipping nothing == baseline (bitwise). ---
    seam0 = _blockskip_seam(skip_ratio=0.0)
    assert not any(seam0.should_skip_block(i) for i in range(BLOCKS)), (
        "skip_ratio=0 must skip no blocks"
    )
    with torch.no_grad(), compute_reducer_seam_context(seam0):
        out_parity = model(latents.clone(), timesteps, context).sample
    parity_max_abs = (out_parity - baseline).abs().max().item()
    assert torch.equal(out_parity, baseline), (
        f"a no-op BlockSkip seam changed the faithful forward (max abs {parity_max_abs:.2e})"
    )

    # --- B. ENGAGEMENT: skip some blocks under faithful (checkpoint OFF). ---
    seam = _blockskip_seam(skip_ratio=0.5)
    skipped = [i for i in range(BLOCKS) if seam.should_skip_block(i)]
    assert skipped, "expected BlockSkip to elect >=1 block to skip at step 0"
    with torch.no_grad(), compute_reducer_seam_context(seam):
        out_skip = model(latents.clone(), timesteps, context).sample
    assert out_skip.shape == latents.shape and torch.isfinite(out_skip).all()
    engage_max_abs = (out_skip - baseline).abs().max().item()
    assert engage_max_abs > 0.0, (
        "BlockSkip under faithful produced an output identical to the full forward "
        "(it did not actually skip / engage)"
    )

    # --- C. THREE-ON: faithful + BlockSkip + block checkpointing + trainable. ---
    lora_model = _build(checkpoint)
    lora_model.train()
    prof = lora_model.set_anima_block_checkpointing(True, "block")
    assert prof["enabled"] and prof["checkpointed_blocks"] == prof["block_count"], prof
    for param in lora_model.parameters():
        param.requires_grad_(False)
    injector = LoRAInjector(rank=2, alpha=2, model_arch="anima")
    injected = injector._inject_model(
        lora_model,
        get_anima_dit_targets(include_llm_adapter=False),
        prefix="net",
    )
    assert injected, "no LoRA layers injected on the native DiT targets"

    seam_c = _blockskip_seam(skip_ratio=0.5)
    skipped_c = [i for i in range(BLOCKS) if seam_c.should_skip_block(i)]
    assert skipped_c, "expected >=1 skipped block in the three-on case"
    with compute_reducer_seam_context(seam_c):
        pred = lora_model(latents.clone(), timesteps, context).sample
        assert torch.isfinite(pred).all(), "three-on faithful+blockskip+checkpoint forward is non-finite"
        loss = torch.nn.functional.mse_loss(pred, torch.randn_like(pred))
        loss.backward()
    grad_hits = [
        name
        for name, param in lora_model.named_parameters()
        if "lora_" in name
        and param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.abs().sum() > 0
    ]
    assert grad_hits, "no LoRA gradients flowed under faithful + BlockSkip + block checkpointing"

    print(
        "Anima faithful x BlockSkip compat smoke passed: "
        f"checkpoint={checkpoint.name}, blocks={BLOCKS}, "
        f"parity_max_abs={parity_max_abs:.2e} (skip_ratio=0 == baseline), "
        f"engaged_skipped_blocks={skipped} drift_vs_full={engage_max_abs:.3e}, "
        f"three_on_skipped={skipped_c} checkpointed_blocks={prof['checkpointed_blocks']}/{prof['block_count']} "
        f"lora_grad_hits={len(grad_hits)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
