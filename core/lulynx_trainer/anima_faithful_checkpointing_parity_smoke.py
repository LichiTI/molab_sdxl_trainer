"""Parity smoke: faithful native-Anima forward is identical with/without block
checkpointing, and remains trainable under checkpointing.

#147 disabled block checkpointing under the faithful 3D-RoPE forward because the
checkpoint wrapper dropped ``rope_emb`` on the recomputed block. #166 threads
``rope_emb`` through ``_checkpoint_block`` so checkpointing — the activation-memory
lever that lets the 4096-token 1024px DiT fit (25G -> ~5-6G) — works under faithful.

This smoke locks that on real weights:

- faithful forward WITHOUT checkpointing == faithful forward WITH block
  checkpointing (RoPE is not dropped on recompute; outputs match bit-for-bit-ish).
- a LoRA injected on the faithful + checkpointed path still receives finite,
  non-zero gradients (the RoPE path is differentiable under recompute).

CPU + float32, small block subset (RoPE / checkpoint behaviour is per-block and
block-count independent). Skips clean when no local checkpoint is present.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/anima_faithful_checkpointing_parity_smoke.py
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
from core.lulynx_trainer.lora_injector import LoRAInjector


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
        block_indices=tuple(range(4)),
        device="cpu",
        dtype=torch.float32,
        faithful=True,
    )
    return model


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

    # --- forward parity: faithful, checkpoint OFF vs ON ---
    model = _build(checkpoint)
    assert getattr(model, "anima_faithful", False), "expected faithful subset"
    model.train()  # train mode so the checkpoint gate can engage

    model.set_anima_block_checkpointing(False)
    x_off = latents.clone().requires_grad_(True)
    with torch.enable_grad():
        out_off = model(x_off, timesteps, context).sample

    prof = model.set_anima_block_checkpointing(True, "block")
    assert prof["enabled"] and prof["checkpointed_blocks"] == prof["block_count"], prof
    x_on = latents.clone().requires_grad_(True)
    with torch.enable_grad():
        out_on = model(x_on, timesteps, context).sample

    assert out_off.shape == out_on.shape == latents.shape
    max_abs = (out_off - out_on).abs().max().item()
    assert torch.allclose(out_off, out_on, atol=1e-5, rtol=0), (
        f"faithful forward diverges under block checkpointing (max abs {max_abs:.2e}); "
        "rope_emb may be dropped on the checkpoint recompute path"
    )

    # --- selective mode + interval=2: same forward, fewer checkpointed blocks ---
    prof_sel = model.set_anima_block_checkpointing(True, "selective", interval=2)
    assert prof_sel["enabled"] and prof_sel["mode"] == "selective", prof_sel
    assert prof_sel["interval"] == 2, prof_sel
    assert prof_sel["checkpointed_blocks"] == (prof_sel["block_count"] + 1) // 2, prof_sel
    x_sel = latents.clone().requires_grad_(True)
    with torch.enable_grad():
        out_sel = model(x_sel, timesteps, context).sample
    max_abs_sel = (out_off - out_sel).abs().max().item()
    assert torch.allclose(out_off, out_sel, atol=1e-5, rtol=0), (
        f"faithful forward diverges under selective/interval checkpointing "
        f"(max abs {max_abs_sel:.2e})"
    )
    model.set_anima_block_checkpointing(True, "block")  # restore for the print profile

    # --- trainable under faithful + checkpointing: LoRA grads must flow ---
    lora_model = _build(checkpoint)
    lora_model.train()
    lora_model.set_anima_block_checkpointing(True, "block")
    for param in lora_model.parameters():
        param.requires_grad_(False)
    injector = LoRAInjector(rank=2, alpha=2, model_arch="anima")
    injected = injector._inject_model(
        lora_model,
        get_anima_dit_targets(include_llm_adapter=False),
        prefix="net",
    )
    assert injected, "no LoRA layers injected on the native DiT targets"
    pred = lora_model(latents.clone(), timesteps, context).sample
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
    assert grad_hits, "no LoRA gradients flowed under faithful + block checkpointing"

    print(
        "Anima faithful+checkpointing parity smoke passed: "
        f"checkpoint={checkpoint.name}, forward_parity_max_abs={max_abs:.2e}, "
        f"selective_interval2_max_abs={max_abs_sel:.2e}, "
        f"checkpointed_blocks={prof['checkpointed_blocks']}/{prof['block_count']}, "
        f"lora_layers={len(injected)}, lora_grad_hits={len(grad_hits)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
