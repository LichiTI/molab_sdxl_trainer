"""Real cached-data Anima full-finetune one-step smoke.

This smoke uses a small executable subset of the real Anima DiT checkpoint.
It validates the Phase-1 full-finetune contract without routing through LoRA:

- cache-first latent/text conditioning
- native DiT parameters are trainable
- one finite forward/backward/optimizer step
- full ``unet.*`` checkpoint save and resume

It is not a throughput or quality test.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file, save_file

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import ModelArch, UnifiedTrainingConfig
from core.lulynx_trainer.anima_flow import AnimaFlowConfig, build_anima_flow_inputs, sample_anima_sigmas
from core.lulynx_trainer.anima_full_finetune import (
    build_anima_full_finetune_state_dict,
    load_anima_full_finetune_state,
    prepare_anima_dit_only_full_finetune,
)
from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset


def _load_cached_latents(path: Path) -> torch.Tensor:
    data = np.load(path)
    latent_keys = sorted(key for key in data.files if key.startswith("latents_"))
    if not latent_keys:
        raise ValueError(f"No latents_* arrays found in {path}")
    latents = torch.from_numpy(data[latent_keys[0]]).float().unsqueeze(0)
    return latents[:, :, :4, :4].contiguous()


def _load_cached_text(path: Path) -> torch.Tensor:
    data = np.load(path)
    if "prompt_embeds" not in data:
        raise ValueError(f"No prompt_embeds found in {path}")
    prompt_embeds = torch.from_numpy(data["prompt_embeds"]).float().unsqueeze(0)
    return prompt_embeds[:, :16, :].contiguous()


def _load_subset(checkpoint: Path) -> torch.nn.Module:
    model, report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=(0,),
        device="cpu",
        dtype=torch.float32,
    )
    if not report.strict_success:
        raise RuntimeError(f"Anima executable subset load failed: {report.to_dict()}")
    return model


def main() -> int:
    torch.manual_seed(42)
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    latent_path = data_dir / "0_1856x2272_anima.npz"
    text_path = data_dir / "0_anima_te.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {checkpoint}")
    if not latent_path.exists() or not text_path.exists():
        raise FileNotFoundError(f"Missing cached Anima smoke data under {data_dir}")

    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        training_type="full_finetune",
        learning_rate=1e-6,
        weight_decay=0.0,
        network_train_unet_only=False,
        network_train_text_encoder_only=False,
    )
    unet = _load_subset(checkpoint)
    model = SimpleNamespace(
        unet=unet,
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        noise_scheduler=None,
        anima_native_train_ready=True,
        anima_cached_training_ready=True,
    )
    setup = prepare_anima_dit_only_full_finetune(config=cfg, model=model)
    optimizer = torch.optim.AdamW(setup.trainable_params, lr=cfg.learning_rate)

    latents = _load_cached_latents(latent_path)
    context = _load_cached_text(text_path)
    noise = torch.randn_like(latents)
    sigmas = sample_anima_sigmas(
        latents.shape[0],
        device=latents.device,
        dtype=latents.dtype,
        config=AnimaFlowConfig(timestep_sampling="sigma"),
    )
    noisy_latents, target, timesteps = build_anima_flow_inputs(latents, noise, sigmas)

    optimizer.zero_grad(set_to_none=True)
    pred = model.unet(noisy_latents, timesteps, context).sample
    loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite Anima full-finetune loss: {loss}")
    loss.backward()

    grad_hits = 0
    finite_grad_tensors = 0
    for param in setup.trainable_params:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise RuntimeError("Anima full-finetune produced a non-finite gradient")
        finite_grad_tensors += 1
        if float(param.grad.detach().abs().sum()) > 0.0:
            grad_hits += 1
    if grad_hits <= 0:
        raise RuntimeError("Anima full-finetune found no non-zero DiT gradients")

    optimizer.step()

    out_dir = repo_root / "backend" / "tmp" / "anima_smokes"
    out_path = out_dir / f"lulynx_anima_full_finetune_real_{int(time.time() * 1000)}.safetensors"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = build_anima_full_finetune_state_dict(unet=model.unet)
    save_file(
        {key: value.detach().cpu() for key, value in state.items()},
        str(out_path),
        metadata={
            "model_family": "anima",
            "training_type": "full_finetune",
            "smoke": "full_finetune_real_step",
            "phase": "dit_only_cache_first",
            "source_latents": latent_path.name,
            "source_text": text_path.name,
        },
    )

    restored = _load_subset(checkpoint)
    first_key = next(iter(restored.state_dict()))
    before = restored.state_dict()[first_key].detach().clone()
    load_report = load_anima_full_finetune_state(
        unet=restored,
        state_dict=load_file(str(out_path), device="cpu"),
    )
    after = restored.state_dict()[first_key].detach()
    if load_report["loaded"] <= 0 or torch.equal(before, after):
        raise RuntimeError(f"Anima full-finetune resume check failed: {load_report}")

    print(
        "Anima full-finetune real smoke passed: "
        f"loss={float(loss.detach()):.6f}, "
        f"trainable_tensors={len(setup.trainable_params)}, "
        f"finite_grad_tensors={finite_grad_tensors}, "
        f"grad_hits={grad_hits}, "
        f"saved={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
