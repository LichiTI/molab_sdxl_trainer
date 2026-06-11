"""Real cached-data Anima LoRA train-step smoke.

This smoke uses existing Anima cache artifacts under `sucai/6_lulu`:

- `*_anima.npz` for Qwen Image VAE latents
- `*_anima_te.npz` for text conditioning

It runs one tiny-token LoRA optimizer step through the full 28-block
real-weight DiT and writes a safetensors adapter.  The spatial crop is
intentionally tiny so the smoke is safe on CPU and does not claim throughput
or quality parity with full-resolution training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.anima_flow import AnimaFlowConfig, build_anima_flow_inputs, sample_anima_sigmas
from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset
from core.lulynx_trainer.anima_targets import get_anima_dit_targets
from core.lulynx_trainer.lora_injector import LoRAInjector
from safetensors.torch import load_file, save_file


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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    latent_path = data_dir / "0_1856x2272_anima.npz"
    text_path = data_dir / "0_anima_te.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {checkpoint}")
    if not latent_path.exists() or not text_path.exists():
        raise FileNotFoundError(f"Missing cached Anima smoke data under {data_dir}")

    model, report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(28)),
        device="cpu",
        dtype=torch.float32,
    )
    assert report.strict_success
    for param in model.parameters():
        param.requires_grad_(False)

    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(
        model,
        get_anima_dit_targets(include_llm_adapter=False),
        prefix="net",
    )
    assert injected

    trainable = injector.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
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
    pred = model(noisy_latents, timesteps, context).sample
    loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    loss.backward()
    grad_hits = [
        name
        for name, param in model.named_parameters()
        if "lora_" in name
        and param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.abs().sum() > 0
    ]
    assert len(grad_hits) == len(injected)
    optimizer.step()

    out_path = Path("H:/tmp/lulynx_anima_cached_train_step.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu() for key, value in injector.get_lora_state_dict().items()},
        str(out_path),
        metadata={
            "model_family": "anima",
            "smoke": "cached_train_step",
            "source_latents": latent_path.name,
            "source_text": text_path.name,
        },
    )
    reloaded = load_file(str(out_path), device="cpu")
    assert reloaded

    print(
        "Anima cached train-step smoke passed: "
        f"loss={float(loss.detach()):.6f}, "
        f"lora_layers={len(injected)}, "
        f"grad_hits={len(grad_hits)}, "
        f"saved={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

