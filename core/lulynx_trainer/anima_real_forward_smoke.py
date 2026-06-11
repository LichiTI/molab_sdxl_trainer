"""Real-weight Anima native DiT forward smoke.

This smoke loads the local Anima preview2 DiT safetensors into the
Warehouse executable subset and verifies:

- full 28-block tiny-token forward can run on CPU
- LoRA injection on native DiT targets receives gradients on the full
  28-block path

It intentionally does not mark the production trainer ready; Qwen3/T5
conditioning, Qwen Image VAE integration, dataset caching, and adapter save
still need to close together before the trainer guard can be lifted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset
from core.lulynx_trainer.anima_targets import get_anima_dit_targets
from core.lulynx_trainer.lora_injector import LoRAInjector


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {checkpoint}")

    start = time.time()
    full_model, full_report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(28)),
        device="cpu",
        dtype=torch.float32,
    )
    latents = torch.randn(1, 16, 4, 4)
    timesteps = torch.tensor([250.0])
    context = torch.randn(1, 4, 1024)
    with torch.no_grad():
        output = full_model(latents, timesteps, context).sample
    assert tuple(output.shape) == tuple(latents.shape)
    assert torch.isfinite(output).all()
    full_elapsed = time.time() - start

    start = time.time()
    lora_model, _report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(28)),
        device="cpu",
        dtype=torch.float32,
    )
    for param in lora_model.parameters():
        param.requires_grad_(False)
    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(
        lora_model,
        get_anima_dit_targets(include_llm_adapter=False),
        prefix="net",
    )
    assert injected
    pred = lora_model(latents, timesteps, context).sample
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
    assert grad_hits
    lora_elapsed = time.time() - start

    print(
        "Anima real forward smoke passed: "
        f"full_keys={full_report.loaded_key_count}, "
        f"full_forward_sec={full_elapsed:.2f}, "
        f"lora_layers={len(injected)}, "
        f"lora_grad_hits={len(grad_hits)}, "
        f"lora_backward_sec={lora_elapsed:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

