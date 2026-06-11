"""Real Anima cached companion + compression train-step smoke.

Uses the existing tiny cached sample under ``sucai/6_lulu`` and the real Anima
DiT checkpoint. It verifies the intended order:

1. inject train adapter slots
2. create/load a companion LoRA checkpoint
3. merge companion into base weights and reset train slots
4. apply frozen weight compression
5. run one optimizer step and save the fresh train adapter
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
from core.lulynx_trainer.compression_companion import apply_compression_companion
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.weight_compression import apply_weight_compression
from safetensors.torch import load_file, save_file


def _load_cached_latents(path: Path) -> torch.Tensor:
    data = np.load(path)
    latent_keys = sorted(key for key in data.files if key.startswith("latents_"))
    if not latent_keys:
        raise ValueError(f"No latents_* arrays found in {path}")
    return torch.from_numpy(data[latent_keys[0]]).float().unsqueeze(0)[:, :, :4, :4].contiguous()


def _load_cached_text(path: Path) -> torch.Tensor:
    data = np.load(path)
    if "prompt_embeds" not in data:
        raise ValueError(f"No prompt_embeds found in {path}")
    return torch.from_numpy(data["prompt_embeds"]).float().unsqueeze(0)[:, :16, :].contiguous()


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

    targets = get_anima_dit_targets(include_llm_adapter=False)
    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(model, targets, prefix="net")
    assert injected

    # Build a deterministic nonzero companion checkpoint in the same injected slots.
    with torch.no_grad():
        for layer in injector.injected_layers.values():
            adapter = layer.lora
            adapter.lora_down.weight.fill_(0.01)
            adapter.lora_up.weight.fill_(0.02)
    companion_path = repo_root / ".tmp" / "lulynx_anima_companion.safetensors"
    companion_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu() for key, value in injector.get_lora_state_dict().items()},
        str(companion_path),
        metadata={"smoke": "compression_companion_source"},
    )

    # Reset before loading the companion so the merge path proves load+merge+reset.
    with torch.no_grad():
        for layer in injector.injected_layers.values():
            layer.lora.lora_down.weight.zero_()
            layer.lora.lora_up.weight.zero_()
    first_layer = next(iter(injector.injected_layers.values()))
    base_before = first_layer.original.weight.detach().float().clone()

    companion = apply_compression_companion(injector, path=str(companion_path), scale=1.0)
    assert companion.merged_layers == len(injected), companion
    assert companion.reset_layers == len(injected), companion
    base_after = first_layer.original.weight.detach().float().clone()
    assert not torch.allclose(base_before, base_after), "companion merge did not alter base weights"
    assert torch.count_nonzero(first_layer.lora.lora_up.weight).item() == 0, "train adapter was not reset"

    compression_probe = apply_weight_compression(
        type("Bundle", (), {"unet": model})(),
        enabled=True,
        target="backbone",
        format="torchao_int8",
        lora_injector=injector,
    )
    assert compression_probe.enabled
    assert compression_probe.compressed_count > 0
    # Native fp8_e4m3 direct-cast is storage-only in the current PyTorch build;
    # restore float32 for the real forward step.
    model.float()

    trainable = injector.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    latents = _load_cached_latents(latent_path)
    context = _load_cached_text(text_path)
    noise = torch.randn_like(latents)
    sigmas = sample_anima_sigmas(latents.shape[0], device=latents.device, dtype=latents.dtype, config=AnimaFlowConfig(timestep_sampling="sigma"))
    noisy_latents, target, timesteps = build_anima_flow_inputs(latents, noise, sigmas)

    optimizer.zero_grad(set_to_none=True)
    pred = model(noisy_latents, timesteps, context).sample
    loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    loss.backward()
    grad_hits = [
        name for name, param in model.named_parameters()
        if "lora_" in name and param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    ]
    assert len(grad_hits) == len(injected)
    optimizer.step()

    out_path = repo_root / ".tmp" / "lulynx_anima_companion_compression_step.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in injector.get_lora_state_dict().items()},
        str(out_path),
        metadata={
            "model_family": "anima",
            "smoke": "companion_compression_train_step",
            "companion_path": str(companion_path),
            "compression_format": compression_probe.format,
        },
    )
    reloaded = load_file(str(out_path), device="cpu")
    assert reloaded

    print(
        "Anima companion+torchao-compression train-step smoke passed: "
        f"loss={float(loss.detach()):.6f}, "
        f"layers={len(injected)}, grad_hits={len(grad_hits)}, "
        f"compressed_params={compression_probe.compressed_count}, "
        f"saved={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



