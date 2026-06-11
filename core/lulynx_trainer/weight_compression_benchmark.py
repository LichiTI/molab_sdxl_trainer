"""Real Anima cached weight-compression benchmark.

Compares a tiny cached Anima train step across baseline and selected frozen
weight-compression presets. It is intentionally small so it can be used as a
local acceptance probe after installing optional runtimes such as torchao.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

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
from core.lulynx_trainer.weight_compression import apply_weight_compression, resolve_weight_compression_preset


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


def _cuda_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_case(name: str, preset_name: str, *, checkpoint: Path, latent_path: Path, text_path: Path, device: str) -> dict[str, Any]:
    _clear_cuda()
    load_device = "cpu"
    model, report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(28)),
        device=load_device,
        dtype=torch.float32,
    )
    if not report.strict_success:
        raise RuntimeError(f"Anima subset load failed for {name}: {report}")
    for param in model.parameters():
        param.requires_grad_(False)

    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(model, get_anima_dit_targets(include_llm_adapter=False), prefix="net")
    if not injected:
        raise RuntimeError("No LoRA targets were injected")

    preset = resolve_weight_compression_preset(preset_name)
    compressed = 0
    estimated_saved = 0.0
    warnings: list[str] = []
    if preset["enabled"]:
        result = apply_weight_compression(
            type("Bundle", (), {"unet": model})(),
            enabled=True,
            target=preset["target"],
            format=preset["format"],
            lora_injector=injector,
        )
        compressed = result.compressed_count
        estimated_saved = result.estimated_saved_mb
        warnings = result.warnings

    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
        latents = _load_cached_latents(latent_path).to("cuda")
        context = _load_cached_text(text_path).to("cuda")
    else:
        latents = _load_cached_latents(latent_path)
        context = _load_cached_text(text_path)

    trainable = injector.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    noise = torch.randn_like(latents)
    sigmas = sample_anima_sigmas(latents.shape[0], device=latents.device, dtype=latents.dtype, config=AnimaFlowConfig(timestep_sampling="sigma"))
    noisy_latents, target, timesteps = build_anima_flow_inputs(latents, noise, sigmas)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    pred = model(noisy_latents, timesteps, context).sample
    loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    grad_hits = sum(
        1
        for _, param in model.named_parameters()
        if "lora_" in _ and param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    )
    out = {
        "case": name,
        "preset": preset_name,
        "format": preset["format"],
        "target": preset["target"],
        "loss": float(loss.detach().cpu()),
        "step_time_sec": elapsed,
        "peak_cuda_mb": _cuda_peak_mb(),
        "compressed_params": compressed,
        "estimated_saved_mb": estimated_saved,
        "layers": len(injected),
        "grad_hits": grad_hits,
        "warnings": warnings,
    }
    del model, optimizer, latents, context, noisy_latents, target, timesteps, pred, loss
    _clear_cuda()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cpu", "cuda"))
    parser.add_argument("--cases", default="off,stable_backbone_int8,aggressive_backbone_uint4")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    latent_path = data_dir / "0_1856x2272_anima.npz"
    text_path = data_dir / "0_anima_te.npz"
    for path in (checkpoint, latent_path, text_path):
        if not path.exists():
            raise FileNotFoundError(path)

    results = []
    for raw in [item.strip() for item in args.cases.split(",") if item.strip()]:
        results.append(_run_case(raw, raw, checkpoint=checkpoint, latent_path=latent_path, text_path=text_path, device=args.device))

    print(json.dumps({"device": args.device, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
