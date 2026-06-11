"""Runtime VRAM probe for a tiny cached Anima training step.

This is the production-friendly version of the old BF16 VRAM probe idea: it
runs the same cached train micro-step under selected dtype/compression cases and
prints JSON with peak CUDA memory, step time, loss, adapter grad coverage, and
compression stats. CPU runs are supported for CI/smoke validation.
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


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name or "fp32").lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


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


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _cuda_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated()) / (1024 * 1024)


def _cuda_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)


def _parse_case(raw: str, default_dtype: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in raw.split(":") if part.strip()]
    if len(parts) == 1:
        return parts[0], parts[0], default_dtype
    if len(parts) == 2:
        return raw, parts[0], parts[1]
    return raw, parts[0], parts[1]


def run_case(
    case_name: str,
    preset_name: str,
    dtype_name: str,
    *,
    checkpoint: Path,
    latent_path: Path,
    text_path: Path,
    device: str,
    blocks: int,
) -> dict[str, Any]:
    _clear_cuda()
    dtype = _dtype_from_name(dtype_name)
    load_dtype = torch.float32 if device == "cpu" else dtype
    model, report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(max(int(blocks), 1))),
        device="cpu",
        dtype=load_dtype,
    )
    if not report.strict_success:
        raise RuntimeError(f"Anima subset load failed for {case_name}: {report}")
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

    actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model.to(actual_device)
    if actual_device == "cuda":
        model.to(dtype=dtype)
    latents = _load_cached_latents(latent_path).to(actual_device, dtype=dtype if actual_device == "cuda" else torch.float32)
    context = _load_cached_text(text_path).to(actual_device, dtype=dtype if actual_device == "cuda" else torch.float32)

    trainable = injector.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    noise = torch.randn_like(latents)
    sigmas = sample_anima_sigmas(latents.shape[0], device=latents.device, dtype=latents.dtype, config=AnimaFlowConfig(timestep_sampling="sigma"))
    noisy_latents, target, timesteps = build_anima_flow_inputs(latents, noise, sigmas)

    if actual_device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    pred = model(noisy_latents, timesteps, context).sample
    loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    loss.backward()
    optimizer.step()
    if actual_device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    grad_hits = sum(
        1
        for name, param in model.named_parameters()
        if "lora_" in name and param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    )
    result = {
        "case": case_name,
        "preset": preset_name,
        "dtype": dtype_name,
        "device": actual_device,
        "blocks": max(int(blocks), 1),
        "loss": float(loss.detach().cpu()),
        "step_time_sec": elapsed,
        "cuda_allocated_mb": _cuda_allocated_mb(),
        "peak_cuda_mb": _cuda_peak_mb(),
        "compressed_params": compressed,
        "estimated_saved_mb": estimated_saved,
        "layers": len(injected),
        "grad_hits": grad_hits,
        "warnings": warnings,
    }
    del model, optimizer, latents, context, noisy_latents, target, timesteps, pred, loss
    _clear_cuda()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--blocks", type=int, default=4, help="Number of Anima transformer blocks to load for the probe.")
    parser.add_argument(
        "--cases",
        default="off,stable_backbone_int8",
        help="Comma list. Each item can be preset or preset:dtype, for example off:bf16,stable_backbone_int8:bf16.",
    )
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
        case_name, preset_name, dtype_name = _parse_case(raw, args.dtype)
        results.append(
            run_case(
                case_name,
                preset_name,
                dtype_name,
                checkpoint=checkpoint,
                latent_path=latent_path,
                text_path=text_path,
                device=args.device,
                blocks=args.blocks,
            )
        )

    print(json.dumps({"device": args.device, "default_dtype": args.dtype, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
