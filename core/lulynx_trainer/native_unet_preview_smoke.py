"""Preview smoke for SDXL native UNet inside the training sampler path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.native_unet import install_sdxl_native_unet_backend
from core.lulynx_trainer.sampler import TrainingSampler
from core.lulynx_trainer.single_file_loader import load_sdxl_single_file_components


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "sdxl" / "silentEraFurrymixNAIXL_v10.safetensors"


def _resolve_dtype(value: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(value or "bfloat16").strip().lower()
    if key not in aliases:
        raise ValueError(f"unsupported dtype: {value}")
    return aliases[key]


def _cuda_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    return {
        "allocated_mb": round(float(torch.cuda.memory_allocated()) / (1024 * 1024), 1),
        "reserved_mb": round(float(torch.cuda.memory_reserved()) / (1024 * 1024), 1),
        "peak_allocated_mb": round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 1),
        "peak_reserved_mb": round(float(torch.cuda.max_memory_reserved()) / (1024 * 1024), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal SDXL native UNet preview smoke.")
    parser.add_argument("--model", default=str(_default_model_path()))
    parser.add_argument("--backend", default="lulynx_native", choices=["diffusers", "lulynx_native"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default="a simple red cube on a table")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = _resolve_dtype(args.dtype)
    started = time.perf_counter()
    components = load_sdxl_single_file_components(args.model, torch_dtype=dtype)
    model = SimpleNamespace(
        model_arch="sdxl",
        unet=components["unet"].to(device=device, dtype=dtype),
        vae=components["vae"],
        text_encoder_1=components["text_encoder_1"],
        text_encoder_2=components["text_encoder_2"],
        tokenizer_1=components["tokenizer_1"],
        tokenizer_2=components["tokenizer_2"],
        noise_scheduler=components["noise_scheduler"],
    )
    native_status: dict[str, Any] | None = None
    if args.backend == "lulynx_native":
        status = install_sdxl_native_unet_backend(
            model,
            backend="lulynx_native",
            model_path=args.model,
        )
        native_status = status.as_dict()
    load_elapsed_ms = (time.perf_counter() - started) * 1000.0

    sampler = TrainingSampler(
        unet=model.unet,
        text_encoder_1=model.text_encoder_1,
        text_encoder_2=model.text_encoder_2,
        vae=model.vae,
        tokenizer_1=model.tokenizer_1,
        tokenizer_2=model.tokenizer_2,
        noise_scheduler=model.noise_scheduler,
        device=str(device),
        dtype=dtype,
        model_arch="sdxl",
        sample_width=int(args.width),
        sample_height=int(args.height),
        sample_seed=int(args.seed),
        preview_device="gpu",
        ephemeral_pipeline=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen_started = time.perf_counter()
    image = sampler.generate(
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt or ""),
        num_inference_steps=max(int(args.steps), 1),
        guidance_scale=1.0,
        width=int(args.width),
        height=int(args.height),
        seed=int(args.seed),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    generate_elapsed_ms = (time.perf_counter() - gen_started) * 1000.0
    ok = image is not None
    output_path = Path(args.output) if args.output else None
    if image is not None and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    report = {
        "ok": bool(ok),
        "backend": str(args.backend),
        "device": str(device),
        "dtype": str(dtype),
        "load_elapsed_ms": round(load_elapsed_ms, 2),
        "generate_elapsed_ms": round(generate_elapsed_ms, 2),
        "size": list(image.size) if image is not None else None,
        "cuda": _cuda_snapshot(),
        "native_unet": native_status,
        "output": str(output_path) if output_path is not None and image is not None else "",
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
