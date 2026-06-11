# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Generate baseline vs Automagic++ LoRA SDXL samples for visual inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def _image_delta(a_path: Path, b_path: Path) -> dict[str, float]:
    from PIL import Image

    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float32)
    diff = np.abs(a - b)
    return {
        "mean_abs_pixel_delta": float(diff.mean()),
        "max_abs_pixel_delta": float(diff.max()),
        "changed_pixel_ratio_gt_5": float((diff.mean(axis=2) > 5.0).mean()),
    }


def _load_sdxl_pipeline(model_path: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    """Load SDXL through the project's offline single-file path.

    Diffusers' generic from_single_file path can trip over some local SDXL
    checkpoints/text encoder combinations, while the trainer already carries a
    compatible offline loader for this exact model family.
    """

    from core.lulynx_trainer.single_file_loader import load_sdxl_single_file_components

    components = load_sdxl_single_file_components(model_path, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline(
        vae=components["vae"],
        text_encoder=components["text_encoder_1"],
        text_encoder_2=components["text_encoder_2"],
        tokenizer=components["tokenizer_1"],
        tokenizer_2=components["tokenizer_2"],
        unet=components["unet"],
        scheduler=components["noise_scheduler"],
        force_zeros_for_empty_prompt=True,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description="Automagic++ LoRA inference comparison smoke.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="low quality, worst quality, blurry, deformed")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.png"
    lora_path = output_dir / "automagic_plus_plus_lora.png"
    report_path = output_dir / "inference_report.json"

    dtype = torch.float16
    pipe = _load_sdxl_pipeline(args.model, dtype)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    generator = torch.Generator(device="cuda").manual_seed(int(args.seed))
    baseline = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=int(args.width),
        height=int(args.height),
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.cfg),
        generator=generator,
    ).images[0]
    baseline.save(baseline_path)

    from core.lulynx_trainer.lora_injector import LoRAInjector

    lora_injector = LoRAInjector(
        rank=max(int(args.rank), 1),
        alpha=float(args.alpha) * float(args.lora_scale),
        model_arch="sdxl",
    )
    lora_injector.inject_unet(pipe.unet)
    lora_injector.load_lora(args.lora)
    pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(int(args.seed))
    lora_image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=int(args.width),
        height=int(args.height),
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.cfg),
        generator=generator,
    ).images[0]
    lora_image.save(lora_path)

    delta = _image_delta(baseline_path, lora_path)
    report = {
        "model": args.model,
        "lora": args.lora,
        "prompt": args.prompt,
        "negative": args.negative,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "width": int(args.width),
        "height": int(args.height),
        "cfg": float(args.cfg),
        "lora_scale": float(args.lora_scale),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "injected_lora_layers": len(lora_injector.injected_layers),
        "baseline": str(baseline_path),
        "automagic_plus_plus_lora": str(lora_path),
        **delta,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
