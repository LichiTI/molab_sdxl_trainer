# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Generate baseline-vs-LoRA preview images for the Automagic++ real-train smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _make_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    cell_w, cell_h = images[0].size
    label_h = 28
    grid = Image.new("RGB", (cell_w * len(images), cell_h + label_h), "white")
    for idx, image in enumerate(images):
        grid.paste(image.convert("RGB"), (idx * cell_w, label_h))
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(grid)
        for idx, label in enumerate(labels):
            draw.text((idx * cell_w + 8, 7), label, fill=(0, 0, 0))
    except Exception:
        pass
    return grid


def _diff_heatmap(a: Image.Image, b: Image.Image) -> tuple[Image.Image, dict[str, float]]:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.int16)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.int16)
    delta = np.abs(arr_a - arr_b).astype(np.float32)
    mean_abs = float(delta.mean())
    max_abs = float(delta.max())
    rms = float(np.sqrt(np.mean((arr_a.astype(np.float32) - arr_b.astype(np.float32)) ** 2)))
    heat = np.clip(delta.mean(axis=2) * 5.0, 0, 255).astype(np.uint8)
    rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=2)
    return Image.fromarray(rgb, mode="RGB"), {
        "mean_abs_pixel_delta": mean_abs,
        "max_abs_pixel_delta": max_abs,
        "rms_pixel_delta": rms,
    }


def _normalized_module_map(root: torch.nn.Module, prefix: str) -> dict[str, torch.nn.Module]:
    result: dict[str, torch.nn.Module] = {}
    for name, module in root.named_modules():
        if name and hasattr(module, "weight"):
            result[prefix + name.replace(".", "_")] = module
    return result


@torch.no_grad()
def _merge_lulynx_lora(pipe, lora_path: str, scale: float) -> dict[str, int]:
    state = load_file(lora_path, device="cpu")
    modules: dict[str, torch.nn.Module] = {}
    modules.update(_normalized_module_map(pipe.unet, "unet_"))
    if getattr(pipe, "text_encoder", None) is not None:
        modules.update(_normalized_module_map(pipe.text_encoder, "text_encoder_"))
    if getattr(pipe, "text_encoder_2", None) is not None:
        modules.update(_normalized_module_map(pipe.text_encoder_2, "text_encoder_2_"))

    merged = 0
    skipped = 0
    for key, down in state.items():
        if not key.endswith(".lora_down.weight"):
            continue
        base = key[: -len(".lora_down.weight")]
        up_key = base + ".lora_up.weight"
        up = state.get(up_key)
        module = modules.get(base)
        if up is None or module is None or not hasattr(module, "weight"):
            skipped += 1
            continue

        weight = module.weight
        down_f = down.to(device=weight.device, dtype=torch.float32)
        up_f = up.to(device=weight.device, dtype=torch.float32)
        if weight.ndim == 2 and down_f.ndim == 2 and up_f.ndim == 2:
            delta = up_f @ down_f
        elif weight.ndim == 4 and down_f.ndim == 4 and up_f.ndim == 4:
            delta = torch.einsum("orxy,riuv->oiuv", up_f, down_f)
        else:
            skipped += 1
            continue
        if tuple(delta.shape) != tuple(weight.shape):
            skipped += 1
            continue
        weight.add_(delta.to(dtype=weight.dtype), alpha=float(scale))
        merged += 1

    return {"merged": merged, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description="Automagic++ LoRA preview smoke")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="low quality, blurry, worst quality")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=5.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--lora-scale-strong", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
    from core.lulynx_trainer.single_file_loader import load_sdxl_single_file_components

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    components = load_sdxl_single_file_components(args.model, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline(
        vae=components["vae"],
        text_encoder=components["text_encoder_1"],
        text_encoder_2=components["text_encoder_2"],
        tokenizer=components["tokenizer_1"],
        tokenizer_2=components["tokenizer_2"],
        unet=components["unet"],
        scheduler=components["noise_scheduler"],
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    def generate(label: str) -> Image.Image:
        generator = torch.Generator(device=pipe.device).manual_seed(int(args.seed))
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=int(args.width),
            height=int(args.height),
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.cfg),
            generator=generator,
        ).images[0]
        image.save(output_dir / f"{label}.png")
        return image

    baseline = generate("baseline")
    merge_1 = _merge_lulynx_lora(pipe, args.lora, float(args.lora_scale))
    lora = generate("automagic_plus_plus_lora_scale_1")
    merge_2 = _merge_lulynx_lora(pipe, args.lora, float(args.lora_scale_strong) - float(args.lora_scale))
    lora_strong = generate("automagic_plus_plus_lora_scale_2")

    heatmap, metrics = _diff_heatmap(baseline, lora)
    heatmap.save(output_dir / "baseline_vs_lora_heatmap.png")
    grid = _make_grid(
        [baseline, lora, lora_strong, heatmap],
        ["baseline", f"LoRA x{args.lora_scale:g}", f"LoRA x{args.lora_scale_strong:g}", "diff heat"],
    )
    grid.save(output_dir / "comparison_grid.png")

    report = {
        "model": str(args.model),
        "lora": str(args.lora),
        "prompt": args.prompt,
        "negative": args.negative,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "cfg": float(args.cfg),
        "width": int(args.width),
        "height": int(args.height),
        "lora_scale": float(args.lora_scale),
        "lora_scale_strong": float(args.lora_scale_strong),
        "metrics": metrics,
        "merge": {
            "scale_1": merge_1,
            "scale_2_increment": merge_2,
        },
        "outputs": {
            "baseline": str(output_dir / "baseline.png"),
            "lora": str(output_dir / "automagic_plus_plus_lora_scale_1.png"),
            "lora_strong": str(output_dir / "automagic_plus_plus_lora_scale_2.png"),
            "heatmap": str(output_dir / "baseline_vs_lora_heatmap.png"),
            "grid": str(output_dir / "comparison_grid.png"),
        },
    }
    (output_dir / "preview_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
