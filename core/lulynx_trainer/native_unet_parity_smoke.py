"""Heavy SDXL native-vs-diffusers U-Net parity smoke.

Runs the same small latent through the reference diffusers U-Net and the
Warehouse native wrapper.  It frees the reference U-Net before loading native
weights so the check can run on 16GB GPUs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.native_unet import build_sdxl_unet_compat_from_manifest
from core.lulynx_trainer.single_file_loader import load_sdxl_single_file_components


def _dtype_from_name(name: str) -> torch.dtype:
    value = str(name or "bf16").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _cleanup_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"),
    )
    parser.add_argument(
        "--model",
        default=str(root / "models" / "sdxl" / "silentEraFurrymixNAIXL_v10.safetensors"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--max-abs-threshold", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    latent_size = max(int(args.latent_size), 8)
    tokens = max(int(args.tokens), 1)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    sample_cpu = torch.randn(1, 4, latent_size, latent_size, generator=generator, dtype=torch.float32)
    encoder_cpu = torch.randn(1, tokens, 2048, generator=generator, dtype=torch.float32)
    text_cpu = torch.randn(1, 1280, generator=generator, dtype=torch.float32)
    time_ids_cpu = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32)
    timestep_cpu = torch.tensor([1])

    load_started = time.perf_counter()
    components = load_sdxl_single_file_components(args.model, torch_dtype=dtype)
    ref_unet = components["unet"].to(device=device, dtype=dtype)
    ref_unet.eval()
    for key in list(components.keys()):
        if key != "unet":
            components[key] = None
    sample = sample_cpu.to(device=device, dtype=dtype)
    encoder = encoder_cpu.to(device=device, dtype=dtype)
    added = {
        "text_embeds": text_cpu.to(device=device, dtype=dtype),
        "time_ids": time_ids_cpu.to(device=device, dtype=dtype),
    }
    with torch.no_grad():
        ref_out = ref_unet(
            sample=sample,
            timestep=timestep_cpu.to(device=device),
            encoder_hidden_states=encoder,
            added_cond_kwargs=added,
            return_dict=False,
        )[0].detach().float().cpu()
    del ref_unet
    del components
    _cleanup_cuda(device)
    ref_seconds = time.perf_counter() - load_started

    native_started = time.perf_counter()
    native_unet = build_sdxl_unet_compat_from_manifest(args.manifest, args.model, device=device, dtype=dtype)
    native_unet.eval()
    with torch.no_grad():
        native_out = native_unet(
            sample=sample_cpu.to(device=device, dtype=dtype),
            timestep=timestep_cpu.to(device=device),
            encoder_hidden_states=encoder_cpu.to(device=device, dtype=dtype),
            added_cond_kwargs={
                "text_embeds": text_cpu.to(device=device, dtype=dtype),
                "time_ids": time_ids_cpu.to(device=device, dtype=dtype),
            },
            return_dict=False,
        )[0].detach().float().cpu()
    del native_unet
    _cleanup_cuda(device)
    native_seconds = time.perf_counter() - native_started

    diff = (ref_out - native_out).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    print(
        "native_unet_parity_smoke: "
        f"shape={tuple(native_out.shape)} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"ref_seconds={ref_seconds:.2f} native_seconds={native_seconds:.2f}"
    )
    if max_abs > float(args.max_abs_threshold):
        raise RuntimeError(f"native parity max_abs {max_abs:.6f} exceeded threshold {args.max_abs_threshold:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

