"""Heavy full-native SDXL UNet forward smoke.

This is intentionally separate from native_unet_smoke.py because it loads the
entire SDXL U-Net.  Use it when validating the opt-in ``lulynx_native`` backend.
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


def _dtype_from_name(name: str) -> torch.dtype:
    value = str(name or "bf16").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


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
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    started = time.perf_counter()
    unet = build_sdxl_unet_compat_from_manifest(
        args.manifest,
        args.model,
        device=device,
        dtype=dtype,
    )
    unet.eval()
    load_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    latent_size = max(int(args.latent_size), 8)
    tokens = max(int(args.tokens), 1)
    sample = torch.randn(1, 4, latent_size, latent_size, device=device, dtype=dtype)
    encoder = torch.randn(1, tokens, 2048, device=device, dtype=dtype)
    added = {
        "text_embeds": torch.randn(1, 1280, device=device, dtype=dtype),
        "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=dtype),
    }
    forward_started = time.perf_counter()
    with torch.no_grad():
        output = unet(
            sample=sample,
            timestep=torch.tensor([1], device=device),
            encoder_hidden_states=encoder,
            added_cond_kwargs=added,
            return_dict=False,
        )[0]
    forward_seconds = time.perf_counter() - forward_started
    peak_mb = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(
        "native_unet_full_forward_smoke: ok "
        f"shape={tuple(output.shape)} dtype={output.dtype} "
        f"load_seconds={load_seconds:.2f} forward_seconds={forward_seconds:.2f} "
        f"peak_mb={peak_mb:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
