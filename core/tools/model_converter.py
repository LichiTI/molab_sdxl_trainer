"""Manual model conversion helpers for the toolbox API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _load_pt_state(path: str) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "state_dict" in data and isinstance(data["state_dict"], dict):
        data = data["state_dict"]
    if not isinstance(data, dict):
        raise ValueError("PyTorch file does not contain a state dict.")
    return {str(k): v for k, v in data.items() if torch.is_tensor(v)}


def convert_tensor_file(input_path: str, output_path: str, output_format: str = "safetensors") -> dict[str, Any]:
    """Convert simple tensor containers between safetensors and PyTorch pt/pth."""
    src = Path(input_path)
    dst = Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    output_format = output_format.lower().strip()
    if output_format not in {"safetensors", "pt", "pth"}:
        raise ValueError("output_format must be safetensors, pt, or pth")

    if src.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(src), device="cpu")
    else:
        state = _load_pt_state(str(src))

    if output_format == "safetensors" or dst.suffix.lower() == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(dst))
    else:
        torch.save(state, str(dst))

    return {
        "success": True,
        "input_path": str(src),
        "output_path": str(dst),
        "output_format": output_format,
        "tensor_count": len(state),
    }


def convert_checkpoint_to_diffusers(
    checkpoint_path: str,
    output_dir: str,
    model_type: str = "sdxl",
    half: bool = True,
) -> dict[str, Any]:
    """Convert a single-file checkpoint into a Diffusers directory."""
    src = Path(checkpoint_path)
    out = Path(output_dir)
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    out.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if half else torch.float32
    model_type = model_type.lower().strip()

    if model_type in {"sdxl", "xl"}:
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_single_file(str(src), torch_dtype=dtype)
    elif model_type in {"sd15", "sd1.5", "sd"}:
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_single_file(str(src), torch_dtype=dtype)
    elif model_type == "sd3":
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_single_file(str(src), torch_dtype=dtype)
    elif model_type == "flux":
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_single_file(str(src), torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipe.save_pretrained(str(out), safe_serialization=True)
    return {"success": True, "input_path": str(src), "output_dir": str(out), "model_type": model_type}

