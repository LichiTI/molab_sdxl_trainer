"""QPiSSA/PiSSA-style residual converter for toolbox use.

This is a manual utility: it reads a safetensors checkpoint, extracts a low
rank approximation from selected 2D weights, writes the residual checkpoint,
and writes a LoRA-like initializer that mirrors the source keys.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def _target_weight(name: str, tensor: torch.Tensor, pattern: re.Pattern[str] | None) -> bool:
    if tensor.ndim != 2:
        return False
    if pattern is not None:
        return bool(pattern.search(name))
    lowered = name.lower()
    return any(token in lowered for token in ("attn", "ff", "mlp", "proj", "linear"))


def _factorize(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_rank = min(weight.shape)
    rank = max(1, min(int(rank), max_rank))
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    s_sqrt = torch.sqrt(s).diag()
    down = s_sqrt @ vh
    up = u @ s_sqrt
    return down, up


def convert_qpissa(
    model_path: str,
    output_dir: str,
    rank: int = 16,
    layers_pattern: str | None = None,
    precision: str = "fp16",
    device: str = "cpu",
) -> dict[str, Any]:
    """Convert a checkpoint into residual weights plus a PiSSA initializer."""
    src = Path(model_path)
    if not src.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(layers_pattern) if layers_pattern else None
    target_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16

    state = load_file(str(src), device="cpu")
    residual: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    converted: list[dict[str, Any]] = []
    skipped = 0

    for key, tensor in state.items():
        if not _target_weight(key, tensor, pattern):
            residual[key] = tensor
            skipped += 1
            continue

        w = tensor.to(target_device, dtype=torch.float32)
        down, up = _factorize(w, rank)
        res = w - (up @ down)
        residual[key] = res.detach().cpu().to(dtype)

        base = key.replace(".", "_")
        adapter[f"lora_unet_{base}.lora_down.weight"] = down.detach().cpu().to(dtype)
        adapter[f"lora_unet_{base}.lora_up.weight"] = up.detach().cpu().to(dtype)
        adapter[f"lora_unet_{base}.alpha"] = torch.tensor(float(down.shape[0]))
        converted.append({"key": key, "shape": list(tensor.shape), "rank": int(down.shape[0])})

    residual_path = out_dir / f"{src.stem}_qpissa_residual.safetensors"
    adapter_path = out_dir / f"{src.stem}_qpissa_init_rank{rank}.safetensors"
    save_file(residual, str(residual_path))
    save_file(adapter, str(adapter_path))

    return {
        "success": True,
        "model_path": str(src),
        "residual_path": str(residual_path),
        "adapter_path": str(adapter_path),
        "rank": rank,
        "converted_layers": len(converted),
        "skipped_tensors": skipped,
        "layers": converted[:200],
    }

