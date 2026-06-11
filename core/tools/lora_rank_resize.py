# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""LoRA rank resize via SVD re-decomposition."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger(__name__)

def resize_lora_rank(
    input_path: str,
    output_path: str,
    target_rank: int,
    device: str = "cpu",
) -> dict:
    """Re-decompose LoRA weights to a different rank.

    For each lora_down/lora_up pair:
    1. Reconstruct W = up @ down
    2. SVD truncate to target_rank
    3. Rescale alpha to preserve effective magnitude

    Returns dict with stats: {layers_resized, original_rank, target_rank}
    """
    from safetensors.torch import load_file, save_file

    tensors = load_file(input_path, device=device)
    output_tensors = {}
    metadata = {}
    layers_resized = 0

    # Group keys by layer prefix
    layer_prefixes = set()
    for key in tensors:
        if ".lora_down.weight" in key:
            prefix = key.replace(".lora_down.weight", "")
            layer_prefixes.add(prefix)

    for prefix in sorted(layer_prefixes):
        down_key = f"{prefix}.lora_down.weight"
        up_key = f"{prefix}.lora_up.weight"
        alpha_key = f"{prefix}.alpha"

        if down_key not in tensors or up_key not in tensors:
            continue

        down = tensors[down_key].float()  # (old_rank, in_features) or 4D for conv
        up = tensors[up_key].float()      # (out_features, old_rank) or 4D for conv

        old_rank = down.shape[0]

        # Handle Conv2d: reshape to 2D for SVD
        is_conv = down.dim() == 4
        if is_conv:
            # down: (old_rank, in_ch, kh, kw) → (old_rank, in_ch*kh*kw)
            down_2d = down.reshape(old_rank, -1)
            # up: (out_ch, old_rank, 1, 1) → (out_ch, old_rank)
            up_2d = up.reshape(up.shape[0], old_rank)
        else:
            down_2d = down
            up_2d = up

        # Reconstruct full weight delta
        W = up_2d @ down_2d  # (out, in)

        # SVD
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        except Exception:
            # Fallback to CPU if CUDA OOM
            U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
            U, S, Vh = U.to(device), S.to(device), Vh.to(device)

        r = min(target_rank, len(S))
        sqrt_s = S[:r].sqrt()

        new_down = (sqrt_s.unsqueeze(1) * Vh[:r])   # (r, in)
        new_up = (U[:, :r] * sqrt_s.unsqueeze(0))    # (out, r)

        # Reshape back for Conv2d
        if is_conv:
            kh, kw = down.shape[2], down.shape[3]
            in_ch = down.shape[1]
            new_down = new_down.reshape(r, in_ch, kh, kw)
            new_up = new_up.reshape(up.shape[0], r, 1, 1)

        output_tensors[down_key] = new_down.contiguous().to(tensors[down_key].dtype)
        output_tensors[up_key] = new_up.contiguous().to(tensors[up_key].dtype)

        # Rescale alpha
        if alpha_key in tensors:
            old_alpha = tensors[alpha_key].item()
            new_alpha = old_alpha * (r / old_rank)
            output_tensors[alpha_key] = torch.tensor(new_alpha)

        layers_resized += 1

    # Copy non-LoRA keys (e.g., metadata tensors)
    for key, val in tensors.items():
        if key not in output_tensors:
            output_tensors[key] = val

    save_file(output_tensors, output_path)
    logger.info(f"Resized {layers_resized} layers to rank {target_rank}")
    return {"layers_resized": layers_resized, "target_rank": target_rank}
