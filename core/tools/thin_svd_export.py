# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Thin-SVD adapter export: compress/merge LoRA deltas for inference.

A dedicated "thin" export route (roadmap line 554).  It reconstructs each LoRA
layer delta ``W = up @ down``, optionally merges several adapters in weight
space, then re-factorizes with a thin SVD:

- ``mode="exact"``  keeps the full effective rank (lossless repackage / merge).
- ``mode="approx"`` truncates to ``target_rank`` and records the retained
  spectral energy so a caller can judge the approximation quality.

The export writes ``thin_svd_mode`` / ``original_rank`` / ``target_rank`` /
``energy_retained`` into the safetensors metadata header.

Cleanroom implementation; shares the SVD re-decomposition approach used by
``lora_rank_resize.resize_lora_rank`` but adds multi-adapter merge + metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_save_dtype(save_precision: str, reference: torch.dtype) -> torch.dtype:
    return _DTYPES.get(str(save_precision or "").strip().lower(), reference)


def _layer_prefixes(tensors: dict[str, torch.Tensor]) -> list[str]:
    prefixes = {
        key[: -len(".lora_down.weight")]
        for key in tensors
        if key.endswith(".lora_down.weight")
    }
    return sorted(prefixes)


def _reconstruct_delta(down: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, Optional[tuple[int, int, int]]]:
    """Return ``W = up @ down`` as 2D plus conv shape ``(in_ch, kh, kw)`` if 4D."""
    old_rank = down.shape[0]
    conv_shape: Optional[tuple[int, int, int]] = None
    if down.dim() == 4:
        conv_shape = (down.shape[1], down.shape[2], down.shape[3])
        down = down.reshape(old_rank, -1)
        up = up.reshape(up.shape[0], old_rank)
    return up.float() @ down.float(), conv_shape


def _svd_factor(weight: torch.Tensor, rank: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Thin-SVD factorize ``weight`` to ``rank``: returns down, up, S(all), r."""
    try:
        U, S, Vh = torch.linalg.svd(weight.to(device), full_matrices=False)
    except Exception:
        U, S, Vh = torch.linalg.svd(weight.cpu(), full_matrices=False)
        U, S, Vh = U.to(device), S.to(device), Vh.to(device)
    r = max(1, min(int(rank), int(S.numel())))
    sqrt_s = S[:r].sqrt()
    new_down = sqrt_s.unsqueeze(1) * Vh[:r]      # (r, in)
    new_up = U[:, :r] * sqrt_s.unsqueeze(0)      # (out, r)
    return new_down, new_up, S, r


def thin_svd_export(
    input_paths: str | Sequence[str],
    output_path: str,
    target_rank: int = 32,
    *,
    mode: str = "approx",
    merge_weights: Optional[Sequence[float]] = None,
    device: str = "cpu",
    save_precision: str = "fp16",
) -> dict:
    """Compress/merge LoRA adapters via thin SVD.

    Args:
        input_paths: one adapter path or several to merge in weight space.
        output_path: destination safetensors.
        target_rank: rank cap for ``mode="approx"`` (ignored by ``exact``).
        mode: ``"approx"`` (truncate) or ``"exact"`` (keep full rank).
        merge_weights: per-input linear weights (defaults to all 1.0).
        device: torch device for the SVD.
        save_precision: output dtype (fp16/bf16/fp32).
    """
    from safetensors.torch import load_file, save_file

    paths = [str(input_paths)] if isinstance(input_paths, (str, Path)) else [str(p) for p in input_paths]
    if not paths:
        raise ValueError("thin_svd_export requires at least one input path")
    weights = list(merge_weights) if merge_weights else [1.0] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("merge_weights length must match input_paths")

    mode_key = str(mode or "approx").strip().lower()
    if mode_key not in {"approx", "exact"}:
        raise ValueError(f"unsupported thin-svd mode: {mode}")

    loaded = [load_file(p, device=device) for p in paths]
    prefixes = _layer_prefixes(loaded[0])

    out_tensors: dict[str, torch.Tensor] = {}
    layers_exported = 0
    original_rank_max = 0
    target_rank_eff = 0
    energy_sum = 0.0

    for prefix in prefixes:
        down_key = f"{prefix}.lora_down.weight"
        up_key = f"{prefix}.lora_up.weight"
        alpha_key = f"{prefix}.alpha"

        merged_w: Optional[torch.Tensor] = None
        conv_shape: Optional[tuple[int, int, int]] = None
        out_features = 0
        layer_old_rank = 0
        layer_total_rank = 0
        for tensors, w in zip(loaded, weights):
            if down_key not in tensors or up_key not in tensors:
                continue
            down, up = tensors[down_key], tensors[up_key]
            layer_old_rank = max(layer_old_rank, int(down.shape[0]))
            layer_total_rank += int(down.shape[0])
            out_features = up.shape[0]
            delta, conv_shape = _reconstruct_delta(down, up)
            merged_w = delta * float(w) if merged_w is None else merged_w + delta * float(w)
        if merged_w is None:
            continue

        full_rank = int(min(merged_w.shape))
        if mode_key == "exact":
            rank = min(full_rank, layer_total_rank)
        else:
            rank = min(int(target_rank), full_rank)
        new_down, new_up, S, r = _svd_factor(merged_w, rank, device)

        total_energy = float((S ** 2).sum().item()) + 1e-12
        energy_sum += float((S[:r] ** 2).sum().item()) / total_energy
        original_rank_max = max(original_rank_max, layer_old_rank)
        target_rank_eff = max(target_rank_eff, r)

        ref_dtype = loaded[0][down_key].dtype
        save_dtype = _resolve_save_dtype(save_precision, ref_dtype)
        if conv_shape is not None:
            in_ch, kh, kw = conv_shape
            new_down = new_down.reshape(r, in_ch, kh, kw)
            new_up = new_up.reshape(out_features, r, 1, 1)
        out_tensors[down_key] = new_down.contiguous().to(save_dtype)
        out_tensors[up_key] = new_up.contiguous().to(save_dtype)

        if alpha_key in loaded[0]:
            old_alpha = float(loaded[0][alpha_key].item())
            ratio = (r / layer_old_rank) if layer_old_rank else 1.0
            out_tensors[alpha_key] = torch.tensor(old_alpha * ratio)
        layers_exported += 1

    # Preserve any non-LoRA keys from the first adapter (metadata tensors, etc).
    for key, val in loaded[0].items():
        out_tensors.setdefault(key, val)

    energy_retained = (energy_sum / layers_exported) if layers_exported else 0.0
    meta = {
        "thin_svd_mode": mode_key,
        "original_rank": str(original_rank_max),
        "target_rank": str(target_rank_eff),
        "energy_retained": f"{energy_retained:.6f}",
        "merged_inputs": str(len(paths)),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, output_path, metadata=meta)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info("thin_svd_export: %s layers, mode=%s, energy=%.4f", layers_exported, mode_key, energy_retained)
    return {
        "layers_exported": layers_exported,
        "original_rank": original_rank_max,
        "target_rank": target_rank_eff,
        "mode": mode_key,
        "energy_retained": round(energy_retained, 6),
        "output_path": str(output_path),
        "output_size_mb": round(size_mb, 3),
    }
