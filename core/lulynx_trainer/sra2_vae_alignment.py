"""SRA2-style VAE self-representation alignment probe.

The primitive builds DiT-token targets from VAE feature maps so alignment losses
can be tested without adding an external teacher encoder. It is not wired into
the trainer yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SRA2VaeAlignmentPolicy:
    enabled: bool = False
    weight: float = 1.0
    loss_type: str = "cosine"
    normalize_targets: bool = True
    stop_grad_target: bool = True

    def normalized(self) -> "SRA2VaeAlignmentPolicy":
        loss_type = str(self.loss_type or "cosine").strip().lower()
        if loss_type not in {"cosine", "l2", "l1"}:
            loss_type = "cosine"
        weight = 1.0 if self.weight is None else float(self.weight)
        return SRA2VaeAlignmentPolicy(
            enabled=bool(self.enabled),
            weight=max(weight, 0.0),
            loss_type=loss_type,
            normalize_targets=bool(self.normalize_targets),
            stop_grad_target=bool(self.stop_grad_target),
        )


def _validate_hidden(hidden_states: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, tokens, hidden]")
    batch_size, token_count, hidden_size = hidden_states.shape
    if batch_size <= 0 or token_count <= 0 or hidden_size <= 0:
        raise ValueError("hidden_states must have non-empty dimensions")
    return int(batch_size), int(token_count), int(hidden_size)


def _flatten_vae_features(vae_features: torch.Tensor) -> torch.Tensor:
    if not isinstance(vae_features, torch.Tensor):
        raise TypeError("vae_features must be a torch.Tensor")
    if vae_features.ndim == 3:
        if min(vae_features.shape) <= 0:
            raise ValueError("vae_features must have non-empty dimensions")
        return vae_features
    if vae_features.ndim == 4:
        batch_size, channels, height, width = vae_features.shape
        if batch_size <= 0 or channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("vae_features must have non-empty dimensions")
        return vae_features.flatten(2).transpose(1, 2)
    raise ValueError("vae_features must have shape [batch, tokens, channels] or [batch, channels, height, width]")


def build_sra2_vae_token_targets(
    vae_features: torch.Tensor,
    *,
    token_count: int,
    hidden_size: int,
    normalize: bool = True,
) -> torch.Tensor:
    tokens = _flatten_vae_features(vae_features).float()
    if token_count <= 0 or hidden_size <= 0:
        raise ValueError("token_count and hidden_size must be positive")
    if tokens.shape[1] != token_count:
        tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), int(token_count)).transpose(1, 2)

    channels = int(tokens.shape[-1])
    if channels > hidden_size:
        tokens = tokens[..., :hidden_size]
    elif channels < hidden_size:
        pad = tokens.new_zeros((*tokens.shape[:-1], hidden_size - channels))
        tokens = torch.cat([tokens, pad], dim=-1)

    if normalize:
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)
    return tokens


def sra2_vae_alignment_loss(
    hidden_states: torch.Tensor,
    vae_features: torch.Tensor,
    policy: SRA2VaeAlignmentPolicy | Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    batch_size, token_count, hidden_size = _validate_hidden(hidden_states)
    if isinstance(policy, Mapping):
        policy = SRA2VaeAlignmentPolicy(**policy)
    cfg = (policy or SRA2VaeAlignmentPolicy()).normalized()
    if not cfg.enabled or cfg.weight <= 0.0:
        return hidden_states.sum() * 0.0, {
            "enabled": False,
            "reason": "disabled",
            "batch_size": batch_size,
            "token_count": token_count,
            "hidden_size": hidden_size,
            "weight": 0.0,
        }

    target = build_sra2_vae_token_targets(
        vae_features,
        token_count=token_count,
        hidden_size=hidden_size,
        normalize=cfg.normalize_targets,
    ).to(device=hidden_states.device, dtype=hidden_states.dtype)
    if target.shape[0] != batch_size:
        raise ValueError("vae_features batch size must match hidden_states")
    if cfg.stop_grad_target:
        target = target.detach()

    if cfg.loss_type == "l2":
        loss = F.mse_loss(hidden_states, target)
    elif cfg.loss_type == "l1":
        loss = F.l1_loss(hidden_states, target)
    else:
        hidden_flat = hidden_states.flatten(0, -2).float()
        target_flat = target.flatten(0, -2).float()
        loss = 1.0 - F.cosine_similarity(hidden_flat, target_flat, dim=-1).mean()

    return loss * cfg.weight, {
        "enabled": True,
        "reason": "active",
        "batch_size": batch_size,
        "token_count": token_count,
        "hidden_size": hidden_size,
        "vae_feature_shape": tuple(int(v) for v in vae_features.shape),
        "target_shape": tuple(int(v) for v in target.shape),
        "loss_type": cfg.loss_type,
        "weight": float(cfg.weight),
        "normalize_targets": bool(cfg.normalize_targets),
        "stop_grad_target": bool(cfg.stop_grad_target),
    }


__all__ = [
    "SRA2VaeAlignmentPolicy",
    "build_sra2_vae_token_targets",
    "sra2_vae_alignment_loss",
]
