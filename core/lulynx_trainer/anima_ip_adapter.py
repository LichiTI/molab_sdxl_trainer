# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Anima IP-Adapter (Phase 8.10 / #118).

Inject image-prompt conditioning into the Anima DiT via an extra
cross-attention path.  Pipeline:

  1. Encode a reference image with a frozen vision encoder (CLIP or DINO).
  2. Project the encoder output into the DiT's conditioning dimension via
     a learnable :class:`ImageProjector`.
  3. Concatenate (or replace) the projected tokens to the existing text
     conditioning tokens before they enter the DiT cross-attention.

Only the projector is trainable; the vision encoder stays frozen.

This module is wiring — it doesn't load specific encoder checkpoints.
The caller passes any callable that turns an image batch into a
``[batch, num_tokens, encoder_dim]`` tensor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class IPAdapterConfig:
    """Configuration for the Anima IP-Adapter."""

    encoder_dim: int = 1024            # vision encoder output dim (e.g. CLIP-L = 768)
    cond_dim: int = 1152               # DiT conditioning dim
    num_image_tokens: int = 16         # number of projected tokens per image
    num_layers: int = 2                # depth of the projector MLP
    dropout: float = 0.0
    scale: float = 1.0                 # multiplier on projected tokens
    cond_mode: str = "concat"          # "concat" or "replace"


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class ImageProjector(nn.Module):
    """Project vision encoder features into DiT conditioning space.

    Input:  ``[batch, encoder_seq, encoder_dim]`` or ``[batch, encoder_dim]``
    Output: ``[batch, num_image_tokens, cond_dim]``
    """

    def __init__(self, config: IPAdapterConfig) -> None:
        super().__init__()
        self.config = config

        # Learnable query tokens that attend over encoder features
        self.queries = nn.Parameter(
            torch.randn(config.num_image_tokens, config.cond_dim) * 0.02
        )
        self.kv_proj = nn.Linear(config.encoder_dim, config.cond_dim * 2, bias=False)
        self.out_proj = nn.Linear(config.cond_dim, config.cond_dim, bias=True)

        layers = []
        for _ in range(max(0, config.num_layers - 1)):
            layers.append(nn.Linear(config.cond_dim, config.cond_dim))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.norm = nn.LayerNorm(config.cond_dim)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        if encoder_features.dim() == 2:
            # [batch, dim] -> [batch, 1, dim]
            encoder_features = encoder_features.unsqueeze(1)

        batch = encoder_features.shape[0]
        # Expand learned queries to batch
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)  # [B, T, D]

        # Project encoder features to keys/values
        kv = self.kv_proj(encoder_features)  # [B, S, 2D]
        k, v = kv.chunk(2, dim=-1)

        # Single-head attention from queries to encoder features
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B, T, D]

        out = self.out_proj(out)
        out = self.dropout(self.mlp(out))
        out = self.norm(out)
        return out


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

class AnimaIPAdapter(nn.Module):
    """High-level wrapper combining a (frozen) vision encoder + projector.

    Parameters
    ----------
    encode_fn : callable
        ``image_tensor -> [batch, encoder_seq, encoder_dim]`` features.
    config : IPAdapterConfig
    """

    def __init__(
        self,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        config: IPAdapterConfig,
    ) -> None:
        super().__init__()
        self.encode_fn = encode_fn
        self.config = config
        self.projector = ImageProjector(config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encode_fn(image)
        return self.projector(features) * self.config.scale

    def merge_with_text_cond(
        self,
        image_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Combine image tokens with the existing text conditioning.

        Returns ``(combined_tokens, combined_attention_mask)``.
        """
        if self.config.cond_mode == "replace" or text_tokens is None:
            return image_tokens, None

        if text_tokens.dim() != image_tokens.dim():
            return image_tokens, None
        if text_tokens.shape[-1] != image_tokens.shape[-1]:
            logger.warning(
                "AnimaIPAdapter: text_tokens dim %d does not match cond_dim %d",
                text_tokens.shape[-1], image_tokens.shape[-1],
            )
            return image_tokens, None

        combined = torch.cat([text_tokens, image_tokens], dim=1)
        new_mask = None
        if text_attention_mask is not None:
            image_mask = text_attention_mask.new_ones(image_tokens.shape[:2])
            new_mask = torch.cat([text_attention_mask, image_mask], dim=1)
        return combined, new_mask

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.projector.parameters())
