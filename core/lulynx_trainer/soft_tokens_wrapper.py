"""Soft Tokens integration for text conditioning.

Soft Tokens are learnable token embeddings prepended to text conditioning,
conditioned on layer index and diffusion timestep.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .soft_tokens import SoftTokenConfig, SoftTokenBank, SoftTokenPrependResult

logger = logging.getLogger(__name__)


class SoftTokensWrapper:
    """Wrapper for integrating Soft Tokens into text encoding pipeline.

    Example
    -------
    >>> config = SoftTokenConfig(num_tokens=4, hidden_size=768, layer_ids=(0, 5, 10))
    >>> wrapper = SoftTokensWrapper(config)
    >>> # During training:
    >>> text_embeds = text_encoder(tokens)
    >>> enhanced = wrapper.prepend_tokens(text_embeds, layer_index=5, timestep=10, total_steps=20)
    """

    def __init__(self, config: SoftTokenConfig):
        """
        Parameters
        ----------
        config : SoftTokenConfig
            Soft token configuration (num_tokens, hidden_size, layer_ids, timestep_boundaries).
        """
        self.config = config
        self.config.validate()
        self.bank = SoftTokenBank(config)
        logger.info(
            f"SoftTokensWrapper initialized: num_tokens={config.num_tokens}, "
            f"hidden_size={config.hidden_size}, layers={config.layer_ids}"
        )

    def prepend_tokens(
        self,
        text_embeds: torch.Tensor,
        layer_index: int,
        timestep: int,
        total_steps: int,
        attention_mask: Optional[torch.Tensor] = None,
        enabled: bool = True,
    ) -> SoftTokenPrependResult:
        """Prepend soft tokens to text embeddings.

        Parameters
        ----------
        text_embeds : torch.Tensor
            Text embeddings of shape [batch, seq_len, hidden_size].
        layer_index : int
            Current DiT block/layer index.
        timestep : int
            Current diffusion timestep.
        total_steps : int
            Total diffusion steps.
        attention_mask : torch.Tensor, optional
            Attention mask of shape [batch, seq_len].
        enabled : bool, default True
            Whether to actually prepend tokens.

        Returns
        -------
        SoftTokenPrependResult
            Result containing enhanced embeddings and updated attention mask.
        """
        return self.bank.prepend(
            text_embeds,
            layer_index=layer_index,
            timestep=timestep,
            total_steps=total_steps,
            attention_mask=attention_mask,
            enabled=enabled,
        )

    def get_trainable_params(self):
        """Get trainable soft token parameters."""
        return [self.bank.tokens]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict of soft token bank."""
        return {"tokens": self.bank.tokens}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load soft token parameters from state dict."""
        if "tokens" in state_dict:
            self.bank.tokens.data.copy_(state_dict["tokens"])

    def metadata(self) -> Dict[str, str]:
        """Get metadata for soft tokens export."""
        return self.bank.metadata()

    def to(self, device):
        """Move soft token bank to device."""
        self.bank = self.bank.to(device)
        return self


class TextEncoderWithSoftTokens(nn.Module):
    """Wrapper around text encoder that integrates soft tokens.

    This wraps a standard text encoder and automatically prepends soft tokens
    based on layer and timestep context.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        soft_tokens_wrapper: SoftTokensWrapper,
    ):
        """
        Parameters
        ----------
        text_encoder : nn.Module
            The base text encoder (e.g., CLIP text encoder).
        soft_tokens_wrapper : SoftTokensWrapper
            Soft tokens wrapper instance.
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.soft_tokens = soft_tokens_wrapper
        self._current_layer = None
        self._current_timestep = None
        self._total_steps = None

    def set_context(self, layer_index: int, timestep: int, total_steps: int):
        """Set current layer and timestep context for soft token selection."""
        self._current_layer = layer_index
        self._current_timestep = timestep
        self._total_steps = total_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """Forward pass with soft token prepending.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape [batch, seq_len].
        attention_mask : torch.Tensor, optional
            Attention mask of shape [batch, seq_len].
        return_dict : bool, default True
            Whether to return a dict or tuple.

        Returns
        -------
        Output with soft tokens prepended to embeddings.
        """
        # Encode text
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        # Extract embeddings
        if return_dict:
            text_embeds = outputs.last_hidden_state
        else:
            text_embeds = outputs[0]

        # Prepend soft tokens if context is set
        if (
            self._current_layer is not None
            and self._current_timestep is not None
            and self._total_steps is not None
        ):
            result = self.soft_tokens.prepend_tokens(
                text_embeds,
                layer_index=self._current_layer,
                timestep=self._current_timestep,
                total_steps=self._total_steps,
                attention_mask=attention_mask,
            )
            text_embeds = result.embeddings
            attention_mask = result.attention_mask

        # Return in same format
        if return_dict:
            outputs.last_hidden_state = text_embeds
            if attention_mask is not None and hasattr(outputs, "attention_mask"):
                outputs.attention_mask = attention_mask
            return outputs
        else:
            return (text_embeds,) + outputs[1:]

    def get_trainable_params(self):
        """Get trainable parameters (soft tokens only, not text encoder)."""
        return self.soft_tokens.get_trainable_params()
