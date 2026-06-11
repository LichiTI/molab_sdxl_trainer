"""Decoder abstraction layer for VAE and external decoders.

LICENSE: PolyForm Noncommercial 1.0.0

This module provides a routing mechanism for image decoders (VAE, external decoders).
It does NOT include any AGPL-licensed decoder implementations.

External decoder implementations (e.g., PiD) must be provided by the user
and loaded dynamically. Users are responsible for ensuring their decoder
checkpoints comply with applicable licenses.

IMPORTANT: This is a cleanroom implementation based on public interfaces only.
No AGPL code was referenced during development.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AbstractDecoder(ABC):
    """Abstract interface for image decoders.

    This interface defines the contract for any decoder (VAE, PiD, etc.)
    that can be used in the training/inference pipeline.
    """

    @abstractmethod
    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Any:
        """Decode latents to images.

        Parameters
        ----------
        latents : torch.Tensor
            Latent representations to decode.
        return_dict : bool, default True
            Whether to return a dict or raw tensor.

        Returns
        -------
        torch.Tensor or object
            Decoded images.
        """
        pass

    @abstractmethod
    def to(self, device: torch.device | str):
        """Move decoder to device."""
        pass

    @abstractmethod
    def eval(self):
        """Set decoder to eval mode."""
        pass


class VAEDecoderWrapper(AbstractDecoder):
    """Wrapper for standard VAE decoder.

    This wraps a diffusers-style VAE to conform to AbstractDecoder interface.
    """

    def __init__(self, vae: nn.Module):
        """
        Parameters
        ----------
        vae : nn.Module
            Diffusers-style VAE model.
        """
        self.vae = vae

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Any:
        """Decode latents using VAE."""
        # VAE expects latents to be scaled
        # Standard diffusers VAE uses scaling factor
        if hasattr(self.vae.config, "scaling_factor"):
            latents = latents / self.vae.config.scaling_factor

        return self.vae.decode(latents, return_dict=return_dict)

    def to(self, device: torch.device | str):
        """Move VAE to device."""
        self.vae = self.vae.to(device)
        return self

    def eval(self):
        """Set VAE to eval mode."""
        self.vae.eval()
        return self


class ExternalDecoderWrapper(AbstractDecoder):
    """Wrapper for externally-provided decoder implementations.

    This class loads decoder implementations from user-provided paths.
    The user is responsible for ensuring the decoder implementation
    complies with applicable licenses.

    IMPORTANT: This wrapper does NOT include any decoder implementation.
    It only provides a loading mechanism for user-provided decoders.
    """

    def __init__(
        self,
        decoder_path: str | Path,
        decoder_type: str = "auto",
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        decoder_path : str or Path
            Path to decoder checkpoint or module.
        decoder_type : str, default "auto"
            Type of decoder ("auto", "pid", "custom").
        device : str, default "cuda"
            Device to load decoder on.
        """
        self.decoder_path = Path(decoder_path)
        self.decoder_type = decoder_type
        self.device = device
        self.decoder = None

        logger.info(
            f"ExternalDecoderWrapper: User is loading decoder from {decoder_path}. "
            f"User is responsible for license compliance."
        )

        self._load_decoder()

    def _load_decoder(self):
        """Load external decoder.

        This is a placeholder that attempts dynamic loading.
        Actual implementation depends on user-provided decoder format.
        """
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder not found: {self.decoder_path}")

        # Try to load as PyTorch checkpoint
        try:
            checkpoint = torch.load(self.decoder_path, map_location=self.device)

            # Check if it's a model state dict
            if isinstance(checkpoint, dict):
                # User needs to provide model architecture separately
                raise ValueError(
                    "Checkpoint is a state dict. Please provide decoder architecture "
                    "or use a different loading mechanism."
                )

            self.decoder = checkpoint
            logger.info(f"Loaded external decoder from {self.decoder_path}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load external decoder from {self.decoder_path}: {e}. "
                f"Please ensure the decoder is in a supported format."
            )

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Any:
        """Decode using external decoder."""
        if self.decoder is None:
            raise RuntimeError("Decoder not loaded")

        # Try common decode methods
        if hasattr(self.decoder, "decode"):
            return self.decoder.decode(latents, return_dict=return_dict)
        elif hasattr(self.decoder, "forward"):
            return self.decoder.forward(latents)
        elif callable(self.decoder):
            return self.decoder(latents)
        else:
            raise RuntimeError("Decoder does not have a callable decode method")

    def to(self, device: torch.device | str):
        """Move decoder to device."""
        if self.decoder is not None:
            self.decoder = self.decoder.to(device)
        self.device = str(device)
        return self

    def eval(self):
        """Set decoder to eval mode."""
        if self.decoder is not None and hasattr(self.decoder, "eval"):
            self.decoder.eval()
        return self


class DecoderRouter:
    """Routes between VAE and external decoders.

    This router allows switching between different decoder implementations
    at runtime based on configuration.

    LICENSE: PolyForm Noncommercial 1.0.0
    This is a cleanroom implementation. No AGPL code was referenced.
    """

    def __init__(
        self,
        vae: nn.Module,
        external_decoder_path: Optional[str | Path] = None,
        external_decoder_type: str = "auto",
        use_external_decoder: bool = False,
    ):
        """
        Parameters
        ----------
        vae : nn.Module
            Standard VAE decoder.
        external_decoder_path : str or Path, optional
            Path to external decoder checkpoint.
        external_decoder_type : str, default "auto"
            Type of external decoder.
        use_external_decoder : bool, default False
            Whether to use external decoder instead of VAE.
        """
        self.vae_decoder = VAEDecoderWrapper(vae)
        self.external_decoder: Optional[AbstractDecoder] = None
        self.use_external = use_external_decoder

        if use_external_decoder:
            if external_decoder_path is None:
                raise ValueError("external_decoder_path required when use_external_decoder=True")

            self.external_decoder = ExternalDecoderWrapper(
                decoder_path=external_decoder_path,
                decoder_type=external_decoder_type,
            )

            logger.info(
                f"DecoderRouter: Using external decoder from {external_decoder_path}. "
                f"User is responsible for license compliance."
            )
        else:
            logger.info("DecoderRouter: Using standard VAE decoder")

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Any:
        """Decode latents using selected decoder.

        Parameters
        ----------
        latents : torch.Tensor
            Latent representations.
        return_dict : bool, default True
            Whether to return dict or raw tensor.

        Returns
        -------
        torch.Tensor or object
            Decoded images.
        """
        if self.use_external and self.external_decoder is not None:
            return self.external_decoder.decode(latents, return_dict=return_dict)
        else:
            return self.vae_decoder.decode(latents, return_dict=return_dict)

    def to(self, device: torch.device | str):
        """Move active decoder to device."""
        if self.use_external and self.external_decoder is not None:
            self.external_decoder.to(device)
        else:
            self.vae_decoder.to(device)
        return self

    def eval(self):
        """Set active decoder to eval mode."""
        if self.use_external and self.external_decoder is not None:
            self.external_decoder.eval()
        else:
            self.vae_decoder.eval()
        return self

    def get_metadata(self) -> Dict[str, Any]:
        """Get decoder metadata for checkpoints.

        Returns
        -------
        dict
            Metadata about decoder configuration.
        """
        return {
            "decoder_type": "external" if self.use_external else "vae",
            "external_decoder_enabled": self.use_external,
        }
