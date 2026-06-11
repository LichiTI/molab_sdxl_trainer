"""Decoder router configuration helper.

LICENSE: PolyForm Noncommercial 1.0.0
Cleanroom implementation - no AGPL code referenced.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn

from .decoder_router import DecoderRouter

logger = logging.getLogger(__name__)


def create_decoder_router_from_config(
    vae: nn.Module,
    config,
) -> DecoderRouter:
    """Create decoder router from training config.

    Parameters
    ----------
    vae : nn.Module
        Standard VAE decoder.
    config : object
        Training configuration object.

    Returns
    -------
    DecoderRouter
        Configured decoder router.

    Examples
    --------
    >>> router = create_decoder_router_from_config(vae, config)
    >>> images = router.decode(latents)
    """
    use_external = getattr(config, "use_external_decoder", False)
    external_path = getattr(config, "external_decoder_path", None)
    external_type = getattr(config, "external_decoder_type", "auto")

    if use_external:
        if not external_path:
            logger.warning(
                "use_external_decoder=True but no external_decoder_path provided. "
                "Falling back to VAE decoder."
            )
            use_external = False

    router = DecoderRouter(
        vae=vae,
        external_decoder_path=external_path,
        external_decoder_type=external_type,
        use_external_decoder=use_external,
    )

    if use_external:
        logger.info(
            f"Created decoder router with external decoder: {external_path}. "
            f"User is responsible for license compliance."
        )
    else:
        logger.info("Created decoder router with standard VAE decoder")

    return router


def get_decoder_config_help() -> str:
    """Get help text for decoder configuration.

    Returns
    -------
    str
        Configuration help text.
    """
    return """
Decoder Configuration
====================

Standard VAE Decoder (default):
    use_external_decoder = false

External Decoder (e.g., PiD):
    use_external_decoder = true
    external_decoder_path = "/path/to/decoder.pth"
    external_decoder_type = "auto"  # or "pid", "custom"

IMPORTANT: When using external decoders, you are responsible for:
1. Providing a compatible decoder checkpoint
2. Ensuring the decoder complies with applicable licenses
3. Verifying the decoder produces correct outputs

The external decoder must implement a `decode(latents)` method
or be callable with latents as input.

Example config.toml:
    [decoder]
    use_external_decoder = true
    external_decoder_path = "./decoders/my_decoder.pth"
    external_decoder_type = "auto"
"""
