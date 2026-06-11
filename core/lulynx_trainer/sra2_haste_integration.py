"""SRA2/HASTE alignment loss integration for training.

This module provides the infrastructure to capture hidden states and apply
alignment loss during training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .sra2_haste_alignment_facade import SRA2HasteAlignmentPolicy, sra2_haste_alignment_loss

logger = logging.getLogger(__name__)


class SRA2HasteAlignmentIntegration:
    """Integrates SRA2/HASTE alignment loss into training loop.

    This integration:
    1. Captures hidden states from specified DiT layers
    2. Manages VAE feature caching
    3. Applies alignment loss with HASTE schedule

    Example
    -------
    >>> policy = SRA2HasteAlignmentPolicy(enabled=True, base_weight=0.1)
    >>> integration = SRA2HasteAlignmentIntegration(
    ...     policy=policy,
    ...     capture_layers=["blocks.10", "blocks.20"]
    ... )
    >>> integration.attach_hooks(model)
    >>> # In training loop:
    >>> loss = integration.compute_alignment_loss(vae_features, step=100)
    """

    def __init__(
        self,
        policy: SRA2HasteAlignmentPolicy,
        capture_layers: List[str],
    ):
        """
        Parameters
        ----------
        policy : SRA2HasteAlignmentPolicy
            Alignment loss policy configuration.
        capture_layers : list of str
            Layer names to capture hidden states from.
        """
        self.policy = policy.normalized()
        self.capture_layers = capture_layers

        # Hidden state storage
        self._hidden_states: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

        # Loss history for HASTE schedule
        self._loss_history: List[float] = []

    def attach_hooks(self, model: nn.Module):
        """Attach forward hooks to capture hidden states.

        Parameters
        ----------
        model : nn.Module
            Model to attach hooks to.
        """
        if not self.policy.enabled:
            return

        for name, module in model.named_modules():
            if name in self.capture_layers:
                hook = module.register_forward_hook(
                    self._make_capture_hook(name)
                )
                self._hooks.append(hook)
                logger.info(f"Attached SRA2 capture hook to {name}")

    def _make_capture_hook(self, layer_name: str):
        """Create a capture hook for a specific layer."""
        def hook(module, input, output):
            # Store output (hidden states)
            if isinstance(output, torch.Tensor):
                self._hidden_states[layer_name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                self._hidden_states[layer_name] = output[0].detach()
        return hook

    def clear_captured_states(self):
        """Clear captured hidden states."""
        self._hidden_states.clear()

    def compute_alignment_loss(
        self,
        vae_features: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute alignment loss using captured hidden states.

        Parameters
        ----------
        vae_features : torch.Tensor
            VAE intermediate features.
        step : int
            Current training step.
        total_steps : int
            Total training steps.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            Alignment loss and profile dict.
        """
        if not self.policy.enabled:
            return torch.tensor(0.0), {"active": False}

        if not self._hidden_states:
            logger.warning("No hidden states captured for SRA2 alignment loss")
            return torch.tensor(0.0), {"active": False, "reason": "no_hidden_states"}

        # Average hidden states from all captured layers
        hidden_list = list(self._hidden_states.values())
        if len(hidden_list) == 1:
            hidden_states = hidden_list[0]
        else:
            # Average across layers
            hidden_states = torch.stack(hidden_list).mean(dim=0)

        # Ensure 3D shape [batch, tokens, hidden]
        if hidden_states.dim() == 2:
            # Add token dimension: [batch, hidden] -> [batch, 1, hidden]
            hidden_states = hidden_states.unsqueeze(1)
        elif hidden_states.dim() != 3:
            logger.warning(f"Unexpected hidden_states shape: {hidden_states.shape}")
            return torch.tensor(0.0), {"active": False, "reason": "invalid_shape"}

        # Ensure vae_features has compatible shape
        if vae_features.dim() == 2:
            vae_features = vae_features.unsqueeze(1)
        elif vae_features.dim() != 3 and vae_features.dim() != 4:
            logger.warning(f"Unexpected vae_features shape: {vae_features.shape}")
            return torch.tensor(0.0), {"active": False, "reason": "invalid_vae_shape"}

        # Compute alignment loss
        loss, profile = sra2_haste_alignment_loss(
            hidden_states=hidden_states,
            vae_features=vae_features,
            policy=self.policy,
            step=step,
            total_steps=total_steps,
            loss_history=self._loss_history,
        )

        # Record loss for HASTE schedule
        if profile.get("active", False):
            self._loss_history.append(profile.get("loss_value", 0.0))

        return loss, profile

    def remove_hooks(self):
        """Remove all attached hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("Removed all SRA2 capture hooks")

    def get_loss_history(self) -> List[float]:
        """Get recorded loss history."""
        return self._loss_history.copy()

    def reset_loss_history(self):
        """Reset loss history."""
        self._loss_history.clear()


class VAEFeatureCache:
    """Cache for VAE intermediate features.

    This cache pre-computes VAE features to avoid recomputing them
    during training.
    """

    def __init__(self):
        self._cache: Dict[str, torch.Tensor] = {}

    def precompute(
        self,
        vae: nn.Module,
        images: torch.Tensor,
        keys: List[str],
    ):
        """Precompute VAE features for a batch.

        Parameters
        ----------
        vae : nn.Module
            VAE model.
        images : torch.Tensor
            Input images.
        keys : list of str
            Keys to identify images (e.g., filenames).
        """
        with torch.no_grad():
            # Encode images
            # Assumes VAE has encode method that returns intermediate features
            if hasattr(vae, "encode"):
                features = vae.encode(images, return_dict=False)[0]
            else:
                # Fallback: just get latents
                features = vae(images)

            # Store features
            for i, key in enumerate(keys):
                self._cache[key] = features[i].detach()

    def get(self, keys: List[str]) -> Optional[torch.Tensor]:
        """Get cached features for keys.

        Parameters
        ----------
        keys : list of str
            Keys to retrieve.

        Returns
        -------
        torch.Tensor or None
            Stacked features, or None if not all keys are cached.
        """
        features = []
        for key in keys:
            if key not in self._cache:
                return None
            features.append(self._cache[key])

        return torch.stack(features)

    def clear(self):
        """Clear all cached features."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


def create_sra2_haste_integration_from_config(
    config,
) -> Optional[SRA2HasteAlignmentIntegration]:
    """Create SRA2/HASTE integration from training config.

    Parameters
    ----------
    config : object
        Training configuration object.

    Returns
    -------
    SRA2HasteAlignmentIntegration or None
        Integration if enabled, None otherwise.
    """
    if not getattr(config, "sra2_alignment_enabled", False):
        return None

    policy = SRA2HasteAlignmentPolicy(
        enabled=True,
        loss_type=getattr(config, "sra2_loss_type", "cosine"),
        normalize_targets=getattr(config, "sra2_normalize_targets", True),
        stop_grad_target=getattr(config, "sra2_stop_grad_target", True),
        base_weight=getattr(config, "sra2_base_weight", 1.0),
        start_step=getattr(config, "sra2_start_step", 0),
        stop_step=getattr(config, "sra2_stop_step", -1),
        decay_start_step=getattr(config, "sra2_decay_start_step", -1),
        decay_end_step=getattr(config, "sra2_decay_end_step", -1),
        min_weight=getattr(config, "sra2_min_weight", 0.0),
        plateau_patience=getattr(config, "sra2_plateau_patience", 0),
        min_relative_improvement=getattr(config, "sra2_min_relative_improvement", 0.0),
    )

    capture_layers = getattr(config, "sra2_capture_layers", [])
    if not capture_layers:
        logger.warning("SRA2 enabled but no capture_layers specified")
        return None

    integration = SRA2HasteAlignmentIntegration(
        policy=policy,
        capture_layers=capture_layers,
    )

    logger.info(
        f"SRA2/HASTE integration initialized: "
        f"base_weight={policy.base_weight}, "
        f"capture_layers={len(capture_layers)}"
    )

    return integration
