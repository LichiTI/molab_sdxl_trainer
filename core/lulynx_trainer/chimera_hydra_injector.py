"""ChimeraHydra dual-pool MoE LoRA adapter integration for training.

ChimeraHydra maintains separate content and frequency expert pools, routing
input features through both pools and combining their outputs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .chimera_hydra import ChimeraHydraConfig, ChimeraHydraLinear

logger = logging.getLogger(__name__)


class ChimeraHydraInjector:
    """Wraps LoRAInjector to inject ChimeraHydra adapters.

    ChimeraHydra uses dual pools: content experts and frequency experts.
    Frequency features can be extracted via FFT or provided externally.

    Example
    -------
    >>> config = ChimeraHydraConfig(content_experts=4, frequency_experts=2, rank=16)
    >>> injector = ChimeraHydraInjector(config, target_modules=["to_q", "to_k"])
    >>> injector.inject_unet(unet)
    >>> # During forward:
    >>> output = injector.forward_with_frequency(module, x, freq_features)
    """

    def __init__(
        self,
        config: ChimeraHydraConfig,
        target_modules: Optional[List[str]] = None,
        model_arch: Optional[str] = None,
        use_fft_features: bool = True,
    ):
        """
        Parameters
        ----------
        config : ChimeraHydraConfig
            ChimeraHydra configuration (content/frequency experts, rank, routing).
        target_modules : list of str, optional
            Target module names to inject.
        model_arch : str, optional
            Model architecture ("anima", "sdxl", etc.).
        use_fft_features : bool, default True
            If True, automatically extract frequency features via FFT.
        """
        self.config = config
        self.config.validate()
        self.use_fft_features = use_fft_features

        # Import model family for defaults
        try:
            from .model_family import get_model_family
            self._family = get_model_family(model_arch)
        except ImportError:
            self._family = None

        self.target_modules = target_modules
        self.injected_layers: Dict[str, ChimeraHydraLinear] = {}

    def inject_unet(self, unet: nn.Module) -> Dict[str, ChimeraHydraLinear]:
        """Inject ChimeraHydra adapters into UNet."""
        if self._family:
            targets = self.target_modules or self._family.unet_target_modules
        else:
            targets = self.target_modules or []
        return self._inject_model(unet, targets, prefix="unet")

    def inject_text_encoder(self, text_encoder: nn.Module, prefix: str = "text_encoder") -> Dict[str, ChimeraHydraLinear]:
        """Inject ChimeraHydra adapters into text encoder."""
        if self._family:
            targets = self.target_modules or self._family.text_encoder_target_modules
        else:
            targets = self.target_modules or []
        return self._inject_model(text_encoder, targets, prefix=prefix)

    def _inject_model(
        self,
        model: nn.Module,
        targets: List[str],
        prefix: str = "model",
    ) -> Dict[str, ChimeraHydraLinear]:
        """Internal injection logic."""
        injected = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            module_name_parts = name.split(".")
            last_part = module_name_parts[-1] if module_name_parts else ""

            if last_part not in targets:
                continue

            # Replace with ChimeraHydra layer
            chimera_layer = ChimeraHydraLinear(
                original=module,
                config=self.config,
            )

            # Replace in parent
            parent_name = ".".join(module_name_parts[:-1])
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            setattr(parent, last_part, chimera_layer)

            full_name = f"{prefix}.{name}"
            injected[full_name] = chimera_layer
            self.injected_layers[full_name] = chimera_layer

            logger.info(
                f"Injected ChimeraHydra: {full_name} "
                f"(content={self.config.content_experts}, freq={self.config.frequency_experts}, rank={self.config.rank})"
            )

        logger.info(f"ChimeraHydra injection complete: {len(injected)} layers injected")
        return injected

    def extract_frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frequency features via FFT.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., features].

        Returns
        -------
        torch.Tensor
            Frequency features of same shape as input.
        """
        # Apply FFT along last dimension
        fft = torch.fft.rfft(x, dim=-1)

        # Take magnitude (absolute value)
        magnitude = torch.abs(fft)

        # Interpolate back to original feature dimension if needed
        if magnitude.shape[-1] != x.shape[-1]:
            magnitude = torch.nn.functional.interpolate(
                magnitude.unsqueeze(1),  # Add channel dim
                size=x.shape[-1],
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        return magnitude

    def forward_with_frequency(
        self,
        layer: ChimeraHydraLinear,
        x: torch.Tensor,
        frequency_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with frequency feature handling.

        Parameters
        ----------
        layer : ChimeraHydraLinear
            The ChimeraHydra layer to call.
        x : torch.Tensor
            Input tensor.
        frequency_features : torch.Tensor, optional
            Pre-computed frequency features. If None and use_fft_features=True,
            will extract via FFT.

        Returns
        -------
        torch.Tensor
            Layer output.
        """
        if frequency_features is None and self.use_fft_features:
            frequency_features = self.extract_frequency_features(x)

        return layer(x, frequency_features=frequency_features)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters from ChimeraHydra layers."""
        params = []
        for layer in self.injected_layers.values():
            params.extend(layer.get_trainable_params())
        return params

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict of all ChimeraHydra parameters."""
        state = {}
        for name, layer in self.injected_layers.items():
            state[f"{name}.content_down"] = layer.content_down
            state[f"{name}.content_up"] = layer.content_up
            state[f"{name}.frequency_down"] = layer.frequency_down
            state[f"{name}.frequency_up"] = layer.frequency_up
            state[f"{name}.content_gate.weight"] = layer.content_gate.weight
            state[f"{name}.frequency_gate.weight"] = layer.frequency_gate.weight
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load ChimeraHydra parameters from state dict."""
        for name, layer in self.injected_layers.items():
            keys = {
                "content_down": f"{name}.content_down",
                "content_up": f"{name}.content_up",
                "frequency_down": f"{name}.frequency_down",
                "frequency_up": f"{name}.frequency_up",
                "content_gate": f"{name}.content_gate.weight",
                "frequency_gate": f"{name}.frequency_gate.weight",
            }

            for attr, key in keys.items():
                if key in state_dict:
                    if attr.endswith("_gate"):
                        getattr(layer, attr).weight.data.copy_(state_dict[key])
                    else:
                        getattr(layer, attr).data.copy_(state_dict[key])

    def metadata(self) -> Dict[str, str]:
        """Get metadata for ChimeraHydra adapter export."""
        if self.injected_layers:
            first_layer = next(iter(self.injected_layers.values()))
            return first_layer.metadata()
        return {}
