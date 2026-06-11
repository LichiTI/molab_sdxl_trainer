"""Step Expert LoRA adapter integration for training.

This module provides a wrapper around LoRAInjector that replaces standard
LoRA layers with StepExpertLoRALinear layers for timestep-conditional routing.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .step_expert_routing import StepExpertConfig, StepExpertLoRALinear

try:
    from .lora_injector import LoRAInjector
except ImportError:
    # Fallback for test environment
    from .model_family import get_model_family

    class LoRAInjector:
        """Minimal LoRAInjector stub for Step Expert."""
        def __init__(self, rank=4, alpha=1.0, target_modules=None, model_arch=None):
            self.rank = rank
            self.alpha = alpha
            self.target_modules = target_modules
            self._family = get_model_family(model_arch)

logger = logging.getLogger(__name__)


class StepExpertInjector:
    """Wraps LoRAInjector to inject Step Expert adapters instead of standard LoRA.

    Step Expert adapters route to different LoRA experts based on the diffusion
    timestep, enabling step-specific feature learning (e.g., Turbo-style models).

    Example
    -------
    >>> config = StepExpertConfig(num_experts=4, rank=16, alpha=16.0)
    >>> injector = StepExpertInjector(config, target_modules=["to_q", "to_k", "to_v"])
    >>> injector.inject_unet(unet)
    >>> # During training:
    >>> injector.set_timestep(current_step, total_steps)
    >>> output = unet(x, t, text_emb)
    """

    def __init__(
        self,
        config: StepExpertConfig,
        target_modules: Optional[List[str]] = None,
        model_arch: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        config : StepExpertConfig
            Step Expert configuration (num_experts, rank, alpha, boundaries).
        target_modules : list of str, optional
            Target module names to inject. If None, uses model family defaults.
        model_arch : str, optional
            Model architecture ("anima", "sdxl", etc.). If None, infers from context.
        """
        self.config = config
        self.config.validate()

        # Create a base LoRAInjector for module discovery
        # We'll replace its layers with Step Expert layers
        self._base_injector = LoRAInjector(
            rank=config.rank,
            alpha=config.alpha,
            target_modules=target_modules,
            model_arch=model_arch,
        )

        self.injected_layers: Dict[str, StepExpertLoRALinear] = {}
        self._current_timestep: Optional[int] = None
        self._total_steps: Optional[int] = None

    def inject_unet(self, unet: nn.Module) -> Dict[str, StepExpertLoRALinear]:
        """Inject Step Expert adapters into UNet.

        Parameters
        ----------
        unet : nn.Module
            The UNet or DiT model to inject.

        Returns
        -------
        dict
            Mapping of module names to injected Step Expert layers.
        """
        targets = self._base_injector.target_modules or self._base_injector._family.unet_target_modules
        return self._inject_model(unet, targets, prefix="unet")

    def inject_text_encoder(self, text_encoder: nn.Module, prefix: str = "text_encoder") -> Dict[str, StepExpertLoRALinear]:
        """Inject Step Expert adapters into text encoder.

        Parameters
        ----------
        text_encoder : nn.Module
            The text encoder model.
        prefix : str
            Prefix for layer naming.

        Returns
        -------
        dict
            Mapping of module names to injected Step Expert layers.
        """
        targets = self._base_injector.target_modules or self._base_injector._family.text_encoder_target_modules
        return self._inject_model(text_encoder, targets, prefix=prefix)

    def _inject_model(
        self,
        model: nn.Module,
        targets: List[str],
        prefix: str = "model",
    ) -> Dict[str, StepExpertLoRALinear]:
        """Internal injection logic."""
        injected = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Check if module name matches any target
            module_name_parts = name.split(".")
            last_part = module_name_parts[-1] if module_name_parts else ""

            if last_part not in targets:
                continue

            # Replace with Step Expert layer
            step_expert_layer = StepExpertLoRALinear(
                original=module,
                config=self.config,
            )

            # Replace in parent
            parent_name = ".".join(module_name_parts[:-1])
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            setattr(parent, last_part, step_expert_layer)

            full_name = f"{prefix}.{name}"
            injected[full_name] = step_expert_layer
            self.injected_layers[full_name] = step_expert_layer

            logger.info(f"Injected Step Expert: {full_name} (experts={self.config.num_experts}, rank={self.config.rank})")

        logger.info(f"Step Expert injection complete: {len(injected)} layers injected")
        return injected

    def set_timestep(self, timestep: int, total_steps: int):
        """Set current diffusion timestep for all Step Expert layers.

        This must be called before each forward pass during training.

        Parameters
        ----------
        timestep : int
            Current diffusion timestep.
        total_steps : int
            Total number of diffusion steps in the schedule.
        """
        self._current_timestep = timestep
        self._total_steps = total_steps

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters from Step Expert layers."""
        params = []
        for layer in self.injected_layers.values():
            params.extend(layer.get_trainable_params())
        return params

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict of all Step Expert parameters."""
        state = {}
        for name, layer in self.injected_layers.items():
            state[f"{name}.lora_down"] = layer.lora_down
            state[f"{name}.lora_up"] = layer.lora_up
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load Step Expert parameters from state dict."""
        for name, layer in self.injected_layers.items():
            down_key = f"{name}.lora_down"
            up_key = f"{name}.lora_up"

            if down_key in state_dict:
                layer.lora_down.data.copy_(state_dict[down_key])
            if up_key in state_dict:
                layer.lora_up.data.copy_(state_dict[up_key])

    def metadata(self) -> Dict[str, str]:
        """Get metadata for Step Expert adapter export."""
        if self.injected_layers:
            # All layers share the same config, so use first layer's metadata
            first_layer = next(iter(self.injected_layers.values()))
            return first_layer.metadata()
        return {}

    def forward_hook(self, module: nn.Module, args, kwargs):
        """Hook to inject timestep into Step Expert layers.

        This is automatically called if registered via register_forward_pre_hook.
        """
        if self._current_timestep is not None:
            kwargs["timestep"] = self._current_timestep
            kwargs["total_steps"] = self._total_steps
        return args, kwargs
