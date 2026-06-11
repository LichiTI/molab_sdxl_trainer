"""
IP-Adapter Attention Processor & Injector

This module handles the injection of custom attention processors into the UNet 
to support image-based conditioning for IP-Adapter training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any

class IPAdapterAttnProcessor(nn.Module):
    """
    Attention processor for IP-Adapter.
    It adds a separate cross-attention path for image features.
    """
    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        # New projection layers for image features (to be trained)
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        ip_adapter_image_embeds=None,  # This is where image features come in
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # Standard Text Cross-Attention
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # IP-Adapter Image Cross-Attention
        if ip_adapter_image_embeds is not None:
            # ip_adapter_image_embeds shape: [B, num_tokens, cross_attention_dim]
            ip_key = self.to_k_ip(ip_adapter_image_embeds)
            ip_value = self.to_v_ip(ip_adapter_image_embeds)

            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

            # Combine text and image features
            hidden_states = hidden_states + ip_hidden_states

        # Linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channel, height, width)

        hidden_states = hidden_states + residual

        return hidden_states


class IPAdapterInjector:
    """
    Handles injection of IPAdapterAttnProcessor into UNet.
    """
    def __init__(self, unet, num_tokens=4):
        self.unet = unet
        self.num_tokens = num_tokens
        self.processors = {}

    def inject(self):
        """Inject processors into all cross-attention layers."""
        for name, module in self.unet.named_modules():
            if name.endswith("attn2"): # attn2 is cross-attention
                cross_attention_dim = module.to_k.in_features
                hidden_size = module.to_q.in_features

                processor = IPAdapterAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=self.num_tokens
                )
                # Move processor to same device/dtype as the UNet module
                proc_device = next(module.parameters()).device
                proc_dtype = next(module.parameters()).dtype
                processor.to(device=proc_device, dtype=proc_dtype)
                # Set per-module to avoid dict-count mismatch with set_attn_processor
                module.set_processor(processor)
                self.processors[name] = processor

        return self.processors

    def get_trainable_params(self):
        """Returns parameters that should be trained (ip layers)."""
        params = []
        for processor in self.processors.values():
            params.extend(list(processor.parameters()))
        return params

    def save_ip_adapter(self, path: str, image_proj_model=None):
        """Save IP-Adapter weights (proj model + attn layers)."""
        state_dict = {}
        
        # Save attention layers
        for name, processor in self.processors.items():
            for k, v in processor.state_dict().items():
                state_dict[f"{name}.{k}"] = v
        
        # Save image projection model if provided
        if image_proj_model is not None:
            for k, v in image_proj_model.state_dict().items():
                state_dict[f"image_proj.{k}"] = v
        
        from safetensors.torch import save_file
        save_file(state_dict, path)

    def load_ip_adapter(self, path: str, image_proj_model=None):
        """Load IP-Adapter weights saved by save_ip_adapter()."""
        path_lower = path.lower()
        if path_lower.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(path, device="cpu")
        else:
            from core.safe_pickle import safe_torch_load

            state_dict = safe_torch_load(path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        if not isinstance(state_dict, dict):
            raise ValueError(f"IP-Adapter weights did not contain a state dict: {path}")

        missing = []
        unexpected = []
        for name, processor in self.processors.items():
            prefix = f"{name}."
            proc_state = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            proc_missing, proc_unexpected = processor.load_state_dict(proc_state, strict=False)
            missing.extend(f"{name}.{key}" for key in proc_missing)
            unexpected.extend(f"{name}.{key}" for key in proc_unexpected)

        if image_proj_model is not None:
            image_proj_state = {
                key[len("image_proj."):]: value
                for key, value in state_dict.items()
                if key.startswith("image_proj.")
            }
            proj_missing, proj_unexpected = image_proj_model.load_state_dict(image_proj_state, strict=False)
            missing.extend(f"image_proj.{key}" for key in proj_missing)
            unexpected.extend(f"image_proj.{key}" for key in proj_unexpected)

        return missing, unexpected
