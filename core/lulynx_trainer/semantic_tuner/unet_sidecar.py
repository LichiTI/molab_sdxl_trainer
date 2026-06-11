"""
Semantic Base-Tuner Sidecar (V3.1)
Implements the "Sidecar Adaptation" logic ("Dual-Stream Cross-Attention").

Architecture:
- Ghost Lane (Channel A): Uses Frozen Original Weights (W_orig) + CLIP Embeddings.
- Neuro Lane (Channel B): Uses Trainable New Weights (W_new) + LLM Embeddings.
- Signal Fusion: Output = Legacy_Out + alpha * Neuro_Out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class NeuroSidecarLinear(nn.Module):
    """
    Lightweight projection layer for the Neuro Lane.
    Mappings: LLM_Dim -> Head_Dim * Heads
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Initialize close to zero to start with "Legacy Dominance"
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

class DualStreamProcessor:
    """
    The "Brain Surgeon" Processor that handles two streams of thought.
    V3.4 Update: Implements K/V LaneNorm Fusion & Scheduled Dropout.
    """
    def __init__(self, original_processor, neuro_k_proj, neuro_v_proj, legacy_norm, neuro_norm, gate, hidden_size, cross_attention_dim):
        self.original_processor = original_processor # The original (frozen) processor logic
        self.neuro_k = neuro_k_proj
        self.neuro_v = neuro_v_proj
        self.ln_legacy = legacy_norm
        self.ln_neuro = neuro_norm
        self.gate = gate
        
        # Dimensions
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        Custom Forward Pass with K/V Fusion.
        """
        # 1. Unpack Inputs & Scheduled Dropout Logic
        # The training loop is responsible for setting ghost_embeds to None based on schedule.
        if isinstance(encoder_hidden_states, dict):
            neuro_embeds = encoder_hidden_states.get("main")
            ghost_embeds = encoder_hidden_states.get("ghost")
        else:
            # Pure Mode or Inference
            neuro_embeds = encoder_hidden_states
            ghost_embeds = None

        batch_size, sequence_length, _ = hidden_states.shape
        
        # 2. Shared Query Projection (W_q is frozen/shared)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        
        # 3. Legacy Lane (CLIP) - K/V Projection
        if ghost_embeds is not None:
             k_legacy = attn.to_k(ghost_embeds)
             v_legacy = attn.to_v(ghost_embeds)
             # P0-05: LaneNorm (Legacy)
             k_legacy = self.ln_legacy(k_legacy)
             v_legacy = self.ln_legacy(v_legacy)
        else:
             k_legacy = None
             v_legacy = None

        # 4. Neuro Lane (LLM) - K/V Projection
        if neuro_embeds is not None:
            # Manually project using W_new
            k_neuro = self.neuro_k(neuro_embeds)
            v_neuro = self.neuro_v(neuro_embeds)
            # P0-05: LaneNorm (Neuro)
            k_neuro = self.ln_neuro(k_neuro)
            v_neuro = self.ln_neuro(v_neuro)
        else:
            # Fallback (Should typically not happen in our architecture)
            k_neuro = None
            v_neuro = None

        # 5. Gated Fusion (P0-05)
        # K_fused = Norm(K_leg) + Gate * Norm(K_neu)
        if k_legacy is not None and k_neuro is not None:
            key = k_legacy + (self.gate * k_neuro)
            value = v_legacy + (self.gate * v_neuro)
        elif k_legacy is not None:
            # Pure CLIP (e.g. initial state or ablation)
            key = k_legacy
            value = v_legacy
        elif k_neuro is not None:
            # Pure mode / ghost-dropout fallback: the neuro lane is the only
            # remaining signal, so do not suppress it behind a zero-init gate.
            key = k_neuro
            value = v_neuro
        else:
             # Just in case
             return torch.zeros_like(hidden_states)

        # 6. Attention & Output
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Linear proj (Shared W_out)
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class NeuroSidecarNetwork(nn.Module):
    """
    The container module that holds all the trainable 'W_new' parameters.
    This is what we save as the 'Semantic Adapter' checkpoint.
    """
    def __init__(self):
        super().__init__()
        self.neuro_modules = nn.ModuleDict() 
        
    def add_layer(self, name, k_proj, v_proj, legacy_norm, neuro_norm, gate):
        # ModuleDict doesn't allow dots in keys — sanitize
        safe_name = name.replace(".", "_")
        self.neuro_modules[safe_name] = nn.ModuleDict({
            "to_k": k_proj,
            "to_v": v_proj,
            "ln_legacy": legacy_norm,
            "ln_neuro": neuro_norm,
        })
        self.neuro_modules[safe_name].add_module("gate", GateWrapper(gate))


class GateWrapper(nn.Module):
    def __init__(self, gate_param):
        super().__init__()
        self.gate = gate_param
    def forward(self):
        return self.gate


def inject_neural_sidecar(unet, llm_dim=2048):
    """
    Injects the Sidecar Processor into the U-Net.
    V3.4: Injects LaneNorm and Learnable Gates.
    """
    sidecar_net = NeuroSidecarNetwork()
    
    processors_to_swap = {}
    
    for name, processor in unet.attn_processors.items():
        module_name = name.replace(".processor", "")
        attn_module = unet.get_submodule(module_name)

        is_cross_attn = "attn2" in name and hasattr(attn_module, "to_k") and attn_module.to_k is not None

        if not is_cross_attn:
            # Self-attention: pass through unchanged (set_attn_processor needs all layers)
            processors_to_swap[name] = processor
            continue

        if is_cross_attn:
            # head_dim may not exist in all diffusers versions; fall back to inner_dim or to_out
            if hasattr(attn_module, 'inner_dim'):
                inner_dim = attn_module.inner_dim
            elif hasattr(attn_module, 'head_dim'):
                inner_dim = attn_module.heads * attn_module.head_dim
            else:
                inner_dim = attn_module.to_out[0].out_features
            param_ref = attn_module.to_k.weight
            module_device = param_ref.device
            module_dtype = param_ref.dtype
            
            # W_new maps from LLM (llm_dim) -> Inner Dim
            k_proj = NeuroSidecarLinear(llm_dim, inner_dim, bias=False).to(device=module_device, dtype=module_dtype)
            v_proj = NeuroSidecarLinear(llm_dim, inner_dim, bias=False).to(device=module_device, dtype=module_dtype)
            
            # P0-05: LaneNorms (Pre-Fusion)
            legacy_norm = nn.LayerNorm(inner_dim, elementwise_affine=True).to(device=module_device, dtype=module_dtype)
            neuro_norm = nn.LayerNorm(inner_dim, elementwise_affine=True).to(device=module_device, dtype=module_dtype)
            
            # P0-05: Learnable Gate (Initialize to 0 for safe fallback)
            gate = nn.Parameter(torch.zeros((), device=module_device, dtype=module_dtype))
            
            # Register to Sidecar Net
            sidecar_net.add_layer(name, k_proj, v_proj, legacy_norm, neuro_norm, gate)
            
            # Create Dual Stream Processor
            new_processor = DualStreamProcessor(
                original_processor=processor,
                neuro_k_proj=k_proj,
                neuro_v_proj=v_proj,
                legacy_norm=legacy_norm,
                neuro_norm=neuro_norm,
                gate=gate,
                hidden_size=inner_dim,
                cross_attention_dim=llm_dim
            )
            
            processors_to_swap[name] = new_processor
            
    # Swap
    unet.set_attn_processor(processors_to_swap)
    
    return unet, sidecar_net
