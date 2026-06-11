
"""
DoRA (Weight-Decomposed Low-Rank Adaptation) Layer
==================================================
Native implementation of DoRA for Lulynx Trainer.

This module provides a DoRALinear layer that explicitly models:
W = m * (W_0 + BA) / ||W_0 + BA||

Where:
- m: Learnable magnitude vector
- W_0: Frozen base weight
- B, A: LoRA weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("DoRA")

class DoRALinear(nn.Module):
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int = 4, 
        alpha: float = 1.0, 
        dropout: float = 0.0,
        mode: str = "full",
        rs_lora_enabled: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.rs_lora_enabled = bool(rs_lora_enabled)
        self.scaling_strategy = "alpha_over_sqrt_rank" if self.rs_lora_enabled else "alpha_over_rank"
        self.scaling = alpha / (rank ** 0.5) if self.rs_lora_enabled else alpha / rank
        self.mode = self._normalize_mode(mode)
        
        # 1. Base Weight (Frozen)
        # We assume base_layer.weight is already correct.
        # We share the weight to avoid duplication if possible, 
        # or clone/detach if we want strict isolation.
        # Lulynx Philosophy: Share memory where possible.
        self.base_weight = base_layer.weight
        self.base_bias = base_layer.bias
        
        # Freeze base
        self.base_weight.requires_grad = False
        if self.base_bias is not None:
            self.base_bias.requires_grad = False
            
        # 2. LoRA Weights (Trainable A & B)
        # B: [out, rank] (0 init), A: [rank, in] (Gaussian)
        # Note: In our notation, deltaW = B @ A * scaling
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Init
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        # 3. Magnitude Vector (Trainable m)
        # Initialized to the norm of W_0
        with torch.no_grad():
            # dim=1 is the input dimension sum for Linear(out, in)
            w0_norm = torch.linalg.norm(self.base_weight, dim=1)
            self.m = nn.Parameter(w0_norm)
        # Tag magnitude param so optimizer can apply weight_decay=0
        self.m._dora_magnitude = True
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._apply_mode()

        # QDoRA Support: Check if base layer is quantized
        self.is_quantized = False
        try:
            import bitsandbytes as bnb
            if isinstance(base_layer, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                self.is_quantized = True
        except ImportError:
            pass


    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode or "full").strip().lower().replace("-", "_")
        aliases = {
            "wd": "full",
            "weight_decomposed": "full",
            "style_lock": "style",
            "magnitude": "style",
            "magnitude_only": "style",
            "structure_lock": "structure",
            "direction": "structure",
            "direction_only": "structure",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"full", "style", "structure"}:
            logger.warning("Unknown DoRA mode %r; falling back to full", mode)
            return "full"
        return normalized

    def _apply_mode(self) -> None:
        train_direction = self.mode in {"full", "structure"}
        train_magnitude = self.mode in {"full", "style"}
        self.lora_A.requires_grad_(train_direction)
        self.lora_B.requires_grad_(train_direction)
        self.m.requires_grad_(train_magnitude)
    def _compute_dora_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        base = base_weight.to(dtype=self.lora_A.dtype)
        weight_eff = torch.addmm(base, self.lora_B, self.lora_A, beta=1.0, alpha=self.scaling)
        norm = torch.linalg.vector_norm(weight_eff, dim=1, keepdim=True)
        row_scale = self.m.to(dtype=weight_eff.dtype).unsqueeze(1) / (norm + 1e-6)
        return weight_eff * row_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle Quantized Weights (QDoRA)
        if self.is_quantized:
            import bitsandbytes as bnb
            # Dequantize base weight
            if hasattr(self.base_layer, "weight") and hasattr(self.base_layer.weight, "quant_state"):
                w0 = bnb.functional.dequantize_4bit(self.base_layer.weight.data, self.base_layer.weight.quant_state)
                weight_dora = self._compute_dora_weight(w0)
                return F.linear(x, weight_dora, self.base_bias)
            else:
                # Fallback if structure unknown
                pass

        # Standard Path (FP16/BF16)
        weight_dora = self._compute_dora_weight(self.base_weight)
        return F.linear(x, weight_dora, self.base_bias)

    def get_weight_matrix(self) -> torch.Tensor:
        """Return the effective DoRA weight delta for analysis/checkpoint stats."""
        return (self.lora_B @ self.lora_A) * self.scaling



class DoRAInjector:
    def __init__(self, rank: int, alpha: float, device: str = "cuda", mode: str = "full"):
        self.rank = rank
        self.alpha = alpha
        self.device = device
        self.mode = mode

    def inject(self, model: nn.Module, target_layers: list = None):
        """
        Replaces nn.Linear layers in the model with DoRALinear layers.
        
        Args:
            model: The base model (e.g. UNet)
            target_layers: List of layer names to replace (if None, heuristics used)
        """
        logger.info(f"[DoRA] 🚀 Injecting DoRA layers (Rank={self.rank})...")
        
        replaced_count = 0
        
        # Implementing recursive replacement logic
        replaced_count = self._recursive_replace(model, "")
        
        logger.info(f"[DoRA] Replaced {replaced_count} layers.")
        return replaced_count

    def _recursive_replace(self, module: nn.Module, prefix: str) -> int:
        count = 0
        for child_name, child in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            
            # Check if this is a target linear layer
            # Heuristics for UNet/Transformers
            is_target = isinstance(child, nn.Linear) and any(k in child_name for k in ["to_q", "to_k", "to_v", "to_out", "ff.net", "proj"])
            
            if is_target:
                # Replace
                dora_layer = DoRALinear(child, rank=self.rank, alpha=self.alpha, mode=self.mode)
                dora_layer.to(self.device)
                
                setattr(module, child_name, dora_layer)
                logger.debug(f"[DoRA] Injected: {full_name}")
                count += 1
            else:
                count += self._recursive_replace(child, full_name)
        return count
