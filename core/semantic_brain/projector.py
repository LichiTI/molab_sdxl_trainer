
"""
Lulynx Universal Projector (LUP)
Part of Lulynx Neuro-Link Architecture

This module defines the projection layer that maps high-dimensional LLM embeddings
to the target dimension expected by the U-Net (e.g., SDXL TE output dimension).
"""

import torch
import torch.nn as nn
from typing import Optional

class LulynxUniversalProjector(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_mult: int = 4,
        bake_in_norm: bool = True
    ):
        """
        Initialize the Universal Projector with bake-in normalization.
        
        Args:
            in_dim: Input dimension from LLM (e.g., 1024, 2048, 4096)
            out_dim: Target dimension for U-Net (e.g., 768 for SDXL TE1, 1280 for TE2, or 2048 for Neuro-Link Mode A)
            hidden_mult: Multiplier for the hidden layer size
            bake_in_norm: Whether to include a final LayerNorm (Recommended: True)
        """
        super().__init__()
        
        # Determine hidden dimension
        hidden_dim = out_dim * hidden_mult
        
        # Architecture: Linear -> GELU -> Linear -> [LayerNorm]
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        ]
        
        if bake_in_norm:
            # The "Bake-in Norm" as per RFC 3.1
            # Ensures output distribution is stable for U-Net
            layers.append(nn.LayerNorm(out_dim, eps=1e-6))
            
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize with small random weights to start close to zero/identity behavior"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor from LLM [Batch, Seq, In_Dim]
            
        Returns:
            Projected tensor [Batch, Seq, Out_Dim]
        """
        return self.net(x)

    def get_config(self):
        """Return configuration for saving/loading"""
        return {
            "in_dim": self.net[0].in_features,
            "out_dim": self.net[2].out_features if len(self.net) > 2 else self.net[0].out_features, # Robust check
            "structure": "linear-gelu-linear-norm"
        }
