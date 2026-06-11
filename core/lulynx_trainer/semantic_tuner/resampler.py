"""
Structured Resampler (Audit P1-11)
Implements a Perceiver-like Resampler with structured query slots 
(Global, Entity, Relation) to preserve semantic hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StructuredResampler(nn.Module):
    """
    Resamples variable-length LLM embeddings [B, N, D_llm] into fixed-length 
    structured CLIP-like embeddings [B, 77, D_clip].
    
    Structure (77 Tokens):
    - 0-7:   Global Context (Scene style, atmosphere) [8 tokens]
    - 8-23:  Entity Slots (Main subjects) [16 tokens]
    - 24-76: Relation/Action Slots (Details & Verbs) [53 tokens]
    """
    def __init__(
        self,
        llm_dim: int = 2048,
        clip_dim: int = 768,
        depth: int = 4,
        num_heads: int = 8
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.clip_dim = clip_dim
        
        # 1. Latent Queries (The "Structure")
        # We define them as learnable parameters
        self.num_global = 8
        self.num_entity = 16
        self.num_relation = 53 # 77 - 8 - 16 = 53
        
        self.query_global = nn.Parameter(torch.randn(1, self.num_global, clip_dim))
        self.query_entity = nn.Parameter(torch.randn(1, self.num_entity, clip_dim))
        self.query_relation = nn.Parameter(torch.randn(1, self.num_relation, clip_dim))
        
        # Initialization
        nn.init.normal_(self.query_global, std=0.02)
        nn.init.normal_(self.query_entity, std=0.02)
        nn.init.normal_(self.query_relation, std=0.02)
        
        # 2. Input Projector (LLM -> Inner Dim)
        # Using CLIP dim as inner dim for efficiency
        self.proj_in = nn.Linear(llm_dim, clip_dim)
        self.ln_in = nn.LayerNorm(llm_dim)
        
        # 3. Transformer Block (Perceiver)
        # Standard Cross-Attention Layer where Q=Latent, K/V=LLM
        # We assume standard TransformerDecoderLayer style
        self.blocks = nn.ModuleList([
            ResamplerBlock(clip_dim, num_heads) for _ in range(depth)
        ])
        
        self.ln_out = nn.LayerNorm(clip_dim)
        
    def forward(self, llm_embeds, attention_mask=None):
        """
        Args:
            llm_embeds: [B, N, llm_dim]
        Returns:
            structured_embeds: [B, 77, clip_dim]
        """
        batch_size = llm_embeds.shape[0]
        
        # 1. Prepare Input (Key/Value)
        x = self.ln_in(llm_embeds)
        x = self.proj_in(x) # [B, N, clip_dim]
        
        # 2. Prepare Queries
        # Expand for batch
        q_g = self.query_global.expand(batch_size, -1, -1)
        q_e = self.query_entity.expand(batch_size, -1, -1)
        q_r = self.query_relation.expand(batch_size, -1, -1)
        
        # Concatenate strictly: Global -> Entity -> Relation
        queries = torch.cat([q_g, q_e, q_r], dim=1) # [B, 77, clip_dim]
        
        # 3. Perceiver Loop
        for block in self.blocks:
            queries = block(queries, x, attention_mask)
            
        return self.ln_out(queries)


class ResamplerBlock(nn.Module):
    """
    A single block of Cross-Attention + FeedForward
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffp = FeedForward(dim)
        
    def forward(self, x, context, context_mask=None):
        # Cross Attention: Query=x, Key/Value=context
        residual = x
        x = self.ln1(x)
        
        # Prepare mask if needed (not implemented for MVP, assuming careful batching)
        attn_out, _ = self.attn(x, context, context)
        x = residual + attn_out
        
        # FFN
        residual = x
        x = self.ln2(x)
        x = self.ffp(x)
        x = residual + x
        
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)
