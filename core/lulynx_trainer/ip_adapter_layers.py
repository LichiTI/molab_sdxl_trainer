"""
IP-Adapter Image Projection Layers

Includes Linear projection and Resampler for mapping image features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageProjModel(nn.Module):
    """Projection model for IP-Adapter."""
    def __init__(
        self,
        cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
    ):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = self.proj(image_embeds)
        embeds = embeds.reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        embeds = self.norm(embeds)
        return embeds

class Resampler(nn.Module):
    """
    Perceiver-style Resampler for IP-Adapter (Used in SDXL and advanced versions).
    """
    def __init__(
        self,
        dim=1024,
        depth=4,
        heads=8,
        dim_head=64,
        num_queries=16,
        embedding_dim=1024,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        b = x.shape[0]
        x = self.proj_in(x)
        latents = self.latents.repeat(b, 1, 1)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = q.reshape(b, -1, h, q.shape[-1] // h).transpose(1, 2)
        k = k.reshape(b, -1, h, k.shape[-1] // h).transpose(1, 2)
        v = v.reshape(b, -1, h, v.shape[-1] // h).transpose(1, 2)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.transpose(1, 2).reshape(b, -1, h * q.shape[-1])
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)
