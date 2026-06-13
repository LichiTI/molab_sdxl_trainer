"""Faithful native-Anima DiT pieces (clean-room Lulynx).

The executable subset in :mod:`anima_native_dit` is a smoke-grade stub that is
already faithful for AdaLN-LoRA modulation, QK-RMSNorm, the three-stage block,
the timestep embedder, the final layer and patchify -- but it is missing the two
pieces the real native-Anima forward needs to actually denoise:

1. **3D RoPE** on self-attention (parameter-free positional encoding).
2. **llm_adapter** -- a 6-layer text-conditioning sub-network whose 118 weights
   already live in the checkpoint under ``net.llm_adapter.*`` but are never
   loaded or run by the stub.

This module supplies both as additive, self-contained components so the native
stack can render faithfully. ``AnimaRope3D`` is wired into the subset's
self-attention (see :mod:`anima_native_dit`); ``AnimaLlmAdapter`` is run once in
the text-encode stage (it depends only on the prompt, never on the noisy latent
or timestep) and its output is injected through the sampler's existing
``prompt_embeds`` seam.

Clean-room: architecture/conventions were studied from the public anima
reference, but no source is shared. PolyForm Noncommercial.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# Native-Anima base config (model_channels=2048 -> 28 blocks / 16 heads).
ANIMA_HEAD_DIM = 128
ANIMA_ROPE_THETA = 10000.0
# RoPE NTK extrapolation ratios for the 16-latent-channel image config.
ANIMA_ROPE_H_RATIO = 4.0
ANIMA_ROPE_W_RATIO = 4.0
ANIMA_ROPE_T_RATIO = 1.0
# llm_adapter geometry.
ANIMA_T5_VOCAB = 32128
ANIMA_ADAPTER_HEADS = 16


# ---------------------------------------------------------------------------
# 3D RoPE (self-attention positional encoding) — parameter-free
# ---------------------------------------------------------------------------

def _rotate_half_noninterleaved(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in two halves: ``[-x2, x1]`` (non-interleaved RoPE)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class AnimaRope3D(nn.Module):
    """Axis-decomposed 3D rotary position embedding for (T, H, W) latent tokens.

    ``head_dim`` is split into temporal / height / width sub-bands; each band gets
    its own NTK-scaled frequency table. ``generate`` returns the per-token *angles*
    ``[L, head_dim]`` (L = T*H*W, row-major over (t, h, w) — the same token order
    the subset's ``unfold`` patchify produces). ``apply`` rotates a ``[B, H, L, D]``
    query/key tensor with those angles.
    """

    def __init__(
        self,
        head_dim: int = ANIMA_HEAD_DIM,
        *,
        h_ratio: float = ANIMA_ROPE_H_RATIO,
        w_ratio: float = ANIMA_ROPE_W_RATIO,
        t_ratio: float = ANIMA_ROPE_T_RATIO,
        theta: float = ANIMA_ROPE_THETA,
    ) -> None:
        super().__init__()
        dim_h = head_dim // 6 * 2
        dim_w = dim_h
        dim_t = head_dim - 2 * dim_h
        if dim_h + dim_w + dim_t != head_dim:
            raise ValueError(f"bad RoPE split: {dim_h}+{dim_w}+{dim_t} != {head_dim}")
        self.head_dim = head_dim
        self._dim_h = dim_h
        self._dim_t = dim_t
        # NTK theta per axis (extrapolation ratio raised to dim/(dim-2)).
        self._h_theta = theta * (h_ratio ** (dim_h / (dim_h - 2)))
        self._w_theta = theta * (w_ratio ** (dim_w / (dim_w - 2)))
        self._t_theta = theta * (t_ratio ** (dim_t / (dim_t - 2)))
        # Cached angle table; rebuilt when the (t, h, w) shape changes.
        self._cache_key: Optional[Tuple[int, int, int]] = None
        self._cache: Optional[torch.Tensor] = None

    def _axis_freqs(self, theta: float, dim: int, device: torch.device) -> torch.Tensor:
        # range = arange(0, dim, 2)[:dim//2] / dim ; freq = 1 / theta**range
        idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: dim // 2] / dim
        return 1.0 / (theta ** idx)

    @torch.no_grad()
    def generate(self, t: int, h: int, w: int, *, device: torch.device) -> torch.Tensor:
        key = (int(t), int(h), int(w))
        if self._cache_key == key and self._cache is not None and self._cache.device == device:
            return self._cache
        h_freqs = self._axis_freqs(self._h_theta, self._dim_h, device)
        w_freqs = self._axis_freqs(self._w_theta, self._dim_h, device)
        t_freqs = self._axis_freqs(self._t_theta, self._dim_t, device)
        seq_t = torch.arange(t, device=device, dtype=torch.float32)
        seq_h = torch.arange(h, device=device, dtype=torch.float32)
        seq_w = torch.arange(w, device=device, dtype=torch.float32)
        emb_t = torch.outer(seq_t, t_freqs)  # [T, dim_t//2]
        emb_h = torch.outer(seq_h, h_freqs)  # [H, dim_h//2]
        emb_w = torch.outer(seq_w, w_freqs)  # [W, dim_w//2]
        te = emb_t.view(t, 1, 1, -1).expand(t, h, w, -1)
        he = emb_h.view(1, h, 1, -1).expand(t, h, w, -1)
        we = emb_w.view(1, 1, w, -1).expand(t, h, w, -1)
        one = torch.cat((te, he, we), dim=-1)            # [T,H,W, head_dim//2]
        angles = torch.cat((one, one), dim=-1)           # [T,H,W, head_dim] (duplicated)
        angles = angles.reshape(t * h * w, self.head_dim).contiguous()
        self._cache_key = key
        self._cache = angles
        return angles

    @staticmethod
    def apply(x_BHLD: torch.Tensor, angles_LD: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` ``[B, heads, L, head_dim]`` by per-token ``angles`` ``[L, head_dim]``."""
        cos = angles_LD.cos()[None, None].to(x_BHLD.dtype)
        sin = angles_LD.sin()[None, None].to(x_BHLD.dtype)
        return x_BHLD * cos + _rotate_half_noninterleaved(x_BHLD) * sin


# ---------------------------------------------------------------------------
# llm_adapter (Qwen3 hidden + T5 token ids -> cross-attention conditioning)
# ---------------------------------------------------------------------------

class _AdapterRMSNorm(nn.Module):
    """T5-style RMSNorm (no mean subtraction); fp32 variance, weight-typed out."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return self.weight * x


def _adapter_rope_table(seq_len: int, head_dim: int, device: torch.device,
                        theta: float = ANIMA_ROPE_THETA) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapter rotary cos/sin tables ``[1, seq, head_dim]`` (standard 1D RoPE)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)              # [seq, head_dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)         # [seq, head_dim]
    return emb.cos()[None], emb.sin()[None]


def _apply_adapter_rope(x_BHSD: torch.Tensor, cos_1SD: torch.Tensor, sin_1SD: torch.Tensor) -> torch.Tensor:
    cos = cos_1SD.unsqueeze(1).to(x_BHSD.dtype)      # [1,1,S,D]
    sin = sin_1SD.unsqueeze(1).to(x_BHSD.dtype)
    return x_BHSD * cos + _rotate_half_noninterleaved(x_BHSD) * sin


class _AdapterAttention(nn.Module):
    """Multi-head attention with QK-RMSNorm + per-q/per-kv RoPE (bias-free)."""

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        inner = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = _AdapterRMSNorm(head_dim)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = _AdapterRMSNorm(head_dim)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        *,
        mask: Optional[torch.Tensor] = None,
        rope_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        ctx = x if context is None else context
        bx, sx, _ = x.shape
        bc, sc, _ = ctx.shape
        q = self.q_norm(self.q_proj(x).view(bx, sx, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(ctx).view(bc, sc, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(ctx).view(bc, sc, self.n_heads, self.head_dim).transpose(1, 2)
        if rope_q is not None:
            q = _apply_adapter_rope(q, rope_q[0], rope_q[1])
        if rope_kv is not None:
            k = _apply_adapter_rope(k, rope_kv[0], rope_kv[1])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(bx, sx, -1)
        return self.o_proj(out)


class _AdapterBlock(nn.Module):
    """Optional self-attn + cross-attn + GELU-MLP, each pre-RMSNorm + residual."""

    def __init__(self, source_dim: int, model_dim: int, n_heads: int = ANIMA_ADAPTER_HEADS,
                 mlp_ratio: float = 4.0, self_attn: bool = True) -> None:
        super().__init__()
        self.has_self_attn = self_attn
        head_dim = model_dim // n_heads
        if self_attn:
            self.norm_self_attn = _AdapterRMSNorm(model_dim)
            self.self_attn = _AdapterAttention(model_dim, model_dim, n_heads, head_dim)
        self.norm_cross_attn = _AdapterRMSNorm(model_dim)
        self.cross_attn = _AdapterAttention(model_dim, source_dim, n_heads, head_dim)
        self.norm_mlp = _AdapterRMSNorm(model_dim)
        hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
        rope_self: Tuple[torch.Tensor, torch.Tensor],
        rope_ctx: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.has_self_attn:
            x = x + self.self_attn(self.norm_self_attn(x), mask=tgt_mask, rope_q=rope_self, rope_kv=rope_self)
        x = x + self.cross_attn(self.norm_cross_attn(x), context, mask=src_mask, rope_q=rope_self, rope_kv=rope_ctx)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class AnimaLlmAdapter(nn.Module):
    """Bridge: Qwen3 hidden states (source) + T5 token ids (target) -> conditioning.

    The T5 ids index a learnable embedding (the "query slots"); each adapter block
    cross-attends those slots to the Qwen3 semantic content. Output length equals
    the T5 sequence length and feeds the DiT cross-attention. Depends only on the
    prompt — never on the noisy latent or timestep — so it is run once per prompt.
    """

    def __init__(
        self,
        source_dim: int = 1024,
        target_dim: int = 1024,
        model_dim: int = 1024,
        *,
        num_layers: int = 6,
        num_heads: int = ANIMA_ADAPTER_HEADS,
        vocab_size: int = ANIMA_T5_VOCAB,
        self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.embed = nn.Embedding(vocab_size, target_dim)
        self.in_proj = nn.Identity() if model_dim == target_dim else nn.Linear(target_dim, model_dim)
        self.blocks = nn.ModuleList(
            [_AdapterBlock(source_dim, model_dim, n_heads=num_heads, self_attn=self_attn) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = _AdapterRMSNorm(target_dim)

    @staticmethod
    def _prep_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        m = mask.to(torch.bool)
        if m.ndim == 2:
            m = m[:, None, None, :]          # [B,1,1,S] -> broadcast over heads/query
        return m

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_mask = self._prep_mask(target_attention_mask)
        src_mask = self._prep_mask(source_attention_mask)
        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states.to(dtype=x.dtype)
        device = x.device
        rope_self = _adapter_rope_table(x.shape[1], self.head_dim, device)
        rope_ctx = _adapter_rope_table(context.shape[1], self.head_dim, device)
        for block in self.blocks:
            x = block(x, context, tgt_mask=tgt_mask, src_mask=src_mask, rope_self=rope_self, rope_ctx=rope_ctx)
        out = self.norm(self.out_proj(x))
        if target_attention_mask is not None:
            out = out * target_attention_mask.to(out.dtype).unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# Checkpoint loader for the llm_adapter sub-network (net.llm_adapter.*)
# ---------------------------------------------------------------------------

def load_anima_llm_adapter(
    dit_path: str | Path,
    *,
    device: str = "cpu",
    dtype: Optional[Any] = None,
    disable_mmap: bool = False,
) -> Tuple[AnimaLlmAdapter, Dict[str, Any]]:
    """Strictly load the 118-key ``net.llm_adapter.*`` sub-network from a native DiT.

    Returns the assembled :class:`AnimaLlmAdapter` plus a small load report. Raises
    if the checkpoint carries no llm_adapter (a stub-only / preview checkpoint).
    """
    try:
        from core.lulynx_trainer.safetensors_loader import open_safetensors
    except ImportError:  # pragma: no cover - direct-file usage
        from .safetensors_loader import open_safetensors

    prefix = "net.llm_adapter."
    sub: Dict[str, torch.Tensor] = {}
    total = 0
    with open_safetensors(str(Path(dit_path)), framework="pt", device="cpu", disable_mmap=disable_mmap) as handle:
        for key in handle.keys():
            total += 1
            if not key.startswith(prefix):
                continue
            tensor = handle.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            sub[key[len(prefix):]] = tensor

    if "embed.weight" not in sub:
        raise ValueError(
            f"Checkpoint {Path(dit_path).name} has no net.llm_adapter.* weights "
            "(stub/preview checkpoint?) — faithful native forward needs them."
        )

    vocab_size, target_dim = (int(v) for v in sub["embed.weight"].shape)
    head_dim = int(sub["blocks.0.self_attn.q_norm.weight"].shape[0]) if "blocks.0.self_attn.q_norm.weight" in sub \
        else target_dim // ANIMA_ADAPTER_HEADS
    num_heads = max(target_dim // head_dim, 1)
    num_layers = 1 + max(
        (int(k.split(".")[1]) for k in sub if k.startswith("blocks.")),
        default=-1,
    )
    has_self_attn = any(k.startswith("blocks.0.self_attn.") for k in sub)

    adapter = AnimaLlmAdapter(
        source_dim=target_dim,
        target_dim=target_dim,
        model_dim=target_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        self_attn=has_self_attn,
    )
    incompatible = adapter.load_state_dict(sub, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"llm_adapter strict load failed: missing={incompatible.missing_keys[:4]} "
            f"unexpected={incompatible.unexpected_keys[:4]}"
        )
    adapter.to(device=device, dtype=dtype) if dtype is not None else adapter.to(device=device)
    adapter.eval()
    report = {
        "loaded_keys": len(sub),
        "total_keys": total,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "model_dim": target_dim,
        "has_self_attn": has_self_attn,
    }
    logger.info("anima llm_adapter loaded: %s", report)
    return adapter, report
