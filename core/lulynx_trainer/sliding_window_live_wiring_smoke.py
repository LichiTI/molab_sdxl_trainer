# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test live DiT sliding-window attention wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.anima_attention import (  # noqa: E402
    patch_anima_attention,
    reset_attention_stats,
    snapshot_attention_stats,
)


class _RmsNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 8
        self.head_dim = 4
        self.num_heads = 2
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.output_proj = nn.Linear(8, 8, bias=False)
        self.q_norm = _RmsNorm()
        self.k_norm = _RmsNorm()

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _heads, tokens, _head_dim = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        source = x if context is None else context
        q = self.q_norm(self._split_heads(self.q_proj(x)))
        k = self.k_norm(self._split_heads(self.k_proj(source)))
        v = self._split_heads(self.v_proj(source))
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.output_proj(self._merge_heads(attn))


def main() -> int:
    torch.manual_seed(7)
    model = nn.Sequential(_TinyAttention())
    profile = SimpleNamespace(
        window_size=3,
        backend="torch_fallback",
        torch_fallback_max_tokens=16,
        launcher_attention_backend="sdpa",
        flex_runtime_active=False,
    )
    patched = patch_anima_attention(model, backend="sdpa", attention_profile=profile)
    assert patched == 1

    attn = model[0]
    assert getattr(attn, "_attention_profile_window_size") == 3
    assert getattr(attn, "_attention_profile_backend") == "torch_fallback"

    reset_attention_stats()
    x = torch.randn(2, 6, 8)
    out = model(x)
    assert out.shape == x.shape
    stats = snapshot_attention_stats()
    assert stats["sliding_window_calls"] == 1
    print("Sliding-window live wiring smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
