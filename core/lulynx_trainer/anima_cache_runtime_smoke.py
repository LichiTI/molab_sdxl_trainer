# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima cache runtime encode callables."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_ROOT = Path(_HERE)
_CORE_ROOT = _TRAINER_ROOT.parent
_BACKEND_ROOT = _CORE_ROOT.parent
for _path in (str(_BACKEND_ROOT), str(_CORE_ROOT), str(_TRAINER_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from .anima_cache_runtime import build_anima_cache_encode_bundle
except ImportError:  # pragma: no cover - direct script execution
    from anima_cache_runtime import build_anima_cache_encode_bundle


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, caption: str, **kwargs: object) -> dict[str, torch.Tensor]:
        self.calls.append({"caption": caption, **kwargs})
        max_length = int(kwargs["max_length"])
        return {
            "input_ids": torch.arange(max_length, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class _FakeQwen3Encoder:
    def __init__(self, hidden_size: int = 6) -> None:
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def to(self, **_kwargs: object) -> "_FakeQwen3Encoder":
        return self

    def __call__(self, *, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        batch, seq_len = input_ids.shape
        hidden = torch.arange(batch * seq_len * self.config.hidden_size, dtype=torch.float32)
        hidden = hidden.reshape(batch, seq_len, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden)


def test_qwen3_and_t5_max_lengths_reach_cache_runtime_tokenizers() -> None:
    qwen3_tokenizer = _FakeTokenizer()
    t5_tokenizer = _FakeTokenizer()
    model = SimpleNamespace(
        vae=object(),
        anima_qwen3_encoder=_FakeQwen3Encoder(),
        anima_qwen3_tokenizer=qwen3_tokenizer,
        anima_t5_tokenizer=t5_tokenizer,
    )
    config = SimpleNamespace(
        anima_qwen3_max_token_length=37,
        anima_t5_max_token_length=53,
    )

    bundle = build_anima_cache_encode_bundle(
        model=model,
        device="cpu",
        dtype=torch.float32,
        config=config,
    )
    encoded = bundle.text_encode_fn("a small anima prompt")

    assert bundle.primary_text_source == "qwen3"
    assert qwen3_tokenizer.calls[-1]["max_length"] == 37
    assert t5_tokenizer.calls[-1]["max_length"] == 53
    assert encoded["prompt_embeds"].shape == (37, 6)
    assert encoded["attn_mask"].shape == (37,)
    assert encoded["t5_input_ids"].shape == (53,)
    assert encoded["t5_attn_mask"].shape == (53,)


def main() -> int:
    test_qwen3_and_t5_max_lengths_reach_cache_runtime_tokenizers()
    print("anima_cache_runtime_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
