# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for SDXL text-encoder-specific conditioning dropout."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.training_loop import TrainingLoop


def _loop(**rates: float) -> TrainingLoop:
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.te_dropout = float(rates.get("te_dropout", 0.0))
    loop.clip_l_dropout_rate = float(rates.get("clip_l_dropout_rate", 0.0))
    loop.clip_g_dropout_rate = float(rates.get("clip_g_dropout_rate", 0.0))
    loop.t5_dropout_rate = float(rates.get("t5_dropout_rate", 0.0))
    loop.text_encoder_1 = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    loop.text_encoder_2 = SimpleNamespace(config=SimpleNamespace(hidden_size=5))
    return loop


def test_route_config_fields() -> None:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "sdxl-lora",
            "clip_l_dropout_rate": 0.2,
            "clip_g_dropout_rate": 0.3,
            "t5_dropout_rate": 0.4,
        }
    )
    assert cfg.clip_l_dropout_rate == 0.2
    assert cfg.clip_g_dropout_rate == 0.3
    assert cfg.t5_dropout_rate == 0.4


def test_clip_l_dropout_zeros_only_clip_l_slice() -> None:
    loop = _loop(clip_l_dropout_rate=1.0)
    embeds = {
        "encoder_hidden_states": torch.ones(2, 4, 8),
        "pooled_prompt_embeds": torch.ones(2, 5),
    }

    dropped = loop._apply_text_encoder_dropout(embeds, do_backward=True)
    hidden = dropped["encoder_hidden_states"]
    assert torch.count_nonzero(hidden[..., :3]) == 0
    assert torch.all(hidden[..., 3:] == 1.0)
    assert torch.all(dropped["pooled_prompt_embeds"] == 1.0)


def test_clip_g_dropout_zeros_clip_g_slice_and_pooled() -> None:
    loop = _loop(clip_g_dropout_rate=1.0)
    embeds = {
        "encoder_hidden_states": torch.ones(2, 4, 8),
        "pooled_prompt_embeds": torch.ones(2, 5),
    }

    dropped = loop._apply_text_encoder_dropout(embeds, do_backward=True)
    hidden = dropped["encoder_hidden_states"]
    assert torch.all(hidden[..., :3] == 1.0)
    assert torch.count_nonzero(hidden[..., 3:8]) == 0
    assert torch.count_nonzero(dropped["pooled_prompt_embeds"]) == 0


def test_generic_te_dropout_still_drops_all_conditioning() -> None:
    loop = _loop(te_dropout=1.0)
    embeds = {
        "encoder_hidden_states": torch.ones(2, 4, 8),
        "pooled_prompt_embeds": torch.ones(2, 5),
        "qwen3_hidden_states": torch.ones(2, 6, 7),
        "qwen3_attention_mask": torch.ones(2, 6),
    }

    dropped = loop._apply_text_encoder_dropout(embeds, do_backward=True)
    assert torch.count_nonzero(dropped["encoder_hidden_states"]) == 0
    assert torch.count_nonzero(dropped["pooled_prompt_embeds"]) == 0
    assert torch.count_nonzero(dropped["qwen3_hidden_states"]) == 0
    assert torch.count_nonzero(dropped["qwen3_attention_mask"]) == 0


def test_t5_dropout_noops_without_t5_tensor() -> None:
    loop = _loop(t5_dropout_rate=1.0)
    embeds = {
        "encoder_hidden_states": torch.ones(2, 4, 8),
        "pooled_prompt_embeds": torch.ones(2, 5),
    }

    dropped = loop._apply_text_encoder_dropout(embeds, do_backward=True)
    assert torch.all(dropped["encoder_hidden_states"] == 1.0)
    assert torch.all(dropped["pooled_prompt_embeds"] == 1.0)


def test_t5_dropout_zeros_future_t5_conditioning_keys() -> None:
    loop = _loop(t5_dropout_rate=1.0)
    embeds = {
        "encoder_hidden_states": torch.ones(2, 4, 8),
        "pooled_prompt_embeds": torch.ones(2, 5),
        "t5_hidden_states": torch.ones(2, 6, 9),
        "t5_attention_mask": torch.ones(2, 6),
    }

    dropped = loop._apply_text_encoder_dropout(embeds, do_backward=True)
    assert torch.count_nonzero(dropped["t5_hidden_states"]) == 0
    assert torch.count_nonzero(dropped["t5_attention_mask"]) == 0
    assert torch.all(dropped["encoder_hidden_states"] == 1.0)


def main() -> int:
    test_route_config_fields()
    test_clip_l_dropout_zeros_only_clip_l_slice()
    test_clip_g_dropout_zeros_clip_g_slice_and_pooled()
    test_generic_te_dropout_still_drops_all_conditioning()
    test_t5_dropout_noops_without_t5_tensor()
    test_t5_dropout_zeros_future_t5_conditioning_keys()
    print("SDXL text encoder dropout smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
