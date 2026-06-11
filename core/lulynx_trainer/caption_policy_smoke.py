# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test targeted caption dropout policies."""

from __future__ import annotations

import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.caption_sidecar import json_caption_to_training_parts
from core.lulynx_trainer.caption_source_mix import normalize_caption_source_mix_config
from core.lulynx_trainer.dataset_loader import CaptionDataset


def _dataset(**kwargs: object) -> CaptionDataset:
    ds = CaptionDataset.__new__(CaptionDataset)
    ds.caption_dropout_rate = float(kwargs.get("caption_dropout_rate", 0.0))
    ds.caption_dropout_every_n_epochs = int(kwargs.get("caption_dropout_every_n_epochs", 0))
    ds.shuffle_caption = bool(kwargs.get("shuffle_caption", False))
    ds.shuffle_caption_tags_only = bool(kwargs.get("shuffle_caption_tags_only", False))
    ds.keep_tokens = int(kwargs.get("keep_tokens", 0))
    ds.keep_tokens_separator = str(kwargs.get("keep_tokens_separator", ""))
    ds.tag_dropout_rate = float(kwargs.get("tag_dropout_rate", 0.0))
    ds.caption_tag_dropout_targets = set(kwargs.get("caption_tag_dropout_targets", set()))
    ds.caption_tag_dropout_target_mode = str(kwargs.get("caption_tag_dropout_target_mode", "drop_all"))
    ds.caption_tag_dropout_target_count = int(kwargs.get("caption_tag_dropout_target_count", 1))
    ds.weighted_captions = bool(kwargs.get("weighted_captions", False))
    ds.token_warmup_min = int(kwargs.get("token_warmup_min", 0))
    ds.token_warmup_max = int(kwargs.get("token_warmup_max", 0))
    ds.token_warmup_steps = int(kwargs.get("token_warmup_steps", 0))
    ds._global_step = int(kwargs.get("_global_step", 0))
    ds._current_epoch = int(kwargs.get("_current_epoch", 0))
    ds.caption_source_mix = normalize_caption_source_mix_config(enabled=False)
    return ds


def main() -> int:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "model_type": "sdxl",
            "caption_tag_dropout_targets": "blue\ngreen",
            "caption_tag_dropout_target_mode": "random_n",
            "caption_tag_dropout_target_count": 1,
            "caption_dropout_every_n_epochs": 2,
            "keep_tokens_separator": "|||",
        }
    )
    assert cfg.caption_tag_dropout_targets == "blue\ngreen"
    assert cfg.caption_tag_dropout_target_mode == "random_n"
    assert cfg.caption_tag_dropout_target_count == 1
    assert cfg.caption_dropout_every_n_epochs == 2
    assert cfg.keep_tokens_separator == "|||"

    drop_all = _dataset(caption_tag_dropout_targets={"blue", "green"}, weighted_captions=True)
    caption, weight = drop_all._process_caption("red, blue, green, weight:2.0")
    assert caption == "red"
    assert weight == 2.0

    random.seed(7)
    random_n = _dataset(
        caption_tag_dropout_targets={"blue", "green"},
        caption_tag_dropout_target_mode="random_n",
        caption_tag_dropout_target_count=1,
    )
    caption, _ = random_n._process_caption("red, blue, green")
    tags = [part.strip() for part in caption.split(",") if part.strip()]
    assert "red" in tags
    assert len([tag for tag in tags if tag in {"blue", "green"}]) == 1

    keep_sep = _dataset(keep_tokens=1, keep_tokens_separator="|||", shuffle_caption=False)
    caption, _ = keep_sep._process_caption("alpha, beta, gamma")
    assert caption == "alpha, |||, beta, gamma"

    random.seed(11)
    parts = json_caption_to_training_parts('{"tags":["alpha","beta","gamma"],"nl":"a calm girl in a classroom"}')
    tags_only = _dataset(shuffle_caption=True, shuffle_caption_tags_only=True)
    caption, _ = tags_only._process_caption(
        str(parts["text"]),
        structured_tags=list(parts["tags"]),
        structured_nl=list(parts["nl"]),
    )
    assert caption.endswith("a calm girl in a classroom")
    assert caption != "alpha, beta, gamma, a calm girl in a classroom"

    epoch_drop = _dataset(caption_dropout_every_n_epochs=2, _current_epoch=1)
    caption, _ = epoch_drop._process_caption("alpha, beta")
    assert caption == ""

    print("Caption policy smoke passed: targeted tag dropout and keep-token separator are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
