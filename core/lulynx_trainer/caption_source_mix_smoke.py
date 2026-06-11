"""Tiny smoke for structured caption source mixing."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.caption_sidecar import json_caption_to_training_parts
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
    ds.caption_tag_dropout_targets = set()
    ds.caption_tag_dropout_target_mode = "drop_all"
    ds.caption_tag_dropout_target_count = 1
    ds.weighted_captions = False
    ds.token_warmup_min = 0
    ds.token_warmup_max = 0
    ds.token_warmup_steps = 0
    ds._global_step = 0
    ds._current_epoch = 0
    ds.caption_source_mix = kwargs["caption_source_mix"]
    return ds


def main() -> int:
    from core.lulynx_trainer.caption_source_mix import normalize_caption_source_mix_config

    structured = json_caption_to_training_parts(
        '{"concept":"lulu","tags":["1girl","red dress"],"nl":"a calm girl in sunlight"}'
    )
    assert structured["triggers"] == ["lulu"]

    nl_only = _dataset(
        caption_source_mix=normalize_caption_source_mix_config(
            enabled=True,
            nl_ratio=100,
            tag_ratio=0,
            trigger_only_ratio=0,
            empty_ratio=0,
        )
    )
    caption, weight = nl_only._process_caption(
        str(structured["text"]),
        structured_tags=list(structured["tags"]),
        structured_nl=list(structured["nl"]),
        structured_triggers=list(structured["triggers"]),
    )
    assert caption == "lulu, a calm girl in sunlight"
    assert weight == 1.0

    tag_only = _dataset(
        caption_source_mix=normalize_caption_source_mix_config(
            enabled=True,
            nl_ratio=0,
            tag_ratio=100,
            trigger_only_ratio=0,
            empty_ratio=0,
        )
    )
    caption, _ = tag_only._process_caption(
        str(structured["text"]),
        structured_tags=list(structured["tags"]),
        structured_nl=list(structured["nl"]),
        structured_triggers=list(structured["triggers"]),
    )
    assert caption == "lulu, 1girl, red dress"

    trigger_only = _dataset(
        caption_source_mix=normalize_caption_source_mix_config(
            enabled=True,
            nl_ratio=0,
            tag_ratio=0,
            trigger_only_ratio=100,
            empty_ratio=0,
        )
    )
    caption, _ = trigger_only._process_caption(
        str(structured["text"]),
        structured_tags=list(structured["tags"]),
        structured_nl=list(structured["nl"]),
        structured_triggers=list(structured["triggers"]),
    )
    assert caption == "lulu"

    empty_only = _dataset(
        caption_source_mix=normalize_caption_source_mix_config(
            enabled=True,
            nl_ratio=0,
            tag_ratio=0,
            trigger_only_ratio=0,
            empty_ratio=100,
        )
    )
    caption, _ = empty_only._process_caption(
        str(structured["text"]),
        structured_tags=list(structured["tags"]),
        structured_nl=list(structured["nl"]),
        structured_triggers=list(structured["triggers"]),
    )
    assert caption == ""

    print("caption_source_mix_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
