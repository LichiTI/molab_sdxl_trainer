from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.lulynx_trainer.anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample
from core.lulynx_trainer.anima_cached_dataset import AnimaCachedDataset, AnimaCacheSchema
from core.lulynx_trainer.caption_source_mix import normalize_caption_source_mix_config
from core.lulynx_trainer.newbie_cached_dataset import NewbieCachedDataset, NewbieCacheSchema


def _write_sample(root: Path) -> Path:
    image_path = root / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 80, 200)).save(image_path)
    (root / "sample.json").write_text(
        '{"trigger":"lulu","tags":["1girl","blue dress"],"nl":"a calm girl standing in sunlight"}',
        encoding="utf-8",
    )
    return image_path


def _fake_text_encode(caption: str) -> dict[str, torch.Tensor]:
    value = float(len(caption))
    return {
        "prompt_embeds": torch.full((3, 2), value),
        "attn_mask": torch.ones(3, dtype=torch.bool),
    }


def test_anima_cache_variants() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_path = _write_sample(root)
        cfg = AnimaCacheBuilderConfig(
            data_dir=str(root),
            output_dir=str(root),
            disk_format="npz",
            caption_source_mix=normalize_caption_source_mix_config(
                enabled=True,
                nl_ratio=65,
                tag_ratio=20,
                trigger_only_ratio=10,
                empty_ratio=5,
                trigger_tokens="lulu",
            ),
        )
        _latent_path, text_path = build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=lambda _image: torch.zeros((1, 16, 4, 4)),
            text_encode_fn=_fake_text_encode,
            config=cfg,
            caption_extension=".json",
            force=True,
        )
        with np.load(text_path) as data:
            keys = set(data.files)
        assert "caption_variant_nl_prompt_embeds" in keys
        assert "caption_variant_tag_prompt_embeds" in keys
        assert "caption_variant_trigger_only_prompt_embeds" in keys
        assert "caption_variant_empty_prompt_embeds" in keys

        dataset = AnimaCachedDataset(
            root,
            caption_extension=".json",
            schema=AnimaCacheSchema(require_loss_mask=False),
            caption_source_mix_enabled=True,
            caption_source_nl_ratio=0,
            caption_source_tag_ratio=0,
            caption_source_trigger_only_ratio=0,
            caption_source_empty_ratio=100,
        )
        item = dataset[0]
        assert torch.all(item["encoder_hidden_states"] == 0), "empty variant should be selected"


def test_newbie_cached_dataset_variant_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        np.savez(
            root / "sample_newbie.npz",
            newbie_cache_schema_version=np.asarray(2, dtype=np.int32),
            latents=np.zeros((16, 4, 4), dtype=np.float32),
            encoder_hidden_states=np.ones((2, 3), dtype=np.float32),
            attention_mask=np.ones((2,), dtype=bool),
            pooled_prompt_embeds=np.ones((4,), dtype=np.float32),
            caption_variant_empty_encoder_hidden_states=np.zeros((2, 3), dtype=np.float32),
            caption_variant_empty_attention_mask=np.zeros((2,), dtype=bool),
            caption_variant_empty_pooled_prompt_embeds=np.zeros((4,), dtype=np.float32),
        )
        dataset = NewbieCachedDataset(
            root,
            schema=NewbieCacheSchema(require_pooled_prompt_embeds=True),
            caption_source_mix_enabled=True,
            caption_source_nl_ratio=0,
            caption_source_tag_ratio=0,
            caption_source_trigger_only_ratio=0,
            caption_source_empty_ratio=100,
        )
        item = dataset[0]
        assert torch.all(item["encoder_hidden_states"] == 0), "empty variant should be selected"
        assert torch.all(item["pooled_prompt_embeds"] == 0), "pooled empty variant should be selected"


if __name__ == "__main__":
    test_anima_cache_variants()
    test_newbie_cached_dataset_variant_selection()
    print("caption_source_mix_cache_smoke: ok")
