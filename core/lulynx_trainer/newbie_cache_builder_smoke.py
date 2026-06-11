# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Newbie cache building without loading the production bundle."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

TRAINER_ROOT = Path(__file__).resolve().parent
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))


def _load_local_module(module_name: str, filename: str):
    full_name = f"_lulynx_{module_name}_smoke_target"
    module = sys.modules.get(full_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(full_name, TRAINER_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


build_newbie_cache = _load_local_module("newbie_cache_builder", "newbie_cache_builder.py").build_newbie_cache
NewbieCachedDataset = _load_local_module("newbie_cached_dataset", "newbie_cached_dataset.py").NewbieCachedDataset


class _FakeLatentDist:
    def sample(self) -> torch.Tensor:
        return torch.ones((1, 16, 4, 4), dtype=torch.float32)


class _FakeVae(torch.nn.Module):
    config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

    def encode(self, _images: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=_FakeLatentDist())


class _FakeTokenizer:
    model_max_length = 8

    def __init__(self) -> None:
        self.last_caption = ""
        self.last_max_length = 0

    def __call__(self, caption: str, **kwargs: object) -> dict[str, torch.Tensor]:
        self.last_caption = caption
        self.last_max_length = int(kwargs.get("max_length", 0) or 0)
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }


class _FakeGemma(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        batch, tokens = input_ids.shape
        hidden = torch.arange(batch * tokens * 8, dtype=torch.float32).reshape(batch, tokens, 8)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeClip(torch.nn.Module):
    def get_text_features(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.ones((input_ids.shape[0], 8), dtype=torch.float32)


def main() -> int:
    root = Path("H:/tmp/lulynx_newbie_cache_builder_smoke")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    Image.new("RGBA", (32, 32), (12, 34, 56, 128)).save(root / "sample.png")
    (root / "sample.caption").write_text("a Warehouse Newbie cache smoke", encoding="utf-8")

    gemma_tokenizer = _FakeTokenizer()
    clip_tokenizer = _FakeTokenizer()
    loaded = SimpleNamespace(
        vae=_FakeVae(),
        text_encoder_1=_FakeGemma(),
        tokenizer_1=gemma_tokenizer,
        text_encoder_2=_FakeClip(),
        tokenizer_2=clip_tokenizer,
    )
    result = build_newbie_cache(
        loaded_model=loaded,
        data_dir=root,
        device="cpu",
        dtype=torch.float32,
        resolution=(32, 32),
        caption_extension=".caption",
        gemma3_prompt="Gemma3 prompt: {caption}",
        gemma_max_token_length=6,
        clip_max_token_length=7,
        alpha_mask=True,
        force=True,
    )
    if result.errors or result.written != 1:
        raise AssertionError(f"Unexpected cache build result: {result}")
    if not result.metadata_fast_path:
        raise AssertionError(f"Expected fast metadata path for full rebuild: {result}")

    dataset = NewbieCachedDataset(root)
    metadata_summary = dataset.get_cache_metadata_summary()
    if metadata_summary["fallback_shape_loads"] != 0:
        raise AssertionError(f"Expected metadata shape hit without fallback scan: {metadata_summary}")
    item = dataset[0]
    if tuple(item["latents"].shape) != (16, 4, 4):  # type: ignore[index]
        raise AssertionError(f"Unexpected latent shape: {item['latents'].shape}")  # type: ignore[index]
    if tuple(item["encoder_hidden_states"].shape) != (4, 8):  # type: ignore[index]
        raise AssertionError(f"Unexpected hidden shape: {item['encoder_hidden_states'].shape}")  # type: ignore[index]
    if tuple(item["pooled_prompt_embeds"].shape) != (8,):  # type: ignore[index]
        raise AssertionError(f"Unexpected pooled shape: {item['pooled_prompt_embeds'].shape}")  # type: ignore[index]
    if tuple(item["loss_mask"].shape) != (4, 4):  # type: ignore[index]
        raise AssertionError(f"Unexpected loss mask shape: {item['loss_mask'].shape}")  # type: ignore[index]
    if gemma_tokenizer.last_caption != "Gemma3 prompt: a Warehouse Newbie cache smoke":
        raise AssertionError(f"Unexpected Gemma prompt: {gemma_tokenizer.last_caption}")
    if gemma_tokenizer.last_max_length != 6 or clip_tokenizer.last_max_length != 7:
        raise AssertionError("Newbie cache builder did not pass configured token limits")

    print("Newbie cache builder smoke passed: wrote schema-v2 cache with latent-sized loss_mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

