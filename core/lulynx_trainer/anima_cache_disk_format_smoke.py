# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima cache-builder disk format / dtype routing."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_ROOT = Path(_HERE)
_CORE_ROOT = _TRAINER_ROOT.parent
_BACKEND_ROOT = _CORE_ROOT.parent
for _path in (str(_BACKEND_ROOT), str(_CORE_ROOT), str(_TRAINER_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from .anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample
except ImportError:  # pragma: no cover - direct script execution
    from anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache_sample


def _write_sample(root: Path) -> Path:
    image_path = root / "sample.png"
    Image.new("RGB", (64, 64), color="navy").save(image_path)
    image_path.with_suffix(".txt").write_text("sample caption", encoding="utf-8")
    return image_path


def _fake_vae_encode(_image: torch.Tensor) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=16 * 4 * 4, dtype=torch.float32).reshape(1, 16, 4, 4)


def _fake_text_encode(_caption: str) -> dict[str, torch.Tensor]:
    return {
        "prompt_embeds": torch.linspace(0.0, 1.0, steps=8 * 6, dtype=torch.float32).reshape(8, 6),
        "attn_mask": torch.ones(8, dtype=torch.bool),
        "qwen3_hidden_states": torch.ones(5, 4, dtype=torch.float32),
        "qwen3_attention_mask": torch.ones(5, dtype=torch.bool),
    }


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    if path.suffix == ".pt":
        return torch.load(str(path), map_location="cpu", weights_only=True)
    with np.load(str(path)) as data:
        return {key: data[key] for key in data.files}


def _payload_dtype(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if isinstance(value, torch.Tensor):
        return str(value.dtype).replace("torch.", "")
    return str(value.dtype)


def test_anima_builder_honors_independent_latent_and_text_cache_format() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        image_path = _write_sample(root)
        config = AnimaCacheBuilderConfig(
            data_dir=str(root),
            output_dir=str(root),
            disk_format="safetensors",
            disk_dtype="float32",
            text_disk_format="pt",
            text_disk_dtype="float16",
        )

        latent_path, text_path = build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=_fake_vae_encode,
            text_encode_fn=_fake_text_encode,
            config=config,
            force=True,
        )

        assert latent_path.name == "sample_4x4_anima.safetensors"
        assert text_path.name == "sample_anima_te.pt"
        latent_payload = _load_payload(latent_path)
        text_payload = _load_payload(text_path)
        assert _payload_dtype(latent_payload, "latents_4x4") == "float32"
        assert _payload_dtype(text_payload, "prompt_embeds") == "float16"
        assert _payload_dtype(text_payload, "qwen3_hidden_states") == "float16"
        assert _payload_dtype(text_payload, "attn_mask") == "bool"


def test_anima_builder_text_cache_defaults_to_latent_format_when_unset() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        image_path = _write_sample(root)
        config = AnimaCacheBuilderConfig(
            data_dir=str(root),
            output_dir=str(root),
            disk_format="pt",
            disk_dtype="float32",
        )

        latent_path, text_path = build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=_fake_vae_encode,
            text_encode_fn=_fake_text_encode,
            config=config,
            force=True,
        )

        assert latent_path.suffix == ".pt"
        assert text_path.suffix == ".pt"
        text_payload = _load_payload(text_path)
        assert _payload_dtype(text_payload, "prompt_embeds") == "float32"


def test_anima_builder_preserves_bfloat16_in_torch_backed_formats() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        image_path = _write_sample(root)
        config = AnimaCacheBuilderConfig(
            data_dir=str(root),
            output_dir=str(root),
            disk_format="safetensors",
            disk_dtype="bfloat16",
            text_disk_format="pt",
            text_disk_dtype="bf16",
        )

        latent_path, text_path = build_anima_cache_sample(
            image_path=image_path,
            vae_encode_fn=_fake_vae_encode,
            text_encode_fn=_fake_text_encode,
            config=config,
            force=True,
        )

        latent_payload = _load_payload(latent_path)
        text_payload = _load_payload(text_path)
        assert _payload_dtype(latent_payload, "latents_4x4") == "bfloat16"
        assert _payload_dtype(text_payload, "prompt_embeds") == "bfloat16"


def main() -> int:
    test_anima_builder_honors_independent_latent_and_text_cache_format()
    test_anima_builder_text_cache_defaults_to_latent_format_when_unset()
    test_anima_builder_preserves_bfloat16_in_torch_backed_formats()
    print("anima_cache_disk_format_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
