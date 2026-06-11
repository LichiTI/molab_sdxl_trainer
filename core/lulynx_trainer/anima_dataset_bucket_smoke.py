# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Anima cached dataset variable-resolution collation.

Proves:
1. The collator pads variable-sized latents to batch-max spatial dimensions.
2. The padding_mask (B, 1, H, W) correctly marks padded positions as True.
3. Mixed-resolution batches produce correct shapes.
4. Fixed visual token budget pads latents to a target patch count.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
ds_mod = _load_module(
    "core.lulynx_trainer.anima_cached_dataset",
    TRAINER_ROOT / "anima_cached_dataset.py",
)
metadata_mod = _load_module(
    "core.lulynx_trainer.cache_metadata",
    TRAINER_ROOT / "cache_metadata.py",
)
AnimaCachedDataset = ds_mod.AnimaCachedDataset
anima_cached_collate = ds_mod.anima_cached_collate
_pad_latents_to_visual_tokens = ds_mod._pad_latents_to_visual_tokens
_collate_latents = ds_mod._collate_latents
write_cache_metadata = metadata_mod.write_cache_metadata


def _write_sample(root: Path, stem: str, h: int, w: int) -> None:
    """Write cache files matching the discovery pattern: {stem}_*_anima.npz."""
    np.savez(
        root / f"{stem}_64x64_anima.npz",
        latents_64=np.zeros((16, h, w), dtype=np.float32),
    )
    np.savez(
        root / f"{stem}_anima_te.npz",
        prompt_embeds=np.zeros((4, 32), dtype=np.float32),
    )


def test_variable_resolution_collation() -> None:
    """Collator pads variable-resolution latents and emits correct padding_mask."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_sample(data_dir, "small", h=4, w=4)
        _write_sample(data_dir, "wide", h=4, w=8)
        _write_sample(data_dir, "tall", h=8, w=4)

        dataset = AnimaCachedDataset(data_dir=data_dir)
        items = [dataset[i] for i in range(3)]
        batch = anima_cached_collate(items)

        latents = batch["latents"]
        padding_mask = batch["padding_mask"]

        assert latents.dim() == 4, f"Expected 4-D BCHW, got {latents.dim()}-D"
        assert latents.shape[0] == 3, f"Batch size 3, got {latents.shape[0]}"
        assert latents.shape[2] == 8, f"Max H should be 8, got {latents.shape[2]}"
        assert latents.shape[3] == 8, f"Max W should be 8, got {latents.shape[3]}"

        assert padding_mask.shape[0] == 3
        assert padding_mask.dim() == 4, f"Expected 4-D B1HW mask, got {padding_mask.dim()}-D"

        # Sample 0 (4x4): padded to 8x8
        assert not padding_mask[0, 0, 0, 0].item(), "small: top-left should be valid"
        assert padding_mask[0, 0, 4, 0].item(), "small: row 4+ should be padded"
        assert padding_mask[0, 0, 0, 4].item(), "small: col 4+ should be padded"

        # Glob returns alphabetical: small, tall, wide
        # Sample 1 = tall (8x4): padded to 8x8
        assert not padding_mask[1, 0, 0, 0].item(), "tall: top-left should be valid"
        assert not padding_mask[1, 0, 7, 0].item(), "tall: bottom edge should be valid"
        assert padding_mask[1, 0, 0, 4].item(), "tall: col 4+ should be padded"

        # Sample 2 = wide (4x8): padded to 8x8
        assert not padding_mask[2, 0, 0, 0].item(), "wide: top-left should be valid"
        assert not padding_mask[2, 0, 0, 7].item(), "wide: right edge should be valid"
        assert padding_mask[2, 0, 4, 0].item(), "wide: row 4+ should be padded"


def test_fixed_visual_token_budget() -> None:
    """_pad_latents_to_visual_tokens pads to target token count."""
    latents = torch.randn(16, 4, 4)
    padded = _pad_latents_to_visual_tokens(latents, target_tokens=16)
    assert padded.shape == (16, 8, 8), f"Expected (16,8,8), got {padded.shape}"


def test_collate_latents_padding() -> None:
    """_collate_latents pads to max spatial dims and returns mask."""
    items = [
        torch.randn(16, 4, 6),
        torch.randn(16, 8, 4),
    ]
    batch, mask = _collate_latents(items)
    assert batch.shape == (2, 16, 8, 6), f"Expected (2,16,8,6), got {batch.shape}"
    assert mask.shape == (2, 1, 8, 6), f"Expected (2,1,8,6) mask, got {mask.shape}"
    assert not mask[0, 0, 0, 0].item()
    assert mask[0, 0, 4, 0].item()
    assert not mask[1, 0, 0, 0].item()
    assert mask[1, 0, 0, 4].item()


def test_bucket_metadata_avoids_shape_file_scan() -> None:
    """Bucket setup can use sidecar metadata instead of opening each cache."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_sample(data_dir, "small", h=4, w=4)
        _write_sample(data_dir, "wide", h=4, w=8)
        metadata = write_cache_metadata(data_dir, family="anima")
        assert metadata.sample_count == 2, metadata

        with patch.object(AnimaCachedDataset, "_load_cache_file", side_effect=AssertionError("metadata miss")):
            dataset = AnimaCachedDataset(data_dir=data_dir, enable_bucket=True)
            bucket_indices = dataset.get_bucket_indices()
            summary = dataset.get_token_bucket_summary()

        assert bucket_indices == {"4x4": [0], "4x8": [1]}, bucket_indices
        assert summary["bucket_count"] == 2, summary


def main() -> int:
    test_variable_resolution_collation()
    print("  Variable-resolution collation with padding_mask -- PASS")

    test_fixed_visual_token_budget()
    print("  Fixed visual token budget padding -- PASS")

    test_collate_latents_padding()
    print("  _collate_latents padding and mask -- PASS")

    test_bucket_metadata_avoids_shape_file_scan()
    print("  Bucket metadata avoids per-cache shape scan -- PASS")

    print(
        "Anima dataset-bucket smoke passed: variable-resolution collation, "
        "padding_mask correctness, fixed visual token budget"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
