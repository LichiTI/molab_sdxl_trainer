# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for SDXL dataset-side feature closures."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.dataset_loader import BucketManager, CaptionDataset, collate_fn


def main() -> int:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "model_type": "sdxl",
            "bucket_selection_mode": "area",
            "bucket_custom_resos": "320x512,512x320",
            "masked_loss": True,
            "alpha_mask": True,
        }
    )
    assert cfg.bucket_selection_mode == "area"
    assert cfg.bucket_custom_resos == "320x512,512x320"
    assert cfg.masked_loss is True
    assert cfg.alpha_mask is True

    buckets = BucketManager(
        base_resolution=512,
        selection_mode="area",
        custom_resos="320x512,512x320",
    )
    assert buckets.buckets == [(320, 512), (512, 320)]
    assert buckets.get_bucket(300, 500) == (320, 512)

    with tempfile.TemporaryDirectory(prefix="lulynx_dataset_feature_smoke_") as tmp:
        root = Path(tmp)
        rgba = Image.new("RGBA", (16, 16), (20, 40, 60, 0))
        for x in range(8):
            for y in range(16):
                rgba.putpixel((x, y), (20, 40, 60, 255))
        rgba.save(root / "sample.png")
        (root / "sample.txt").write_text("cat, clean, weight:2.5", encoding="utf-8")

        dataset = CaptionDataset(
            data_dir=str(root),
            resolution=16,
            enable_bucket=False,
            weighted_captions=True,
            masked_loss=True,
            alpha_mask=True,
            shuffle_caption=False,
        )
        item = dataset[0]
        assert item["caption"] == "cat, clean"
        assert abs(float(item["caption_weight"]) - 2.5) < 1e-6
        assert item["loss_mask"] is not None
        batch = collate_fn([item])
        assert torch.is_tensor(batch["loss_masks"])
        assert tuple(batch["loss_masks"].shape) == (1, 1, 16, 16)
        assert batch["loss_masks"][0, 0, :, :8].mean() > 0.9
        assert batch["loss_masks"][0, 0, :, 8:].mean() < 0.1

    print("Dataset feature smoke passed: custom buckets, alpha masks, and caption weights are wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
