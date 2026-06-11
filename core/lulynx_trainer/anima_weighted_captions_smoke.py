"""Anima weighted-captions smoke.

This proves the cache-first Anima path does more than preserve the field:

1. ``AnimaCachedDataset`` reads caption sidecars and extracts a trailing
   ``weight:X`` token when ``weighted_captions=True``.
2. The cached Anima collate path emits ``caption_weights``.
3. The real Anima train-step weighting path changes the effective loss.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.anima_cached_dataset import AnimaCachedDataset, anima_cached_collate
from core.lulynx_trainer.training_loop import TrainingLoop


def _write_sample(root: Path, stem: str, weight: float) -> None:
    np.savez(
        root / f"{stem}_64x64_anima.npz",
        latents_64=np.zeros((16, 8, 8), dtype=np.float32),
    )
    np.savez(
        root / f"{stem}_anima_te.npz",
        prompt_embeds=np.zeros((4, 32), dtype=np.float32),
        attn_mask=np.ones((4,), dtype=np.bool_),
    )
    (root / f"{stem}.txt").write_text(f"tag_a, tag_b, weight:{weight}", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_sample(data_dir, "sample_a", 2.0)
        _write_sample(data_dir, "sample_b", 1.0)

        dataset = AnimaCachedDataset(
            data_dir=data_dir,
            caption_extension=".txt",
            weighted_captions=True,
        )
        item0 = dataset[0]
        item1 = dataset[1]
        assert item0["captions"] == "tag_a, tag_b"
        assert item1["captions"] == "tag_a, tag_b"
        assert float(item0["caption_weight"]) == 2.0
        assert float(item1["caption_weight"]) == 1.0

        batch = anima_cached_collate([item0, item1])
        assert torch.allclose(batch["caption_weights"], torch.tensor([2.0, 1.0], dtype=torch.float32))

        loop = TrainingLoop.__new__(TrainingLoop)
        per_sample = torch.tensor([10.0, 1.0], dtype=torch.float32)
        weighted = loop._weighted_mean_loss(per_sample, batch)
        unweighted = per_sample.mean()
        assert torch.allclose(weighted, torch.tensor(7.0))
        assert not torch.allclose(weighted, unweighted)

    print("Anima weighted-captions smoke passed: cache captions produce caption_weights and change effective loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
