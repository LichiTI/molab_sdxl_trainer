"""Lightweight Anima alias/cache-boundary smoke.

This stays below trainer/model loading.  It proves the cache-first dataloader
honors persistent workers and documents the raw/online boundary separately
from heavy Anima checkpoint smokes.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    smoke_dir = Path(__file__).resolve().parent
    if str(smoke_dir) not in sys.path:
        sys.path.insert(0, str(smoke_dir))

from anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        np.savez(
            data_dir / "sample_64x64_anima.npz",
            latents_64=np.zeros((16, 8, 8), dtype=np.float32),
        )
        np.savez(
            data_dir / "sample_anima_te.npz",
            prompt_embeds=np.zeros((4, 32), dtype=np.float32),
            attn_mask=np.ones((4,), dtype=np.bool_),
        )

        dataset = AnimaCachedDataset(data_dir)
        loader = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )
        assert loader.persistent_workers is True
        single_process_loader = create_anima_cached_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            persistent_workers=True,
        )
        batch = next(iter(single_process_loader))
        assert batch["latents"].shape == (1, 16, 8, 8)
        assert batch["encoder_hidden_states"].shape == (1, 4, 32)

    print("Anima alias/cache-boundary smoke passed: persistent cached dataloader is wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
