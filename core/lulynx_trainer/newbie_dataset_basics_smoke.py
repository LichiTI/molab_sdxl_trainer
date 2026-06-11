# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test raw-image dataset basics used by Newbie training preparation."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.dataset_loader import CaptionDataset, create_dataloader


def main() -> int:
    root = Path("H:/tmp/lulynx_newbie_dataset_basics")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    samples = [
        ((128, 64), "alpha beta gamma delta"),
        ((64, 128), "one two"),
    ]
    for index, (size, caption) in enumerate(samples):
        image_path = root / f"sample_{index}.png"
        Image.new("RGB", size, (20 * (index + 1), 40, 60)).save(image_path)
        image_path.with_suffix(".caption").write_text(
            caption,
            encoding="utf-8",
        )

    dataset = CaptionDataset(
        data_dir=root,
        resolution=128,
        caption_extension=".caption",
        enable_bucket=True,
        min_bucket_reso=64,
        max_bucket_reso=128,
        bucket_reso_steps=64,
        caption_length_bucket_size=2,
    )
    if len(dataset) != 2:
        raise AssertionError(f"Expected 2 dataset samples, got {len(dataset)}")

    sample0 = dataset.samples[0]
    sample1 = dataset.samples[1]
    if sample0.target_size not in {(128, 64), (64, 128), (128, 128)}:
        raise AssertionError(f"Unexpected target_size for sample0: {sample0.target_size}")
    if sample1.target_size not in {(128, 64), (64, 128), (128, 128)}:
        raise AssertionError(f"Unexpected target_size for sample1: {sample1.target_size}")
    if sample0.caption_token_length <= 0 or sample1.caption_token_length <= 0:
        raise AssertionError("Caption token lengths were not precomputed for bucketing")

    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    seen_captions: list[str] = []
    seen_target_sizes: list[tuple[int, int]] = []
    for batch in dataloader:
        seen_captions.extend(str(caption) for caption in batch["captions"])
        seen_target_sizes.extend(tuple(size) for size in batch["target_sizes"])
    if len(seen_captions) != 2:
        raise AssertionError(f"Dataloader did not yield both caption samples: {seen_captions}")
    if len(seen_target_sizes) != 2:
        raise AssertionError(f"Dataloader did not preserve per-sample target sizes: {seen_target_sizes}")

    print(
        "Newbie dataset basics smoke passed: "
        f"targets={[sample.target_size for sample in dataset.samples]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
