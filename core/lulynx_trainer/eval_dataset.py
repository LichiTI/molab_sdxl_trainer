# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Independent evaluation dataset helpers.

Eval datasets should be stable and explainable: no caption shuffle, no caption
dropout, no random tag dropout, and no image augmentation.  This keeps
generalization checks independent from training-only stochastic transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .dataset_loader import CaptionDataset, create_dataloader


@dataclass(frozen=True)
class EvalDatasetConfig:
    data_dir: str
    resolution: Union[int, str]
    caption_extension: str = ".txt"
    enable_bucket: bool = True
    min_bucket_reso: int = 512
    max_bucket_reso: int = 2048
    bucket_reso_steps: int = 64
    bucket_selection_mode: str = "aspect"
    bucket_custom_resos: Optional[Any] = None
    keep_tokens: int = 0
    keep_tokens_separator: str = ""
    weighted_captions: bool = False
    masked_loss: bool = False
    alpha_mask: bool = False
    caption_length_bucket_size: int = 0


def create_eval_caption_dataset(config: EvalDatasetConfig) -> CaptionDataset:
    """Create a deterministic caption dataset for eval/validation."""

    data_dir = Path(str(config.data_dir or ""))
    if not data_dir.is_dir():
        raise FileNotFoundError(f"eval_data_dir does not exist: {data_dir}")

    dataset = CaptionDataset(
        data_dir=str(data_dir),
        resolution=config.resolution,
        caption_extension=config.caption_extension,
        enable_bucket=config.enable_bucket,
        min_bucket_reso=config.min_bucket_reso,
        max_bucket_reso=config.max_bucket_reso,
        bucket_reso_steps=config.bucket_reso_steps,
        bucket_selection_mode=config.bucket_selection_mode,
        bucket_custom_resos=config.bucket_custom_resos,
        shuffle_caption=False,
        shuffle_caption_tags_only=False,
        keep_tokens=config.keep_tokens,
        keep_tokens_separator=config.keep_tokens_separator,
        flip_augment=False,
        color_augment=False,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        tag_dropout_rate=0.0,
        caption_tag_dropout_targets="",
        caption_tag_dropout_target_mode="drop_all",
        caption_tag_dropout_target_count=1,
        token_warmup_min=0,
        token_warmup_max=0,
        token_warmup_steps=0,
        weighted_captions=config.weighted_captions,
        masked_loss=config.masked_loss,
        alpha_mask=config.alpha_mask,
        caption_length_bucket_size=config.caption_length_bucket_size,
    )
    return dataset


def create_eval_dataloader(
    dataset: CaptionDataset,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    """Create a non-shuffled DataLoader for eval."""

    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
