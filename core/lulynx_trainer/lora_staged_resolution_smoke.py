"""Smoke test for LoRA staged resolution dataset rebuilds."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

from core.configs import ModelArch, UnifiedTrainingConfig
from core.lulynx_trainer.staged_resolution import stages_to_summary
from core.lulynx_trainer.trainer import LulynxTrainer


def _write_dataset(root: Path) -> None:
    for index, size in enumerate(((640, 640), (960, 960)), start=1):
        image_path = root / f"sample_{index}.png"
        Image.new("RGB", size, (32 * index, 64, 96)).save(image_path)
        image_path.with_suffix(".txt").write_text(f"test sample {index}", encoding="utf-8")


def test_lora_staged_resolution_rebuilds_caption_dataset() -> None:
    with tempfile.TemporaryDirectory(prefix="lulynx_lora_staged_") as tmp:
        data_dir = Path(tmp) / "data"
        out_dir = Path(tmp) / "out"
        data_dir.mkdir()
        out_dir.mkdir()
        _write_dataset(data_dir)

        cfg = UnifiedTrainingConfig(
            model_type=ModelArch.SDXL,
            train_data_dir=str(data_dir),
            output_dir=str(out_dir),
            resolution=768,
            train_batch_size=1,
            max_train_epochs=2,
            enable_bucket=True,
            min_bucket_reso=256,
            max_bucket_reso=1024,
            bucket_reso_steps=64,
            enable_mixed_resolution_training=True,
            staged_resolution_ratio_512=50,
            staged_resolution_ratio_768=50,
            staged_resolution_stage_batch_sizes="512:2,768:1",
            cache_latents=False,
            cache_text_encoder_outputs=False,
            validation_split=0.0,
        )
        trainer = LulynxTrainer(cfg)
        trainer._resolve_data_backend_profile(
            model_arch="sdxl",
            anima_cached_training=False,
            newbie_cached_training=False,
        )

        plan = trainer._build_lora_staged_resolution_plan(
            model_arch="sdxl",
            anima_cached_training=False,
            newbie_cached_training=False,
        )
        assert [stage.resolution for stage in plan] == [512, 768], stages_to_summary(plan)

        first_index, first_stage = trainer._select_lora_staged_resolution_stage(0)
        assert first_stage is not None
        trainer._set_lora_stage_resolution(first_stage.resolution)
        first_input = trainer._create_caption_training_input(
            data_dir=str(data_dir),
            model_arch="sdxl",
            batch_size=first_stage.batch_size or cfg.batch_size,
            drop_last=False,
        )
        first_sizes = {sample.target_size for sample in first_input.dataset.samples}
        assert first_index == 0
        assert first_sizes == {(512, 512)}

        second_index, second_stage = trainer._select_lora_staged_resolution_stage(1)
        assert second_stage is not None
        trainer._set_lora_stage_resolution(second_stage.resolution)
        second_input = trainer._create_caption_training_input(
            data_dir=str(data_dir),
            model_arch="sdxl",
            batch_size=second_stage.batch_size or cfg.batch_size,
            drop_last=False,
        )
        second_sizes = {sample.target_size for sample in second_input.dataset.samples}
        assert second_index == 1
        assert second_sizes == {(768, 768)}
        assert first_sizes != second_sizes


if __name__ == "__main__":
    test_lora_staged_resolution_rebuilds_caption_dataset()
    print("PASS: LoRA staged resolution rebuilds CaptionDataset target_size")
