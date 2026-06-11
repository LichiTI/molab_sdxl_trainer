# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Newbie save/training basics on the native cached route."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.newbie_trainer_cached_smoke import (
    _TinyNewbieDenoiser,
    _write_cache,
)
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.model_loader import LoadedModel
from core.lulynx_trainer.trainer import LulynxTrainer


def main() -> int:
    cache_dir = Path("H:/tmp/lulynx_newbie_save_cache")
    output_dir = Path("H:/tmp/lulynx_newbie_save_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_cache(cache_dir)

    cfg = UnifiedTrainingConfig(
        model_type="newbie",
        pretrained_model_name_or_path="H:/tmp/newbie-tiny-placeholder",
        train_data_dir=str(cache_dir),
        output_dir=str(output_dir),
        output_name="newbie_save_smoke",
        mixed_precision="no",
        optimizer_type="AdamW",
        lr_scheduler="constant",
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=2,
        max_train_steps=2,
        network_dim=1,
        network_alpha=1,
        learning_rate=1e-4,
        save_every_n_epochs=1,
        save_every_n_steps=1,
        save_state=True,
        save_state_on_train_end=True,
        save_model_as="pt",
        use_cache=True,
        newbie_target_modules="attention.qkv\nattention.out\nfeed_forward.w1\nfeed_forward.w2\nfeed_forward.w3",
    )

    model = LoadedModel(
        unet=_TinyNewbieDenoiser(),
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        model_arch="newbie",
    )
    model.newbie_scaffold_mode = False
    model.newbie_native_conditioning_ready = True
    model.newbie_transport_ready = True
    model.newbie_forward_smoke_passed = True
    model.newbie_gradient_smoke_passed = True

    injector = LoRAInjector(rank=1, alpha=1, model_arch="newbie")
    injected = injector._inject_model(model.unet, cfg.newbie_target_modules.splitlines(), prefix="unet")
    if not injected:
        raise RuntimeError("Failed to inject LoRA targets for Newbie save smoke")

    trainer = LulynxTrainer(cfg)
    trainer.device = "cpu"
    trainer.dtype = torch.float32
    trainer.model = model
    trainer.lora_injector = injector
    trainer._adapter_cpu_residency = None
    trainer._ensure_native_family_training_ready()

    with patch("torch.cuda.is_available", return_value=False):
        ok = trainer.start()
    if not ok:
        raise RuntimeError("Newbie save/training smoke failed")

    expected_files = {
        "newbie_save_smoke-step000001.pt",
        "newbie_save_smoke-step000001-state.pt",
        "newbie_save_smoke-step000002.pt",
        "newbie_save_smoke-step000002-state.pt",
        "newbie_save_smoke.pt",
        "newbie_save_smoke-last-state.pt",
    }
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = sorted(expected_files - actual_files)
    if missing:
        raise AssertionError(f"Missing expected Newbie save artifacts: {missing}; actual={sorted(actual_files)}")

    if trainer.training_loop is None or trainer.training_loop.global_step != 2:
        raise AssertionError(
            f"Expected cached-Newbie training to complete 2 steps, got {getattr(trainer.training_loop, 'global_step', None)}"
        )

    print(f"Newbie save/training smoke passed: files={sorted(actual_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
