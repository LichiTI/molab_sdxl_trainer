"""Trainer-level cache-first Anima smoke.

This validates the production LulynxTrainer path, not just the standalone
native DiT module:

- load Anima single-file checkpoint
- swap in the native executable DiT when paired cache files exist
- inject LoRA
- train one cached latent/text batch
- save a LoRA adapter

The crop/token limits keep this safe for local CPU verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import ModelArch, MixedPrecision, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.trainer import LulynxTrainer


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    output_dir = Path("H:/tmp/lulynx_anima_trainer_cached_smoke")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {checkpoint}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Anima cached data not found: {data_dir}")

    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path=str(checkpoint),
        anima_model_path=str(checkpoint),
        train_data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name="anima_trainer_cached_smoke",
        mixed_precision=MixedPrecision.NO,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=1,
        max_train_steps=1,
        network_dim=1,
        network_alpha=1,
        learning_rate=1e-4,
        save_every_n_epochs=1,
        save_state=False,
        sample_every=0,
        sample_every_n_epochs=0,
        gradient_checkpointing=True,
        anima_cached_training=True,
        anima_cached_latent_crop_size=4,
        anima_cached_text_token_limit=16,
        anima_native_block_count=28,
    )

    logs: list[str] = []
    trainer = LulynxTrainer(cfg)
    trainer.set_callbacks(on_log=logs.append)
    ok = trainer.start()
    if not ok:
        raise RuntimeError("Anima trainer cached smoke failed")

    expected = output_dir / "anima_trainer_cached_smoke.safetensors"
    if not expected.exists():
        raise FileNotFoundError(f"Expected adapter was not saved: {expected}")

    print(f"Anima trainer cached smoke passed: saved={expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
