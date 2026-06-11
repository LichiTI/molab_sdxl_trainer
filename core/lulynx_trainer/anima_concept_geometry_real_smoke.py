"""Real local Concept Geometry smoke on the cache-first Anima trainer path.

This uses the repository's local ``models/anima`` and ``sucai/6_lulu`` assets
to validate that the trainer can:

1. load a real Anima checkpoint
2. consume a real cached Anima dataset
3. enable Concept Geometry geometry-aware sampling/weighting
4. train a single tiny cached step
5. save an adapter artifact
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import ModelArch, MixedPrecision, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.concept_geometry_prep import build_concept_geometry
from core.lulynx_trainer.trainer import LulynxTrainer


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    geometry_path = data_dir / "concept_geometry.json"
    output_dir = repo_root / "tmp" / "lulynx_anima_concept_geometry_real_smoke"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {checkpoint}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Anima cached data not found: {data_dir}")

    if not geometry_path.exists():
        build_concept_geometry(data_dir, backend="lexical")

    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path=str(checkpoint),
        anima_model_path=str(checkpoint),
        train_data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name="anima_concept_geometry_real_smoke",
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
        concept_geometry_enabled=True,
        concept_geometry_path=str(geometry_path),
        concept_geometry_sampler_mode="density_curriculum",
        concept_geometry_loss_weighting=True,
        concept_geometry_density_power=1.0,
        seed=42,
    )

    logs: list[str] = []
    trainer = LulynxTrainer(cfg)
    trainer.set_callbacks(on_log=logs.append)
    ok = trainer.start()
    if not ok:
        raise RuntimeError("Anima Concept Geometry real smoke failed")

    expected = output_dir / "anima_concept_geometry_real_smoke.safetensors"
    if not expected.exists():
        raise FileNotFoundError(f"Expected adapter was not saved: {expected}")

    matched_logs = [
        line for line in logs
        if "Concept Geometry" in str(line).lower() or "geometry" in str(line).lower()
    ]
    print(
        "Anima Concept Geometry real smoke passed: "
        f"saved={expected} "
        f"geometry_logs={len(matched_logs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


