"""Real-model smoke for the experimental Anima full-core compile path.

This keeps the run intentionally small:

- cache-first native Anima executable subset
- `compile_anima_full_core_enabled=true`
- fixed token budgets for static-shape friendliness
- one optimizer step
- save adapter and print compile-related log lines
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
    output_dir = Path("H:/tmp/lulynx_anima_full_core_compile_smoke")

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
        output_name="anima_full_core_compile_smoke",
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
        gradient_checkpointing=False,
        newbie_safe_fallback=False,
        anima_cached_training=True,
        anima_cached_latent_crop_size=4,
        anima_cached_text_token_limit=16,
        anima_fixed_text_tokens=16,
        anima_fixed_visual_tokens=4,
        anima_native_block_count=28,
        anima_compile_scope="full_cudagraph",
        torch_compile=False,
        torch_compile_scope="",
        compile_probe_enabled=False,
        compile_anima_full_core_enabled=True,
        compile_contract_strict=True,
        compile_static_shape_drop_last=True,
        compile_require_cache_first=True,
        cached_dataloader_auto_policy=False,
        dataloader_num_workers=0,
        persistent_data_loader_workers=False,
    )

    logs: list[str] = []
    trainer = LulynxTrainer(cfg)
    trainer.set_callbacks(on_log=logs.append)
    ok = trainer.start()
    if not ok:
        raise RuntimeError("Anima full-core compile smoke failed")

    expected = output_dir / "anima_full_core_compile_smoke.safetensors"
    if not expected.exists():
        raise FileNotFoundError(f"Expected adapter was not saved: {expected}")

    for line in logs:
        text = str(line)
        if any(token in text.lower() for token in ("compile", "cache-contract", "runtime-opt", "dataloader-policy")):
            print(text)

    print(f"Anima full-core compile smoke passed: saved={expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
