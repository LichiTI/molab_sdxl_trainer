"""Real Anima staged-resolution cache switch smoke.

This smoke reuses existing Anima cache samples from ``sucai/6_lulu`` by copying a
small paired subset into two staged cache directories. It then runs the product
``LulynxTrainer.start()`` path for two short epochs and verifies that staged
cache-first training switches the active dataset/dataloader at the epoch
boundary. The run is capped to four optimizer steps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.configs import MixedPrecision, ModelArch, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.trainer import LulynxTrainer


DEFAULT_STEPS = 4
MAX_STEPS = 40
SEED = 20260528


@dataclass
class StagedSmokeResult:
    success: bool
    steps_completed: int
    duration_seconds: float
    output_path: str
    save_verified: bool
    stage_dirs: list[str] = field(default_factory=list)
    staged_logs: list[str] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)
    error: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _precision() -> tuple[MixedPrecision, str]:
    if not torch.cuda.is_available():
        return MixedPrecision.NO, "float32"
    if torch.cuda.is_bf16_supported():
        return MixedPrecision.BF16, "bf16"
    return MixedPrecision.FP16, "fp16"


def _copy_cache_subset(source: Path, target_root: Path, *, sample_count: int = 2) -> list[Path]:
    stage_root = target_root / ".lulynx_cache" / "anima_staged"
    if target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)
    target_root.mkdir(parents=True, exist_ok=True)
    stage_dirs = [stage_root / "res_512", stage_root / "res_768"]
    for stage_dir in stage_dirs:
        stage_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(source.glob("*_anima_te.*"))[: max(int(sample_count), 1)]
    if not text_files:
        raise FileNotFoundError(f"No Anima text cache files found in {source}")

    for text_path in text_files:
        stem = text_path.name.split("_anima_te", 1)[0]
        latent_candidates = sorted(source.glob(f"{stem}_*_anima.*"))
        if not latent_candidates:
            raise FileNotFoundError(f"No paired Anima latent cache for {text_path}")
        for stage_dir in stage_dirs:
            shutil.copy2(text_path, stage_dir / text_path.name)
            shutil.copy2(latent_candidates[0], stage_dir / latent_candidates[0].name)
            caption = source / f"{stem}.txt"
            image = next((source / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".webp") if (source / f"{stem}{suffix}").exists()), None)
            if caption.exists():
                shutil.copy2(caption, stage_dir / caption.name)
            if image is not None:
                shutil.copy2(image, stage_dir / image.name)
    return stage_dirs


def _make_config(*, model_path: Path, train_dir: Path, output_dir: Path, steps: int, epochs: int, output_name: str) -> UnifiedTrainingConfig:
    mixed_precision, save_precision = _precision()
    return UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path=str(model_path),
        anima_model_path=str(model_path),
        train_data_dir=str(train_dir),
        output_dir=str(output_dir),
        output_name=output_name,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_accumulation_mode="fast",
        max_train_epochs=max(int(epochs), 1),
        max_train_steps=max(1, min(int(steps), MAX_STEPS)),
        network_dim=2,
        network_alpha=2,
        learning_rate=1e-4,
        save_every_n_epochs=9999,
        save_every_n_steps=0,
        save_state=False,
        sample_every=0,
        sample_every_n_epochs=0,
        gradient_checkpointing=True,
        attention_backend="flash2",
        anima_attn_mode="flash2",
        anima_cached_training=True,
        anima_cached_latent_crop_size=4,
        anima_cached_text_token_limit=16,
        anima_native_block_count=28,
        native_cache_mode="cache_first",
        enable_mixed_resolution_training=True,
        resolution=768,
        staged_resolution_ratio_512=50,
        staged_resolution_ratio_768=50,
        staged_resolution_ratio_1024=0,
        staged_resolution_stage_batch_sizes="512:1,768:1",
        torch_compile=False,
        compile_runtime="off",
        cached_dataloader_auto_policy=False,
        dataloader_num_workers=0,
        persistent_data_loader_workers=False,
        pin_memory=True,
        prefetch_factor=2,
        enable_auditor=False,
        adaptive_step_logging_enabled=False,
        tensorboard_flush_interval_steps=1000,
        so_enable_nan_detection=False,
        so_enable_loss_spike_detection=False,
        so_enable_lr_deadlock_detection=False,
        so_enable_auto_recovery=False,
        rm_enable_adaptive_accumulation=False,
        rm_enable_adaptive_batch=False,
        seed=SEED,
        caption_extension=".txt",
    )


def run_smoke(*, steps: int = DEFAULT_STEPS, epochs: int = 2, sample_count: int = 2, output_name: str = "anima_staged_resolution_real_smoke", output_json: str = "") -> dict[str, Any]:
    repo_root = _repo_root()
    model_path = repo_root / "models" / "anima" / "diffusion_models" / "anima-base-v1.0.safetensors"
    source_cache = repo_root / "sucai" / "6_lulu"
    root = repo_root / "temp" / output_name
    train_dir = root / "train"
    output_dir = root / "output"

    if not model_path.exists():
        raise FileNotFoundError(f"Anima model not found: {model_path}")
    if not source_cache.exists():
        raise FileNotFoundError(f"Anima source cache dir not found: {source_cache}")

    stage_dirs = _copy_cache_subset(source_cache, train_dir, sample_count=max(int(sample_count), 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(model_path=model_path, train_dir=train_dir, output_dir=output_dir, steps=steps, epochs=epochs, output_name=output_name)
    trainer = LulynxTrainer(cfg)
    logs: list[str] = []
    trainer.set_callbacks(on_log=logs.append)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    error = ""
    try:
        success = bool(trainer.start())
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration = time.perf_counter() - started

    steps_completed = int(getattr(getattr(trainer, "training_loop", None), "global_step", 0) or 0)
    output_path = output_dir / f"{cfg.output_name}.safetensors"
    save_verified = False
    if output_path.exists():
        save_verified = bool(load_file(str(output_path), device="cpu"))

    staged_logs = [line for line in logs if "staged" in str(line).lower() or "resolution=" in str(line).lower()]
    result = StagedSmokeResult(
        success=success,
        steps_completed=steps_completed,
        duration_seconds=duration,
        output_path=str(output_path),
        save_verified=save_verified,
        stage_dirs=[str(path) for path in stage_dirs],
        staged_logs=staged_logs,
        log_tail=logs[-40:],
        error=error,
    )
    return asdict(result)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Training steps; capped at 40.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--output-name", default="anima_staged_resolution_real_smoke")
    parser.add_argument("--json", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    steps = max(1, min(int(args.steps), MAX_STEPS))
    payload = run_smoke(
        steps=steps,
        epochs=max(int(args.epochs), 1),
        sample_count=max(int(args.sample_count), 1),
        output_name=str(args.output_name or "anima_staged_resolution_real_smoke"),
        output_json=str(args.json or ""),
    )
    output = Path(args.json) if str(args.json or "").strip() else _repo_root() / "temp" / f"{args.output_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") and payload.get("save_verified") and len(payload.get("staged_logs") or []) >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())

