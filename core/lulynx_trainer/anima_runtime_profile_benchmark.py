"""Anima standard vs aggressive runtime profile benchmark.

This script runs a small real cache-first Anima training workload with identical
training parameters and compares only the native runtime profile.  It is meant
for local Pareto checks before promoting aggressive defaults.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.configs import MixedPrecision, ModelArch, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.trainer import LulynxTrainer


@dataclass
class ProfileRunResult:
    label: str
    profile: str
    success: bool
    steps_completed: int
    total_wall_seconds: float
    first_step_ms: float
    mean_step_ms: float
    median_step_ms: float
    steady_mean_step_ms: float
    steady_median_step_ms: float
    samples_per_second: float
    steady_samples_per_second: float
    peak_vram_mb: float
    final_loss: float
    attention_backend: str
    compile_requested: bool
    compile_scope: str
    output_path: str
    output_size_bytes: int
    step_times_ms: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _precision() -> tuple[MixedPrecision, str]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return MixedPrecision.BF16, "bf16"
    if torch.cuda.is_available():
        return MixedPrecision.FP16, "fp16"
    return MixedPrecision.NO, "float32"


def _make_config(
    *,
    profile: str,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    output_name: str,
    steps: int,
    network_dim: int,
    cached_workers: int,
    compile_cache_root: Path,
) -> UnifiedTrainingConfig:
    mixed_precision, save_precision = _precision()
    return UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path=str(model_path),
        anima_model_path=str(model_path),
        train_data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name=output_name,
        native_runtime_profile=profile,
        attention_backend="auto",
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_accumulation_mode="fast",
        max_train_epochs=1,
        max_train_steps=steps,
        network_dim=network_dim,
        network_alpha=network_dim,
        learning_rate=1e-4,
        save_every_n_epochs=1,
        save_every_n_steps=0,
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
        compile_probe_enabled=False,
        compile_anima_full_core_enabled=False,
        compile_contract_strict=True,
        compile_static_shape_drop_last=True,
        compile_require_cache_first=True,
        compile_cache_enabled=True,
        compile_cache_root=str(compile_cache_root),
        compile_cache_reuse=True,
        cached_dataloader_auto_policy=True,
        cached_dataloader_workers=cached_workers,
        dataloader_num_workers=0,
        persistent_data_loader_workers=False,
        pin_memory=True,
        prefetch_factor=2,
        data_transfer_profile_enabled=False,
        enable_auditor=False,
        adaptive_step_logging_enabled=False,
        tensorboard_flush_interval_steps=1000,
        so_enable_nan_detection=False,
        so_enable_loss_spike_detection=False,
        so_enable_lr_deadlock_detection=False,
        so_enable_auto_recovery=False,
        rm_enable_adaptive_accumulation=False,
        rm_enable_adaptive_batch=False,
        seed=1337,
        caption_extension=".txt",
    )


def _run_profile(
    *,
    label: str,
    profile: str,
    model_path: Path,
    data_dir: Path,
    run_root: Path,
    steps: int,
    steady_warmup: int,
    network_dim: int,
    cached_workers: int,
) -> ProfileRunResult:
    output_dir = run_root / label / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_cache_root = run_root / label / "compile_cache"
    compile_cache_root.mkdir(parents=True, exist_ok=True)
    output_name = f"anima_profile_{label}"

    cfg = _make_config(
        profile=profile,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        output_name=output_name,
        steps=steps,
        network_dim=network_dim,
        cached_workers=cached_workers,
        compile_cache_root=compile_cache_root,
    )
    trainer = LulynxTrainer(cfg)
    logs: list[str] = []
    step_times_ms: list[float] = []
    losses: list[float] = []
    peak_vram_mb = 0.0
    original_on_step_end = trainer._on_step_end

    def _on_step_end(step: int, loss: float, info: dict[str, Any]) -> None:
        nonlocal peak_vram_mb
        step_ms = float(info.get("step_wall_seconds", 0.0) or 0.0) * 1000.0
        step_times_ms.append(step_ms)
        losses.append(float(loss))
        if torch.cuda.is_available():
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        original_on_step_end(step, loss, info)

    trainer._on_step_end = _on_step_end  # type: ignore[assignment]
    trainer.set_callbacks(on_log=logs.append)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    success = trainer.start()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    total_wall_seconds = time.perf_counter() - start

    output_path = output_dir / f"{output_name}.safetensors"
    output_size = output_path.stat().st_size if output_path.exists() else 0
    steady = step_times_ms[steady_warmup:] if len(step_times_ms) > steady_warmup else step_times_ms
    mean_step = _mean(step_times_ms)
    steady_mean = _mean(steady)
    plan = getattr(trainer, "runtime_optimization_plan", None)

    return ProfileRunResult(
        label=label,
        profile=profile,
        success=bool(success),
        steps_completed=int(getattr(trainer.training_loop, "global_step", 0) or 0),
        total_wall_seconds=total_wall_seconds,
        first_step_ms=step_times_ms[0] if step_times_ms else 0.0,
        mean_step_ms=mean_step,
        median_step_ms=_median(step_times_ms),
        steady_mean_step_ms=steady_mean,
        steady_median_step_ms=_median(steady),
        samples_per_second=1000.0 / mean_step if mean_step > 0 else 0.0,
        steady_samples_per_second=1000.0 / steady_mean if steady_mean > 0 else 0.0,
        peak_vram_mb=peak_vram_mb,
        final_loss=losses[-1] if losses else 0.0,
        attention_backend=str(getattr(plan, "attention_backend", "unknown") or "unknown"),
        compile_requested=bool(getattr(plan, "torch_compile", False)),
        compile_scope=str(getattr(plan, "torch_compile_scope", "") or ""),
        output_path=str(output_path),
        output_size_bytes=output_size,
        step_times_ms=step_times_ms,
        losses=losses,
        log_tail=logs[-40:],
    )


def _compare(old: ProfileRunResult, new: ProfileRunResult) -> dict[str, float]:
    return {
        "mean_step_speedup": (old.mean_step_ms or 1.0) / max(new.mean_step_ms or 1.0, 1e-9),
        "steady_step_speedup": (old.steady_mean_step_ms or 1.0) / max(new.steady_mean_step_ms or 1.0, 1e-9),
        "end_to_end_speedup": (old.total_wall_seconds or 1.0) / max(new.total_wall_seconds or 1.0, 1e-9),
        "peak_vram_ratio": (new.peak_vram_mb or 0.0) / max(old.peak_vram_mb or 1.0, 1e-9),
        "peak_vram_delta_mb": (new.peak_vram_mb or 0.0) - (old.peak_vram_mb or 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--steady-warmup", type=int, default=2)
    parser.add_argument("--network-dim", type=int, default=1)
    parser.add_argument("--cached-workers", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo = _repo_root()
    model_path = args.model or repo / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = args.data or repo / "sucai" / "6_lulu"
    run_root = args.out or repo / "temp" / "anima_runtime_profile_benchmark" / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Anima cached data not found: {data_dir}")

    print(
        f"Running Anima profile benchmark: steps={args.steps} steady_warmup={args.steady_warmup} "
        f"model={model_path.name}",
        flush=True,
    )
    standard = _run_profile(
        label="standard",
        profile="standard",
        model_path=model_path,
        data_dir=data_dir,
        run_root=run_root,
        steps=max(args.steps, 1),
        steady_warmup=max(args.steady_warmup, 0),
        network_dim=max(args.network_dim, 1),
        cached_workers=max(args.cached_workers, 0),
    )
    aggressive = _run_profile(
        label="aggressive",
        profile="aggressive",
        model_path=model_path,
        data_dir=data_dir,
        run_root=run_root,
        steps=max(args.steps, 1),
        steady_warmup=max(args.steady_warmup, 0),
        network_dim=max(args.network_dim, 1),
        cached_workers=max(args.cached_workers, 0),
    )

    payload = {
        "benchmark": {
            "route": "anima",
            "steps": max(args.steps, 1),
            "steady_warmup": max(args.steady_warmup, 0),
            "model": str(model_path),
            "data": str(data_dir),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "runs": {
            "standard": asdict(standard),
            "aggressive": asdict(aggressive),
        },
        "comparison": _compare(standard, aggressive),
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def line(run: ProfileRunResult) -> str:
        return (
            f"[benchmark] label={run.label} profile={run.profile} success={run.success} "
            f"steps={run.steps_completed} peak_vram={run.peak_vram_mb:.1f}MB "
            f"avg_step_time={run.mean_step_ms:.2f}ms steady_step_time={run.steady_mean_step_ms:.2f}ms "
            f"samples_per_sec={run.samples_per_second:.3f} steady_samples_per_sec={run.steady_samples_per_second:.3f} "
            f"attention={run.attention_backend} compile={run.compile_scope or 'off'}"
        )

    print(line(standard), flush=True)
    print(line(aggressive), flush=True)
    print(f"[benchmark] comparison={json.dumps(payload['comparison'], sort_keys=True)}", flush=True)
    print(f"[benchmark] summary={summary_path}", flush=True)
    return 0 if standard.success and aggressive.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
