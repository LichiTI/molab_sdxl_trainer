"""Real Anima compile/runtime strategy benchmark.

Runs the native cache-first Anima trainer against local production assets and
compares selectable compile/runtime strategies. The default run stays capped at
40 steps and writes a JSON summary under ``temp/anima_compile_benchmark``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
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


MAX_BENCH_STEPS = 40
DEFAULT_BENCH_STEPS = 40
STEADY_STATE_WARMUP_STEPS = 5
BENCH_LATENT_CROP = 4
FIXED_TEXT_TOKENS = 64
FIXED_VISUAL_TOKENS = 4
SEED = 1337


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    compile_runtime: str
    compile_shape_strategy: str
    compile_target_strategy: str
    torch_compile: bool | None
    torch_compile_scope: str = ""
    compile_cache_enabled: bool | None = False
    description: str = ""


CASES: dict[str, BenchmarkCase] = {
    "off": BenchmarkCase(
        label="off",
        compile_runtime="off",
        compile_shape_strategy="auto",
        compile_target_strategy="auto",
        torch_compile=False,
        description="Eager baseline; mirrors compile_runtime=off.",
    ),
    "compile_fixed_block": BenchmarkCase(
        label="compile_fixed_block",
        compile_runtime="compile_cache",
        compile_shape_strategy="fixed_pad",
        compile_target_strategy="block",
        torch_compile=True,
        torch_compile_scope="per_block",
        compile_cache_enabled=True,
        description="Compile cache with fixed token padding and block-level targets.",
    ),
    "compile_token_inner": BenchmarkCase(
        label="compile_token_inner",
        compile_runtime="compile_cache",
        compile_shape_strategy="token_flatten",
        compile_target_strategy="inner_forward",
        torch_compile=True,
        torch_compile_scope="per_block",
        compile_cache_enabled=True,
        description="Compile cache using native token buckets and stable inner forward callables.",
    ),
    "auto": BenchmarkCase(
        label="auto",
        compile_runtime="auto",
        compile_shape_strategy="auto",
        compile_target_strategy="auto",
        torch_compile=None,
        compile_cache_enabled=None,
        description="Route-aware compile_runtime=auto resolution; explicit fields stay unset.",
    ),
}
DEFAULT_CASES = ("off", "compile_token_inner")


@dataclass
class BenchmarkRunResult:
    label: str
    description: str
    requested_compile_runtime: str
    requested_shape_strategy: str
    requested_target_strategy: str
    requested_torch_compile: bool
    requested_compile_scope: str
    success: bool
    steps_completed: int
    total_wall_seconds: float
    first_step_ms: float
    mean_step_ms: float
    median_step_ms: float
    steady_mean_step_ms: float
    steady_median_step_ms: float
    peak_vram_mb: float
    final_loss: float
    output_path: str
    output_size_bytes: int
    save_verified: bool
    attention_backend: str = "unknown"
    effective_compile_runtime: dict[str, Any] = field(default_factory=dict)
    runtime_plan: dict[str, Any] = field(default_factory=dict)
    step_times_ms: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    compile_cache_root: str = ""
    log_tail: list[str] = field(default_factory=list)
    error: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _precision_triplet() -> tuple[MixedPrecision, str, torch.dtype]:
    if not torch.cuda.is_available():
        return MixedPrecision.NO, "float32", torch.float32
    if torch.cuda.is_bf16_supported():
        return MixedPrecision.BF16, "bf16", torch.bfloat16
    return MixedPrecision.FP16, "fp16", torch.float16


def _make_config(
    *,
    case: BenchmarkCase,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    compile_cache_root: Path,
    output_name: str,
    steps: int,
) -> UnifiedTrainingConfig:
    mixed_precision, save_precision, _dtype = _precision_triplet()
    cfg_kwargs: dict[str, Any] = dict(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path=str(model_path),
        anima_model_path=str(model_path),
        train_data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name=output_name,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_accumulation_mode="fast",
        max_train_epochs=1,
        max_train_steps=steps,
        network_dim=4,
        network_alpha=4,
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
        anima_cached_latent_crop_size=BENCH_LATENT_CROP,
        anima_cached_text_token_limit=0,
        anima_native_block_count=28,
        anima_fixed_text_tokens=FIXED_TEXT_TOKENS,
        anima_fixed_visual_tokens=FIXED_VISUAL_TOKENS,
        compile_runtime=case.compile_runtime,
        compile_shape_strategy=case.compile_shape_strategy,
        compile_target_strategy=case.compile_target_strategy,
        compile_probe_enabled=False,
        compile_contract_strict=True,
        compile_static_shape_drop_last=True,
        compile_require_cache_first=True,
        newbie_safe_fallback=False,
        compile_cache_root=str(compile_cache_root),
        compile_cache_reuse=True,
        compile_cache_prewarm=False,
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
    if case.torch_compile is not None:
        cfg_kwargs["torch_compile"] = case.torch_compile
    if case.torch_compile_scope:
        cfg_kwargs["torch_compile_scope"] = case.torch_compile_scope
        cfg_kwargs["anima_compile_scope"] = case.torch_compile_scope
    if case.compile_cache_enabled is not None:
        cfg_kwargs["compile_cache_enabled"] = case.compile_cache_enabled
    return UnifiedTrainingConfig(**cfg_kwargs)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _runtime_plan_payload(trainer: LulynxTrainer) -> dict[str, Any]:
    plan = getattr(trainer, "runtime_optimization_plan", None)
    if plan is None:
        return {}
    payload: dict[str, Any] = {}
    for key in (
        "attention_backend",
        "torch_compile",
        "torch_compile_scope",
        "torch_compile_dynamic",
        "compile_shape_strategy",
        "compile_target_strategy",
        "compile_cache_enabled",
    ):
        payload[key] = getattr(plan, key, None)
    payload["reasons_tail"] = list(getattr(plan, "reasons", []) or [])[-20:]
    payload["warnings_tail"] = list(getattr(plan, "warnings", []) or [])[-20:]
    return payload


def _run_once(*, case: BenchmarkCase, steps: int, bench_root: Path, keep_existing: bool = False) -> BenchmarkRunResult:
    repo_root = _repo_root()
    model_path = repo_root / "models" / "anima" / "diffusion_models" / "anima-base-v1.0.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    output_dir = bench_root / case.label / "output"
    compile_cache_root = bench_root / case.label / "compile_cache"

    if not model_path.exists():
        raise FileNotFoundError(f"Anima model not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Anima data dir not found: {data_dir}")

    if output_dir.parent.exists() and not keep_existing:
        shutil.rmtree(output_dir.parent, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_cache_root.mkdir(parents=True, exist_ok=True)

    cfg = _make_config(
        case=case,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        compile_cache_root=compile_cache_root,
        output_name=f"anima_compile_bench_{case.label}",
        steps=steps,
    )
    trainer = LulynxTrainer(cfg)

    logs: list[str] = []
    step_times_ms: list[float] = []
    losses: list[float] = []
    peak_vram_mb_seen = 0.0

    original_on_step_end = trainer._on_step_end

    def _wrapped_on_step_end(step: int, loss: float, info: dict[str, Any]) -> None:
        nonlocal peak_vram_mb_seen
        step_wall_ms = float(info.get("step_wall_seconds", 0.0) or 0.0) * 1000.0
        step_times_ms.append(step_wall_ms)
        losses.append(float(loss))
        if torch.cuda.is_available():
            peak_vram_mb_seen = max(
                peak_vram_mb_seen,
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            )
        original_on_step_end(step, loss, info)

    trainer._on_step_end = _wrapped_on_step_end  # type: ignore[assignment]
    trainer.set_callbacks(on_log=logs.append)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    error = ""
    try:
        success = bool(trainer.start())
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb_seen = max(
            peak_vram_mb_seen,
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        )
    total_wall_seconds = time.perf_counter() - start

    output_path = output_dir / f"{cfg.output_name}.safetensors"
    save_verified = False
    output_size = 0
    if output_path.exists():
        output_size = output_path.stat().st_size
        loaded = load_file(str(output_path), device="cpu")
        save_verified = bool(loaded)

    steady = step_times_ms[STEADY_STATE_WARMUP_STEPS:] if len(step_times_ms) > STEADY_STATE_WARMUP_STEPS else step_times_ms
    steps_completed = int(getattr(getattr(trainer, "training_loop", None), "global_step", 0) or 0)
    runtime_profile = dict(getattr(trainer, "_compile_runtime_profile", {}) or {})
    runtime_plan = _runtime_plan_payload(trainer)

    return BenchmarkRunResult(
        label=case.label,
        description=case.description,
        requested_compile_runtime=case.compile_runtime,
        requested_shape_strategy=case.compile_shape_strategy,
        requested_target_strategy=case.compile_target_strategy,
        requested_torch_compile=case.torch_compile,
        requested_compile_scope=case.torch_compile_scope,
        success=success,
        steps_completed=steps_completed,
        total_wall_seconds=total_wall_seconds,
        first_step_ms=step_times_ms[0] if step_times_ms else 0.0,
        mean_step_ms=_mean(step_times_ms),
        median_step_ms=_median(step_times_ms),
        steady_mean_step_ms=_mean(steady),
        steady_median_step_ms=_median(steady),
        peak_vram_mb=peak_vram_mb_seen,
        final_loss=losses[-1] if losses else 0.0,
        output_path=str(output_path),
        output_size_bytes=output_size,
        save_verified=save_verified,
        attention_backend=str(runtime_plan.get("attention_backend") or "unknown"),
        effective_compile_runtime=runtime_profile,
        runtime_plan=runtime_plan,
        step_times_ms=step_times_ms,
        losses=losses,
        compile_cache_root=str(compile_cache_root),
        log_tail=logs[-40:],
        error=error,
    )


def _comparison_payload(baseline: BenchmarkRunResult, run: BenchmarkRunResult) -> dict[str, float]:
    base_mean = baseline.mean_step_ms or 1.0
    run_mean = run.mean_step_ms or 1.0
    base_steady = baseline.steady_mean_step_ms or 1.0
    run_steady = run.steady_mean_step_ms or 1.0
    base_wall = baseline.total_wall_seconds or 1.0
    run_wall = run.total_wall_seconds or 1.0
    return {
        "all_step_speedup_vs_baseline": base_mean / run_mean,
        "steady_state_speedup_vs_baseline": base_steady / run_steady,
        "end_to_end_speedup_vs_baseline": base_wall / run_wall,
        "first_step_ratio_vs_baseline": (run.first_step_ms or 0.0) / max(baseline.first_step_ms or 1.0, 1e-9),
        "peak_vram_ratio_vs_baseline": (run.peak_vram_mb or 0.0) / max(baseline.peak_vram_mb or 1.0, 1e-9),
    }


def _parse_case_names(raw: str) -> list[str]:
    if raw.strip().lower() in {"all", "*"}:
        return list(CASES.keys())
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        available = ", ".join(CASES)
        raise ValueError(f"Unknown case(s): {', '.join(unknown)}. Available: {available}")
    return names or list(DEFAULT_CASES)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real Anima compile/runtime strategy benchmark.")
    parser.add_argument("--steps", type=int, default=DEFAULT_BENCH_STEPS, help="Training steps per case; capped at 40.")
    parser.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help="Comma-separated case names, or 'all'. Available: " + ", ".join(CASES),
    )
    parser.add_argument("--list-cases", action="store_true", help="Print available cases and exit.")
    parser.add_argument("--output-root", default="", help="Optional benchmark output root.")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing case output/cache directories to probe cache reuse.")
    return parser.parse_args(argv)


def _line(run: BenchmarkRunResult) -> str:
    profile = run.effective_compile_runtime or {}
    effective_shape = profile.get("compile_shape_strategy") or run.runtime_plan.get("compile_shape_strategy") or "n/a"
    effective_target = profile.get("compile_target_strategy") or run.runtime_plan.get("compile_target_strategy") or "n/a"
    static_source = profile.get("effective_static_shape_source") or "n/a"
    return (
        f"[{run.label}] success={run.success} steps={run.steps_completed} "
        f"wall={run.total_wall_seconds:.2f}s first={run.first_step_ms:.2f}ms "
        f"mean={run.mean_step_ms:.2f}ms steady_mean={run.steady_mean_step_ms:.2f}ms "
        f"peak={run.peak_vram_mb:.1f}MB saved={run.save_verified} "
        f"shape={effective_shape} target={effective_target} static={static_source}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_cases:
        for case in CASES.values():
            print(
                f"{case.label}: runtime={case.compile_runtime}, shape={case.compile_shape_strategy}, "
                f"target={case.compile_target_strategy}, torch_compile={case.torch_compile} - {case.description}",
                flush=True,
            )
        return 0

    steps = max(1, min(int(args.steps), MAX_BENCH_STEPS))
    case_names = _parse_case_names(str(args.cases))
    bench_root = Path(args.output_root) if args.output_root else _repo_root() / "temp" / "anima_compile_benchmark"
    bench_root.mkdir(parents=True, exist_ok=True)

    print(
        "Running Anima real benchmark: "
        f"steps={steps}, cases={','.join(case_names)}, fixed_text={FIXED_TEXT_TOKENS}, fixed_visual={FIXED_VISUAL_TOKENS}",
        flush=True,
    )

    runs: dict[str, BenchmarkRunResult] = {}
    for name in case_names:
        case = CASES[name]
        print(
            f"Starting case {case.label}: runtime={case.compile_runtime}, "
            f"shape={case.compile_shape_strategy}, target={case.compile_target_strategy}",
            flush=True,
        )
        result = _run_once(case=case, steps=steps, bench_root=bench_root, keep_existing=bool(args.keep_existing))
        runs[case.label] = result
        print(_line(result), flush=True)
        if result.error:
            print(f"[{case.label}] error={result.error}", flush=True)

    baseline = runs.get("off") or next(iter(runs.values()))
    comparisons = {
        label: _comparison_payload(baseline, run)
        for label, run in runs.items()
        if label != baseline.label
    }
    payload = {
        "benchmark": {
            "route": "anima",
            "steps": steps,
            "max_steps_cap": MAX_BENCH_STEPS,
            "steady_state_warmup_steps": STEADY_STATE_WARMUP_STEPS,
            "seed": SEED,
            "fixed_text_tokens": FIXED_TEXT_TOKENS,
            "fixed_visual_tokens": FIXED_VISUAL_TOKENS,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "cases": case_names,
        },
        "runs": {label: asdict(run) for label, run in runs.items()},
        "comparison_baseline": baseline.label,
        "comparisons": comparisons,
    }

    summary_path = bench_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}", flush=True)
    return 0 if all(run.success for run in runs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
