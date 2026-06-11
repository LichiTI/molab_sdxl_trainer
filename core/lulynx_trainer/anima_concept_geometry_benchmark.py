"""Real local benchmark: baseline random sampler vs Concept Geometry Sampling on Anima cache-first.

This is a trainer-side benchmark for the local Concept Geometry MVP. It keeps the model,
dataset, optimizer, and training route identical, and changes only the dataset
sampling / weighting strategy:

- baseline: normal shuffled Anima cache-first training
- concept_geometry: concept geometry sampling + optional loss weighting
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.safe_pickle import safe_torch_load
from core.configs import MixedPrecision, ModelArch, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.adapter_evaluator import evaluate_adapter
from core.lulynx_trainer.dataset_discovery import discover_smart_subsets
from core.lulynx_trainer.dataset_inspector import InspectorOptions, inspect_dataset
from core.lulynx_trainer.concept_geometry_prep import build_concept_geometry
from core.lulynx_trainer.trainer import LulynxTrainer


@dataclass
class ConceptGeometryBenchRunResult:
    label: str
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
    peak_torch_allocated_mb: float
    peak_torch_reserved_mb: float
    initial_loss: float
    final_loss: float
    loss_delta: float
    output_path: str
    output_size_bytes: int
    attention_backend: str
    concept_geometry_enabled: bool
    concept_geometry_sampler_mode: str
    concept_geometry_loss_weighting: bool
    step_times_ms: list[float] = field(default_factory=list)
    layer_monitor_times_ms: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)
    adapter_evaluation: dict[str, object] = field(default_factory=dict)


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


def _estimate_cached_sample_count(data_dir: Path) -> int:
    stems: set[str] = set()
    for subset in discover_smart_subsets(data_dir):
        for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
            for path in subset.root.glob(f"*{suffix}"):
                stems.add(f"{subset.root.as_posix()}::{path.name[: -len(suffix)]}")
    return max(len(stems), 1)


def _infer_cached_text_tokens(data_dir: Path, *, sample_limit: int = 256) -> int:
    max_tokens = 0
    inspected = 0
    for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
        for path in sorted(data_dir.rglob(f"*{suffix}")):
            if inspected >= sample_limit:
                return max_tokens
            inspected += 1
            try:
                if path.suffix.lower() == ".npz":
                    import numpy as np
                    with np.load(str(path), mmap_mode="r") as data:
                        if "prompt_embeds" in data.files:
                            max_tokens = max(max_tokens, int(data["prompt_embeds"].shape[0]))
                elif path.suffix.lower() == ".safetensors":
                    from safetensors import safe_open
                    with safe_open(str(path), framework="pt", device="cpu") as data:
                        if "prompt_embeds" in data.keys():
                            max_tokens = max(max_tokens, int(data.get_slice("prompt_embeds").get_shape()[0]))
                elif path.suffix.lower() == ".pt":
                    payload = safe_torch_load(str(path), map_location="cpu")
                    if isinstance(payload, dict) and "prompt_embeds" in payload:
                        max_tokens = max(max_tokens, int(payload["prompt_embeds"].shape[0]))
            except Exception:
                continue
    return max_tokens


def _make_config(
    *,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    output_name: str,
    steps: int,
    batch_size: int,
    network_dim: int,
    latent_crop_size: int,
    cached_text_token_limit: int,
    fixed_text_tokens: int,
    fixed_visual_tokens: int,
    attention_backend: str,
    mn_lora_enabled: bool,
    mn_lora_preset: str,
    concept_geometry_enabled: bool,
    concept_geometry_path: Path,
    concept_geometry_sampler_mode: str,
    concept_geometry_loss_weighting: bool,
    cached_workers: int,
    gradient_checkpointing: bool,
    cpu_offload_checkpointing: bool,
    fused_optimizer: bool,
    blocks_to_swap: int,
    swap_granularity: str,
    swap_ratio: float,
    swap_count: int,
    module_offload_enabled: bool,
    module_offload_ratio: int,
    layer_monitor_enabled: bool,
    layer_monitor_interval: int,
    layer_monitor_mode: str,
    layer_monitor_sample_size: int,
    layer_monitor_max_layers: int,
    seed: int,
) -> UnifiedTrainingConfig:
    mixed_precision, save_precision = _precision()
    effective_batch_size = max(int(batch_size or 1), 1)
    sample_count = _estimate_cached_sample_count(data_dir)
    steps_per_epoch = max(int(math.ceil(sample_count / effective_batch_size)), 1)
    required_epochs = max(int(math.ceil(max(int(steps or 1), 1) / steps_per_epoch)), 1)
    return UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path=str(model_path),
        anima_model_path=str(model_path),
        train_data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name=output_name,
        native_runtime_profile="standard",
        attention_backend=str(attention_backend or "auto"),
        anima_attn_mode=str(attention_backend or "auto"),
        mn_lora_enabled=bool(mn_lora_enabled),
        mn_lora_preset=str(mn_lora_preset or "slim"),
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        optimizer_type=OptimizerType.ADAMW,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=effective_batch_size,
        gradient_accumulation_steps=1,
        gradient_accumulation_mode="fast",
        max_train_epochs=required_epochs,
        max_train_steps=steps,
        network_dim=network_dim,
        network_alpha=network_dim,
        learning_rate=1e-4,
        save_every_n_epochs=1,
        save_every_n_steps=0,
        save_state=False,
        sample_every=0,
        sample_every_n_epochs=0,
        gradient_checkpointing=bool(gradient_checkpointing),
        newbie_safe_fallback=False,
        anima_cached_training=True,
        anima_cached_latent_crop_size=max(int(latent_crop_size or 0), 0),
        anima_cached_text_token_limit=max(int(cached_text_token_limit or 0), 0),
        anima_fixed_text_tokens=max(int(fixed_text_tokens or 0), 0),
        anima_fixed_visual_tokens=max(int(fixed_visual_tokens or 0), 0),
        anima_native_block_count=28,
        compile_probe_enabled=False,
        compile_anima_full_core_enabled=False,
        compile_contract_strict=True,
        compile_static_shape_drop_last=True,
        compile_require_cache_first=True,
        cached_dataloader_auto_policy=True,
        cached_dataloader_workers=cached_workers,
        dataloader_num_workers=0,
        persistent_data_loader_workers=False,
        pin_memory=True,
        prefetch_factor=2,
        data_transfer_profile_enabled=False,
        layer_monitor_enabled=bool(layer_monitor_enabled),
        layer_monitor_interval=max(int(layer_monitor_interval or 1), 1),
        layer_monitor_mode=str(layer_monitor_mode or "sampled"),
        layer_monitor_sample_size=max(int(layer_monitor_sample_size or 4096), 128),
        layer_monitor_max_layers=max(int(layer_monitor_max_layers or 0), 0),
        cpu_offload_checkpointing=bool(cpu_offload_checkpointing),
        fused_optimizer=bool(fused_optimizer),
        blocks_to_swap=max(int(blocks_to_swap or 0), 0),
        swap_granularity=str(swap_granularity or "off"),
        swap_ratio=max(float(swap_ratio or 0.0), 0.0),
        swap_count=max(int(swap_count or 0), 0),
        module_offload_enabled=bool(module_offload_enabled),
        module_offload_ratio=max(int(module_offload_ratio or 0), 0),
        enable_auditor=False,
        adaptive_step_logging_enabled=False,
        tensorboard_flush_interval_steps=1000,
        so_enable_nan_detection=False,
        so_enable_loss_spike_detection=False,
        so_enable_lr_deadlock_detection=False,
        so_enable_auto_recovery=False,
        rm_enable_adaptive_accumulation=False,
        rm_enable_adaptive_batch=False,
        seed=seed,
        caption_extension=".txt",
        concept_geometry_enabled=concept_geometry_enabled,
        concept_geometry_path=str(concept_geometry_path),
        concept_geometry_sampler_mode=concept_geometry_sampler_mode,
        concept_geometry_loss_weighting=concept_geometry_loss_weighting,
        concept_geometry_density_power=1.0,
    )


def _run_case(
    *,
    label: str,
    model_path: Path,
    data_dir: Path,
    run_root: Path,
    steps: int,
    steady_warmup: int,
    batch_size: int,
    network_dim: int,
    latent_crop_size: int,
    cached_text_token_limit: int,
    fixed_text_tokens: int,
    fixed_visual_tokens: int,
    attention_backend: str,
    mn_lora_enabled: bool,
    mn_lora_preset: str,
    concept_geometry_enabled: bool,
    concept_geometry_path: Path,
    concept_geometry_sampler_mode: str,
    concept_geometry_loss_weighting: bool,
    cached_workers: int,
    gradient_checkpointing: bool,
    cpu_offload_checkpointing: bool,
    fused_optimizer: bool,
    blocks_to_swap: int,
    swap_granularity: str,
    swap_ratio: float,
    swap_count: int,
    module_offload_enabled: bool,
    module_offload_ratio: int,
    layer_monitor_enabled: bool,
    layer_monitor_interval: int,
    layer_monitor_mode: str,
    layer_monitor_sample_size: int,
    layer_monitor_max_layers: int,
    seed: int,
) -> ConceptGeometryBenchRunResult:
    output_dir = run_root / label / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"anima_concept_geometry_{label}"

    cfg = _make_config(
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        output_name=output_name,
        steps=steps,
        batch_size=batch_size,
        network_dim=network_dim,
        latent_crop_size=latent_crop_size,
        cached_text_token_limit=cached_text_token_limit,
        fixed_text_tokens=fixed_text_tokens,
        fixed_visual_tokens=fixed_visual_tokens,
        attention_backend=attention_backend,
        mn_lora_enabled=mn_lora_enabled,
        mn_lora_preset=mn_lora_preset,
        concept_geometry_enabled=concept_geometry_enabled,
        concept_geometry_path=concept_geometry_path,
        concept_geometry_sampler_mode=concept_geometry_sampler_mode,
        concept_geometry_loss_weighting=concept_geometry_loss_weighting,
        cached_workers=cached_workers,
        gradient_checkpointing=gradient_checkpointing,
        cpu_offload_checkpointing=cpu_offload_checkpointing,
        fused_optimizer=fused_optimizer,
        blocks_to_swap=blocks_to_swap,
        swap_granularity=swap_granularity,
        swap_ratio=swap_ratio,
        swap_count=swap_count,
        module_offload_enabled=module_offload_enabled,
        module_offload_ratio=module_offload_ratio,
        layer_monitor_enabled=layer_monitor_enabled,
        layer_monitor_interval=layer_monitor_interval,
        layer_monitor_mode=layer_monitor_mode,
        layer_monitor_sample_size=layer_monitor_sample_size,
        layer_monitor_max_layers=layer_monitor_max_layers,
        seed=seed,
    )
    trainer = LulynxTrainer(cfg)
    logs: list[str] = []
    step_times_ms: list[float] = []
    layer_monitor_times_ms: list[float] = []
    losses: list[float] = []
    peak_vram_mb = 0.0
    peak_torch_allocated_mb = 0.0
    peak_torch_reserved_mb = 0.0
    original_on_step_end = trainer._on_step_end

    def _on_step_end(step: int, loss: float, info: dict[str, object]) -> None:
        nonlocal peak_vram_mb, peak_torch_allocated_mb, peak_torch_reserved_mb
        step_ms = float(info.get("step_wall_seconds", 0.0) or 0.0) * 1000.0
        step_times_ms.append(step_ms)
        layer_monitor = info.get("layer_monitor") if isinstance(info, dict) else None
        if isinstance(layer_monitor, dict):
            elapsed = float(layer_monitor.get("elapsed_seconds", 0.0) or 0.0) * 1000.0
            if elapsed > 0:
                layer_monitor_times_ms.append(elapsed)
        losses.append(float(loss))
        if torch.cuda.is_available():
            peak_torch_allocated_mb = max(peak_torch_allocated_mb, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            peak_torch_reserved_mb = max(peak_torch_reserved_mb, torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            device_used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0)
            peak_vram_mb = max(peak_vram_mb, device_used_mb)
        original_on_step_end(step, loss, info)

    trainer._on_step_end = _on_step_end  # type: ignore[assignment]
    trainer.set_callbacks(on_log=logs.append)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    success = trainer.start()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_torch_allocated_mb = max(peak_torch_allocated_mb, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        peak_torch_reserved_mb = max(peak_torch_reserved_mb, torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        peak_vram_mb = max(peak_vram_mb, (total_bytes - free_bytes) / (1024.0 * 1024.0))
    total_wall_seconds = time.perf_counter() - start

    output_path = output_dir / f"{output_name}.safetensors"
    output_size = output_path.stat().st_size if output_path.exists() else 0
    adapter_eval = evaluate_adapter(output_path) if output_path.exists() else {}
    steady = step_times_ms[steady_warmup:] if len(step_times_ms) > steady_warmup else step_times_ms
    mean_step = _mean(step_times_ms)
    steady_mean = _mean(steady)
    runtime_plan = getattr(trainer, "runtime_optimization_plan", None)

    initial_loss = losses[0] if losses else 0.0
    final_loss = losses[-1] if losses else 0.0
    result = ConceptGeometryBenchRunResult(
        label=label,
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
        peak_torch_allocated_mb=peak_torch_allocated_mb,
        peak_torch_reserved_mb=peak_torch_reserved_mb,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_delta=final_loss - initial_loss,
        output_path=str(output_path),
        output_size_bytes=output_size,
        attention_backend=str(getattr(runtime_plan, "attention_backend", "unknown") or "unknown"),
        concept_geometry_enabled=concept_geometry_enabled,
        concept_geometry_sampler_mode=concept_geometry_sampler_mode if concept_geometry_enabled else "off",
        concept_geometry_loss_weighting=concept_geometry_loss_weighting if concept_geometry_enabled else False,
        step_times_ms=step_times_ms,
        layer_monitor_times_ms=layer_monitor_times_ms,
        losses=losses,
        log_tail=logs[-60:],
        adapter_evaluation=adapter_eval,
    )
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _compare(baseline: ConceptGeometryBenchRunResult, candidate: ConceptGeometryBenchRunResult) -> dict[str, float]:
    return {
        "mean_step_speedup": (baseline.mean_step_ms or 1.0) / max(candidate.mean_step_ms or 1.0, 1e-9),
        "steady_step_speedup": (baseline.steady_mean_step_ms or 1.0) / max(candidate.steady_mean_step_ms or 1.0, 1e-9),
        "end_to_end_speedup": (baseline.total_wall_seconds or 1.0) / max(candidate.total_wall_seconds or 1.0, 1e-9),
        "peak_vram_ratio": (candidate.peak_vram_mb or 0.0) / max(baseline.peak_vram_mb or 1.0, 1e-9),
        "peak_vram_delta_mb": (candidate.peak_vram_mb or 0.0) - (baseline.peak_vram_mb or 0.0),
        "final_loss_delta": (candidate.final_loss or 0.0) - (baseline.final_loss or 0.0),
        "loss_delta_gap": (candidate.loss_delta or 0.0) - (baseline.loss_delta or 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--steady-warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--network-dim", type=int, default=4)
    parser.add_argument("--latent-crop-size", type=int, default=0)
    parser.add_argument("--cached-text-token-limit", type=int, default=0)
    parser.add_argument("--fixed-text-tokens", type=int, default=0)
    parser.add_argument("--fixed-visual-tokens", type=int, default=0)
    parser.add_argument("--attention-backend", default="auto", choices=("auto", "flash2", "sdpa", "torch"))
    parser.add_argument("--cached-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu-offload-checkpointing", action="store_true")
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--blocks-to-swap", type=int, default=0)
    parser.add_argument("--swap-granularity", default="off", choices=("off", "auto", "block", "merged_block", "layer"))
    parser.add_argument("--swap-ratio", type=float, default=0.0)
    parser.add_argument("--swap-count", type=int, default=0)
    parser.add_argument("--module-offload-ratio", type=int, default=0)
    parser.add_argument("--layer-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layer-monitor-interval", type=int, default=3)
    parser.add_argument("--layer-monitor-mode", default="sampled", choices=("sampled", "exact"))
    parser.add_argument("--layer-monitor-sample-size", type=int, default=4096)
    parser.add_argument("--layer-monitor-max-layers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sampler-mode", default="density_curriculum", choices=("curriculum", "density", "density_curriculum", "concept_batch"))
    parser.add_argument("--geometry-backend", default="latent_tags", choices=("latent_tags", "clip", "dino", "hybrid"))
    parser.add_argument("--clip-model-path", default="", help="Local CLIP checkpoint path for Concept Geometry geometry")
    parser.add_argument("--dino-model-path", default="", help="Local DINO/DINOv2 checkpoint path for Concept Geometry geometry")
    parser.add_argument("--mn-lora", action="store_true", help="Enable the current MN-LoRA preset path during the run")
    parser.add_argument("--mn-lora-preset", default="slim", choices=("slim", "fast", "balanced", "quality"))
    parser.add_argument("--disable-loss-weighting", action="store_true")
    parser.add_argument("--case", default="both", choices=("both", "baseline", "concept_geometry", "h_lora"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--geometry", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo = _repo_root()
    model_path = (args.model or (repo / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors")).expanduser().resolve()
    data_dir = (args.data or (repo / "sucai" / "6_lulu")).expanduser().resolve()
    geometry_path = (
        args.geometry.expanduser().resolve()
        if args.geometry is not None
        else (data_dir / "concept_geometry.json").resolve()
    )
    run_root = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (repo / "temp" / "anima_concept_geometry_benchmark" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    )
    run_root.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Anima cached data not found: {data_dir}")
    if not geometry_path.exists():
        build_concept_geometry(
            data_dir,
            backend=str(args.geometry_backend or "latent_tags"),
            output_path=geometry_path,
            clip_model_path=str(args.clip_model_path or ""),
            dino_model_path=str(args.dino_model_path or ""),
        )
    try:
        geometry_meta = json.loads(geometry_path.read_text(encoding="utf-8")).get("meta", {})
    except Exception:
        geometry_meta = {}
    print(
        f"[concept-geometry-ablation] geometry_backend_requested={args.geometry_backend} "
        f"backend_resolved={geometry_meta.get('backend_resolved', '')} "
        f"sources={','.join(str(item) for item in geometry_meta.get('feature_sources', []))}",
        flush=True,
    )
    for reason in geometry_meta.get("fallback_reasons", []) if isinstance(geometry_meta.get("fallback_reasons", []), list) else []:
        print(f"[concept-geometry-fallback] {reason}", flush=True)
    inspector_report = inspect_dataset(
        InspectorOptions(
            data_dir=data_dir,
            caption_extension=".txt",
            geometry_path=geometry_path,
            concept_geometry_sampler_mode=str(args.sampler_mode or ""),
            batch_size=max(args.batch_size, 1),
        )
    )
    dataset_summary = inspector_report.get("dataset", {})
    cache_summary = inspector_report.get("cache", {})
    concept_geometry_preflight = inspector_report.get("concept_geometry", inspector_report.get("concept_geometry", {}))
    print(
        "[dataset-inspector] "
        f"scan_mode={inspector_report.get('scan_mode')} subsets={inspector_report.get('subset_count')} "
        f"samples={dataset_summary.get('sample_count')} images={dataset_summary.get('image_count')} "
        f"captions={dataset_summary.get('caption_count')} coverage={float(dataset_summary.get('caption_coverage', 0.0) or 0.0):.2f} "
        f"concepts={dataset_summary.get('concept_group_count')}",
        flush=True,
    )
    print(
        "[dataset-cache] "
        f"pairs={cache_summary.get('cache_pair_count')} missing_latent={cache_summary.get('missing_latent_count')} "
        f"missing_text={cache_summary.get('missing_text_count')} attach={float(cache_summary.get('cache_attach_rate', 0.0) or 0.0):.2f}",
        flush=True,
    )
    if concept_geometry_preflight.get("enabled"):
        print(
            "[concept-geometry-preflight] "
            f"attach={float(concept_geometry_preflight.get('geometry_attach_rate', 0.0) or 0.0):.2f} "
            f"concepts={concept_geometry_preflight.get('concept_group_count')} "
            f"neighbor_same={float(concept_geometry_preflight.get('neighbor_same_ratio', 0.0) or 0.0):.2f} "
            f"risk={concept_geometry_preflight.get('risk')}",
            flush=True,
        )
    for warning in inspector_report.get("warnings", [])[:12]:
        print(f"[dataset-warning] {warning}", flush=True)

    fixed_text_tokens = max(args.fixed_text_tokens, 0)
    if fixed_text_tokens <= 0:
        fixed_text_tokens = _infer_cached_text_tokens(data_dir)
        if fixed_text_tokens > 0:
            print(f"[benchmark] fixed_text_tokens auto-sized to {fixed_text_tokens} from cached text metadata", flush=True)
    cached_text_token_limit = max(args.cached_text_token_limit, 0)
    if cached_text_token_limit <= 0 and fixed_text_tokens > 0:
        cached_text_token_limit = fixed_text_tokens

    print(
        f"Running Anima Concept Geometry benchmark: steps={max(args.steps, 1)} steady_warmup={max(args.steady_warmup, 0)} "
        f"batch_size={max(args.batch_size, 1)} latent_crop={max(args.latent_crop_size, 0)} "
        f"fixed_visual_tokens={max(args.fixed_visual_tokens, 0)} model={model_path.name} data={data_dir.name}",
        flush=True,
    )

    requested_case = str(args.case or "both").strip().lower()
    if requested_case == "h_lora":
        requested_case = "concept_geometry"
    sampler_mode = str(args.sampler_mode or "density_curriculum").strip().lower()
    concept_geometry_label = f"concept_geometry_{sampler_mode}"
    baseline = None
    concept_geometry = None
    if requested_case in {"both", "baseline"}:
        baseline = _run_case(
            label="baseline_random",
            model_path=model_path,
            data_dir=data_dir,
            run_root=run_root,
            steps=max(args.steps, 1),
            steady_warmup=max(args.steady_warmup, 0),
            batch_size=max(args.batch_size, 1),
            network_dim=max(args.network_dim, 1),
            latent_crop_size=max(args.latent_crop_size, 0),
            cached_text_token_limit=cached_text_token_limit,
            fixed_text_tokens=fixed_text_tokens,
            fixed_visual_tokens=max(args.fixed_visual_tokens, 0),
            attention_backend=str(args.attention_backend or "auto"),
            mn_lora_enabled=bool(args.mn_lora),
            mn_lora_preset=str(args.mn_lora_preset or "slim"),
            concept_geometry_enabled=False,
            concept_geometry_path=geometry_path,
            concept_geometry_sampler_mode="off",
            concept_geometry_loss_weighting=False,
            cached_workers=max(args.cached_workers, 0),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            cpu_offload_checkpointing=bool(args.cpu_offload_checkpointing),
            fused_optimizer=bool(args.fused_optimizer),
            blocks_to_swap=max(args.blocks_to_swap, 0),
            swap_granularity=str(args.swap_granularity or "off"),
            swap_ratio=max(float(args.swap_ratio or 0.0), 0.0),
            swap_count=max(args.swap_count, 0),
            module_offload_enabled=max(args.module_offload_ratio, 0) > 0,
            module_offload_ratio=max(args.module_offload_ratio, 0),
            layer_monitor_enabled=bool(args.layer_monitor),
            layer_monitor_interval=max(args.layer_monitor_interval, 1),
            layer_monitor_mode=str(args.layer_monitor_mode or "sampled"),
            layer_monitor_sample_size=max(args.layer_monitor_sample_size, 128),
            layer_monitor_max_layers=max(args.layer_monitor_max_layers, 0),
            seed=int(args.seed),
        )
    if requested_case in {"both", "concept_geometry"}:
        concept_geometry = _run_case(
            label=concept_geometry_label,
            model_path=model_path,
            data_dir=data_dir,
            run_root=run_root,
            steps=max(args.steps, 1),
            steady_warmup=max(args.steady_warmup, 0),
            batch_size=max(args.batch_size, 1),
            network_dim=max(args.network_dim, 1),
            latent_crop_size=max(args.latent_crop_size, 0),
            cached_text_token_limit=cached_text_token_limit,
            fixed_text_tokens=fixed_text_tokens,
            fixed_visual_tokens=max(args.fixed_visual_tokens, 0),
            attention_backend=str(args.attention_backend or "auto"),
            mn_lora_enabled=bool(args.mn_lora),
            mn_lora_preset=str(args.mn_lora_preset or "slim"),
            concept_geometry_enabled=True,
            concept_geometry_path=geometry_path,
            concept_geometry_sampler_mode=sampler_mode,
            concept_geometry_loss_weighting=not bool(args.disable_loss_weighting),
            cached_workers=max(args.cached_workers, 0),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            cpu_offload_checkpointing=bool(args.cpu_offload_checkpointing),
            fused_optimizer=bool(args.fused_optimizer),
            blocks_to_swap=max(args.blocks_to_swap, 0),
            swap_granularity=str(args.swap_granularity or "off"),
            swap_ratio=max(float(args.swap_ratio or 0.0), 0.0),
            swap_count=max(args.swap_count, 0),
            module_offload_enabled=max(args.module_offload_ratio, 0) > 0,
            module_offload_ratio=max(args.module_offload_ratio, 0),
            layer_monitor_enabled=bool(args.layer_monitor),
            layer_monitor_interval=max(args.layer_monitor_interval, 1),
            layer_monitor_mode=str(args.layer_monitor_mode or "sampled"),
            layer_monitor_sample_size=max(args.layer_monitor_sample_size, 128),
            layer_monitor_max_layers=max(args.layer_monitor_max_layers, 0),
            seed=int(args.seed),
        )

    payload = {
        "benchmark": {
            "route": "anima",
            "kind": "concept_geometry_sampler",
            "case": requested_case,
            "steps": max(args.steps, 1),
            "steady_warmup": max(args.steady_warmup, 0),
            "batch_size": max(args.batch_size, 1),
            "latent_crop_size": max(args.latent_crop_size, 0),
            "cached_text_token_limit": cached_text_token_limit,
            "fixed_text_tokens": fixed_text_tokens,
            "fixed_visual_tokens": max(args.fixed_visual_tokens, 0),
            "attention_backend_requested": str(args.attention_backend or "auto"),
            "layer_monitor_enabled": bool(args.layer_monitor),
            "layer_monitor_interval": max(args.layer_monitor_interval, 1),
            "layer_monitor_mode": str(args.layer_monitor_mode or "sampled"),
            "layer_monitor_sample_size": max(args.layer_monitor_sample_size, 128),
            "layer_monitor_max_layers": max(args.layer_monitor_max_layers, 0),
            "mn_lora_enabled": bool(args.mn_lora),
            "mn_lora_preset": str(args.mn_lora_preset or "slim"),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "cpu_offload_checkpointing": bool(args.cpu_offload_checkpointing),
            "fused_optimizer": bool(args.fused_optimizer),
            "blocks_to_swap": max(args.blocks_to_swap, 0),
            "swap_granularity": str(args.swap_granularity or "off"),
            "swap_ratio": max(float(args.swap_ratio or 0.0), 0.0),
            "swap_count": max(args.swap_count, 0),
            "module_offload_ratio": max(args.module_offload_ratio, 0),
            "model": str(model_path),
            "data": str(data_dir),
            "geometry": str(geometry_path),
            "geometry_backend_requested": str(args.geometry_backend or "latent_tags"),
            "geometry_backend_resolved": geometry_meta.get("backend_resolved", ""),
            "geometry_feature_sources": geometry_meta.get("feature_sources", []),
            "geometry_fallback_reasons": geometry_meta.get("fallback_reasons", []),
            "dataset_inspector": inspector_report,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "runs": {},
        "comparison": {},
    }
    if baseline is not None:
        payload["runs"]["baseline_random"] = asdict(baseline)
    if concept_geometry is not None:
        payload["runs"][concept_geometry_label] = asdict(concept_geometry)
    if baseline is not None and concept_geometry is not None:
        payload["comparison"] = _compare(baseline, concept_geometry)
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def line(run: ConceptGeometryBenchRunResult) -> str:
        return (
            f"[benchmark] label={run.label} success={run.success} steps={run.steps_completed} "
            f"peak_vram={run.peak_vram_mb:.1f}MB torch_alloc={run.peak_torch_allocated_mb:.1f}MB "
            f"torch_reserved={run.peak_torch_reserved_mb:.1f}MB avg_step_time={run.mean_step_ms:.2f}ms "
            f"steady_step_time={run.steady_mean_step_ms:.2f}ms samples_per_sec={run.samples_per_second:.3f} "
            f"steady_samples_per_sec={run.steady_samples_per_second:.3f} "
            f"initial_loss={run.initial_loss:.4f} final_loss={run.final_loss:.4f} "
            f"concept_geometry={run.concept_geometry_enabled} mode={run.concept_geometry_sampler_mode}"
        )

    if baseline is not None:
        print(line(baseline), flush=True)
    if concept_geometry is not None:
        print(line(concept_geometry), flush=True)
    if payload["comparison"]:
        print(f"[benchmark] comparison={json.dumps(payload['comparison'], sort_keys=True)}", flush=True)
    print(f"[benchmark] summary={summary_path}", flush=True)
    if baseline is not None and not baseline.success:
        return 1
    if concept_geometry is not None and not concept_geometry.success:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
