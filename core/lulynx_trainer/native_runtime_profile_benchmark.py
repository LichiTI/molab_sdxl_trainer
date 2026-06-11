"""SD15/SDXL/Newbie native runtime profile benchmark.

Runs the product trainer path twice with identical training settings and only
switches ``native_runtime_profile`` between ``standard`` and ``aggressive``.
The script uses local ``models/`` and ``sucai/`` assets and writes a JSON summary
under ``temp/native_runtime_profile_benchmark`` by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (str(repo_root), str(backend_root)):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)

from core.configs import MixedPrecision, ModelArch, OptimizerType, SchedulerType, UnifiedTrainingConfig
from core.lulynx_trainer.dataset_discovery import caption_candidates_for_stem
from core.lulynx_trainer.newbie_cache_contract import find_newbie_cache_files, newbie_cache_contract_for_files
from core.lulynx_trainer.trainer import LulynxTrainer
from core.lulynx_trainer.runtime_feature_summary import load_runtime_feature_summary_from_manifest
from core.lulynx_trainer.step_phase_profile import build_step_phase_bubble_profile
from core.turbocore_native_update_performance_report import build_native_update_profile_performance_report


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
BLOCK_RESIDENCY_MODE_CHOICES = (
    "resident",
    "streaming_offload",
    "block_cpu_pinned",
    "hot_aware",
    "balanced",
    "steaming_offload",
)

SD15_CHECKPOINT_CANDIDATES = (
    "v1-5-pruned-emaonly.safetensors",
    "v1-5-pruned.safetensors",
    "sd15.safetensors",
    "model.safetensors",
)


def _resolve_sd15_checkpoint(model_root: Path) -> Path:
    sd15_dir = model_root / "sd15"
    candidates = [sd15_dir / name for name in SD15_CHECKPOINT_CANDIDATES]
    candidates.extend(sorted(sd15_dir.glob("*.safetensors")))
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        expected = ", ".join(str(path) for path in candidates[: len(SD15_CHECKPOINT_CANDIDATES)])
        raise FileNotFoundError(
            "SD15 benchmark requires a local base checkpoint under models/sd15 "
            f"(expected one of: {expected})"
        )
    return checkpoint


@dataclass
class RuntimeProfileRun:
    family: str
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
    setup_peak_vram_mb: float
    training_peak_vram_mb: float
    final_loss: float
    attention_backend: str
    compile_requested: bool
    compile_scope: str
    optimizer_type: str
    optimizer_args: str
    output_path: str
    output_size_bytes: int
    step_times_ms: list[float] = field(default_factory=list)
    phase_profiles: list[dict[str, Any]] = field(default_factory=list)
    steady_phase_summary: dict[str, float] = field(default_factory=dict)
    steady_bubble_profile: dict[str, Any] = field(default_factory=dict)
    runtime_feature_summary: dict[str, Any] = field(default_factory=dict)
    native_adamw_gain_estimates: dict[str, float] = field(default_factory=dict)
    update_shadow_reports: list[dict[str, Any]] = field(default_factory=list)
    native_update_dispatch_arming_reports: list[dict[str, Any]] = field(default_factory=list)
    native_update_runtime_recovery_observations: list[dict[str, Any]] = field(default_factory=list)
    native_update_dispatch_runtime_reports: list[dict[str, Any]] = field(default_factory=list)
    native_update_diagnostic_replay_reports: list[dict[str, Any]] = field(default_factory=list)
    native_update_dispatch_arming_observations: list[dict[str, Any]] = field(default_factory=list)
    native_update_gate_reports: list[dict[str, Any]] = field(default_factory=list)
    native_update_loop_timings: list[dict[str, Any]] = field(default_factory=list)
    native_update_readiness: dict[str, Any] = field(default_factory=dict)
    bubble_controlled_rollback_observations: list[dict[str, Any]] = field(default_factory=list)
    bubble_controlled_data_wait_observations: list[dict[str, Any]] = field(default_factory=list)
    bubble_benchmark_data_wait_stall: dict[str, Any] = field(default_factory=dict)
    losses: list[float] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _phase_value(profile: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = profile
    for key in path:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _amdahl_speedup_pct(share: float, local_speedup: float) -> float:
    share = min(max(float(share or 0.0), 0.0), 0.999999)
    local_speedup = max(float(local_speedup or 1.0), 1.0)
    total_speedup = 1.0 / ((1.0 - share) + share / local_speedup)
    return (total_speedup - 1.0) * 100.0


def _summarize_phase_profiles(
    phase_profiles: list[dict[str, Any]],
    *,
    steady_warmup: int,
) -> tuple[dict[str, float], dict[str, float]]:
    steady = phase_profiles[steady_warmup:] if len(phase_profiles) > steady_warmup else phase_profiles
    optimizer_step_ms = [_phase_value(item, ("phases_ms", "optimizer_step")) for item in steady]
    zero_grad_ms = [_phase_value(item, ("phases_ms", "zero_grad")) for item in steady]
    update_total_ms = [_phase_value(item, ("phases_ms", "optimizer_update_total")) for item in steady]
    step_wall_ms = [_phase_value(item, ("step_wall_ms",)) for item in steady]
    optimizer_step_share = [_phase_value(item, ("optimizer_step_share",)) for item in steady]
    optimizer_plus_zero_share = [_phase_value(item, ("optimizer_plus_zero_grad_share",)) for item in steady]
    optimizer_update_share = [_phase_value(item, ("optimizer_update_share",)) for item in steady]
    summary = {
        "steady_profiled_steps": float(len(steady)),
        "step_wall_ms_mean": _mean(step_wall_ms),
        "optimizer_step_ms_mean": _mean(optimizer_step_ms),
        "zero_grad_ms_mean": _mean(zero_grad_ms),
        "optimizer_plus_zero_grad_ms_mean": _mean(
            [left + right for left, right in zip(optimizer_step_ms, zero_grad_ms)]
        ),
        "optimizer_update_total_ms_mean": _mean(update_total_ms),
        "optimizer_step_share_mean": _mean(optimizer_step_share),
        "optimizer_plus_zero_grad_share_mean": _mean(optimizer_plus_zero_share),
        "optimizer_update_share_mean": _mean(optimizer_update_share),
    }
    optimizer_share = summary["optimizer_step_share_mean"]
    optimizer_plus_zero_share = summary["optimizer_plus_zero_grad_share_mean"]
    update_share = summary["optimizer_update_share_mean"] or optimizer_plus_zero_share
    estimates: dict[str, float] = {}
    for speedup in (10.0, 20.0, 30.0):
        label = f"native_{speedup:g}x"
        estimates[f"{label}_optimizer_step_speedup_pct"] = _amdahl_speedup_pct(optimizer_share, speedup)
        estimates[f"{label}_optimizer_plus_zero_grad_speedup_pct"] = _amdahl_speedup_pct(
            optimizer_plus_zero_share,
            speedup,
        )
        estimates[f"{label}_update_upper_bound_speedup_pct"] = _amdahl_speedup_pct(update_share, speedup)
    return summary, estimates


def _load_runtime_feature_summary(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return load_runtime_feature_summary_from_manifest(manifest_path)
    except Exception as exc:
        return {"manifest_error": f"{type(exc).__name__}: {exc}", "manifest_path": str(manifest_path)}


def _controlled_data_wait_runtime_features(*, share: float, mean_ms: float) -> dict[str, Any]:
    data_wait = min(max(float(share or 0.0), 0.0), 0.95)
    train_share = max(1.0 - data_wait, 0.0)
    step_ms = max(float(mean_ms or 100.0), 1.0)
    return {
        "training_loop_runtime": {
            "step_phase_profile": {
                "enabled": False,
                "scope": "benchmark_controlled_data_wait",
                "gpu_bubble_profile": {
                    "schema_version": 1,
                    "profile": "step_phase_bubble_profile_v0",
                    "step_count": 1,
                    "mean_step_ms": round(step_ms, 4),
                    "phase_mean_ms": {
                        "data_wait": round(step_ms * data_wait, 4),
                        "train_step_total": round(step_ms * train_share, 4),
                    },
                    "phase_share": {
                        "data_wait": round(data_wait, 6),
                        "train_step_total": round(train_share, 6),
                    },
                    "bubble_ratio_estimate": round(data_wait, 6),
                    "dominant_bottleneck": "data_bound",
                    "bottlenecks": ["data_bound"],
                    "recommendations": [
                        "benchmark-controlled data_wait probe for DataLoader rebuild evidence"
                    ],
                    "evidence": {
                        "train_step_share": round(train_share, 6),
                        "optimizer_share": 0.0,
                        "data_wait_share": round(data_wait, 6),
                        "h2d_transfer_share": 0.0,
                        "logging_checkpoint_share": 0.0,
                        "host_gap_share": 0.0,
                        "top_phases": [
                            {
                                "label": "data_wait",
                                "mean_ms": round(step_ms * data_wait, 4),
                                "share": round(data_wait, 6),
                            },
                            {
                                "label": "train_step_total",
                                "mean_ms": round(step_ms * train_share, 4),
                                "share": round(train_share, 6),
                            },
                        ],
                    },
                },
            }
        },
        "bubble_controller_controlled_data_wait": {
            "schema_version": 1,
            "kind": "bubble_controller_controlled_data_wait_v0",
            "data_wait_share": round(data_wait, 6),
            "mean_step_ms": round(step_ms, 4),
            "scope": "benchmark_only",
        },
    }


def _precision() -> tuple[MixedPrecision, str, torch.dtype]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return MixedPrecision.BF16, "bf16", torch.bfloat16
    if torch.cuda.is_available():
        return MixedPrecision.FP16, "fp16", torch.float16
    return MixedPrecision.NO, "float32", torch.float32


def _normalize_block_residency_mode(value: Any) -> str:
    mode = str(value or "resident").strip().lower().replace("-", "_")
    aliases = {
        "off": "resident",
        "gpu": "resident",
        "none": "resident",
        "balanced": "streaming_offload",
        "hot": "streaming_offload",
        "hotaware": "streaming_offload",
        "hot_aware": "streaming_offload",
        "hot_aware_cpu_pinned": "streaming_offload",
        "streaming": "streaming_offload",
        "streaming_cpu_offload": "streaming_offload",
        "steaming": "streaming_offload",
        "steaming_offload": "streaming_offload",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in {"resident", "streaming_offload", "block_cpu_pinned"} else "resident"


def _dit_block_checkpoint_recommended(residency_mode: str, *, resolution: int, fixed_visual_tokens: int) -> bool:
    """Mirror the trainer's safety rule for high-token non-resident DiT runs."""

    mode = _normalize_block_residency_mode(residency_mode)
    if mode == "resident":
        return False
    high_resolution = int(resolution) >= 1024
    high_visual_tokens = int(fixed_visual_tokens) >= 4096 or (int(fixed_visual_tokens) <= 0 and high_resolution)
    return high_resolution or high_visual_tokens


def _iter_images(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            yield child


def _prepare_sample_dir(source: Path, target: Path, *, sample_count: int) -> None:
    target.mkdir(parents=True, exist_ok=True)
    images = list(_iter_images(source))[: max(int(sample_count), 1)]
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    for image in images:
        dst_image = target / image.name
        shutil.copy2(image, dst_image)
        for caption in caption_candidates_for_stem(image.parent, image.stem, ".txt"):
            if caption.is_file():
                shutil.copy2(caption, target / caption.name)
        stem = image.stem
        cache_patterns = (
            f"{stem}_*_anima.npz",
            f"{stem}_anima_te.npz",
            f"{stem}_*_newbie.npz",
            f"{stem}_newbie.npz",
            f"{stem}_newbie.safetensors",
            f"{stem}_newbie.pt",
            f"{stem}_te_outputs.npz",
            f"{stem}.jpg.safetensors",
            f"{stem}.png.safetensors",
            f"{stem}.txt.safetensors",
        )
        for pattern in cache_patterns:
            for cache_file in source.glob(pattern):
                if cache_file.is_file():
                    shutil.copy2(cache_file, target / cache_file.name)


def _cache_inventory(root: Path) -> dict[str, Any]:
    patterns = {
        "anima_latent": "*_anima.npz",
        "anima_text": "*_anima_te.npz",
        "newbie_npz": "*_newbie.npz",
        "newbie_safetensors": "*_newbie.safetensors",
        "newbie_pt": "*_newbie.pt",
        "text_outputs": "*_te_outputs.npz",
        "sdxl_cache": "*.safetensors",
    }
    counts = {
        key: sum(1 for path in root.glob(pattern) if path.is_file())
        for key, pattern in patterns.items()
    }
    newbie_contract = newbie_cache_contract_for_files(find_newbie_cache_files(root, recursive=False))
    return {
        "schema_version": 1,
        "cache_inventory": "native_runtime_profile_cache_inventory_v0",
        "root": str(root),
        "counts": counts,
        "has_anima_cache": bool(counts["anima_latent"] and counts["anima_text"]),
        "has_newbie_cache": bool(newbie_contract.get("ok")),
        "newbie_contract_ok": bool(newbie_contract.get("ok")),
        "newbie_contract_reasons": list(newbie_contract.get("reasons") or []),
        "newbie_pooled_shapes": list(newbie_contract.get("pooled_shapes") or []),
    }


def _common_config(
    *,
    family: str,
    profile: str,
    train_dir: Path,
    output_dir: Path,
    output_name: str,
    steps: int,
    resolution: int,
    network_dim: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float | None,
) -> dict[str, Any]:
    mixed_precision, save_precision, _dtype = _precision()
    lr = float(learning_rate) if learning_rate is not None and float(learning_rate) > 0.0 else (1e-5 if family == "newbie" else 1e-4)
    return {
        "train_data_dir": str(train_dir),
        "output_dir": str(output_dir),
        "output_name": output_name,
        "native_runtime_profile": profile,
        "mixed_precision": mixed_precision,
        "save_precision": save_precision,
        "optimizer_type": OptimizerType.ADAMW,
        "lr_scheduler": SchedulerType.CONSTANT,
        "train_batch_size": max(int(train_batch_size), 1),
        "gradient_accumulation_steps": max(int(gradient_accumulation_steps), 1),
        "gradient_accumulation_mode": "fast",
        "max_train_epochs": max(int(steps), 1),
        "max_train_steps": max(int(steps), 1),
        "network_dim": max(int(network_dim), 1),
        "network_alpha": max(int(network_dim), 1),
        "learning_rate": lr,
        "save_every_n_epochs": 1,
        "save_every_n_steps": 0,
        "save_state": False,
        "sample_every": 0,
        "sample_every_n_epochs": 0,
        "gradient_checkpointing": True,
        "resolution": int(resolution),
        "caption_extension": ".txt",
        "seed": 1337,
        "mem_efficient_save": True,
        "compile_probe_enabled": False,
        "compile_anima_full_core_enabled": False,
        "compile_contract_strict": True,
        "compile_static_shape_drop_last": True,
        "compile_require_cache_first": True,
        "cached_dataloader_auto_policy": True,
        "cached_dataloader_workers": 0,
        "dataloader_num_workers": 0,
        "persistent_data_loader_workers": False,
        "pin_memory": True,
        "prefetch_factor": 2,
        "enable_auditor": False,
        "adaptive_step_logging_enabled": False,
        "tensorboard_flush_interval_steps": 1000,
        "so_enable_nan_detection": False,
        "so_enable_loss_spike_detection": False,
        "so_enable_lr_deadlock_detection": False,
        "so_enable_auto_recovery": False,
        "rm_enable_adaptive_accumulation": False,
        "rm_enable_adaptive_batch": False,
    }


def _make_config(
    *,
    family: str,
    profile: str,
    model_root: Path,
    train_dir: Path,
    output_dir: Path,
    output_name: str,
    steps: int,
    resolution: int,
    network_dim: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float | None,
    anima_latent_crop_size: int,
    anima_fixed_text_tokens: int,
    anima_fixed_visual_tokens: int,
    anima_block_residency: str,
    anima_block_residency_min_params: int,
    anima_block_checkpointing: bool,
    anima_block_checkpointing_mode: str,
    anima_block_prefetch: bool,
    anima_block_prefetch_depth: int,
    newbie_latent_crop_size: int,
    newbie_fixed_text_tokens: int,
    newbie_fixed_visual_tokens: int,
    newbie_block_residency: str,
    newbie_block_residency_min_params: int,
    newbie_block_checkpointing: bool,
    newbie_block_checkpointing_mode: str,
    newbie_block_prefetch: bool,
    newbie_block_prefetch_depth: int,
    phase_profile: bool,
    bubble_controller: Mapping[str, Any] | None,
    turbocore_update_shadow: str,
    turbocore_update_shadow_direct_grad: bool,
    turbocore_update_shadow_compare_sample_params: int,
    turbocore_update_shadow_stop_after_consecutive_passes: int,
    turbocore_update_shadow_checkpoint_contract: bool,
    turbocore_update_shadow_copyback_probe: bool,
    turbocore_update_shadow_copyback_dispatch_experimental: bool,
    turbocore_update_shadow_native_binding_probe: bool,
    turbocore_update_shadow_owner_native_launch_probe: bool,
    turbocore_update_shadow_owner_native_launch_max_numel: int,
    turbocore_update_shadow_owner_native_event_chain_probe: bool,
    turbocore_update_shadow_save_owner_state: bool,
    turbocore_native_update_mode: str,
    turbocore_native_update_required_shadow_passes: int,
    turbocore_native_update_allow_missing_kernel: bool,
    turbocore_native_update_dispatch_enabled: bool,
    turbocore_native_update_training_path_enabled: bool,
    turbocore_native_update_require_native_cuda: bool,
    turbocore_native_update_diagnostic_executor_replay: bool,
    turbocore_native_update_defer_state_sync: bool,
    turbocore_native_update_runtime_synchronization_policy: str,
    native_cache_mode: str,
) -> UnifiedTrainingConfig:
    common = _common_config(
        family=family,
        profile=profile,
        train_dir=train_dir,
        output_dir=output_dir,
        output_name=output_name,
        steps=steps,
        resolution=resolution,
        network_dim=network_dim,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
    )
    bubble = dict(bubble_controller or {})
    if bubble:
        common["bubble_controller_enabled"] = bool(bubble.get("enabled", False))
        common["bubble_controller_mode"] = str(bubble.get("mode", "report_only") or "report_only")
        common["bubble_controller_warmup_steps"] = max(int(bubble.get("warmup_steps", 8) or 0), 0)
        common["bubble_controller_tune_interval_steps"] = max(int(bubble.get("tune_interval_steps", 32) or 1), 1)
        common["bubble_controller_max_actions_per_run"] = max(int(bubble.get("max_actions_per_run", 3) or 0), 0)
        common["bubble_controller_min_throughput_gain"] = max(float(bubble.get("min_throughput_gain", 0.03) or 0.0), 0.0)
        common["bubble_controller_allow_dataloader_rebuild_current_run"] = bool(
            bubble.get("allow_dataloader_rebuild_current_run", False)
        )
    common["turbocore_update_shadow_mode"] = str(turbocore_update_shadow or "off")
    common["turbocore_update_shadow_direct_grad"] = bool(turbocore_update_shadow_direct_grad)
    common["turbocore_update_shadow_compare_interval"] = 1
    common["turbocore_update_shadow_compare_sample_params"] = max(int(turbocore_update_shadow_compare_sample_params or 0), 0)
    common["turbocore_update_shadow_stop_after_consecutive_passes"] = max(int(turbocore_update_shadow_stop_after_consecutive_passes or 0), 0)
    common["turbocore_update_shadow_checkpoint_contract"] = bool(turbocore_update_shadow_checkpoint_contract)
    common["turbocore_update_shadow_copyback_probe"] = bool(turbocore_update_shadow_copyback_probe)
    common["turbocore_update_shadow_copyback_dispatch_experimental"] = bool(
        turbocore_update_shadow_copyback_dispatch_experimental
    )
    common["turbocore_update_shadow_native_binding_probe"] = bool(turbocore_update_shadow_native_binding_probe)
    common["turbocore_update_shadow_owner_native_launch_probe"] = bool(
        turbocore_update_shadow_owner_native_launch_probe
    )
    common["turbocore_update_shadow_owner_native_launch_max_numel"] = max(
        int(turbocore_update_shadow_owner_native_launch_max_numel or 0),
        1,
    )
    common["turbocore_update_shadow_owner_native_event_chain_probe"] = bool(
        turbocore_update_shadow_owner_native_event_chain_probe
    )
    common["turbocore_update_shadow_save_owner_state"] = bool(turbocore_update_shadow_save_owner_state)
    common["turbocore_native_update_mode"] = str(turbocore_native_update_mode or "off")
    common["turbocore_native_update_required_shadow_passes"] = max(
        int(turbocore_native_update_required_shadow_passes or 3),
        1,
    )
    common["turbocore_native_update_allow_missing_kernel"] = bool(turbocore_native_update_allow_missing_kernel)
    common["turbocore_native_update_dispatch_enabled"] = bool(turbocore_native_update_dispatch_enabled)
    common["turbocore_native_update_training_path_enabled"] = bool(turbocore_native_update_training_path_enabled)
    common["turbocore_native_update_require_native_cuda"] = bool(turbocore_native_update_require_native_cuda)
    common["turbocore_native_update_diagnostic_executor_replay"] = bool(
        turbocore_native_update_diagnostic_executor_replay
    )
    common["turbocore_native_update_defer_state_sync"] = bool(turbocore_native_update_defer_state_sync)
    common["turbocore_native_update_runtime_synchronization_policy"] = str(
        turbocore_native_update_runtime_synchronization_policy or "context_synchronize"
    )
    normalized_cache_mode = str(native_cache_mode or "").strip().lower().replace("-", "_")
    if normalized_cache_mode:
        common["native_cache_mode"] = normalized_cache_mode
        if family == "anima":
            common["anima_cache_mode"] = normalized_cache_mode
    if family == "sd15":
        common["step_phase_profile_enabled"] = bool(phase_profile)
        checkpoint = _resolve_sd15_checkpoint(model_root)
        return UnifiedTrainingConfig(
            model_type=ModelArch.SD15,
            pretrained_model_name_or_path=str(checkpoint),
            attention_backend="auto",
            xformers=True,
            enable_bucket=True,
            min_bucket_reso=int(resolution),
            max_bucket_reso=int(resolution),
            bucket_reso_steps=64,
            **common,
        )
    if family == "sdxl":
        common["step_phase_profile_enabled"] = bool(phase_profile)
        checkpoint = model_root / "sdxl" / "silentEraFurrymixNAIXL_v10.safetensors"
        return UnifiedTrainingConfig(
            model_type=ModelArch.SDXL,
            pretrained_model_name_or_path=str(checkpoint),
            attention_backend="auto",
            xformers=True,
            enable_bucket=True,
            min_bucket_reso=int(resolution),
            max_bucket_reso=int(resolution),
            bucket_reso_steps=64,
            **common,
        )
    if family == "anima":
        common["step_phase_profile_enabled"] = bool(phase_profile)
        common["native_cache_mode"] = normalized_cache_mode or "cache_first"
        common["anima_cache_mode"] = normalized_cache_mode or "cache_first"
        checkpoint = model_root / "anima" / "diffusion_models" / "anima-preview2.safetensors"
        if not checkpoint.exists():
            checkpoint = model_root / "anima" / "diffusion_models" / "anima-base-v1.0.safetensors"
        qwen3_path = model_root / "anima" / "text_encoders" / "qwen_3_06b_base.safetensors"
        vae_path = model_root / "anima" / "vae" / "qwen_image_vae.safetensors"
        text_token_limit = max(int(anima_fixed_text_tokens or 0), 16)
        return UnifiedTrainingConfig(
            model_type=ModelArch.ANIMA,
            pretrained_model_name_or_path=str(checkpoint),
            anima_model_path=str(checkpoint),
            anima_qwen3_path=str(qwen3_path),
            vae_path=str(vae_path),
            attention_backend="auto",
            anima_cached_training=True,
            anima_cached_latent_crop_size=max(int(anima_latent_crop_size), 0),
            anima_qwen3_max_token_length=text_token_limit,
            anima_text_token_limit=text_token_limit,
            anima_cached_text_token_limit=text_token_limit,
            anima_fixed_text_tokens=max(int(anima_fixed_text_tokens), 0),
            anima_fixed_visual_tokens=max(int(anima_fixed_visual_tokens), 0),
            anima_block_residency=str(anima_block_residency or "resident"),
            anima_block_residency_min_params=max(int(anima_block_residency_min_params), 0),
            anima_block_checkpointing=bool(anima_block_checkpointing),
            anima_block_checkpointing_mode=str(anima_block_checkpointing_mode or "block"),
            anima_block_prefetch=bool(anima_block_prefetch),
            anima_block_prefetch_depth=max(int(anima_block_prefetch_depth), 0),
            anima_native_block_count=28,
            **common,
        )
    if family == "newbie":
        common["step_phase_profile_enabled"] = bool(phase_profile)
        newbie_dir = model_root / "newbie"
        return UnifiedTrainingConfig(
            model_type=ModelArch.NEWBIE,
            pretrained_model_name_or_path=str(newbie_dir),
            attention_backend="auto",
            trust_remote_code=True,
            use_cache=True,
            newbie_safe_fallback=torch.cuda.is_available(),
            newbie_run_native_smoke=False,
            newbie_transformer_path=str(newbie_dir / "transformer"),
            newbie_gemma_model_path=str(newbie_dir / "text_encoder"),
            newbie_clip_model_path=str(newbie_dir / "clip_model"),
            newbie_vae_path=str(newbie_dir / "vae"),
            newbie_gemma_max_token_length=64,
            newbie_clip_max_token_length=128,
            newbie_cached_latent_crop_size=max(int(newbie_latent_crop_size), 0),
            newbie_cached_text_token_limit=64,
            newbie_fixed_text_tokens=max(int(newbie_fixed_text_tokens), 0),
            newbie_fixed_visual_tokens=max(int(newbie_fixed_visual_tokens), 0),
            newbie_block_residency=str(newbie_block_residency or "resident"),
            newbie_block_residency_min_params=max(int(newbie_block_residency_min_params), 0),
            newbie_block_checkpointing=bool(newbie_block_checkpointing),
            newbie_block_checkpointing_mode=str(newbie_block_checkpointing_mode or "block"),
            newbie_block_prefetch=bool(newbie_block_prefetch),
            newbie_block_prefetch_depth=max(int(newbie_block_prefetch_depth), 0),
            newbie_target_modules="layers.0.attention.qkv\nlayers.0.attention.out",
            **common,
        )
    raise ValueError(f"Unsupported family: {family}")


def _run_profile(
    *,
    family: str,
    label: str,
    profile: str,
    model_root: Path,
    source_data: Path,
    run_root: Path,
    steps: int,
    steady_warmup: int,
    resolution: int,
    network_dim: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float | None,
    sample_count: int,
    fused_adamw: bool,
    attention_backend: str,
    sdpa_backend_policy: str,
    dataloader_workers: int,
    dataloader_prefetch_factor: int,
    dataloader_pin_memory: bool,
    data_transfer_profile: bool,
    data_transfer_profile_mode: str,
    anima_latent_crop_size: int,
    anima_fixed_text_tokens: int,
    anima_fixed_visual_tokens: int,
    anima_block_residency: str,
    anima_block_residency_min_params: int,
    anima_block_checkpointing: bool,
    anima_block_checkpointing_mode: str,
    anima_block_prefetch: bool,
    anima_block_prefetch_depth: int,
    newbie_latent_crop_size: int,
    newbie_fixed_text_tokens: int,
    newbie_fixed_visual_tokens: int,
    newbie_block_residency: str,
    newbie_block_residency_min_params: int,
    newbie_block_checkpointing: bool,
    newbie_block_checkpointing_mode: str,
    newbie_block_prefetch: bool,
    newbie_block_prefetch_depth: int,
    lora_activation_recompute: str,
    phase_profile: bool,
    bubble_controller: Mapping[str, Any] | None,
    turbocore_update_shadow: str,
    turbocore_update_shadow_direct_grad: bool,
    turbocore_update_shadow_compare_sample_params: int,
    turbocore_update_shadow_stop_after_consecutive_passes: int,
    turbocore_update_shadow_checkpoint_contract: bool,
    turbocore_update_shadow_copyback_probe: bool,
    turbocore_update_shadow_copyback_dispatch_experimental: bool,
    turbocore_update_shadow_native_binding_probe: bool,
    turbocore_update_shadow_owner_native_launch_probe: bool,
    turbocore_update_shadow_owner_native_launch_max_numel: int,
    turbocore_update_shadow_owner_native_event_chain_probe: bool,
    turbocore_update_shadow_save_owner_state: bool,
    turbocore_native_update_mode: str,
    turbocore_native_update_required_shadow_passes: int,
    turbocore_native_update_allow_missing_kernel: bool,
    turbocore_native_update_dispatch_enabled: bool,
    turbocore_native_update_training_path_enabled: bool,
    turbocore_native_update_require_native_cuda: bool,
    turbocore_native_update_diagnostic_executor_replay: bool,
    turbocore_native_update_defer_state_sync: bool,
    turbocore_native_update_runtime_synchronization_policy: str,
    bubble_controller_controlled_rollback_slowdown_ratio: float,
    bubble_controller_controlled_rollback_after_apply_steps: int,
    bubble_controller_controlled_data_wait_share: float,
    bubble_controller_controlled_data_wait_mean_ms: float,
    bubble_controller_benchmark_data_wait_stall_ms: float,
    bubble_controller_benchmark_data_wait_direct_action: bool,
    native_cache_mode: str,
    torch_compile: bool,
    torch_compile_scope: str,
    torch_compile_backend: str,
    torch_compile_mode: str,
    torch_compile_dynamic: bool,
    torch_compile_fullgraph: bool,
    compile_cache_enabled: bool,
    compile_cache_reuse: bool,
    compile_cache_root: str,
) -> RuntimeProfileRun:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    case_root = run_root / family / label
    train_dir = case_root / "train"
    output_dir = case_root / "output"
    if case_root.exists():
        shutil.rmtree(case_root, ignore_errors=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_sample_dir(source_data, train_dir, sample_count=sample_count)
    cache_inventory_before = _cache_inventory(train_dir)
    output_name = f"{family}_profile_{label}"
    cfg = _make_config(
        family=family,
        profile=profile,
        model_root=model_root,
        train_dir=train_dir,
        output_dir=output_dir,
        output_name=output_name,
        steps=steps,
        resolution=resolution,
        network_dim=network_dim,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        anima_latent_crop_size=anima_latent_crop_size,
        anima_fixed_text_tokens=anima_fixed_text_tokens,
        anima_fixed_visual_tokens=anima_fixed_visual_tokens,
        anima_block_residency=anima_block_residency,
        anima_block_residency_min_params=anima_block_residency_min_params,
        anima_block_checkpointing=anima_block_checkpointing,
        anima_block_checkpointing_mode=anima_block_checkpointing_mode,
        anima_block_prefetch=anima_block_prefetch,
        anima_block_prefetch_depth=anima_block_prefetch_depth,
        newbie_latent_crop_size=newbie_latent_crop_size,
        newbie_fixed_text_tokens=newbie_fixed_text_tokens,
        newbie_fixed_visual_tokens=newbie_fixed_visual_tokens,
        newbie_block_residency=newbie_block_residency,
        newbie_block_residency_min_params=newbie_block_residency_min_params,
        newbie_block_checkpointing=newbie_block_checkpointing,
        newbie_block_checkpointing_mode=newbie_block_checkpointing_mode,
        newbie_block_prefetch=newbie_block_prefetch,
        newbie_block_prefetch_depth=newbie_block_prefetch_depth,
        phase_profile=phase_profile,
        bubble_controller=bubble_controller,
        turbocore_update_shadow=turbocore_update_shadow,
        turbocore_update_shadow_direct_grad=turbocore_update_shadow_direct_grad,
        turbocore_update_shadow_compare_sample_params=turbocore_update_shadow_compare_sample_params,
        turbocore_update_shadow_stop_after_consecutive_passes=turbocore_update_shadow_stop_after_consecutive_passes,
        turbocore_update_shadow_checkpoint_contract=turbocore_update_shadow_checkpoint_contract,
        turbocore_update_shadow_copyback_probe=turbocore_update_shadow_copyback_probe,
        turbocore_update_shadow_copyback_dispatch_experimental=turbocore_update_shadow_copyback_dispatch_experimental,
        turbocore_update_shadow_native_binding_probe=turbocore_update_shadow_native_binding_probe,
        turbocore_update_shadow_owner_native_launch_probe=turbocore_update_shadow_owner_native_launch_probe,
        turbocore_update_shadow_owner_native_launch_max_numel=turbocore_update_shadow_owner_native_launch_max_numel,
        turbocore_update_shadow_owner_native_event_chain_probe=turbocore_update_shadow_owner_native_event_chain_probe,
        turbocore_update_shadow_save_owner_state=turbocore_update_shadow_save_owner_state,
        turbocore_native_update_mode=turbocore_native_update_mode,
        turbocore_native_update_required_shadow_passes=turbocore_native_update_required_shadow_passes,
        turbocore_native_update_allow_missing_kernel=turbocore_native_update_allow_missing_kernel,
        turbocore_native_update_dispatch_enabled=turbocore_native_update_dispatch_enabled,
        turbocore_native_update_training_path_enabled=turbocore_native_update_training_path_enabled,
        turbocore_native_update_require_native_cuda=turbocore_native_update_require_native_cuda,
        turbocore_native_update_diagnostic_executor_replay=turbocore_native_update_diagnostic_executor_replay,
        turbocore_native_update_defer_state_sync=turbocore_native_update_defer_state_sync,
        turbocore_native_update_runtime_synchronization_policy=(
            turbocore_native_update_runtime_synchronization_policy
        ),
        native_cache_mode=native_cache_mode,
    )
    requested_attention = str(attention_backend or "auto").strip().lower().replace("-", "_")
    if requested_attention not in {"auto", "torch", "xformers", "sdpa", "flash2", "sageattn", "flexattn", "spargeattn2"}:
        requested_attention = "auto"
    cfg.attention_backend = requested_attention
    cfg.sdpa_backend_policy = str(sdpa_backend_policy or "cutlass").strip().lower()
    if requested_attention == "xformers":
        cfg.xformers = True
        cfg.sdpa = False
    elif requested_attention == "sdpa":
        cfg.xformers = False
        cfg.sdpa = True
    elif requested_attention != "auto":
        cfg.xformers = False
    resolved_workers = max(int(dataloader_workers or 0), 0)
    resolved_prefetch = max(int(dataloader_prefetch_factor or 2), 1)
    cfg.dataloader_num_workers = resolved_workers
    cfg.cached_dataloader_workers = resolved_workers
    cfg.prefetch_factor = resolved_prefetch
    cfg.cached_dataloader_prefetch_factor = resolved_prefetch
    cfg.pin_memory = bool(dataloader_pin_memory)
    cfg.cached_dataloader_pin_memory = bool(dataloader_pin_memory)
    cfg.persistent_data_loader_workers = resolved_workers > 0
    cfg.data_transfer_profile_enabled = bool(data_transfer_profile)
    cfg.data_transfer_profile_mode = str(data_transfer_profile_mode or "event")
    cfg.data_transfer_profile_window = max(int(steady_warmup or 0) + 1, 1)
    cfg.torch_compile = bool(torch_compile)
    if cfg.torch_compile:
        scope = str(torch_compile_scope or "per_block").strip().lower().replace("-", "_")
        cfg.torch_compile_scope = scope if scope in {"per_block", "full", "full_core"} else "per_block"
        cfg.torch_compile_backend = str(torch_compile_backend or "inductor")
        cfg.torch_compile_mode = str(torch_compile_mode or "default")
        cfg.torch_compile_dynamic = bool(torch_compile_dynamic)
        cfg.torch_compile_fullgraph = bool(torch_compile_fullgraph)
        cfg.compile_cache_enabled = bool(compile_cache_enabled)
        cfg.compile_cache_reuse = bool(compile_cache_reuse)
        if str(compile_cache_root or "").strip():
            cfg.compile_cache_root = str(compile_cache_root)
    if fused_adamw:
        cfg.optimizer_args = "fused=True"
    recompute_mode = str(lora_activation_recompute or "auto").strip().lower()
    if recompute_mode not in {"auto", "on", "off"}:
        recompute_mode = "auto"
    cfg.lora_activation_recompute_mode = recompute_mode
    if recompute_mode in {"on", "true", "1", "yes"}:
        cfg.lora_activation_recompute = True
    elif recompute_mode in {"off", "false", "0", "no"}:
        cfg.lora_activation_recompute = False
    benchmark_stall_ms = max(float(bubble_controller_benchmark_data_wait_stall_ms or 0.0), 0.0)
    cfg.bubble_controller_benchmark_data_wait_stall_ms = benchmark_stall_ms
    cfg.bubble_controller_benchmark_data_wait_direct_action = bool(
        bubble_controller_benchmark_data_wait_direct_action
    )

    trainer = LulynxTrainer(cfg)
    controlled_data_wait_share = max(float(bubble_controller_controlled_data_wait_share or 0.0), 0.0)
    controlled_data_wait_observations: list[dict[str, Any]] = []
    if controlled_data_wait_share > 0.0:
        controlled_data_wait_features = _controlled_data_wait_runtime_features(
            share=controlled_data_wait_share,
            mean_ms=max(float(bubble_controller_controlled_data_wait_mean_ms or 100.0), 1.0),
        )
        trainer._bubble_controller_external_runtime_features = controlled_data_wait_features
        controlled_data_wait_observations.append(
            dict(controlled_data_wait_features["bubble_controller_controlled_data_wait"])
        )
    logs: list[str] = []
    step_times_ms: list[float] = []
    phase_profiles: list[dict[str, Any]] = []
    update_shadow_reports: list[dict[str, Any]] = []
    native_update_dispatch_arming_reports: list[dict[str, Any]] = []
    native_update_runtime_recovery_observations: list[dict[str, Any]] = []
    native_update_dispatch_runtime_reports: list[dict[str, Any]] = []
    native_update_diagnostic_replay_reports: list[dict[str, Any]] = []
    native_update_dispatch_arming_observations: list[dict[str, Any]] = []
    native_update_gate_reports: list[dict[str, Any]] = []
    native_update_loop_timings: list[dict[str, Any]] = []
    bubble_controlled_rollback_observations: list[dict[str, Any]] = []
    losses: list[float] = []
    peak_vram_mb = 0.0
    setup_peak_vram_mb = 0.0
    training_peak_vram_mb = 0.0
    training_peak_reset = False
    original_on_step_end = trainer._on_step_end

    def _cuda_peak_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    def _on_log(message: str) -> None:
        nonlocal setup_peak_vram_mb, training_peak_reset
        logs.append(message)
        if training_peak_reset or "Starting training" not in str(message):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            setup_peak_vram_mb = max(setup_peak_vram_mb, _cuda_peak_mb())
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        training_peak_reset = True

    slowdown_ratio = max(float(bubble_controller_controlled_rollback_slowdown_ratio or 1.0), 1.0)
    slowdown_after_steps = max(int(bubble_controller_controlled_rollback_after_apply_steps or 0), 0)

    def _active_closed_loop_action() -> Mapping[str, Any]:
        state = getattr(trainer, "_bubble_closed_loop_state", {}) or {}
        if not isinstance(state, Mapping):
            return {}
        active = state.get("active_action")
        return active if isinstance(active, Mapping) else {}

    def _maybe_inject_controlled_rollback_slowdown(step: int, info: dict[str, Any]) -> None:
        if slowdown_ratio <= 1.0:
            return
        active = _active_closed_loop_action()
        if str(active.get("status") or "") not in {"applied", "cooldown"}:
            return
        applied_step = int(active.get("applied_step", step) or step)
        if int(step) < applied_step + slowdown_after_steps:
            return
        original_wall = float(info.get("step_wall_seconds", 0.0) or 0.0)
        if original_wall <= 0.0:
            return
        adjusted_wall = original_wall * slowdown_ratio
        info["step_wall_seconds"] = adjusted_wall
        observation = {
            "schema_version": 1,
            "kind": "bubble_controller_controlled_rollback_slowdown_v0",
            "step": int(step),
            "action_id": str(active.get("action_id") or ""),
            "action_kind": str(active.get("action_kind") or ""),
            "ratio": slowdown_ratio,
            "original_step_wall_seconds": original_wall,
            "adjusted_step_wall_seconds": adjusted_wall,
        }
        info["bubble_controlled_rollback_slowdown"] = dict(observation)
        bubble_controlled_rollback_observations.append(observation)

    def _on_step_end(step: int, loss: float, info: dict[str, Any]) -> None:
        nonlocal training_peak_vram_mb
        _maybe_inject_controlled_rollback_slowdown(step, info)
        step_ms = float(info.get("step_wall_seconds", 0.0) or 0.0) * 1000.0
        step_times_ms.append(step_ms)
        phase_profile_info = info.get("step_phase_profile")
        if isinstance(phase_profile_info, dict):
            phase_profiles.append(dict(phase_profile_info))
        update_shadow = info.get("turbocore_update_shadow")
        if isinstance(update_shadow, dict):
            update_shadow_reports.append(dict(update_shadow))
        dispatch_arming = info.get("turbocore_native_update_dispatch_arming")
        if isinstance(dispatch_arming, dict):
            native_update_dispatch_arming_reports.append(dict(dispatch_arming))
        recovery_observation = info.get("turbocore_native_update_runtime_recovery_observation")
        if isinstance(recovery_observation, dict):
            native_update_runtime_recovery_observations.append(dict(recovery_observation))
        dispatch_runtime = info.get("turbocore_native_update_dispatch_runtime")
        if isinstance(dispatch_runtime, dict):
            native_update_dispatch_runtime_reports.append(dict(dispatch_runtime))
        diagnostic_replay = info.get("turbocore_native_update_diagnostic_replay")
        if isinstance(diagnostic_replay, dict):
            native_update_diagnostic_replay_reports.append(dict(diagnostic_replay))
        native_update_gate = info.get("turbocore_native_update_gate")
        if isinstance(native_update_gate, dict):
            native_update_gate_reports.append(dict(native_update_gate))
        loop_timing = info.get("turbocore_native_update_loop_timing")
        if isinstance(loop_timing, dict):
            native_update_loop_timings.append(dict(loop_timing))
        dispatch_arming_observation = info.get("turbocore_native_update_dispatch_arming_observation")
        if isinstance(dispatch_arming_observation, dict):
            native_update_dispatch_arming_observations.append(dict(dispatch_arming_observation))
        losses.append(float(loss))
        if torch.cuda.is_available():
            training_peak_vram_mb = max(training_peak_vram_mb, _cuda_peak_mb())
        original_on_step_end(step, loss, info)

    trainer._on_step_end = _on_step_end  # type: ignore[assignment]
    trainer.set_callbacks(on_log=_on_log)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    success = trainer.start()
    cache_inventory_after = _cache_inventory(train_dir)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if training_peak_reset:
            training_peak_vram_mb = max(training_peak_vram_mb, _cuda_peak_mb())
        else:
            setup_peak_vram_mb = max(setup_peak_vram_mb, _cuda_peak_mb())
        peak_vram_mb = max(setup_peak_vram_mb, training_peak_vram_mb)
        torch.cuda.empty_cache()
    total_wall_seconds = time.perf_counter() - start

    output_path = output_dir / f"{output_name}.safetensors"
    output_size = output_path.stat().st_size if output_path.exists() else 0
    steady = step_times_ms[steady_warmup:] if len(step_times_ms) > steady_warmup else step_times_ms
    mean_step = _mean(step_times_ms)
    steady_mean = _mean(steady)
    phase_summary, native_gain_estimates = _summarize_phase_profiles(
        phase_profiles,
        steady_warmup=steady_warmup,
    ) if phase_profiles else ({}, {})
    bubble_profile = build_step_phase_bubble_profile(
        phase_profiles,
        steady_warmup=steady_warmup,
    ) if phase_profiles else {}
    runtime_feature_summary = _load_runtime_feature_summary(output_dir)
    effective_batch = max(int(getattr(cfg, "train_batch_size", train_batch_size) or train_batch_size), 1)
    plan = getattr(trainer, "runtime_optimization_plan", None)
    loop = getattr(trainer, "training_loop", None)

    result = RuntimeProfileRun(
        family=family,
        label=label,
        profile=profile,
        success=bool(success),
        steps_completed=int(getattr(loop, "global_step", 0) or 0),
        total_wall_seconds=total_wall_seconds,
        first_step_ms=step_times_ms[0] if step_times_ms else 0.0,
        mean_step_ms=mean_step,
        median_step_ms=_median(step_times_ms),
        steady_mean_step_ms=steady_mean,
        steady_median_step_ms=_median(steady),
        samples_per_second=(1000.0 * effective_batch) / mean_step if mean_step > 0 else 0.0,
        steady_samples_per_second=(1000.0 * effective_batch) / steady_mean if steady_mean > 0 else 0.0,
        peak_vram_mb=peak_vram_mb,
        setup_peak_vram_mb=setup_peak_vram_mb,
        training_peak_vram_mb=training_peak_vram_mb,
        final_loss=losses[-1] if losses else 0.0,
        attention_backend=str(getattr(plan, "attention_backend", "unknown") or "unknown"),
        compile_requested=bool(getattr(plan, "torch_compile", False)),
        compile_scope=str(getattr(plan, "torch_compile_scope", "") or ""),
        optimizer_type=type(getattr(loop, "optimizer", None)).__name__ if loop is not None else "",
        optimizer_args=str(getattr(cfg, "optimizer_args", "") or ""),
        output_path=str(output_path),
        output_size_bytes=output_size,
        step_times_ms=step_times_ms,
        phase_profiles=phase_profiles,
        steady_phase_summary=phase_summary,
        steady_bubble_profile=bubble_profile,
        runtime_feature_summary=runtime_feature_summary,
        native_adamw_gain_estimates=native_gain_estimates,
        update_shadow_reports=update_shadow_reports,
        native_update_dispatch_arming_reports=native_update_dispatch_arming_reports,
        native_update_runtime_recovery_observations=native_update_runtime_recovery_observations,
        native_update_dispatch_runtime_reports=native_update_dispatch_runtime_reports,
        native_update_diagnostic_replay_reports=native_update_diagnostic_replay_reports,
        native_update_dispatch_arming_observations=native_update_dispatch_arming_observations,
        native_update_gate_reports=native_update_gate_reports,
        native_update_loop_timings=native_update_loop_timings,
        native_update_readiness=dict(getattr(loop, "_turbocore_native_update_readiness", {}) or {}) if loop is not None else {},
        bubble_controlled_rollback_observations=bubble_controlled_rollback_observations,
        bubble_controlled_data_wait_observations=controlled_data_wait_observations,
        bubble_benchmark_data_wait_stall={
            "schema_version": 1,
            "kind": "bubble_controller_benchmark_data_wait_stall_v0",
            "scope": "benchmark_only",
            "stall_ms_per_item": round(benchmark_stall_ms, 4),
            "direct_action": bool(bubble_controller_benchmark_data_wait_direct_action),
        } if benchmark_stall_ms > 0.0 else {},
        losses=losses,
        log_tail=logs[-40:],
    )
    result.runtime_feature_summary.setdefault("benchmark_cache_inventory", {})
    result.runtime_feature_summary["benchmark_cache_inventory"] = {
        "before": cache_inventory_before,
        "after": cache_inventory_after,
    }
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


def _compare(standard: RuntimeProfileRun, aggressive: RuntimeProfileRun) -> dict[str, float]:
    return {
        "mean_step_speedup": (standard.mean_step_ms or 1.0) / max(aggressive.mean_step_ms or 1.0, 1e-9),
        "steady_step_speedup": (standard.steady_mean_step_ms or 1.0) / max(aggressive.steady_mean_step_ms or 1.0, 1e-9),
        "end_to_end_speedup": (standard.total_wall_seconds or 1.0) / max(aggressive.total_wall_seconds or 1.0, 1e-9),
        "peak_vram_ratio": (aggressive.peak_vram_mb or 0.0) / max(standard.peak_vram_mb or 1.0, 1e-9),
        "peak_vram_delta_mb": (aggressive.peak_vram_mb or 0.0) - (standard.peak_vram_mb or 0.0),
        "setup_peak_vram_delta_mb": (aggressive.setup_peak_vram_mb or 0.0) - (standard.setup_peak_vram_mb or 0.0),
        "training_peak_vram_ratio": (aggressive.training_peak_vram_mb or 0.0)
        / max(standard.training_peak_vram_mb or 1.0, 1e-9),
        "training_peak_vram_delta_mb": (aggressive.training_peak_vram_mb or 0.0)
        - (standard.training_peak_vram_mb or 0.0),
    }


def _line(run: RuntimeProfileRun) -> str:
    line = (
        f"[benchmark] family={run.family} label={run.label} profile={run.profile} success={run.success} "
        f"steps={run.steps_completed} peak_vram={run.peak_vram_mb:.1f}MB "
        f"setup_peak_vram={run.setup_peak_vram_mb:.1f}MB training_peak_vram={run.training_peak_vram_mb:.1f}MB "
        f"avg_step_time={run.mean_step_ms:.2f}ms steady_step_time={run.steady_mean_step_ms:.2f}ms "
        f"samples_per_sec={run.samples_per_second:.3f} steady_samples_per_sec={run.steady_samples_per_second:.3f} "
        f"attention={run.attention_backend} compile={run.compile_scope or 'off'} "
        f"optimizer={run.optimizer_type} optimizer_args={run.optimizer_args or '-'}"
    )
    phase = run.steady_phase_summary or {}
    if phase:
        line += (
            f" optimizer_update={phase.get('optimizer_update_total_ms_mean', 0.0):.3f}ms"
            f" opt_step_share={phase.get('optimizer_step_share_mean', 0.0) * 100.0:.2f}%"
            f" update_share={phase.get('optimizer_update_share_mean', 0.0) * 100.0:.2f}%"
            f" est_native20x_opt={run.native_adamw_gain_estimates.get('native_20x_optimizer_step_speedup_pct', 0.0):.2f}%"
            f" upper={run.native_adamw_gain_estimates.get('native_20x_update_upper_bound_speedup_pct', 0.0):.2f}%"
        )
    bubble = run.steady_bubble_profile or {}
    if bubble:
        line += (
            f" bubble={bubble.get('bubble_ratio_estimate', 0.0) * 100.0:.2f}%"
            f" bottleneck={bubble.get('dominant_bottleneck', 'unknown')}"
        )
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("sd15", "sdxl", "anima", "newbie"), required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--steady-warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--network-dim", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument(
        "--native-cache-mode",
        choices=("cache_first", "rebuild_cache", "online_cache", "raw_online", "force_cache_only"),
        default="",
        help="Runtime/request cache mode passed through to Anima/Newbie trainers.",
    )
    parser.add_argument("--data-transfer-profile", action="store_true")
    parser.add_argument("--data-transfer-profile-mode", choices=("event", "sync", "off"), default="event")
    parser.add_argument("--fused-adamw", action="store_true", help="Force optimizer_args=fused=True for both benchmark profiles.")
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "torch", "xformers", "sdpa", "flash2", "sageattn", "flexattn", "spargeattn2"),
        default="auto",
        help="Dev-only: request an attention backend through the normal trainer runtime contract.",
    )
    parser.add_argument(
        "--sdpa-backend-policy",
        choices=("auto", "cutlass", "flash", "cudnn", "math"),
        default="cutlass",
        help="Dev-only: SDPA kernel policy passed through when attention_backend resolves to sdpa.",
    )
    parser.add_argument("--torch-compile", action="store_true", help="Dev-only: enable torch.compile through the normal trainer runtime contract.")
    parser.add_argument("--torch-compile-scope", choices=("per_block", "full", "full_core"), default="per_block")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--torch-compile-dynamic", action="store_true")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--compile-cache-root", default="")
    parser.add_argument("--no-compile-cache", action="store_true")
    parser.add_argument("--no-compile-cache-reuse", action="store_true")
    parser.add_argument(
        "--phase-profile",
        action="store_true",
        help="Synchronize CUDA and record coarse train_step/optimizer phase timing.",
    )
    parser.add_argument("--bubble-controller-enabled", action="store_true")
    parser.add_argument(
        "--bubble-controller-mode",
        choices=("report_only", "advisor_patch", "auto_apply"),
        default="report_only",
    )
    parser.add_argument("--bubble-controller-warmup-steps", type=int, default=8)
    parser.add_argument("--bubble-controller-tune-interval-steps", type=int, default=32)
    parser.add_argument("--bubble-controller-max-actions-per-run", type=int, default=3)
    parser.add_argument("--bubble-controller-min-throughput-gain", type=float, default=0.03)
    parser.add_argument(
        "--bubble-controller-allow-dataloader-rebuild-current-run",
        action="store_true",
        help="Experimental benchmark gate: allow epoch-boundary current-run DataLoader rebuild actions.",
    )
    parser.add_argument(
        "--bubble-controller-controlled-rollback-slowdown-ratio",
        type=float,
        default=1.0,
        help="Benchmark-only hook: inflate observed post-apply step time to exercise rollback evidence.",
    )
    parser.add_argument(
        "--bubble-controller-controlled-rollback-after-apply-steps",
        type=int,
        default=1,
        help="Benchmark-only hook: start the controlled rollback slowdown N steps after auto-apply.",
    )
    parser.add_argument(
        "--bubble-controller-controlled-data-wait-share",
        type=float,
        default=0.0,
        help="Benchmark-only hook: inject a controlled data_wait_share into the bubble controller evidence.",
    )
    parser.add_argument(
        "--bubble-controller-controlled-data-wait-mean-ms",
        type=float,
        default=100.0,
        help="Benchmark-only hook: synthetic step mean used with controlled data_wait_share evidence.",
    )
    parser.add_argument(
        "--bubble-controller-benchmark-data-wait-stall-ms",
        type=float,
        default=0.0,
        help="Benchmark-only hook: sleep inside Anima cache item loading to create real profiler data_wait.",
    )
    parser.add_argument(
        "--bubble-controller-benchmark-data-wait-direct-action",
        action="store_true",
        help="Benchmark-only hook: let real data_wait stall bypass the sync-profiler host guard.",
    )
    parser.add_argument("--turbocore-update-shadow", choices=("off", "profile", "shadow"), default="off")
    parser.add_argument("--turbocore-update-shadow-direct-grad", action="store_true")
    parser.add_argument("--turbocore-update-shadow-compare-sample-params", type=int, default=0)
    parser.add_argument("--turbocore-update-shadow-stop-after-consecutive-passes", type=int, default=0)
    parser.add_argument("--turbocore-update-shadow-checkpoint-contract", action="store_true")
    parser.add_argument("--turbocore-update-shadow-copyback-probe", action="store_true")
    parser.add_argument("--turbocore-update-shadow-copyback-dispatch-experimental", action="store_true")
    parser.add_argument("--turbocore-update-shadow-native-binding-probe", action="store_true")
    parser.add_argument("--turbocore-update-shadow-owner-native-launch-probe", action="store_true")
    parser.add_argument("--turbocore-update-shadow-owner-native-launch-max-numel", type=int, default=1048576)
    parser.add_argument("--turbocore-update-shadow-owner-native-event-chain-probe", action="store_true")
    parser.add_argument("--turbocore-update-shadow-save-owner-state", action="store_true")
    parser.add_argument("--turbocore-native-update-mode", choices=("off", "profile", "native_experimental"), default="off")
    parser.add_argument("--turbocore-native-update-required-shadow-passes", type=int, default=3)
    parser.add_argument("--turbocore-native-update-allow-missing-kernel", action="store_true")
    parser.add_argument("--turbocore-native-update-dispatch-enabled", action="store_true")
    parser.add_argument("--turbocore-native-update-training-path-enabled", action="store_true")
    parser.add_argument("--turbocore-native-update-require-native-cuda", action="store_true")
    parser.add_argument("--turbocore-native-update-diagnostic-executor-replay", action="store_true")
    parser.add_argument("--turbocore-native-update-defer-state-sync", action="store_true")
    parser.add_argument(
        "--turbocore-native-update-runtime-synchronization-policy",
        choices=("context_synchronize", "borrowed_stream_event_chain"),
        default="context_synchronize",
    )
    parser.add_argument("--anima-latent-crop-size", type=int, default=4, help="Anima cached latent crop size. Use 0 for full cached latents.")
    parser.add_argument("--anima-fixed-text-tokens", type=int, default=16)
    parser.add_argument("--anima-fixed-visual-tokens", type=int, default=4, help="Use 0 to avoid fixed-token padding.")
    parser.add_argument("--anima-block-residency", default="resident", choices=BLOCK_RESIDENCY_MODE_CHOICES)
    parser.add_argument("--anima-block-residency-min-params", type=int, default=0)
    parser.add_argument("--anima-block-checkpointing", action="store_true", help="Recompute native Anima DiT blocks during backward.")
    parser.add_argument("--anima-block-checkpointing-mode", default="block", choices=("block",))
    parser.add_argument("--anima-block-prefetch", action="store_true", help="Async prefetch CPU-pinned Anima DiT Linear weights for streaming_offload.")
    parser.add_argument("--anima-block-prefetch-depth", type=int, default=1)
    parser.add_argument("--newbie-latent-crop-size", type=int, default=4, help="Newbie cached latent crop size. Use 0 for full cached latents.")
    parser.add_argument("--newbie-fixed-text-tokens", type=int, default=64)
    parser.add_argument("--newbie-fixed-visual-tokens", type=int, default=4, help="Use 0 to avoid fixed-token padding.")
    parser.add_argument("--newbie-block-residency", default="resident", choices=BLOCK_RESIDENCY_MODE_CHOICES)
    parser.add_argument("--newbie-block-residency-min-params", type=int, default=0)
    parser.add_argument("--newbie-block-checkpointing", action="store_true", help="Recompute native Newbie DiT blocks during backward.")
    parser.add_argument("--newbie-block-checkpointing-mode", default="block", choices=("block",))
    parser.add_argument("--newbie-block-prefetch", action="store_true", help="Async prefetch CPU-pinned Newbie DiT Linear weights for streaming_offload.")
    parser.add_argument("--newbie-block-prefetch-depth", type=int, default=1)
    parser.add_argument("--source-data", type=Path, default=None, help="Optional dataset/cache source directory.")
    parser.add_argument("--profiles", nargs="+", choices=("standard", "aggressive"), default=("standard", "aggressive"))
    parser.add_argument(
        "--lora-activation-recompute",
        choices=("auto", "on", "off"),
        default="auto",
        help="Override lora_activation_recompute for A/B memory tests.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo = _repo_root()
    run_root = args.out or repo / "temp" / "native_runtime_profile_benchmark" / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    resolution = int(args.resolution or (64 if args.family in {"anima", "newbie"} else 512))
    if args.family == "sd15":
        try:
            _resolve_sd15_checkpoint(repo / "models")
        except FileNotFoundError as exc:
            print(f"[benchmark] family=sd15 status=missing_resource reason={exc}", flush=True)
            return 2

    anima_block_residency = _normalize_block_residency_mode(args.anima_block_residency)
    newbie_block_residency = _normalize_block_residency_mode(args.newbie_block_residency)
    anima_fixed_visual_tokens = max(int(args.anima_fixed_visual_tokens), 0)
    newbie_fixed_visual_tokens = max(int(args.newbie_fixed_visual_tokens), 0)
    anima_block_checkpointing = bool(args.anima_block_checkpointing)
    newbie_block_checkpointing = bool(args.newbie_block_checkpointing)
    native_update_mode = str(args.turbocore_native_update_mode or "off")
    effective_update_shadow = str(args.turbocore_update_shadow or "off")
    if native_update_mode != "off" and effective_update_shadow == "off":
        effective_update_shadow = "shadow"
    anima_checkpoint_source = "cli" if anima_block_checkpointing else "off"
    newbie_checkpoint_source = "cli" if newbie_block_checkpointing else "off"
    if (
        args.family == "anima"
        and not anima_block_checkpointing
        and _dit_block_checkpoint_recommended(
            anima_block_residency,
            resolution=resolution,
            fixed_visual_tokens=anima_fixed_visual_tokens,
        )
    ):
        anima_block_checkpointing = True
        anima_checkpoint_source = "auto_for_dit_residency"
    if (
        args.family == "newbie"
        and not newbie_block_checkpointing
        and _dit_block_checkpoint_recommended(
            newbie_block_residency,
            resolution=resolution,
            fixed_visual_tokens=newbie_fixed_visual_tokens,
        )
    ):
        newbie_block_checkpointing = True
        newbie_checkpoint_source = "auto_for_dit_residency"
    bubble_controller = {
        "enabled": bool(args.bubble_controller_enabled),
        "mode": str(args.bubble_controller_mode or "report_only"),
        "warmup_steps": max(int(args.bubble_controller_warmup_steps), 0),
        "tune_interval_steps": max(int(args.bubble_controller_tune_interval_steps), 1),
        "max_actions_per_run": max(int(args.bubble_controller_max_actions_per_run), 0),
        "min_throughput_gain": max(float(args.bubble_controller_min_throughput_gain), 0.0),
        "allow_dataloader_rebuild_current_run": bool(
            args.bubble_controller_allow_dataloader_rebuild_current_run
        ),
    }

    print(
        f"Running {args.family} profile benchmark: steps={args.steps} steady_warmup={args.steady_warmup} "
        f"samples={args.samples} resolution={resolution} batch_size={max(int(args.train_batch_size), 1)} "
        f"grad_accum={max(int(args.gradient_accumulation_steps), 1)} "
        f"lr={float(args.learning_rate) if float(args.learning_rate) > 0.0 else 'default'} "
        f"phase_profile={bool(args.phase_profile)} update_shadow={effective_update_shadow} "
        f"native_update_gate={native_update_mode} "
        f"native_update_sync_policy={args.turbocore_native_update_runtime_synchronization_policy} "
        f"attention_backend={args.attention_backend} sdpa_policy={args.sdpa_backend_policy} "
        f"torch_compile={bool(args.torch_compile)} "
        f"compile_scope={args.torch_compile_scope if args.torch_compile else 'off'} "
        f"bubble_controller={bubble_controller['mode'] if bubble_controller['enabled'] else 'off'}",
        flush=True,
    )
    runs: dict[str, RuntimeProfileRun] = {}
    for profile in dict.fromkeys(args.profiles):
        runs[profile] = _run_profile(
            family=args.family,
            label=profile,
            profile=profile,
            model_root=repo / "models",
            source_data=args.source_data or repo / "sucai" / "6_lulu",
            run_root=run_root,
            steps=max(args.steps, 1),
            steady_warmup=max(args.steady_warmup, 0),
            resolution=resolution,
            network_dim=max(args.network_dim, 1),
            train_batch_size=max(int(args.train_batch_size), 1),
            gradient_accumulation_steps=max(int(args.gradient_accumulation_steps), 1),
            learning_rate=float(args.learning_rate) if float(args.learning_rate) > 0.0 else None,
            sample_count=max(args.samples, 1),
            fused_adamw=bool(args.fused_adamw),
            attention_backend=str(args.attention_backend or "auto"),
            sdpa_backend_policy=str(args.sdpa_backend_policy or "cutlass"),
            dataloader_workers=max(int(args.dataloader_workers), 0),
            dataloader_prefetch_factor=max(int(args.dataloader_prefetch_factor), 1),
            dataloader_pin_memory=not bool(args.no_pin_memory),
            data_transfer_profile=bool(args.data_transfer_profile),
            data_transfer_profile_mode=str(args.data_transfer_profile_mode),
            anima_latent_crop_size=max(int(args.anima_latent_crop_size), 0),
            anima_fixed_text_tokens=max(int(args.anima_fixed_text_tokens), 0),
            anima_fixed_visual_tokens=anima_fixed_visual_tokens,
            anima_block_residency=anima_block_residency,
            anima_block_residency_min_params=max(int(args.anima_block_residency_min_params), 0),
            anima_block_checkpointing=anima_block_checkpointing,
            anima_block_checkpointing_mode=str(args.anima_block_checkpointing_mode),
            anima_block_prefetch=bool(args.anima_block_prefetch),
            anima_block_prefetch_depth=max(int(args.anima_block_prefetch_depth), 0),
            newbie_latent_crop_size=max(int(args.newbie_latent_crop_size), 0),
            newbie_fixed_text_tokens=max(int(args.newbie_fixed_text_tokens), 0),
            newbie_fixed_visual_tokens=newbie_fixed_visual_tokens,
            newbie_block_residency=newbie_block_residency,
            newbie_block_residency_min_params=max(int(args.newbie_block_residency_min_params), 0),
            newbie_block_checkpointing=newbie_block_checkpointing,
            newbie_block_checkpointing_mode=str(args.newbie_block_checkpointing_mode),
            newbie_block_prefetch=bool(args.newbie_block_prefetch),
            newbie_block_prefetch_depth=max(int(args.newbie_block_prefetch_depth), 0),
            lora_activation_recompute=str(args.lora_activation_recompute),
            phase_profile=bool(args.phase_profile),
            bubble_controller=bubble_controller,
            turbocore_update_shadow=effective_update_shadow,
            turbocore_update_shadow_direct_grad=bool(args.turbocore_update_shadow_direct_grad),
            turbocore_update_shadow_compare_sample_params=max(int(args.turbocore_update_shadow_compare_sample_params), 0),
            turbocore_update_shadow_stop_after_consecutive_passes=max(int(args.turbocore_update_shadow_stop_after_consecutive_passes), 0),
            turbocore_update_shadow_checkpoint_contract=bool(args.turbocore_update_shadow_checkpoint_contract),
            turbocore_update_shadow_copyback_probe=bool(args.turbocore_update_shadow_copyback_probe),
            turbocore_update_shadow_copyback_dispatch_experimental=bool(
                args.turbocore_update_shadow_copyback_dispatch_experimental
            ),
            turbocore_update_shadow_native_binding_probe=bool(args.turbocore_update_shadow_native_binding_probe),
            turbocore_update_shadow_owner_native_launch_probe=bool(
                args.turbocore_update_shadow_owner_native_launch_probe
            ),
            turbocore_update_shadow_owner_native_launch_max_numel=max(
                int(args.turbocore_update_shadow_owner_native_launch_max_numel),
                1,
            ),
            turbocore_update_shadow_owner_native_event_chain_probe=bool(
                args.turbocore_update_shadow_owner_native_event_chain_probe
            ),
            turbocore_update_shadow_save_owner_state=bool(args.turbocore_update_shadow_save_owner_state),
            turbocore_native_update_mode=native_update_mode,
            turbocore_native_update_required_shadow_passes=max(
                int(args.turbocore_native_update_required_shadow_passes),
                1,
            ),
            turbocore_native_update_allow_missing_kernel=bool(
                args.turbocore_native_update_allow_missing_kernel
            ),
            turbocore_native_update_dispatch_enabled=bool(args.turbocore_native_update_dispatch_enabled),
            turbocore_native_update_training_path_enabled=bool(
                args.turbocore_native_update_training_path_enabled
            ),
            turbocore_native_update_require_native_cuda=bool(args.turbocore_native_update_require_native_cuda),
            turbocore_native_update_diagnostic_executor_replay=bool(
                args.turbocore_native_update_diagnostic_executor_replay
            ),
            turbocore_native_update_defer_state_sync=bool(args.turbocore_native_update_defer_state_sync),
            turbocore_native_update_runtime_synchronization_policy=str(
                args.turbocore_native_update_runtime_synchronization_policy
            ),
            bubble_controller_controlled_rollback_slowdown_ratio=max(
                float(args.bubble_controller_controlled_rollback_slowdown_ratio),
                1.0,
            ),
            bubble_controller_controlled_rollback_after_apply_steps=max(
                int(args.bubble_controller_controlled_rollback_after_apply_steps),
                0,
            ),
            bubble_controller_controlled_data_wait_share=max(
                float(args.bubble_controller_controlled_data_wait_share),
                0.0,
            ),
            bubble_controller_controlled_data_wait_mean_ms=max(
                float(args.bubble_controller_controlled_data_wait_mean_ms),
                1.0,
            ),
            bubble_controller_benchmark_data_wait_stall_ms=max(
                float(args.bubble_controller_benchmark_data_wait_stall_ms),
                0.0,
            ),
            bubble_controller_benchmark_data_wait_direct_action=bool(
                args.bubble_controller_benchmark_data_wait_direct_action
            ),
            native_cache_mode=str(args.native_cache_mode or ""),
            torch_compile=bool(args.torch_compile),
            torch_compile_scope=str(args.torch_compile_scope or "per_block"),
            torch_compile_backend=str(args.torch_compile_backend or "inductor"),
            torch_compile_mode=str(args.torch_compile_mode or "default"),
            torch_compile_dynamic=bool(args.torch_compile_dynamic),
            torch_compile_fullgraph=bool(args.torch_compile_fullgraph),
            compile_cache_enabled=not bool(args.no_compile_cache),
            compile_cache_reuse=not bool(args.no_compile_cache_reuse),
            compile_cache_root=str(args.compile_cache_root or ""),
        )
    comparison = (
        _compare(runs["standard"], runs["aggressive"])
        if "standard" in runs and "aggressive" in runs
        else {}
    )

    payload = {
        "benchmark": {
            "family": args.family,
            "steps": max(args.steps, 1),
            "steady_warmup": max(args.steady_warmup, 0),
            "samples": max(args.samples, 1),
            "resolution": resolution,
            "train_batch_size": max(int(args.train_batch_size), 1),
            "gradient_accumulation_steps": max(int(args.gradient_accumulation_steps), 1),
            "learning_rate": float(args.learning_rate) if float(args.learning_rate) > 0.0 else (1e-5 if args.family == "newbie" else 1e-4),
            "dataloader_workers": max(int(args.dataloader_workers), 0),
            "dataloader_prefetch_factor": max(int(args.dataloader_prefetch_factor), 1),
            "pin_memory": not bool(args.no_pin_memory),
            "native_cache_mode": str(args.native_cache_mode or ""),
            "data_transfer_profile": bool(args.data_transfer_profile),
            "data_transfer_profile_mode": str(args.data_transfer_profile_mode),
            "fused_adamw": bool(args.fused_adamw),
            "attention_backend": str(args.attention_backend or "auto"),
            "sdpa_backend_policy": str(args.sdpa_backend_policy or "cutlass"),
            "torch_compile": bool(args.torch_compile),
            "torch_compile_scope": str(args.torch_compile_scope or "per_block") if args.torch_compile else "",
            "torch_compile_backend": str(args.torch_compile_backend or "inductor"),
            "torch_compile_mode": str(args.torch_compile_mode or "default"),
            "torch_compile_dynamic": bool(args.torch_compile_dynamic),
            "torch_compile_fullgraph": bool(args.torch_compile_fullgraph),
            "compile_cache_enabled": not bool(args.no_compile_cache),
            "compile_cache_reuse": not bool(args.no_compile_cache_reuse),
            "compile_cache_root": str(args.compile_cache_root or ""),
            "phase_profile": bool(args.phase_profile),
            "bubble_controller_enabled": bool(args.bubble_controller_enabled),
            "bubble_controller_mode": str(args.bubble_controller_mode or "report_only"),
            "bubble_controller_warmup_steps": max(int(args.bubble_controller_warmup_steps), 0),
            "bubble_controller_tune_interval_steps": max(int(args.bubble_controller_tune_interval_steps), 1),
            "bubble_controller_max_actions_per_run": max(int(args.bubble_controller_max_actions_per_run), 0),
            "bubble_controller_min_throughput_gain": max(float(args.bubble_controller_min_throughput_gain), 0.0),
            "bubble_controller_allow_dataloader_rebuild_current_run": bool(
                args.bubble_controller_allow_dataloader_rebuild_current_run
            ),
            "bubble_controller_controlled_rollback_slowdown_ratio": max(
                float(args.bubble_controller_controlled_rollback_slowdown_ratio),
                1.0,
            ),
            "bubble_controller_controlled_rollback_after_apply_steps": max(
                int(args.bubble_controller_controlled_rollback_after_apply_steps),
                0,
            ),
            "bubble_controller_controlled_data_wait_share": max(
                float(args.bubble_controller_controlled_data_wait_share),
                0.0,
            ),
            "bubble_controller_controlled_data_wait_mean_ms": max(
                float(args.bubble_controller_controlled_data_wait_mean_ms),
                1.0,
            ),
            "bubble_controller_benchmark_data_wait_stall_ms": max(
                float(args.bubble_controller_benchmark_data_wait_stall_ms),
                0.0,
            ),
            "bubble_controller_benchmark_data_wait_direct_action": bool(
                args.bubble_controller_benchmark_data_wait_direct_action
            ),
            "turbocore_update_shadow": effective_update_shadow,
            "turbocore_update_shadow_requested": str(args.turbocore_update_shadow),
            "turbocore_update_shadow_direct_grad": bool(args.turbocore_update_shadow_direct_grad),
            "turbocore_update_shadow_compare_sample_params": max(int(args.turbocore_update_shadow_compare_sample_params), 0),
            "turbocore_update_shadow_stop_after_consecutive_passes": max(int(args.turbocore_update_shadow_stop_after_consecutive_passes), 0),
            "turbocore_update_shadow_checkpoint_contract": bool(args.turbocore_update_shadow_checkpoint_contract),
            "turbocore_update_shadow_copyback_probe": bool(args.turbocore_update_shadow_copyback_probe),
            "turbocore_update_shadow_copyback_dispatch_experimental": bool(
                args.turbocore_update_shadow_copyback_dispatch_experimental
            ),
            "turbocore_update_shadow_native_binding_probe": bool(args.turbocore_update_shadow_native_binding_probe),
            "turbocore_update_shadow_owner_native_launch_probe": bool(
                args.turbocore_update_shadow_owner_native_launch_probe
            ),
            "turbocore_update_shadow_owner_native_launch_max_numel": max(
                int(args.turbocore_update_shadow_owner_native_launch_max_numel),
                1,
            ),
            "turbocore_update_shadow_owner_native_event_chain_probe": bool(
                args.turbocore_update_shadow_owner_native_event_chain_probe
            ),
            "turbocore_update_shadow_save_owner_state": bool(args.turbocore_update_shadow_save_owner_state),
            "turbocore_native_update_mode": native_update_mode,
            "turbocore_native_update_required_shadow_passes": max(
                int(args.turbocore_native_update_required_shadow_passes),
                1,
            ),
            "turbocore_native_update_allow_missing_kernel": bool(
                args.turbocore_native_update_allow_missing_kernel
            ),
            "turbocore_native_update_dispatch_enabled": bool(args.turbocore_native_update_dispatch_enabled),
            "turbocore_native_update_training_path_enabled": bool(
                args.turbocore_native_update_training_path_enabled
            ),
            "turbocore_native_update_require_native_cuda": bool(args.turbocore_native_update_require_native_cuda),
            "turbocore_native_update_diagnostic_executor_replay": bool(
                args.turbocore_native_update_diagnostic_executor_replay
            ),
            "turbocore_native_update_defer_state_sync": bool(args.turbocore_native_update_defer_state_sync),
            "turbocore_native_update_runtime_synchronization_policy": str(
                args.turbocore_native_update_runtime_synchronization_policy
            ),
            "anima_latent_crop_size": max(int(args.anima_latent_crop_size), 0),
            "anima_fixed_text_tokens": max(int(args.anima_fixed_text_tokens), 0),
            "anima_fixed_visual_tokens": anima_fixed_visual_tokens,
            "anima_block_residency": anima_block_residency,
            "anima_block_residency_min_params": max(int(args.anima_block_residency_min_params), 0),
            "anima_block_checkpointing": anima_block_checkpointing,
            "anima_block_checkpointing_requested": bool(args.anima_block_checkpointing),
            "anima_block_checkpointing_source": anima_checkpoint_source,
            "anima_block_checkpointing_mode": str(args.anima_block_checkpointing_mode),
            "anima_block_prefetch": bool(args.anima_block_prefetch),
            "anima_block_prefetch_depth": max(int(args.anima_block_prefetch_depth), 0),
            "newbie_latent_crop_size": max(int(args.newbie_latent_crop_size), 0),
            "newbie_fixed_text_tokens": max(int(args.newbie_fixed_text_tokens), 0),
            "newbie_fixed_visual_tokens": newbie_fixed_visual_tokens,
            "newbie_block_residency": newbie_block_residency,
            "newbie_block_residency_min_params": max(int(args.newbie_block_residency_min_params), 0),
            "newbie_block_checkpointing": newbie_block_checkpointing,
            "newbie_block_checkpointing_requested": bool(args.newbie_block_checkpointing),
            "newbie_block_checkpointing_source": newbie_checkpoint_source,
            "newbie_block_checkpointing_mode": str(args.newbie_block_checkpointing_mode),
            "newbie_block_prefetch": bool(args.newbie_block_prefetch),
            "newbie_block_prefetch_depth": max(int(args.newbie_block_prefetch_depth), 0),
            "lora_activation_recompute": str(args.lora_activation_recompute),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "runs": {key: asdict(value) for key, value in runs.items()},
        "comparison": comparison,
    }
    payload["native_update_performance_report"] = build_native_update_profile_performance_report(payload)
    summary_path = run_root / f"{args.family}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for run in runs.values():
        print(_line(run), flush=True)
    print(f"[benchmark] family={args.family} comparison={json.dumps(payload['comparison'], sort_keys=True)}", flush=True)
    print(f"[benchmark] family={args.family} summary={summary_path}", flush=True)
    return 0 if all(run.success for run in runs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
