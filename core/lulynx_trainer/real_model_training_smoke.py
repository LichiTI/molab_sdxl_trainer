"""Real-model local training runner for /models + /sucai assets.

This script is intentionally heavier than the tiny matrix / cached smokes:

1. it uses real local model assets under ``models/``
2. it uses real local training material under ``sucai/``
3. it runs the product ``LulynxTrainer.start()`` path
4. it defaults to at least 40 training steps

It is still a smoke-oriented runner rather than a full production recipe:
- it copies a bounded subset of samples into a unique work directory
- it keeps resolution / ranks conservative by default
- it avoids mutating the source dataset tree
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.configs import (
    LyCORISAlgo,
    MixedPrecision,
    ModelArch,
    NetworkType,
    OptimizerType,
    SchedulerType,
    UnifiedTrainingConfig,
)
from core.lulynx_trainer.trainer import LulynxTrainer


SUPPORTED_FAMILIES = ("newbie", "anima", "sdxl")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEFAULT_OUTPUT_FALLBACK_FREE_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_ADAPTERS = (
    "lora",
    "lora_plus",
    "dora",
    "lora_fa",
    "vera",
    "tlora",
    "hydralora",
    "fera",
    "loha",
    "locon",
    "lokr",
    "ia3",
    "full",
    "diag_oft",
)


@dataclass
class RealTrainCaseResult:
    family: str
    adapter: str
    ok: bool
    work_dir: str
    output_dir: str
    train_dir: str
    artifact: str = ""
    global_step: int = 0
    resolved_steps: int = 0
    resolved_epochs: int = 0
    duration_seconds: float = 0.0
    copy_report: dict[str, object] | None = None
    optimizer_runtime: dict[str, object] | None = None
    memory_optimization: dict[str, object] | None = None
    native_unet: dict[str, object] | None = None
    anima_block_residency: dict[str, object] | None = None
    newbie_block_residency: dict[str, object] | None = None
    newbie_cache_first_profile: dict[str, object] | None = None
    sample_files: list[str] | None = None
    runtime_event_tail: list[dict[str, object]] | None = None
    log_tail: list[str] | None = None
    error: str = ""


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _disk_free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return 0


def _resolve_session_parent(repo_root: Path, explicit_output_root: str) -> tuple[Path, str]:
    if explicit_output_root:
        parent = Path(explicit_output_root)
        parent.mkdir(parents=True, exist_ok=True)
        return parent, "explicit"

    repo_tmp = repo_root / "backend" / "tmp"
    repo_tmp.mkdir(parents=True, exist_ok=True)
    repo_free = _disk_free_bytes(repo_tmp)
    if repo_free >= DEFAULT_OUTPUT_FALLBACK_FREE_BYTES:
        return repo_tmp, "repo_tmp"

    fallback = Path.home() / ".codex" / "memories" / "lulynx_real_adapter_train"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, f"fallback_low_disk:{repo_free}"


def _create_session_root(parent: Path) -> Path:
    """Create a readable session directory without tempfile's restrictive ACL quirks."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for attempt in range(1000):
        suffix = f"_{attempt:03d}" if attempt else ""
        candidate = parent / f"lulynx_real_adapter_train_{stamp}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Unable to allocate session directory under {parent}")


def _iter_images(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            yield child


def _pick_samples(source_dir: Path, limit: int) -> list[Path]:
    images = list(_iter_images(source_dir))
    if not images:
        raise FileNotFoundError(f"No images found in {source_dir}")
    return images[: max(int(limit), 1)]


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _normalize_adapter(adapter: str) -> str:
    normalized = str(adapter or "").strip().lower().replace("-", "_")
    alias_map = {
        "": "lora",
        "standard": "lora",
        "lora+": "lora_plus",
        "loraplus": "lora_plus",
        "hydra_lora": "hydralora",
        "hydra-lora": "hydralora",
        "diag_oft": "diag-oft",
        "diag-oft": "diag-oft",
        "oft": "diag-oft",
        "lycoris_diag_oft": "diag-oft",
        "lycoris_lokr": "lokr",
        "lycoris_loha": "loha",
        "lycoris_locon": "locon",
        "lycoris_ia3": "ia3",
        "lycoris_full": "full",
    }
    return alias_map.get(normalized, normalized)


def _adapter_tag(adapter: str) -> str:
    return _normalize_adapter(adapter).replace("-", "_")


def _resolve_selected_families(args: argparse.Namespace) -> list[str]:
    raw = list(args.families) if args.families else [args.family]
    seen: set[str] = set()
    families: list[str] = []
    for family in raw:
        normalized = str(family or "").strip().lower()
        if normalized not in SUPPORTED_FAMILIES or normalized in seen:
            continue
        seen.add(normalized)
        families.append(normalized)
    if not families:
        raise ValueError("No valid families selected")
    return families


def _resolve_selected_adapters(args: argparse.Namespace) -> list[str]:
    raw = list(args.adapters) if args.adapters else (list(DEFAULT_ADAPTERS) if args.matrix else [args.adapter])
    seen: set[str] = set()
    adapters: list[str] = []
    for adapter in raw:
        normalized = _normalize_adapter(adapter)
        if normalized in seen:
            continue
        seen.add(normalized)
        adapters.append(normalized)
    if not adapters:
        raise ValueError("No adapters selected")
    return adapters


def _cleanup_runtime() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_adapter_overrides(config: UnifiedTrainingConfig, *, family: str, adapter: str) -> None:
    normalized = _normalize_adapter(adapter)
    lycoris_map = {
        "loha": LyCORISAlgo.LOHA,
        "locon": LyCORISAlgo.LOCON,
        "lokr": LyCORISAlgo.LOKR,
        "ia3": LyCORISAlgo.IA3,
        "full": LyCORISAlgo.FULL,
        "diag-oft": LyCORISAlgo.DIAG_OFT,
        "glora": LyCORISAlgo.GLORA,
        "glokr": LyCORISAlgo.GLOKR,
    }

    config.network_module = NetworkType.LORA
    config.lycoris_algo = LyCORISAlgo.LOHA
    config.use_dora = False
    config.dora_enabled = False
    config.lora_fa_enabled = False
    config.vera_enabled = False
    config.lora_plus_enabled = False
    config.hydralora_enabled = False
    config.fera_enabled = False

    if family == "newbie":
        config.newbie_adapter_type = normalized

    if normalized in lycoris_map:
        config.network_module = NetworkType.LYCORIS
        config.lycoris_algo = lycoris_map[normalized]
        return

    if normalized == "lora":
        return
    if normalized == "lora_plus":
        config.lora_plus_enabled = True
        return
    if normalized == "dora":
        config.use_dora = True
        config.dora_enabled = True
        return
    if normalized == "lora_fa":
        config.network_module = NetworkType.LORA_FA
        config.lora_fa_enabled = True
        return
    if normalized == "vera":
        config.network_module = NetworkType.VERA
        config.vera_enabled = True
        return
    if normalized == "tlora":
        config.network_module = NetworkType.TLORA
        return
    if normalized == "hydralora":
        config.hydralora_enabled = True
        return
    if normalized == "fera":
        config.fera_enabled = True
        return

    raise ValueError(f"Unsupported adapter: {adapter}")


def _materialize_dataset_subset(
    source_dir: Path,
    target_dir: Path,
    *,
    family: str,
    sample_limit: int,
    caption_extension: str,
) -> dict[str, object]:
    target_dir.mkdir(parents=True, exist_ok=True)
    picked = _pick_samples(source_dir, sample_limit)

    copied = 0
    cached_pairs = 0
    missing_captions: list[str] = []

    for image in picked:
        stem = image.stem
        copied_image = target_dir / image.name
        shutil.copy2(image, copied_image)
        copied += 1

        caption_src = image.with_suffix(caption_extension)
        if caption_src.exists():
            shutil.copy2(caption_src, target_dir / caption_src.name)
        else:
            missing_captions.append(image.name)

        # Keep sidecar safetensors when present; some local workflows may rely on them.
        _copy_if_exists(image.with_name(f"{image.name}.safetensors"), target_dir / f"{image.name}.safetensors")
        _copy_if_exists(caption_src.with_name(f"{caption_src.name}.safetensors"), target_dir / f"{caption_src.name}.safetensors")

        if family == "anima":
            latent_candidates = sorted(source_dir.glob(f"{stem}_*_anima.npz"))
            if latent_candidates:
                shutil.copy2(latent_candidates[0], target_dir / latent_candidates[0].name)
            _copy_if_exists(source_dir / f"{stem}_anima_te.npz", target_dir / f"{stem}_anima_te.npz")
            if latent_candidates and (source_dir / f"{stem}_anima_te.npz").exists():
                cached_pairs += 1
        elif family == "sdxl":
            latent_candidates = sorted(source_dir.glob(f"{stem}_*_sdxl.npz"))
            if latent_candidates:
                shutil.copy2(latent_candidates[0], target_dir / latent_candidates[0].name)
            _copy_if_exists(source_dir / f"{stem}_te_outputs.npz", target_dir / f"{stem}_te_outputs.npz")
            if latent_candidates and (source_dir / f"{stem}_te_outputs.npz").exists():
                cached_pairs += 1
        elif family == "newbie":
            cache_copied = False
            for suffix in ("_newbie.npz", "_newbie.safetensors", "_newbie.pt"):
                cache_src = source_dir / f"{stem}{suffix}"
                if _copy_if_exists(cache_src, target_dir / cache_src.name):
                    cache_copied = True
            if cache_copied:
                cached_pairs += 1

    if family == "newbie" and cached_pairs:
        for metadata_name in ("lulynx_cache_metadata_newbie.json", "lulynx_cache_manifest_newbie.json"):
            _copy_if_exists(source_dir / metadata_name, target_dir / metadata_name)

    return {
        "copied_images": copied,
        "cached_pairs": cached_pairs,
        "missing_captions": missing_captions,
        "picked_names": [path.name for path in picked],
    }


def _resolve_runtime() -> tuple[str, torch.dtype, MixedPrecision]:
    detected_device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_device = str(os.environ.get("LULYNX_REAL_TRAIN_DEVICE", detected_device) or detected_device).strip().lower()
    if runtime_device == "cuda" and not torch.cuda.is_available():
        runtime_device = "cpu"
    if runtime_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported LULYNX_REAL_TRAIN_DEVICE={runtime_device!r}")

    if runtime_device == "cpu":
        return "cpu", torch.float32, MixedPrecision.NO

    dtype_name = str(
        os.environ.get(
            "LULYNX_REAL_TRAIN_DTYPE",
            "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        )
        or ""
    ).strip().lower()
    if dtype_name in {"", "bf16", "bfloat16"}:
        return "cuda", torch.bfloat16, MixedPrecision.BF16
    if dtype_name in {"fp16", "float16", "half"}:
        return "cuda", torch.float16, MixedPrecision.FP16
    if dtype_name in {"fp32", "float32", "no"}:
        return "cuda", torch.float32, MixedPrecision.NO
    raise ValueError(f"Unsupported LULYNX_REAL_TRAIN_DTYPE={dtype_name!r}")


def _phase(label: str, *, start_time: float, last_time: float) -> float:
    now = time.perf_counter()
    print(
        f"[real-train] phase={label} dt={now - last_time:.2f}s total={now - start_time:.2f}s",
        flush=True,
    )
    return now


def _mn_lora_summary(telemetry: dict[str, object]) -> dict[str, object]:
    """Compact MN-LoRA telemetry for real smoke reports."""
    kfac = telemetry.get("kfac_lite") if isinstance(telemetry.get("kfac_lite"), dict) else {}
    gsp = telemetry.get("gsp") if isinstance(telemetry.get("gsp"), dict) else {}
    effective_delta = (
        telemetry.get("effective_delta") if isinstance(telemetry.get("effective_delta"), dict) else {}
    )
    fisher_ewc = telemetry.get("fisher_ewc") if isinstance(telemetry.get("fisher_ewc"), dict) else {}
    gradient_conflict = (
        telemetry.get("gradient_conflict") if isinstance(telemetry.get("gradient_conflict"), dict) else {}
    )
    return {
        "step_count": telemetry.get("step_count", 0),
        "gsp_enabled": telemetry.get("gsp_enabled", False),
        "gsp_precondition_mode": gsp.get("mode", ""),
        "gsp_v_cache_layers": gsp.get("v_cache_layers", 0),
        "gsp_coord_curv_layers": gsp.get("coord_curv_layers", 0),
        "kfac_lite_enabled": telemetry.get("kfac_lite_enabled", False),
        "kfac_registered_modules": kfac.get("registered_modules", 0),
        "kfac_factor_layers": kfac.get("factor_layers", 0),
        "kfac_updates": kfac.get("updates", 0),
        "kfac_preconditioned": kfac.get("preconditioned", 0),
        "kfac_preconditioned_params": kfac.get("preconditioned_params", 0),
        "kfac_precondition_hit_rate": kfac.get("precondition_hit_rate", 0.0),
        "kfac_grad_norm_ratio_avg": kfac.get("grad_norm_ratio_avg", 1.0),
        "kfac_grad_norm_ratio_max": kfac.get("grad_norm_ratio_max", 1.0),
        "kfac_stacked_with_gsp": kfac.get("stacked_with_gsp", False),
        "kfac_stacked_grad_clip": kfac.get("stacked_grad_clip", 0.0),
        "effective_delta_enabled": telemetry.get("effective_delta_enabled", False),
        "effective_delta_registered_pairs": effective_delta.get("registered_pairs", 0),
        "effective_delta_pairs_seen": effective_delta.get("pairs_seen", 0),
        "effective_delta_fisher_pairs": effective_delta.get("fisher_pairs", 0),
        "effective_delta_fisher_updates": effective_delta.get("fisher_updates", 0),
        "fisher_ewc_enabled": telemetry.get("fisher_ewc_enabled", False),
        "fisher_ewc_registered_params": fisher_ewc.get("registered_params", 0),
        "fisher_ewc_fisher_layers": fisher_ewc.get("fisher_layers", 0),
        "fisher_ewc_penalty_applications": fisher_ewc.get("penalty_applications", 0),
        "fisher_ewc_last_penalty_loss": fisher_ewc.get("last_penalty_loss", 0.0),
        "gradient_conflict_enabled": telemetry.get("gradient_conflict_enabled", False),
        "gradient_conflict_pairs": gradient_conflict.get("conflict_pairs", 0),
        "gradient_conflict_projections": gradient_conflict.get("projections", 0),
    }


def _build_config(
    *,
    family: str,
    adapter: str,
    model_dir: Path,
    train_dir: Path,
    output_dir: Path,
    runtime_device: str,
    mixed_precision: MixedPrecision,
    steps: int,
    epochs: int,
    resolution: int,
    rank: int,
    learning_rate: float,
    optimizer_type: OptimizerType,
    mn_lora_enabled: bool,
    mn_lora_precondition_mode: str,
    mn_lora_adaptive_sparse: bool,
    mn_lora_update_interval: int,
    mn_lora_adaptive_sparse_hot_ratio: float,
    mn_lora_adaptive_sparse_refresh_interval: int,
    mn_lora_trust_region: bool,
    mn_lora_trust_region_hotspot_only: bool,
    mn_lora_effective_delta: bool,
    mn_lora_effective_delta_fisher: bool,
    mn_lora_kfac_lite: bool,
    mn_lora_fisher_ewc: bool,
    mn_lora_fisher_ewc_update_interval: int,
    mn_lora_gradient_conflict: bool,
    output_name: str,
    keep_intermediate_saves: bool,
    save_training_state: bool,
    attention_backend: str,
    te_vae_offload_strategy: str,
    precision_swap: bool,
    precision_swap_strategy: str,
    preview_device: str,
    sample_every: int,
    sample_every_n_epochs: int,
    sample_width: int,
    sample_height: int,
    sample_steps: int,
    sample_seed: int,
    sample_prompt: str,
    sample_negative: str,
    sdxl_unet_backend: str,
    lulynx_weight_residency: str,
    lulynx_weight_residency_min_params: int,
    anima_block_residency: str,
    anima_block_residency_min_params: int,
    anima_block_prefetch: bool,
    anima_block_prefetch_depth: int,
    anima_block_prefetch_mode: str,
    newbie_block_residency: str,
    newbie_block_residency_min_params: int,
    newbie_block_prefetch: bool,
    newbie_block_prefetch_depth: int,
    vram_auto_enhance_enabled: bool,
    enhanced_protection_mode: bool,
    pcie_transfer_format: str,
    sparse_swap_enabled: bool,
    sparse_swap_budget_mb: float,
    sparse_swap_warm_fraction: float,
    pcie_delta_cache_enabled: bool,
    pcie_delta_cache_mode: str,
    pcie_delta_cache_budget_mb: float,
    vram_smart_sensing_delta_cache_enabled: bool,
    peak_vram_diagnostics: bool,
    cuda_cache_release_strategy: str,
    cuda_cache_release_interval: int,
    checkpoint_policy: str,
    advanced_optimizer_strategy: str,
    svd_grad_proj_rank: int,
    svd_grad_proj_update_interval: int,
) -> UnifiedTrainingConfig:
    save_every_n_epochs = 1 if keep_intermediate_saves else max(int(epochs) + 1, 10_000)
    save_every_n_steps = max(1, int(steps // 2)) if keep_intermediate_saves else 0
    common = dict(
        train_data_dir=str(train_dir),
        output_dir=str(output_dir),
        output_name=output_name,
        mixed_precision=mixed_precision,
        optimizer_type=optimizer_type,
        lr_scheduler=SchedulerType.CONSTANT,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        max_train_epochs=max(int(epochs), 1),
        max_train_steps=max(int(steps), 1),
        network_dim=max(int(rank), 1),
        network_alpha=max(int(rank), 1),
        learning_rate=float(learning_rate),
        save_every_n_epochs=save_every_n_epochs,
        save_every_n_steps=save_every_n_steps,
        checkpoint_keep_last=1,
        save_last_n_epochs=1 if keep_intermediate_saves else 0,
        save_last_n_steps=1 if keep_intermediate_saves else 0,
        save_state=bool(save_training_state),
        save_state_on_train_end=bool(save_training_state),
        save_last_n_epochs_state=0,
        save_last_n_steps_state=0,
        sample_every=max(int(sample_every or 0), 0),
        sample_every_n_epochs=max(int(sample_every_n_epochs or 0), 0),
        sample_width=max(int(sample_width or 0), 0),
        sample_height=max(int(sample_height or 0), 0),
        sample_steps=max(int(sample_steps or 1), 1),
        sample_seed=int(sample_seed or 0),
        sample_prompts=str(sample_prompt or "a simple red cube on a table"),
        sample_negative=str(sample_negative or ""),
        preview_device=str(preview_device or "off"),
        sdxl_unet_backend=str(sdxl_unet_backend or "diffusers"),
        lulynx_weight_residency=str(lulynx_weight_residency or "resident"),
        lulynx_weight_residency_min_params=max(int(lulynx_weight_residency_min_params or 0), 0),
        anima_block_residency=str(anima_block_residency or "resident"),
        anima_block_residency_min_params=max(int(anima_block_residency_min_params or 0), 0),
        anima_block_prefetch=bool(anima_block_prefetch),
        anima_block_prefetch_depth=max(int(anima_block_prefetch_depth or 0), 0),
        anima_block_prefetch_mode=str(anima_block_prefetch_mode or "original"),
        newbie_block_residency=str(newbie_block_residency or "resident"),
        newbie_block_residency_min_params=max(int(newbie_block_residency_min_params or 0), 0),
        newbie_block_prefetch=bool(newbie_block_prefetch),
        newbie_block_prefetch_depth=max(int(newbie_block_prefetch_depth or 0), 0),
        vram_auto_enhance_enabled=bool(vram_auto_enhance_enabled),
        enhanced_protection_mode=bool(enhanced_protection_mode),
        pcie_transfer_format=str(pcie_transfer_format or "off"),
        sparse_swap_enabled=bool(sparse_swap_enabled),
        sparse_swap_budget_mb=max(float(sparse_swap_budget_mb or 0.0), 0.0),
        sparse_swap_warm_fraction=min(max(float(sparse_swap_warm_fraction or 0.35), 0.0), 1.0),
        pcie_delta_cache_enabled=bool(pcie_delta_cache_enabled),
        pcie_delta_cache_mode=str(pcie_delta_cache_mode or "observe"),
        pcie_delta_cache_budget_mb=max(float(pcie_delta_cache_budget_mb or 0.0), 0.0),
        vram_smart_sensing_delta_cache_enabled=bool(vram_smart_sensing_delta_cache_enabled),
        advanced_monitoring_enabled=bool(peak_vram_diagnostics),
        peak_vram_diagnostics_interval=1,
        cuda_cache_release_strategy=str(cuda_cache_release_strategy or "off"),
        cuda_cache_release_interval=max(int(cuda_cache_release_interval or 1), 1),
        gradient_checkpointing=True,
        checkpoint_policy=str(checkpoint_policy or "auto"),
        advanced_optimizer_strategy=str(advanced_optimizer_strategy or "auto"),
        svd_grad_proj_rank=max(int(svd_grad_proj_rank or 128), 1),
        svd_grad_proj_update_interval=max(int(svd_grad_proj_update_interval or 200), 1),
        resolution=int(resolution),
        caption_extension=".txt",
        seed=42,
        attention_backend=str(attention_backend or "auto"),
        te_vae_offload_strategy=str(te_vae_offload_strategy or "phase"),
        model_to_condition_enabled=True,
        lulynx_precision_swap_enabled=bool(precision_swap),
        lulynx_precision_swap_strategy=str(precision_swap_strategy or "balanced"),
        mem_efficient_save=True,
        mn_lora_enabled=bool(mn_lora_enabled),
        mn_lora_precondition_mode=str(mn_lora_precondition_mode or "grad_ema"),
        mn_lora_gsp_enabled=True,
        mn_lora_tgwd_enabled=True,
        mn_lora_update_interval=max(1, int(mn_lora_update_interval)),
        mn_lora_precond_clip=3.0,
        mn_lora_adaptive_sparse_enabled=bool(mn_lora_enabled or mn_lora_adaptive_sparse),
        mn_lora_adaptive_sparse_hot_ratio=float(mn_lora_adaptive_sparse_hot_ratio),
        mn_lora_adaptive_sparse_refresh_interval=max(1, int(mn_lora_adaptive_sparse_refresh_interval)),
        mn_lora_trust_region_enabled=bool(mn_lora_trust_region),
        mn_lora_trust_region_hotspot_only=bool(mn_lora_trust_region_hotspot_only),
        mn_lora_effective_delta_enabled=bool(mn_lora_effective_delta),
        mn_lora_effective_delta_fisher_weighted=bool(mn_lora_effective_delta_fisher),
        mn_lora_kfac_lite_enabled=bool(mn_lora_kfac_lite),
        mn_lora_fisher_ewc_enabled=bool(mn_lora_fisher_ewc),
        mn_lora_fisher_ewc_update_interval=max(1, int(mn_lora_fisher_ewc_update_interval)),
        mn_lora_gradient_conflict_enabled=bool(mn_lora_gradient_conflict),
    )

    if family == "newbie":
        config = UnifiedTrainingConfig(
            model_type=ModelArch.NEWBIE,
            pretrained_model_name_or_path=str(model_dir),
            trust_remote_code=True,
            use_cache=True,
            newbie_safe_fallback=(runtime_device == "cuda"),
            newbie_transformer_path=str(model_dir / "transformer"),
            newbie_gemma_model_path=str(model_dir / "text_encoder"),
            newbie_clip_model_path=str(model_dir / "clip_model"),
            newbie_vae_path=str(model_dir / "vae"),
            newbie_gemma_max_token_length=64,
            newbie_clip_max_token_length=128,
            newbie_target_modules="layers.0.attention.qkv\nlayers.0.attention.out",
            **common,
        )
        _apply_adapter_overrides(config, family=family, adapter=adapter)
        return config

    if family == "anima":
        checkpoint = model_dir / "diffusion_models" / "anima-preview2.safetensors"
        if not checkpoint.exists():
            checkpoint = model_dir / "diffusion_models" / "anima-base-v1.0.safetensors"
        config = UnifiedTrainingConfig(
            model_type=ModelArch.ANIMA,
            pretrained_model_name_or_path=str(checkpoint),
            anima_model_path=str(checkpoint),
            anima_cached_training=True,
            anima_cached_latent_crop_size=4,
            anima_cached_text_token_limit=16,
            anima_native_block_count=28,
            native_cache_mode="cache_first",
            **common,
        )
        _apply_adapter_overrides(config, family=family, adapter=adapter)
        return config

    if family == "sdxl":
        checkpoint = model_dir / "silentEraFurrymixNAIXL_v10.safetensors"
        config = UnifiedTrainingConfig(
            model_type=ModelArch.SDXL,
            pretrained_model_name_or_path=str(checkpoint),
            xformers=(runtime_device == "cuda"),
            enable_bucket=True,
            min_bucket_reso=int(resolution),
            max_bucket_reso=int(resolution),
            bucket_reso_steps=64,
            **common,
        )
        _apply_adapter_overrides(config, family=family, adapter=adapter)
        return config

    raise ValueError(f"Unsupported family: {family}")


def _run_case(
    *,
    family: str,
    adapter: str,
    model_dir: Path,
    train_dir: Path,
    case_root: Path,
    copy_report: dict[str, object],
    runtime_device: str,
    runtime_dtype: torch.dtype,
    mixed_precision: MixedPrecision,
    steps: int,
    epochs: int,
    resolution: int,
    rank: int,
    learning_rate: float,
    optimizer_type: OptimizerType,
    keep_intermediate_saves: bool,
    save_training_state: bool,
    mn_lora_enabled: bool,
    mn_lora_precondition_mode: str,
    mn_lora_adaptive_sparse: bool,
    mn_lora_update_interval: int,
    mn_lora_adaptive_sparse_hot_ratio: float,
    mn_lora_adaptive_sparse_refresh_interval: int,
    mn_lora_trust_region: bool,
    mn_lora_trust_region_hotspot_only: bool,
    mn_lora_effective_delta: bool,
    mn_lora_effective_delta_fisher: bool,
    mn_lora_kfac_lite: bool,
    mn_lora_fisher_ewc: bool,
    mn_lora_fisher_ewc_update_interval: int,
    mn_lora_gradient_conflict: bool,
    attention_backend: str,
    te_vae_offload_strategy: str,
    precision_swap: bool,
    precision_swap_strategy: str,
    preview_device: str,
    sample_every: int,
    sample_every_n_epochs: int,
    sample_width: int,
    sample_height: int,
    sample_steps: int,
    sample_seed: int,
    sample_prompt: str,
    sample_negative: str,
    sdxl_unet_backend: str,
    lulynx_weight_residency: str,
    lulynx_weight_residency_min_params: int,
    anima_block_residency: str,
    anima_block_residency_min_params: int,
    anima_block_prefetch: bool,
    anima_block_prefetch_depth: int,
    anima_block_prefetch_mode: str,
    newbie_block_residency: str,
    newbie_block_residency_min_params: int,
    newbie_block_prefetch: bool,
    newbie_block_prefetch_depth: int,
    vram_auto_enhance_enabled: bool,
    enhanced_protection_mode: bool,
    pcie_transfer_format: str,
    sparse_swap_enabled: bool,
    sparse_swap_budget_mb: float,
    sparse_swap_warm_fraction: float,
    pcie_delta_cache_enabled: bool,
    pcie_delta_cache_mode: str,
    pcie_delta_cache_budget_mb: float,
    vram_smart_sensing_delta_cache_enabled: bool,
    peak_vram_diagnostics: bool,
    cuda_cache_release_strategy: str,
    cuda_cache_release_interval: int,
    checkpoint_policy: str,
    advanced_optimizer_strategy: str,
    svd_grad_proj_rank: int,
    svd_grad_proj_update_interval: int,
) -> RealTrainCaseResult:
    adapter_tag = _adapter_tag(adapter)
    case_start = time.perf_counter()
    output_dir = case_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{family}_{adapter_tag}_{steps}steps"
    logs: list[str] = []
    runtime_events: list[dict[str, object]] = []

    cfg = _build_config(
        family=family,
        adapter=adapter,
        model_dir=model_dir,
        train_dir=train_dir,
        output_dir=output_dir,
        runtime_device=runtime_device,
        mixed_precision=mixed_precision,
        steps=steps,
        epochs=epochs,
        resolution=resolution,
        rank=rank,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        mn_lora_enabled=mn_lora_enabled,
        mn_lora_precondition_mode=mn_lora_precondition_mode,
        mn_lora_adaptive_sparse=mn_lora_adaptive_sparse,
        mn_lora_update_interval=mn_lora_update_interval,
        mn_lora_adaptive_sparse_hot_ratio=mn_lora_adaptive_sparse_hot_ratio,
        mn_lora_adaptive_sparse_refresh_interval=mn_lora_adaptive_sparse_refresh_interval,
        mn_lora_trust_region=mn_lora_trust_region,
        mn_lora_trust_region_hotspot_only=mn_lora_trust_region_hotspot_only,
        mn_lora_effective_delta=mn_lora_effective_delta,
        mn_lora_effective_delta_fisher=mn_lora_effective_delta_fisher,
        mn_lora_kfac_lite=mn_lora_kfac_lite,
        mn_lora_fisher_ewc=mn_lora_fisher_ewc,
        mn_lora_fisher_ewc_update_interval=mn_lora_fisher_ewc_update_interval,
        mn_lora_gradient_conflict=mn_lora_gradient_conflict,
        output_name=output_name,
        keep_intermediate_saves=keep_intermediate_saves,
        save_training_state=save_training_state,
        attention_backend=attention_backend,
        te_vae_offload_strategy=te_vae_offload_strategy,
        precision_swap=precision_swap,
        precision_swap_strategy=precision_swap_strategy,
        preview_device=preview_device,
        sample_every=sample_every,
        sample_every_n_epochs=sample_every_n_epochs,
        sample_width=sample_width,
        sample_height=sample_height,
        sample_steps=sample_steps,
        sample_seed=sample_seed,
        sample_prompt=sample_prompt,
        sample_negative=sample_negative,
        sdxl_unet_backend=sdxl_unet_backend,
        lulynx_weight_residency=lulynx_weight_residency,
        lulynx_weight_residency_min_params=lulynx_weight_residency_min_params,
        anima_block_residency=anima_block_residency,
        anima_block_residency_min_params=anima_block_residency_min_params,
        anima_block_prefetch=anima_block_prefetch,
        anima_block_prefetch_depth=anima_block_prefetch_depth,
        anima_block_prefetch_mode=anima_block_prefetch_mode,
        newbie_block_residency=newbie_block_residency,
        newbie_block_residency_min_params=newbie_block_residency_min_params,
        newbie_block_prefetch=newbie_block_prefetch,
        newbie_block_prefetch_depth=newbie_block_prefetch_depth,
        vram_auto_enhance_enabled=vram_auto_enhance_enabled,
        enhanced_protection_mode=enhanced_protection_mode,
        pcie_transfer_format=pcie_transfer_format,
        sparse_swap_enabled=sparse_swap_enabled,
        sparse_swap_budget_mb=sparse_swap_budget_mb,
        sparse_swap_warm_fraction=sparse_swap_warm_fraction,
        pcie_delta_cache_enabled=pcie_delta_cache_enabled,
        pcie_delta_cache_mode=pcie_delta_cache_mode,
        pcie_delta_cache_budget_mb=pcie_delta_cache_budget_mb,
        vram_smart_sensing_delta_cache_enabled=vram_smart_sensing_delta_cache_enabled,
        peak_vram_diagnostics=peak_vram_diagnostics,
        cuda_cache_release_strategy=cuda_cache_release_strategy,
        cuda_cache_release_interval=cuda_cache_release_interval,
        checkpoint_policy=checkpoint_policy,
        advanced_optimizer_strategy=advanced_optimizer_strategy,
        svd_grad_proj_rank=svd_grad_proj_rank,
        svd_grad_proj_update_interval=svd_grad_proj_update_interval,
    )

    trainer = LulynxTrainer(cfg)
    trainer.device = runtime_device
    trainer.dtype = runtime_dtype
    trainer.set_callbacks(
        on_log=logs.append,
        on_runtime_event=lambda payload: runtime_events.append(payload),
    )

    print(
        "[real-train] "
        f"family={family} adapter={_normalize_adapter(adapter)} device={runtime_device} dtype={runtime_dtype} "
        f"work_dir={case_root}",
        flush=True,
    )

    try:
        ok = trainer.start()
        if not ok:
            raise RuntimeError(f"{family}/{adapter} real local training failed")

        output_candidates = sorted(output_dir.glob(f"{output_name}*.safetensors"))
        if not output_candidates:
            raise FileNotFoundError(f"No adapter artifact found under {output_dir} for {output_name}")

        global_step = getattr(getattr(trainer, "training_loop", None), "global_step", None)
        if global_step is None or int(global_step) < steps:
            raise AssertionError(f"Expected at least {steps} steps, got {global_step}")

        optimizer = getattr(getattr(trainer, "training_loop", None), "optimizer", None)
        memory_optimization = getattr(getattr(trainer, "training_loop", None), "memory_optimization_state", None)
        native_unet = getattr(trainer, "_native_unet_status", None)
        anima_block_residency = getattr(trainer, "_anima_block_residency_profile", None)
        newbie_block_residency = getattr(trainer, "_newbie_block_residency_profile", None)
        unet = getattr(getattr(trainer, "model", None), "unet", None)
        controller = getattr(unet, "_lulynx_dit_prefetch_controller", None)
        if controller is not None and hasattr(controller, "as_dict"):
            try:
                if isinstance(anima_block_residency, dict) and family == "anima":
                    anima_block_residency = dict(anima_block_residency)
                    anima_block_residency["prefetch"] = controller.as_dict()
                if isinstance(newbie_block_residency, dict) and family == "newbie":
                    newbie_block_residency = dict(newbie_block_residency)
                    newbie_block_residency["prefetch"] = controller.as_dict()
            except Exception:
                pass
        if bool(getattr(cfg, "pcie_delta_cache_enabled", False)) and unet is not None:
            try:
                from core.lulynx_trainer.pcie_cache_profiler import build_active_module_pcie_cache_profile

                if isinstance(anima_block_residency, dict) and family == "anima":
                    profile = build_active_module_pcie_cache_profile(
                        unet,
                        enabled=True,
                        family="anima",
                        mode=str(anima_block_residency.get("mode", "")),
                    ).as_dict()
                    profile["reason"] = "observe_only_post_training"
                    anima_block_residency = dict(anima_block_residency)
                    anima_block_residency["pcie_delta_cache"] = profile
                if isinstance(newbie_block_residency, dict) and family == "newbie":
                    profile = build_active_module_pcie_cache_profile(
                        unet,
                        enabled=True,
                        family="newbie",
                        mode=str(newbie_block_residency.get("mode", "")),
                    ).as_dict()
                    profile["reason"] = "observe_only_post_training"
                    newbie_block_residency = dict(newbie_block_residency)
                    newbie_block_residency["pcie_delta_cache"] = profile
            except Exception as exc:
                if isinstance(anima_block_residency, dict) and family == "anima":
                    anima_block_residency = dict(anima_block_residency)
                    anima_block_residency["pcie_delta_cache_refresh_error"] = f"{type(exc).__name__}: {exc}"
                if isinstance(newbie_block_residency, dict) and family == "newbie":
                    newbie_block_residency = dict(newbie_block_residency)
                    newbie_block_residency["pcie_delta_cache_refresh_error"] = f"{type(exc).__name__}: {exc}"
        newbie_cache_first_profile = getattr(trainer, "_newbie_cache_first_profile", None)
        sample_dir = output_dir / "samples"
        sample_files = sorted(str(path) for path in sample_dir.glob("*") if path.is_file()) if sample_dir.exists() else []
        gsp = getattr(optimizer, "gsp", None)
        base_optimizer = getattr(optimizer, "base_optimizer", None) or getattr(optimizer, "_base", None)
        optimizer_runtime = {
            "type": type(optimizer).__name__ if optimizer is not None else "",
            "base_type": type(base_optimizer).__name__ if base_optimizer is not None else "",
            "svd_grad_projection": bool(type(optimizer).__name__ == "SVDGradientProjectionWrapper" or hasattr(optimizer, "_projectors")),
            "gsp_enabled": gsp is not None,
            "gsp_precondition_mode": str(getattr(gsp, "precondition_mode", "")) if gsp is not None else "",
            "gsp_v_cache_layers": len(getattr(gsp, "V_cache", {}) or {}) if gsp is not None else 0,
            "gsp_s_cache_layers": len(getattr(gsp, "S_cache", {}) or {}) if gsp is not None else 0,
            "gsp_coord_curv_layers": len(getattr(gsp, "coord_curv_cache", {}) or {}) if gsp is not None else 0,
        }
        if optimizer is not None and hasattr(optimizer, "get_telemetry_snapshot"):
            try:
                telemetry = optimizer.get_telemetry_snapshot()
                optimizer_runtime["telemetry"] = telemetry
                optimizer_runtime["mn_lora_summary"] = _mn_lora_summary(telemetry)
            except Exception as exc:
                optimizer_runtime["telemetry_error"] = f"{type(exc).__name__}: {exc}"

        return RealTrainCaseResult(
            family=family,
            adapter=_normalize_adapter(adapter),
            ok=True,
            work_dir=str(case_root),
            output_dir=str(output_dir),
            train_dir=str(train_dir),
            artifact=str(output_candidates[-1]),
            global_step=int(global_step),
            resolved_steps=int(steps),
            resolved_epochs=int(epochs),
            duration_seconds=round(time.perf_counter() - case_start, 3),
            copy_report=copy_report,
            optimizer_runtime=optimizer_runtime,
            memory_optimization=memory_optimization if isinstance(memory_optimization, dict) else None,
            native_unet=native_unet if isinstance(native_unet, dict) else None,
            anima_block_residency=anima_block_residency if isinstance(anima_block_residency, dict) else None,
            newbie_block_residency=newbie_block_residency if isinstance(newbie_block_residency, dict) else None,
            newbie_cache_first_profile=newbie_cache_first_profile if isinstance(newbie_cache_first_profile, dict) else None,
            sample_files=sample_files,
            runtime_event_tail=runtime_events[-10:],
            log_tail=logs[-30:],
        )
    except Exception as exc:
        return RealTrainCaseResult(
            family=family,
            adapter=_normalize_adapter(adapter),
            ok=False,
            work_dir=str(case_root),
            output_dir=str(output_dir),
            train_dir=str(train_dir),
            resolved_steps=int(steps),
            resolved_epochs=int(epochs),
            duration_seconds=round(time.perf_counter() - case_start, 3),
            copy_report=copy_report,
            newbie_cache_first_profile=getattr(trainer, "_newbie_cache_first_profile", None) if isinstance(getattr(trainer, "_newbie_cache_first_profile", None), dict) else None,
            runtime_event_tail=runtime_events[-10:],
            log_tail=logs[-30:],
            error=str(exc),
        )
    finally:
        del trainer
        _cleanup_runtime()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real local adapter training against /models and /sucai.")
    parser.add_argument("--family", default="newbie", choices=list(SUPPORTED_FAMILIES))
    parser.add_argument("--families", nargs="*", choices=list(SUPPORTED_FAMILIES), default=None)
    parser.add_argument("--adapter", default="lora")
    parser.add_argument("--adapters", nargs="*", default=None)
    parser.add_argument("--matrix", action="store_true", help="Run the default adapter matrix for selected families.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=0, help="0 = auto-resolve enough epochs to cover requested steps")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "AdamW8bit", "Automagic++"])
    parser.add_argument("--output-root", default="")
    parser.add_argument("--source-data", default="")
    parser.add_argument("--json", default="", help="Optional JSON report path.")
    parser.add_argument("--allow-short-steps", action="store_true", help="Allow --steps below the default 40-step smoke floor.")
    parser.add_argument("--mn-lora", action="store_true", help="Enable MN-LoRA optimizer wrapper during the real training smoke.")
    parser.add_argument(
        "--mn-lora-precondition-mode",
        default="grad_ema",
        choices=["none", "svd", "grad_ema", "hybrid"],
        help="MN-LoRA GSP subspace preconditioner mode.",
    )
    parser.add_argument(
        "--mn-lora-adaptive-sparse",
        action="store_true",
        help="Enable adaptive sparse GSP projection during the real training smoke.",
    )
    parser.add_argument("--mn-lora-update-interval", type=int, default=20, help="MN-LoRA GSP SVD refresh interval.")
    parser.add_argument(
        "--mn-lora-adaptive-sparse-hot-ratio",
        type=float,
        default=0.20,
        help="Fraction of MN-LoRA layers kept in the hot GSP tier.",
    )
    parser.add_argument(
        "--mn-lora-adaptive-sparse-refresh-interval",
        type=int,
        default=20,
        help="MN-LoRA adaptive sparse tier refresh interval.",
    )
    parser.add_argument(
        "--mn-lora-trust-region",
        action="store_true",
        help="Enable MN-LoRA P3 trust-region update clipping during the real training smoke.",
    )
    parser.add_argument(
        "--mn-lora-trust-region-hotspot-only",
        action="store_true",
        help="Apply MN-LoRA trust-region clipping only to GSP hot-tier parameters.",
    )
    parser.add_argument(
        "--mn-lora-effective-delta",
        action="store_true",
        help="Enable MN-LoRA P3.1 effective ΔW telemetry/clipping during the real training smoke.",
    )
    parser.add_argument(
        "--mn-lora-effective-delta-fisher",
        action="store_true",
        help="Enable lightweight Fisher-weighted effective ΔW constraints.",
    )
    parser.add_argument(
        "--mn-lora-kfac-lite",
        action="store_true",
        help="Enable MN-LoRA P4 LoRA-KFAC-Lite gradient preconditioning during the real training smoke.",
    )
    parser.add_argument(
        "--mn-lora-fisher-ewc",
        action="store_true",
        help="Enable MN-LoRA diagonal Fisher/EWC gradient regularization during the real training smoke.",
    )
    parser.add_argument(
        "--mn-lora-fisher-ewc-update-interval",
        type=int,
        default=5,
        help="MN-LoRA Fisher/EWC diagonal Fisher update interval.",
    )
    parser.add_argument(
        "--mn-lora-gradient-conflict",
        action="store_true",
        help="Enable MN-LoRA objective-gradient conflict surgery for regularizer gradients.",
    )
    parser.add_argument("--keep-intermediate-saves", action="store_true", help="Keep epoch/step checkpoints during the smoke run.")
    parser.add_argument("--save-training-state", action="store_true", help="Also save optimizer/scheduler state snapshots.")
    parser.add_argument("--attention-backend", default="auto", help="Runtime attention backend, e.g. auto/sdpa/xformers/flash2.")
    parser.add_argument("--te-vae-offload-strategy", default="phase", choices=["resident", "phase", "aggressive"])
    parser.add_argument("--precision-swap", action="store_true", help="Enable Lulynx precision swap planning for the smoke run.")
    parser.add_argument("--precision-swap-strategy", default="balanced", choices=["balanced", "aggressive", "off"])
    parser.add_argument("--preview-device", default="off", choices=["off", "cpu", "gpu"])
    parser.add_argument("--sample-every", type=int, default=0, help="Generate samples every N training steps.")
    parser.add_argument("--sample-every-n-epochs", type=int, default=0, help="Generate samples every N epochs.")
    parser.add_argument("--sample-width", type=int, default=0, help="Preview width override; 0 uses trainer default.")
    parser.add_argument("--sample-height", type=int, default=0, help="Preview height override; 0 uses trainer default.")
    parser.add_argument("--sample-steps", type=int, default=1, help="Preview inference steps.")
    parser.add_argument("--sample-seed", type=int, default=1234, help="Preview seed; 0 lets the sampler choose.")
    parser.add_argument("--sample-prompt", default="a simple red cube on a table")
    parser.add_argument("--sample-negative", default="")
    parser.add_argument("--sdxl-unet-backend", default="diffusers", choices=["diffusers", "native_shadow", "native_proxy", "native_skeleton", "lulynx_native"])
    parser.add_argument("--lulynx-weight-residency", default="resident", choices=["resident", "linear_cpu_pinned", "linear_conv_cpu_pinned"])
    parser.add_argument("--lulynx-weight-residency-min-params", type=int, default=0)
    parser.add_argument("--anima-block-residency", default="resident", choices=["resident", "streaming_offload", "block_cpu_pinned"])
    parser.add_argument("--anima-block-residency-min-params", type=int, default=0)
    parser.add_argument("--anima-block-prefetch", action="store_true")
    parser.add_argument("--anima-block-prefetch-depth", type=int, default=1)
    parser.add_argument("--anima-block-prefetch-mode", default="original", choices=["original", "adaptive"])
    parser.add_argument("--newbie-block-residency", default="resident", choices=["resident", "streaming_offload", "block_cpu_pinned"])
    parser.add_argument("--newbie-block-residency-min-params", type=int, default=0)
    parser.add_argument("--newbie-block-prefetch", action="store_true")
    parser.add_argument("--newbie-block-prefetch-depth", type=int, default=1)
    parser.add_argument("--disable-vram-auto-enhance", action="store_true")
    parser.add_argument("--enhanced-protection-mode", action="store_true")
    parser.add_argument("--pcie-transfer-format", default="off", choices=["off", "raw_fp16", "raw_bf16", "fp8_e4m3", "int8_rowwise", "uint4_rowwise"])
    parser.add_argument("--sparse-swap-enabled", action="store_true")
    parser.add_argument("--sparse-swap-budget-mb", type=float, default=0.0)
    parser.add_argument("--sparse-swap-warm-fraction", type=float, default=0.35)
    parser.add_argument("--pcie-delta-cache", action="store_true", help="Enable observe-only PCIe Delta/Cache candidate profiling.")
    parser.add_argument("--pcie-delta-cache-mode", default="observe", choices=["observe", "cache_v0"])
    parser.add_argument("--pcie-delta-cache-budget-mb", type=float, default=256.0)
    parser.add_argument("--vram-smart-sensing-delta-cache", action="store_true", help="Allow VRAM smart sensing to enable observe-only PCIe Delta/Cache profiling.")
    parser.add_argument("--peak-vram-diagnostics", action="store_true", help="Enable per-step peak VRAM diagnostics in the smoke report.")
    parser.add_argument("--cuda-cache-release-strategy", default="off", choices=["off", "after_optimizer", "every_step"])
    parser.add_argument("--cuda-cache-release-interval", type=int, default=1)
    parser.add_argument("--checkpoint-policy", default="auto", choices=["auto", "off", "full", "offloaded", "selective"])
    parser.add_argument("--advanced-optimizer-strategy", default="auto", choices=["auto", "off", "profile_only", "lora_plus", "rs_lora", "galore"])
    parser.add_argument("--svd-grad-proj-rank", type=int, default=128)
    parser.add_argument("--svd-grad-proj-update-interval", type=int, default=200)
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop the matrix after the first failed case.")
    args = parser.parse_args()

    start_time = time.perf_counter()

    repo_root = _resolve_repo_root()
    selected_families = _resolve_selected_families(args)
    selected_adapters = _resolve_selected_adapters(args)
    source_dir = Path(args.source_data) if args.source_data else (repo_root / "sucai" / "6_lulu")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source data dir not found: {source_dir}")

    runtime_device, runtime_dtype, mixed_precision = _resolve_runtime()

    session_parent, session_parent_reason = _resolve_session_parent(repo_root, args.output_root)
    session_root = _create_session_root(session_parent)
    if session_parent_reason.startswith("fallback_low_disk:"):
        free_bytes = int(session_parent_reason.split(":", 1)[1])
        print(
            "[real-train] "
            f"repo tmp free space too low ({round(free_bytes / (1024 ** 3), 3)} GiB); using fallback output root {session_parent}",
            flush=True,
        )

    resolved_steps = max(int(args.steps), 1 if args.allow_short_steps else 40)
    resolution = max(int(args.resolution), 64)
    rank = max(int(args.rank), 1)
    learning_rate = float(args.learning_rate)
    optimizer_type = OptimizerType(args.optimizer)

    results: list[RealTrainCaseResult] = []
    for family in selected_families:
        family_root = session_root / family
        family_root.mkdir(parents=True, exist_ok=True)
        model_dir = repo_root / "models" / family
        if not model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {model_dir}")

        train_dir = family_root / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        copy_report = _materialize_dataset_subset(
            source_dir,
            train_dir,
            family=family,
            sample_limit=max(int(args.sample_limit), 1),
            caption_extension=".txt",
        )
        copied_images = max(int(copy_report.get("copied_images", 0) or 0), 1)
        auto_epochs = max(int(math.ceil(resolved_steps / copied_images)), 1)
        resolved_epochs = max(int(args.epochs), auto_epochs) if int(args.epochs) > 0 else auto_epochs
        _phase(f"{family}-dataset-prepared", start_time=start_time, last_time=start_time)

        for adapter in selected_adapters:
            case_root = family_root / _adapter_tag(adapter)
            case_root.mkdir(parents=True, exist_ok=True)
            result = _run_case(
                family=family,
                adapter=adapter,
                model_dir=model_dir,
                train_dir=train_dir,
                case_root=case_root,
                copy_report=copy_report,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
                mixed_precision=mixed_precision,
                steps=resolved_steps,
                epochs=resolved_epochs,
                resolution=resolution,
                rank=rank,
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                keep_intermediate_saves=bool(args.keep_intermediate_saves),
                save_training_state=bool(args.save_training_state),
                mn_lora_enabled=bool(args.mn_lora),
                mn_lora_precondition_mode=str(args.mn_lora_precondition_mode or "grad_ema"),
                mn_lora_adaptive_sparse=bool(args.mn_lora_adaptive_sparse),
                mn_lora_update_interval=int(args.mn_lora_update_interval),
                mn_lora_adaptive_sparse_hot_ratio=float(args.mn_lora_adaptive_sparse_hot_ratio),
                mn_lora_adaptive_sparse_refresh_interval=int(args.mn_lora_adaptive_sparse_refresh_interval),
                mn_lora_trust_region=bool(args.mn_lora or args.mn_lora_trust_region),
                mn_lora_trust_region_hotspot_only=bool(args.mn_lora_trust_region_hotspot_only),
                mn_lora_effective_delta=bool(args.mn_lora or args.mn_lora_effective_delta),
                mn_lora_effective_delta_fisher=bool(args.mn_lora or args.mn_lora_effective_delta_fisher),
                mn_lora_kfac_lite=bool(args.mn_lora_kfac_lite),
                mn_lora_fisher_ewc=bool(args.mn_lora or args.mn_lora_fisher_ewc),
                mn_lora_fisher_ewc_update_interval=int(args.mn_lora_fisher_ewc_update_interval),
                mn_lora_gradient_conflict=bool(args.mn_lora_gradient_conflict),
                attention_backend=str(args.attention_backend or "auto"),
                te_vae_offload_strategy=str(args.te_vae_offload_strategy or "phase"),
                precision_swap=bool(args.precision_swap),
                precision_swap_strategy=str(args.precision_swap_strategy or "balanced"),
                preview_device=str(args.preview_device or "off"),
                sample_every=max(int(args.sample_every or 0), 0),
                sample_every_n_epochs=max(int(args.sample_every_n_epochs or 0), 0),
                sample_width=max(int(args.sample_width or 0), 0),
                sample_height=max(int(args.sample_height or 0), 0),
                sample_steps=max(int(args.sample_steps or 1), 1),
                sample_seed=int(args.sample_seed or 0),
                sample_prompt=str(args.sample_prompt or "a simple red cube on a table"),
                sample_negative=str(args.sample_negative or ""),
                sdxl_unet_backend=str(args.sdxl_unet_backend or "diffusers"),
                lulynx_weight_residency=str(args.lulynx_weight_residency or "resident"),
                lulynx_weight_residency_min_params=max(int(args.lulynx_weight_residency_min_params or 0), 0),
                anima_block_residency=str(args.anima_block_residency or "resident"),
                anima_block_residency_min_params=max(int(args.anima_block_residency_min_params or 0), 0),
                anima_block_prefetch=bool(args.anima_block_prefetch),
                anima_block_prefetch_depth=max(int(args.anima_block_prefetch_depth or 0), 0),
                anima_block_prefetch_mode=str(args.anima_block_prefetch_mode or "original"),
                newbie_block_residency=str(args.newbie_block_residency or "resident"),
                newbie_block_residency_min_params=max(int(args.newbie_block_residency_min_params or 0), 0),
                newbie_block_prefetch=bool(args.newbie_block_prefetch),
                newbie_block_prefetch_depth=max(int(args.newbie_block_prefetch_depth or 0), 0),
                vram_auto_enhance_enabled=not bool(args.disable_vram_auto_enhance),
                enhanced_protection_mode=bool(args.enhanced_protection_mode),
                pcie_transfer_format=str(args.pcie_transfer_format or "off"),
                sparse_swap_enabled=bool(args.sparse_swap_enabled),
                sparse_swap_budget_mb=max(float(args.sparse_swap_budget_mb or 0.0), 0.0),
                sparse_swap_warm_fraction=min(max(float(args.sparse_swap_warm_fraction or 0.35), 0.0), 1.0),
                pcie_delta_cache_enabled=bool(args.pcie_delta_cache),
                pcie_delta_cache_mode=str(args.pcie_delta_cache_mode or "observe"),
                pcie_delta_cache_budget_mb=max(float(args.pcie_delta_cache_budget_mb or 0.0), 0.0),
                vram_smart_sensing_delta_cache_enabled=bool(args.vram_smart_sensing_delta_cache),
                peak_vram_diagnostics=bool(args.peak_vram_diagnostics),
                cuda_cache_release_strategy=str(args.cuda_cache_release_strategy or "off"),
                cuda_cache_release_interval=max(int(args.cuda_cache_release_interval or 1), 1),
                checkpoint_policy=str(args.checkpoint_policy or "auto"),
                advanced_optimizer_strategy=str(args.advanced_optimizer_strategy or "auto"),
                svd_grad_proj_rank=max(int(args.svd_grad_proj_rank or 128), 1),
                svd_grad_proj_update_interval=max(int(args.svd_grad_proj_update_interval or 200), 1),
            )
            results.append(result)
            status = "PASS" if result.ok else "FAIL"
            tail = f" error={result.error}" if result.error else f" artifact={result.artifact}"
            print(f"[real-train] {status}: {family}/{result.adapter} steps={resolved_steps}{tail}", flush=True)
            if args.stop_on_failure and not result.ok:
                break
        if args.stop_on_failure and any(not item.ok for item in results if item.family == family):
            break

    ok = all(result.ok for result in results)
    report = {
        "ok": ok,
        "runtime_device": runtime_device,
        "runtime_dtype": str(runtime_dtype),
        "source_dir": str(source_dir),
        "session_root": str(session_root),
        "families": selected_families,
        "adapters": selected_adapters,
        "resolved_steps": resolved_steps,
        "optimizer": args.optimizer,
        "anima_block_residency": str(args.anima_block_residency or "resident"),
        "anima_block_prefetch": bool(args.anima_block_prefetch),
        "anima_block_prefetch_depth": max(int(args.anima_block_prefetch_depth or 0), 0),
        "anima_block_prefetch_mode": str(args.anima_block_prefetch_mode or "original"),
        "newbie_block_residency": str(args.newbie_block_residency or "resident"),
        "newbie_block_prefetch": bool(args.newbie_block_prefetch),
        "newbie_block_prefetch_depth": max(int(args.newbie_block_prefetch_depth or 0), 0),
        "vram_auto_enhance_enabled": not bool(args.disable_vram_auto_enhance),
        "enhanced_protection_mode": bool(args.enhanced_protection_mode),
        "pcie_transfer_format": str(args.pcie_transfer_format or "off"),
        "sparse_swap_enabled": bool(args.sparse_swap_enabled),
        "sparse_swap_budget_mb": max(float(args.sparse_swap_budget_mb or 0.0), 0.0),
        "sparse_swap_warm_fraction": min(max(float(args.sparse_swap_warm_fraction or 0.35), 0.0), 1.0),
        "pcie_delta_cache_enabled": bool(args.pcie_delta_cache),
        "pcie_delta_cache_mode": str(args.pcie_delta_cache_mode or "observe"),
        "pcie_delta_cache_budget_mb": max(float(args.pcie_delta_cache_budget_mb or 0.0), 0.0),
        "vram_smart_sensing_delta_cache_enabled": bool(args.vram_smart_sensing_delta_cache),
        "mn_lora_enabled": bool(args.mn_lora),
        "mn_lora_precondition_mode": str(args.mn_lora_precondition_mode or "grad_ema"),
        "mn_lora_adaptive_sparse": bool(args.mn_lora_adaptive_sparse),
        "mn_lora_update_interval": int(args.mn_lora_update_interval),
        "mn_lora_adaptive_sparse_hot_ratio": float(args.mn_lora_adaptive_sparse_hot_ratio),
        "mn_lora_adaptive_sparse_refresh_interval": int(args.mn_lora_adaptive_sparse_refresh_interval),
        "mn_lora_trust_region": bool(args.mn_lora or args.mn_lora_trust_region),
        "mn_lora_trust_region_hotspot_only": bool(args.mn_lora_trust_region_hotspot_only),
        "mn_lora_effective_delta": bool(args.mn_lora or args.mn_lora_effective_delta),
        "mn_lora_effective_delta_fisher": bool(args.mn_lora or args.mn_lora_effective_delta_fisher),
        "mn_lora_kfac_lite": bool(args.mn_lora_kfac_lite),
        "mn_lora_fisher_ewc": bool(args.mn_lora or args.mn_lora_fisher_ewc),
        "mn_lora_fisher_ewc_update_interval": int(args.mn_lora_fisher_ewc_update_interval),
        "mn_lora_gradient_conflict": bool(args.mn_lora_gradient_conflict),
        "checkpoint_policy": str(args.checkpoint_policy or "auto"),
        "advanced_optimizer_strategy": str(args.advanced_optimizer_strategy or "auto"),
        "svd_grad_proj_rank": max(int(args.svd_grad_proj_rank or 128), 1),
        "svd_grad_proj_update_interval": max(int(args.svd_grad_proj_update_interval or 200), 1),
        "case_count": len(results),
        "duration_seconds": round(time.perf_counter() - start_time, 3),
        "results": [asdict(result) for result in results],
    }

    default_name = "real_train_matrix_report.json" if len(results) != 1 else "real_train_report.json"
    report_path = Path(args.json) if args.json else (session_root / default_name)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[real-train] "
        f"completed cases={len(results)} ok={ok} report={report_path}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


