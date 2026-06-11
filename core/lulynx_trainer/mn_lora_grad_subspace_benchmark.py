# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Real-model 40-step convergence benchmark for the gradient-subspace optimizer.

Keeps the model / dataset / seed / route identical and changes ONLY the
optimizer "arm", so the loss-vs-step curves are directly comparable:

  * ``adamw``          : plain AdamW (reference line)
  * ``mn_lora``        : current MN-LoRA bundle (weight-SVD GSP, the status quo)
  * ``galore``         : AdamW + SVD gradient projection (gradient-SVD "guide")
  * ``mn_lora_galore`` : MN-LoRA but the gradient stage is galore instead of
                         weight-SVD GSP (Phase 2 wiring; mutually exclusive)

It reuses the real training path (``LulynxTrainer.start()``) and the dataset /
config scaffolding from ``real_model_training_smoke``; per-step loss is captured
by wrapping ``trainer._on_step_end`` (same pattern as the concept-geometry
benchmark). Real preview quality stays the user's job — this only measures
convergence speed (loss vs step), wall time and peak VRAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.configs import OptimizerType
from core.lulynx_trainer.trainer import LulynxTrainer
from core.lulynx_trainer.real_model_training_smoke import (
    _adapter_tag,
    _build_config,
    _create_session_root,
    _materialize_dataset_subset,
    _resolve_repo_root,
    _resolve_runtime,
    _resolve_session_parent,
)

ARMS = (
    "adamw",
    "mn_lora",
    "mn_lora_no_gsp",
    "mn_lora_no_fisher",
    "mn_lora_no_gsp_no_fisher",
    "mn_lora_bare",
    "mn_lora_gsp_only",
    "galore",
    "mn_lora_galore",
    # LoRA-native recombination arms: bare wrapper + the right tools for LoRA.
    "adamw_lora_plus",       # AdamW + LoRA+ (isolates the LoRA+ effect)
    "mn_lora_lora_plus",     # bare MN-LoRA + LoRA+ (candidate LoRA preset)
    "mn_lora_lora_plus_pp",  # + plus_plus automagic per-param LR multiplier
)

# Arms that strip MN-LoRA down to a transparent wrapper (the full-FT-oriented
# GSP / Fisher / conservative controllers are all net-harmful on LoRA).
_BARE_ARMS = ("mn_lora_bare", "mn_lora_lora_plus", "mn_lora_lora_plus_pp")


@dataclass
class ArmResult:
    arm: str
    mode: str
    adapter: str
    success: bool
    failed_reason: str = ""
    steps_completed: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
    min_loss: float = 0.0
    loss_delta: float = 0.0
    mean_step_ms: float = 0.0
    total_wall_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    optimizer_runtime_type: str = ""
    base_optimizer_type: str = ""
    svd_grad_projection_active: bool = False
    losses: list = field(default_factory=list)
    log_tail: list = field(default_factory=list)


def _mn_lora_flags(enabled: bool) -> dict:
    """Mirror the realistic ``--mn-lora`` bundle from real_model_training_smoke."""
    return dict(
        mn_lora_enabled=enabled,
        mn_lora_precondition_mode="grad_ema",
        mn_lora_adaptive_sparse=False,
        mn_lora_update_interval=20,
        mn_lora_adaptive_sparse_hot_ratio=0.20,
        mn_lora_adaptive_sparse_refresh_interval=20,
        mn_lora_trust_region=enabled,
        mn_lora_trust_region_hotspot_only=False,
        mn_lora_effective_delta=enabled,
        mn_lora_effective_delta_fisher=enabled,
        mn_lora_kfac_lite=False,
        mn_lora_fisher_ewc=enabled,
        mn_lora_fisher_ewc_update_interval=5,
        mn_lora_gradient_conflict=False,
    )


def _base_config_kwargs(
    *,
    family: str,
    adapter: str,
    model_dir: Path,
    train_dir: Path,
    output_dir: Path,
    runtime_device: str,
    mixed_precision,
    steps: int,
    epochs: int,
    resolution: int,
    rank: int,
    learning_rate: float,
    mn_lora_enabled: bool,
    output_name: str,
    block_residency: str,
) -> dict:
    kwargs = dict(
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
        optimizer_type=OptimizerType("AdamW"),
        output_name=output_name,
        keep_intermediate_saves=False,
        save_training_state=False,
        attention_backend="auto",
        te_vae_offload_strategy="phase",
        precision_swap=False,
        precision_swap_strategy="balanced",
        preview_device="off",
        sample_every=0,
        sample_every_n_epochs=0,
        sample_width=0,
        sample_height=0,
        sample_steps=1,
        sample_seed=0,
        sample_prompt="a simple red cube on a table",
        sample_negative="",
        sdxl_unet_backend="diffusers",
        lulynx_weight_residency="resident",
        lulynx_weight_residency_min_params=0,
        anima_block_residency=block_residency,
        anima_block_residency_min_params=0,
        anima_block_prefetch=False,
        anima_block_prefetch_depth=1,
        newbie_block_residency=block_residency,
        newbie_block_residency_min_params=0,
        newbie_block_prefetch=False,
        newbie_block_prefetch_depth=1,
        vram_auto_enhance_enabled=True,
        enhanced_protection_mode=False,
        pcie_transfer_format="off",
        sparse_swap_enabled=False,
        sparse_swap_budget_mb=0.0,
        sparse_swap_warm_fraction=0.35,
        pcie_delta_cache_enabled=False,
        pcie_delta_cache_mode="observe",
        pcie_delta_cache_budget_mb=0.0,
        vram_smart_sensing_delta_cache_enabled=False,
        peak_vram_diagnostics=False,
        cuda_cache_release_strategy="off",
        cuda_cache_release_interval=1,
        checkpoint_policy="auto",
        advanced_optimizer_strategy="auto",
        svd_grad_proj_rank=128,
        svd_grad_proj_update_interval=200,
    )
    kwargs.update(_mn_lora_flags(mn_lora_enabled))
    return kwargs


def _apply_arm_overrides(
    cfg,
    arm: str,
    *,
    seed: int,
    proj_rank: int,
    proj_update_interval: int,
    proj_scale: float,
    proj_warmup: int,
    projection_mode: str,
) -> None:
    if hasattr(cfg, "seed"):
        cfg.seed = int(seed)
    # All arms start from a clean grad-space stage; set per arm below.
    if arm == "galore":
        cfg.svd_grad_proj_enabled = True
        cfg.svd_grad_proj_rank = int(proj_rank)
        cfg.svd_grad_proj_update_interval = int(proj_update_interval)
        cfg.svd_grad_proj_scale = float(proj_scale)
        cfg.svd_grad_proj_warmup_steps = int(proj_warmup)
        if projection_mode:
            setattr(cfg, "svd_grad_proj_projection_mode", projection_mode)
    elif arm == "mn_lora_galore":
        # Galore becomes MN-LoRA's gradient stage (Phase 2 honors this);
        # keep the standalone wrapper OFF to avoid a double projection wrap.
        cfg.svd_grad_proj_enabled = False
        cfg.svd_grad_proj_rank = int(proj_rank)
        cfg.svd_grad_proj_update_interval = int(proj_update_interval)
        cfg.svd_grad_proj_scale = float(proj_scale)
        cfg.svd_grad_proj_warmup_steps = int(proj_warmup)
        if projection_mode:
            setattr(cfg, "svd_grad_proj_projection_mode", projection_mode)
        setattr(cfg, "mn_lora_grad_subspace", "gradient_galore")


def _apply_mn_lora_ablation(cfg, arm: str) -> None:
    """Toggle individual MN-LoRA components to isolate what helps vs hurts.

    The ``mn_lora`` arm leaves the full bundle (built by _build_config) intact;
    every other ``mn_lora_*`` arm subtracts components from it.
    """
    if not arm.startswith("mn_lora") or arm in ("mn_lora", "mn_lora_galore"):
        return
    drop_gsp = arm in (("mn_lora_no_gsp", "mn_lora_no_gsp_no_fisher", "mn_lora_gsp_only") + _BARE_ARMS)
    drop_fisher = arm in (("mn_lora_no_fisher", "mn_lora_no_gsp_no_fisher", "mn_lora_gsp_only") + _BARE_ARMS)
    drop_controllers = arm in (("mn_lora_gsp_only",) + _BARE_ARMS)
    if arm == "mn_lora_gsp_only":
        cfg.mn_lora_gsp_enabled = True  # keep GSP, drop everything else
        drop_gsp = False
    if drop_gsp:
        cfg.mn_lora_gsp_enabled = False
    if drop_fisher:
        cfg.mn_lora_fisher_ewc_enabled = False
    if drop_controllers:
        cfg.mn_lora_tgwd_enabled = False
        cfg.mn_lora_trust_region_enabled = False
        cfg.mn_lora_effective_delta_enabled = False


def _apply_lora_plus(cfg, arm: str, lr_ratio: float = 16.0) -> None:
    """Enable LoRA-native tools for ``*lora_plus*`` arms.

    LoRA+ puts the B/up matrices on ``lr_ratio x`` the base LR via param groups
    built before optimizer construction, so the per-group LRs survive the
    MNLoRAOptimizer wrap (LoRA+ and MN-LoRA compose). ``*_pp`` arms additionally
    turn on the automagic per-parameter LR multiplier.
    """
    if "lora_plus" not in arm:
        return
    cfg.lora_plus_enabled = True
    cfg.lora_plus_lr_ratio = float(lr_ratio)
    if arm.endswith("_pp"):
        cfg.mn_lora_plus_plus_enabled = True


def _run_arm(
    arm: str,
    *,
    mode: str,
    adapter: str,
    model_dir: Path,
    train_dir: Path,
    case_root: Path,
    runtime_device: str,
    runtime_dtype,
    mixed_precision,
    steps: int,
    epochs: int,
    resolution: int,
    rank: int,
    learning_rate: float,
    seed: int,
    proj_rank: int,
    proj_update_interval: int,
    proj_scale: float,
    proj_warmup: int,
    projection_mode: str,
    block_residency: str,
) -> ArmResult:
    output_dir = case_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    mn_lora_enabled = arm.startswith("mn_lora")
    cfg = _build_config(
        **_base_config_kwargs(
            family="anima",
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
            mn_lora_enabled=mn_lora_enabled,
            output_name=f"anima_{_adapter_tag(adapter)}_{arm}",
            block_residency=block_residency,
        )
    )
    _apply_arm_overrides(
        cfg,
        arm,
        seed=seed,
        proj_rank=proj_rank,
        proj_update_interval=proj_update_interval,
        proj_scale=proj_scale,
        proj_warmup=proj_warmup,
        projection_mode=projection_mode,
    )
    if mode == "full":
        # Full fine-tune Adam state (m+v fp32 for the whole DiT) does not fit in
        # 16 GB; offload optimizer state to CPU. Applies to every arm (the paging
        # wrap sits after _create_optimizer, so it wraps AdamW / MNLoRAOptimizer).
        cfg.optimizer_state_paging_enabled = True
        cfg.optimizer_state_paging_min_tensor_mb = 1.0
        cfg.optimizer_state_paging_pin_memory = False
    _apply_mn_lora_ablation(cfg, arm)
    _apply_lora_plus(cfg, arm)

    trainer = LulynxTrainer(cfg)
    trainer.device = runtime_device
    trainer.dtype = runtime_dtype
    logs: list = []
    losses: list = []
    step_times_ms: list = []
    peak_vram_mb = 0.0
    original_on_step_end = trainer._on_step_end

    def _on_step_end(step: int, loss: float, info: dict) -> None:
        nonlocal peak_vram_mb
        step_times_ms.append(float(info.get("step_wall_seconds", 0.0) or 0.0) * 1000.0)
        losses.append(float(loss))
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            peak_vram_mb = max(peak_vram_mb, (total_bytes - free_bytes) / (1024.0 * 1024.0))
        original_on_step_end(step, loss, info)

    trainer._on_step_end = _on_step_end  # type: ignore[assignment]
    trainer.set_callbacks(on_log=logs.append)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    try:
        success = bool(trainer.start())
        failed_reason = "" if success else "trainer.start() returned False"
    except Exception as exc:  # noqa: BLE001 - benchmark must record, not crash the matrix
        success = False
        failed_reason = f"{type(exc).__name__}: {exc}"
    total_wall = time.perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    optimizer = getattr(getattr(trainer, "training_loop", None), "optimizer", None)
    base_optimizer = getattr(optimizer, "base_optimizer", None) or getattr(optimizer, "_base", None)
    steps_completed = int(getattr(getattr(trainer, "training_loop", None), "global_step", 0) or 0)
    mean_step_ms = sum(step_times_ms) / len(step_times_ms) if step_times_ms else 0.0
    result = ArmResult(
        arm=arm,
        mode=mode,
        adapter=adapter,
        success=success,
        failed_reason=failed_reason,
        steps_completed=steps_completed,
        initial_loss=float(losses[0]) if losses else 0.0,
        final_loss=float(losses[-1]) if losses else 0.0,
        min_loss=float(min(losses)) if losses else 0.0,
        loss_delta=float(losses[-1] - losses[0]) if losses else 0.0,
        mean_step_ms=mean_step_ms,
        total_wall_seconds=round(total_wall, 3),
        peak_vram_mb=round(peak_vram_mb, 1),
        optimizer_runtime_type=type(optimizer).__name__ if optimizer is not None else "",
        base_optimizer_type=type(base_optimizer).__name__ if base_optimizer is not None else "",
        svd_grad_projection_active=bool(
            optimizer is not None
            and (type(optimizer).__name__ == "SVDGradientProjectionWrapper" or hasattr(optimizer, "_projectors"))
        ),
        losses=[round(float(v), 8) for v in losses],
        log_tail=logs[-30:],
    )

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_benchmark(
    *,
    mode: str,
    arms: tuple,
    steps: int,
    seed: int,
    rank: int,
    learning_rate: float,
    resolution: int,
    sample_limit: int,
    proj_rank: int,
    proj_update_interval: int,
    proj_scale: float,
    proj_warmup: int,
    projection_mode: str,
    output_root: str,
) -> dict:
    repo_root = _resolve_repo_root()
    model_dir = repo_root / "models" / "anima"
    if not model_dir.exists():
        raise FileNotFoundError(f"Anima model dir not found: {model_dir}")
    source_dir = repo_root / "sucai" / "6_lulu"
    if not source_dir.exists():
        raise FileNotFoundError(f"Source data dir not found: {source_dir}")

    runtime_device, runtime_dtype, mixed_precision = _resolve_runtime()
    session_parent, _reason = _resolve_session_parent(repo_root, output_root)
    session_root = _create_session_root(session_parent)
    mode_root = session_root / mode
    train_dir = mode_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    copy_report = _materialize_dataset_subset(
        source_dir, train_dir, family="anima", sample_limit=max(int(sample_limit), 1), caption_extension=".txt"
    )
    copied_images = max(int(copy_report.get("copied_images", 0) or 0), 1)
    epochs = max((int(steps) + copied_images - 1) // copied_images, 1)
    adapter = "full" if mode == "full" else "lora"
    block_residency = "streaming_offload" if mode == "full" else "resident"

    results: list = []
    for arm in arms:
        case_root = mode_root / arm
        case_root.mkdir(parents=True, exist_ok=True)
        print(f"[grad-subspace-bench] mode={mode} arm={arm} adapter={adapter} steps={steps} -> running", flush=True)
        result = _run_arm(
            arm,
            mode=mode,
            adapter=adapter,
            model_dir=model_dir,
            train_dir=train_dir,
            case_root=case_root,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
            mixed_precision=mixed_precision,
            steps=steps,
            epochs=epochs,
            resolution=resolution,
            rank=rank,
            learning_rate=learning_rate,
            seed=seed,
            proj_rank=proj_rank,
            proj_update_interval=proj_update_interval,
            proj_scale=proj_scale,
            proj_warmup=proj_warmup,
            projection_mode=projection_mode,
            block_residency=block_residency,
        )
        status = "OK" if result.success else f"FAIL ({result.failed_reason})"
        print(
            f"[grad-subspace-bench] {arm}: {status} init={result.initial_loss:.4f} "
            f"final={result.final_loss:.4f} min={result.min_loss:.4f} "
            f"step_ms={result.mean_step_ms:.1f} vram_mb={result.peak_vram_mb:.0f} "
            f"opt={result.optimizer_runtime_type}",
            flush=True,
        )
        results.append(result)

    baseline = next((r for r in results if r.arm == "adamw" and r.success), None)
    comparison = {}
    if baseline is not None:
        for r in results:
            if r.arm == "adamw" or not r.success:
                continue
            comparison[r.arm] = {
                "final_loss_minus_adamw": round(r.final_loss - baseline.final_loss, 6),
                "min_loss_minus_adamw": round(r.min_loss - baseline.min_loss, 6),
                "loss_delta_minus_adamw": round(r.loss_delta - baseline.loss_delta, 6),
                "step_ms_ratio_vs_adamw": round(r.mean_step_ms / baseline.mean_step_ms, 4) if baseline.mean_step_ms else None,
                "peak_vram_delta_mb": round(r.peak_vram_mb - baseline.peak_vram_mb, 1),
            }

    return {
        "benchmark": "mn_lora_grad_subspace_benchmark",
        "mode": mode,
        "device": runtime_device,
        "dtype": str(runtime_dtype),
        "steps": int(steps),
        "seed": int(seed),
        "adapter": adapter,
        "rank": int(rank),
        "learning_rate": float(learning_rate),
        "resolution": int(resolution),
        "projection": {
            "rank": int(proj_rank),
            "update_interval": int(proj_update_interval),
            "scale": float(proj_scale),
            "warmup_steps": int(proj_warmup),
            "mode": projection_mode or "(code default)",
        },
        "copy_report": copy_report,
        "session_root": str(session_root),
        "results": [asdict(r) for r in results],
        "comparison_vs_adamw": comparison,
        "interpretation": (
            "Lower final/min loss vs adamw = better convergence. On LoRA the "
            "projection is near no-op (params already low-rank); the strong "
            "signal is expected on full fine-tune."
        ),
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="lora", choices=["lora", "full"])
    parser.add_argument("--arms", nargs="*", default=None, choices=list(ARMS))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0, help="0 = mode default (lora 1e-4, full 5e-6)")
    parser.add_argument("--resolution", type=int, default=0, help="0 = mode default (lora 512, full 256)")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--proj-rank", type=int, default=0, help="0 = mode default (lora 8, full 128)")
    parser.add_argument("--proj-update-interval", type=int, default=8)
    parser.add_argument("--proj-scale", type=float, default=1.0)
    parser.add_argument("--proj-warmup", type=int, default=0)
    parser.add_argument("--projection-mode", default="", choices=["", "one_sided", "two_sided"])
    parser.add_argument("--output-root", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    arms = tuple(args.arms) if args.arms else ARMS
    learning_rate = args.learning_rate if args.learning_rate > 0 else (1e-4 if args.mode == "lora" else 5e-6)
    proj_rank = args.proj_rank if args.proj_rank > 0 else (8 if args.mode == "lora" else 128)
    resolution = args.resolution if args.resolution > 0 else (512 if args.mode == "lora" else 256)

    payload = run_benchmark(
        mode=args.mode,
        arms=arms,
        steps=max(int(args.steps), 1),
        seed=int(args.seed),
        rank=max(int(args.rank), 1),
        learning_rate=float(learning_rate),
        resolution=max(int(resolution), 64),
        sample_limit=max(int(args.sample_limit), 1),
        proj_rank=int(proj_rank),
        proj_update_interval=max(int(args.proj_update_interval), 1),
        proj_scale=float(args.proj_scale),
        proj_warmup=max(int(args.proj_warmup), 0),
        projection_mode=str(args.projection_mode or ""),
        output_root=str(args.output_root or ""),
    )

    out_path = Path(args.json) if args.json else (Path(payload["session_root"]) / f"grad_subspace_bench_{args.mode}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[grad-subspace-bench] report -> {out_path}", flush=True)
    print(json.dumps(payload["comparison_vs_adamw"], indent=2, sort_keys=True), flush=True)
    ok = any(r["success"] for r in payload["results"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
