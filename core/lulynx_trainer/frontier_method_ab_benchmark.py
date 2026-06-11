# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Real-model single-variable A/B benchmark for runtime-wired frontier methods.

The backend already wires four roadmap frontier techniques into the live trainer
as **default-off, strategy-selectable** options; their disabled-path parity is
proven by local wiring smokes. What is still missing is a **real-GPU signal**:
does each switch run on a real Anima model without exploding, how does it move
the loss curve, and what does it cost in step time / peak VRAM. This harness
produces exactly that, keeping model / dataset / seed / resolution / steps fixed
and flipping **exactly one** frontier switch per arm against an AdamW + LoRA
baseline anchor:

  * ``baseline``          : AdamW + plain LoRA, every frontier switch OFF (anchor)
  * ``adapter_gradient``  : #70 adapter_target_policy="gradient_selected" (subset)
  * ``sra2_haste``        : #71 sra2_haste_enabled (additive aux loss)
  * ``tlora_linear``      : #72 t_lora_enabled + linear rank schedule
  * ``tlora_geometric``   : #72 t_lora_enabled + geometric rank schedule
  * ``reducer_tread``     : #73 dit_compute_reducer_strategy="tread"
  * ``reducer_diffcr``    : #73 dit_compute_reducer_strategy="diffcr"
  * ``reducer_blockskip`` : #73 dit_compute_reducer_strategy="blockskip"

The ``baseline`` arm runs first and does triple duty: it is the parity anchor,
it **introspects the real unet block-module names** (so #71 picks a capture
layer that emits 3D features instead of silently no-op'ing on a wrong name),
and it accumulates **per module-type LoRA gradient norms** to write the #70
profile JSON that ``adapter_gradient`` consumes. Both products flow to later
arms via the ``shared`` dict — one self-contained matrix run, no manual stages.

It reuses the real training path (``LulynxTrainer.start()``) and the dataset /
config scaffolding from ``real_model_training_smoke`` + ``_base_config_kwargs``
from the grad-subspace benchmark (so the baseline is identical to that harness's
``adamw`` arm). Real preview quality stays the operator's job — this measures
runnability, convergence (loss vs step), wall time and peak VRAM.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/frontier_method_ab_benchmark.py --arms baseline --steps 8
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/frontier_method_ab_benchmark.py --mode lora --steps 40
"""

from __future__ import annotations

import argparse
import gc
import json
import re
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
from core.lulynx_trainer.mn_lora_grad_subspace_benchmark import _base_config_kwargs

ARMS = (
    "baseline",
    "adapter_gradient",
    "sra2_haste",
    "tlora_linear",
    "tlora_geometric",
    "reducer_tread",
    "reducer_diffcr",
    "reducer_blockskip",
)

# A real DiT block container looks like ``...blocks.<int>`` and its forward
# output is the 3D ``[B, tokens, hidden]`` feature SRA2/HASTE wants to capture.
_BLOCK_NAME_RE = re.compile(r"(?:^|\.)blocks\.\d+$")


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
    switch: str = ""              # the single frontier switch this arm flips
    effect_evidence: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)
    log_tail: list = field(default_factory=list)


def _collect_block_layer_names(model) -> list:
    """Real unet block-container module names (``...blocks.<int>``), in order."""
    if model is None:
        return []
    names = [name for name, _ in model.named_modules() if _BLOCK_NAME_RE.search(name)]
    # named_modules yields in registration order; keep that (depth order).
    return names


def _pick_capture_layer(block_names: list) -> str:
    """A mid-depth block whose forward output is a stable 3D feature map."""
    if not block_names:
        return ""
    return block_names[len(block_names) // 2]


def _save_adapter_profile(grad_sums: dict, grad_counts: dict, param_counts: dict, path: Path) -> dict:
    """Write a #70 profile: one row per module type with its mean grad norm.

    ``grad_sums`` holds GPU scalar tensors (accumulated without host sync); we
    only ``.item()`` here, once, at save time.
    """
    layers = []
    for module_type in sorted(grad_counts.keys()):
        count = grad_counts.get(module_type, 0)
        if count <= 0:
            continue
        total = grad_sums[module_type]
        mean_norm = float(total.detach().item()) / float(count) if hasattr(total, "detach") else float(total) / count
        layers.append(
            {
                "name": module_type,
                "gradient_norm": mean_norm,
                "parameter_count": int(param_counts.get(module_type, 0)),
            }
        )
    profile = {"layers": layers, "source": "frontier_method_ab_benchmark/baseline_lora_grad"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return profile


def _apply_frontier_overrides(
    cfg,
    arm: str,
    *,
    profile_path: str,
    capture_layer: str,
    top_k: int,
    rank: int,
    module_types: list,
) -> dict:
    """Flip exactly one frontier switch for ``arm``; return evidence of what was set.

    ``baseline`` leaves cfg untouched (all frontier fields keep their default-off
    config values), so it is the bitwise-parity anchor the smokes already proved.
    """
    if arm == "baseline":
        return {"switch": "none (parity anchor)"}

    if arm == "adapter_gradient":
        cfg.adapter_target_policy = "gradient_selected"
        cfg.adapter_target_policy_profile_path = str(profile_path or "")
        cfg.adapter_target_policy_top_k = int(top_k)
        evidence = {
            "policy": "gradient_selected",
            "profile_path": str(profile_path or ""),
            "top_k": int(top_k),
        }
        # Preview which module types the policy would select (config-level proof
        # that gradient_selected really narrows the injection set vs all).
        try:
            from core.lulynx_trainer.adapter_target_policy_consumer import (
                load_policy_consumer_from_config,
            )

            consumer = load_policy_consumer_from_config(cfg)
            if consumer is not None:
                from core.lulynx_trainer.model_family import get_model_family

                # Use the exact module list the runtime feeds select_targets so the
                # evidence preview matches what trainer.py will actually inject.
                available = list(get_model_family("anima").unet_target_modules)
                selected, rank_map = consumer.select_targets(available, base_rank=int(rank))
                evidence["available_module_types"] = available
                evidence["profiled_module_types"] = list(module_types)
                evidence["selected_types"] = list(selected)
                evidence["rank_map"] = {k: int(v) for k, v in rank_map.items()}
            else:
                evidence["selected_types"] = "consumer_none(profile_missing->runtime_falls_back_to_all)"
        except Exception as exc:  # noqa: BLE001 - evidence only, never fail the arm
            evidence["selected_types_error"] = f"{type(exc).__name__}: {exc}"
        return evidence

    if arm == "sra2_haste":
        cfg.sra2_haste_enabled = True
        cfg.sra2_haste_capture_layers = str(capture_layer or "")
        return {
            "sra2_haste_enabled": True,
            "capture_layer": str(capture_layer or ""),
            "note": "empty capture_layer -> aux loss is a silent no-op (introspect failed)",
        }

    if arm in ("tlora_linear", "tlora_geometric"):
        schedule = "linear" if arm == "tlora_linear" else "geometric"
        cfg.t_lora_enabled = True
        cfg.tlora_rank_schedule = schedule
        cfg.tlora_min_rank = 4
        return {"t_lora_enabled": True, "tlora_rank_schedule": schedule, "tlora_min_rank": 4}

    if arm == "reducer_tread":
        cfg.dit_compute_reducer_strategy = "tread"
        cfg.dit_compute_reducer_keep_ratio = 0.5
        return {"dit_compute_reducer_strategy": "tread", "keep_ratio": 0.5}

    if arm == "reducer_diffcr":
        cfg.dit_compute_reducer_strategy = "diffcr"
        cfg.dit_compute_reducer_compression_ratio = 0.5
        return {"dit_compute_reducer_strategy": "diffcr", "compression_ratio": 0.5}

    if arm == "reducer_blockskip":
        cfg.dit_compute_reducer_strategy = "blockskip"
        cfg.dit_compute_reducer_skip_every = 2
        cfg.dit_compute_reducer_min_block = 1
        return {"dit_compute_reducer_strategy": "blockskip", "skip_every": 2, "min_block": 1}

    return {"switch": f"unknown_arm:{arm}"}


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
    block_residency: str,
    top_k: int,
    shared: dict,
) -> ArmResult:
    output_dir = case_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    is_baseline = arm == "baseline"

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
            mn_lora_enabled=False,  # pure AdamW + LoRA anchor; one variable = frontier switch
            output_name=f"anima_{_adapter_tag(adapter)}_{arm}",
            block_residency=block_residency,
        )
    )
    if hasattr(cfg, "seed"):
        cfg.seed = int(seed)
    if mode == "full":
        cfg.optimizer_state_paging_enabled = True
        cfg.optimizer_state_paging_min_tensor_mb = 1.0
        cfg.optimizer_state_paging_pin_memory = False

    evidence = _apply_frontier_overrides(
        cfg,
        arm,
        profile_path=shared.get("profile_path", ""),
        capture_layer=shared.get("capture_layer", ""),
        top_k=top_k,
        rank=rank,
        module_types=shared.get("adapter_module_types", []),
    )

    trainer = LulynxTrainer(cfg)
    trainer.device = runtime_device
    trainer.dtype = runtime_dtype
    logs: list = []
    losses: list = []
    step_times_ms: list = []
    peak_vram_mb = 0.0
    # Baseline-only #70/#71 collection state (GPU-resident, no per-step host sync).
    grad_sums: dict = {}
    grad_counts: dict = {}
    param_counts: dict = {}
    collected = {"hooks": False, "layers": False}
    original_on_step_end = trainer._on_step_end

    def _install_baseline_probes() -> None:
        unet = getattr(getattr(trainer, "training_loop", None), "unet", None)
        if unet is None:
            return
        if not collected["layers"]:
            block_names = _collect_block_layer_names(unet)
            shared.setdefault("block_layer_names", block_names)
            shared.setdefault("capture_layer", _pick_capture_layer(block_names))
            collected["layers"] = True
        if not collected["hooks"]:
            from core.lulynx_trainer.model_family import get_model_family

            # Map each injected LoRALinear to the model-family target NAME the #70
            # consumer matches on (the same list trainer.py feeds select_targets),
            # so the profile keys line up with the runtime available_modules
            # exactly -- including dotted targets like "to_out.0"/"ff.net.0.proj".
            targets = list(get_model_family("anima").unet_target_modules)
            type_module_counts: dict = {}
            for name, module in unet.named_modules():
                if type(module).__name__ != "LoRALinear":
                    continue
                target = next(
                    (t for t in targets if name.endswith(t) or name.split(".")[-1] == t),
                    None,
                )
                if target is None:
                    continue
                type_module_counts[target] = type_module_counts.get(target, 0) + 1
                for param in module.parameters():
                    if not param.requires_grad:
                        continue
                    param_counts[target] = param_counts.get(target, 0) + int(param.numel())

                    def _accumulate(grad, h=target):
                        norm = grad.detach().norm()
                        grad_sums[h] = norm if h not in grad_sums else grad_sums[h] + norm
                        grad_counts[h] = grad_counts.get(h, 0) + 1

                    param.register_hook(_accumulate)
            shared.setdefault("adapter_module_types", sorted(type_module_counts.keys()))
            shared.setdefault("adapter_module_type_counts", type_module_counts)
            collected["hooks"] = True

    def _on_step_end(step: int, loss: float, info: dict) -> None:
        nonlocal peak_vram_mb
        if is_baseline:
            _install_baseline_probes()
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
    except Exception as exc:  # noqa: BLE001 - record failures, never crash the matrix
        success = False
        failed_reason = f"{type(exc).__name__}: {exc}"
    total_wall = time.perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Baseline publishes the #70 profile + #71 capture layer for the later arms.
    if is_baseline:
        profile_path = case_root.parent / "adapter_target_profile.json"
        profile = _save_adapter_profile(grad_sums, grad_counts, param_counts, profile_path)
        shared["profile_path"] = str(profile_path)
        shared["adapter_profile"] = profile
        evidence["adapter_profile"] = profile
        evidence["adapter_module_types"] = shared.get("adapter_module_types", [])
        evidence["adapter_module_type_counts"] = shared.get("adapter_module_type_counts", {})
        evidence["capture_layer"] = shared.get("capture_layer", "")
        evidence["block_layer_count"] = len(shared.get("block_layer_names", []) or [])

    optimizer = getattr(getattr(trainer, "training_loop", None), "optimizer", None)
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
        switch=str(evidence.get("switch", arm)) if arm == "baseline" else arm,
        effect_evidence=evidence,
        losses=[round(float(v), 8) for v in losses],
        log_tail=logs[-30:],
    )

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _order_arms(arms: tuple) -> list:
    """Force ``baseline`` first so its #70/#71 products exist for later arms."""
    ordered = [a for a in ARMS if a in arms]
    return ordered or list(arms)


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
    top_k: int,
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

    shared: dict = {}
    results: list = []
    for arm in _order_arms(arms):
        case_root = mode_root / arm
        case_root.mkdir(parents=True, exist_ok=True)
        print(f"[frontier-ab] mode={mode} arm={arm} adapter={adapter} steps={steps} -> running", flush=True)
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
            block_residency=block_residency,
            top_k=top_k,
            shared=shared,
        )
        status = "OK" if result.success else f"FAIL ({result.failed_reason})"
        print(
            f"[frontier-ab] {arm}: {status} init={result.initial_loss:.4f} "
            f"final={result.final_loss:.4f} min={result.min_loss:.4f} "
            f"step_ms={result.mean_step_ms:.1f} vram_mb={result.peak_vram_mb:.0f}",
            flush=True,
        )
        results.append(result)

    baseline = next((r for r in results if r.arm == "baseline" and r.success), None)
    comparison = {}
    if baseline is not None:
        for r in results:
            if r.arm == "baseline" or not r.success:
                continue
            comparison[r.arm] = {
                "final_loss_minus_baseline": round(r.final_loss - baseline.final_loss, 6),
                "min_loss_minus_baseline": round(r.min_loss - baseline.min_loss, 6),
                "loss_delta_minus_baseline": round(r.loss_delta - baseline.loss_delta, 6),
                "step_ms_ratio_vs_baseline": round(r.mean_step_ms / baseline.mean_step_ms, 4) if baseline.mean_step_ms else None,
                "peak_vram_delta_mb": round(r.peak_vram_mb - baseline.peak_vram_mb, 1),
            }

    return {
        "benchmark": "frontier_method_ab_benchmark",
        "mode": mode,
        "device": runtime_device,
        "dtype": str(runtime_dtype),
        "steps": int(steps),
        "seed": int(seed),
        "adapter": adapter,
        "rank": int(rank),
        "learning_rate": float(learning_rate),
        "resolution": int(resolution),
        "top_k": int(top_k),
        "capture_layer": shared.get("capture_layer", ""),
        "block_layer_names": shared.get("block_layer_names", []),
        "adapter_module_types": shared.get("adapter_module_types", []),
        "adapter_module_type_counts": shared.get("adapter_module_type_counts", {}),
        "adapter_profile": shared.get("adapter_profile", {}),
        "copy_report": copy_report,
        "session_root": str(session_root),
        "results": [asdict(r) for r in results],
        "comparison_vs_baseline": comparison,
        "interpretation": (
            "baseline = AdamW + plain LoRA, all frontier switches off (parity "
            "anchor). Each arm flips one switch. Read: success=runs on real Anima; "
            "loss_*_minus_baseline = convergence impact at this step budget; "
            "step_ms_ratio / peak_vram_delta = runtime cost. T-LoRA's rank "
            "schedule barely climbs in 40 steps (total_steps floored at 1000), so "
            "its main signal here is runs/parity/cost, not loss."
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
    parser.add_argument("--top-k", type=int, default=2, help="#70: keep the top-k module types by grad norm")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    arms = tuple(args.arms) if args.arms else ARMS
    learning_rate = args.learning_rate if args.learning_rate > 0 else (1e-4 if args.mode == "lora" else 5e-6)
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
        top_k=max(int(args.top_k), 1),
        output_root=str(args.output_root or ""),
    )

    out_path = Path(args.json) if args.json else (Path(payload["session_root"]) / f"frontier_ab_{args.mode}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[frontier-ab] report -> {out_path}", flush=True)
    print(json.dumps(payload["comparison_vs_baseline"], indent=2, sort_keys=True), flush=True)
    ok = any(r["success"] for r in payload["results"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
