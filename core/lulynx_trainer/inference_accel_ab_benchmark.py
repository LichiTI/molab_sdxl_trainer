# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Real-model A/B benchmark for the inference-acceleration cache seam, at the
DiT-block level (no image decode).

The block-level cache seam (``spectrum`` / ``smoothcache``) is wired into the
native Anima DiT block loop (``anima_native_dit._run_blocks``) as a default-off,
strategy-selectable option whose disabled-path parity is proven by the unit
smokes. The real-GPU question is: on the real Anima DiT, how much wall time does
each scheme save per denoising step, what does its cache cost in peak VRAM, and
how far does the denoised latent drift from the exact (no-skip) trajectory.

Why block-level and not end-to-end image generation: the only code path that can
truly exercise the seam is the native Anima DiT forward, but the surrounding
image-generation stack is half-built — the finished-product CLI only introspects
native checkpoints (scaffold, no runnable UNet), and the training-time path runs
cache-first and releases the text encoder / VAE after cache build, which disables
preview rendering. So this harness loads ONLY the executable DiT subset
(``load_anima_native_executable_subset`` — no TE / VAE / cache needed), feeds a
fixed synthetic latent + condition, and runs a real flow-matching denoising loop
with the seam active. It measures the seam mechanism itself; latent drift is the
upstream proxy for the image drift the operator would otherwise see.

Each arm flips one scheme against an exact-compute anchor:

  * ``baseline``   : scheme none, every block computed (exact trajectory anchor)
  * ``spectrum``   : Spectrum block-cache linear extrapolation (probe + seam on)
  * ``smoothcache``: SmoothCache error-guided cache reuse (probe + seam on)

A real skip needs BOTH the per-step decision probe (built here exactly as
``sample_anima`` builds it) AND the execution seam (``cache_seam_context``); the
``resolve_inference_accel_scheme`` helper flips them together per scheme.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/inference_accel_ab_benchmark.py --arms baseline --steps 4 --latent-hw 16
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/inference_accel_ab_benchmark.py --steps 20 --latent-hw 32
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset
from core.lulynx_trainer.unified_cache_seam import (
    build_cache_seam,
    cache_seam_context,
    resolve_inference_accel_scheme,
)
from core.lulynx_trainer.real_model_training_smoke import (
    _create_session_root,
    _resolve_repo_root,
    _resolve_runtime,
    _resolve_session_parent,
)

ARMS = ("baseline", "spectrum", "smoothcache")
SCHEME_BY_ARM = {"baseline": "none", "spectrum": "spectrum", "smoothcache": "smoothcache"}


@dataclass
class ArmResult:
    arm: str
    scheme: str
    success: bool
    failed_reason: str = ""
    num_steps: int = 0
    latent_hw: int = 0
    latent_channels: int = 0
    context_seq: int = 0
    block_count: int = 0
    spectrum_probe: bool = False
    smoothcache_probe: bool = False
    cache_seam_backend: str = "none"
    mean_step_ms: float = 0.0
    median_step_ms: float = 0.0
    total_wall_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    output_mse_vs_baseline: float = 0.0     # 0 for the baseline arm (self-anchor)
    output_max_abs_vs_baseline: float = 0.0
    step_ms: list = field(default_factory=list)


def _reset_vram(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def _peak_vram_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    return 0.0


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_probe_states(scheme: str, num_steps: int, window_size: float):
    """Build the per-step decision source for a scheme, exactly as sample_anima does.

    Returns (spectrum_state, smoothcache_state, spectrum_ctx_factory,
    smoothcache_ctx_factory); unused slots are None. A real skip needs this probe
    plus the execution seam — missing either degrades to passthrough (no skip).
    """
    spectrum_state = smoothcache_state = None
    spectrum_factory = smoothcache_factory = None
    if scheme == "spectrum":
        from core.lulynx_trainer.spectrum_probe import (
            SpectrumProbePolicy,
            SpectrumProbeState,
            spectrum_step_context,
        )
        spectrum_state = SpectrumProbeState(
            SpectrumProbePolicy(enabled=True, window_size=window_size, warmup_steps=6),
            total_steps=num_steps,
        )
        spectrum_factory = spectrum_step_context
    elif scheme == "smoothcache":
        from core.lulynx_trainer.smoothcache import (
            SmoothCachePolicy,
            SmoothCacheState,
            smoothcache_step_context,
        )
        smoothcache_state = SmoothCacheState(
            SmoothCachePolicy(enabled=True, error_threshold=0.08, warmup_steps=2, schedule=None),
            total_steps=num_steps,
        )
        smoothcache_factory = smoothcache_step_context
    return spectrum_state, smoothcache_state, spectrum_factory, smoothcache_factory


def _run_arm(
    arm: str,
    *,
    native_unet,
    device: str,
    dtype,
    latents0: torch.Tensor,
    context: torch.Tensor,
    num_steps: int,
    num_train_timesteps: int,
    discrete_flow_shift: float,
    window_size: float,
    baseline_final: "torch.Tensor | None",
) -> "tuple[ArmResult, torch.Tensor | None]":
    scheme = SCHEME_BY_ARM[arm]
    accel = resolve_inference_accel_scheme(scheme)
    result = ArmResult(
        arm=arm,
        scheme=scheme,
        success=False,
        num_steps=int(num_steps),
        latent_hw=int(latents0.shape[-1]),
        latent_channels=int(latents0.shape[1]),
        context_seq=int(context.shape[1]),
        block_count=len(getattr(native_unet.net, "blocks", []) or []),
        spectrum_probe=accel.spectrum_probe,
        smoothcache_probe=accel.smoothcache_probe,
        cache_seam_backend=accel.cache_seam_backend,
    )
    _reset_vram(device)
    # Flow-matching sigma schedule, identical in shape to sample_anima's loop.
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    if discrete_flow_shift != 1.0:
        sigmas = (sigmas * discrete_flow_shift) / (1.0 + (discrete_flow_shift - 1.0) * sigmas)

    seam = build_cache_seam(
        enabled=accel.enabled, backend=accel.cache_seam_backend,
        spectrum_window_size=int(window_size or 3),
    )
    spectrum_state, smoothcache_state, spectrum_factory, smoothcache_factory = _build_probe_states(
        scheme, num_steps, window_size,
    )
    latents = latents0.clone()
    t0 = perf_counter()
    try:
        with torch.no_grad(), cache_seam_context(seam):
            for i in range(num_steps):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                timestep = (sigma * float(num_train_timesteps)).long().reshape(1)
                spectrum_ctx = (
                    spectrum_factory(spectrum_state.decide(i))
                    if spectrum_state is not None else nullcontext()
                )
                smoothcache_ctx = (
                    smoothcache_factory(smoothcache_state.decide(i))
                    if smoothcache_state is not None else nullcontext()
                )
                _sync(device)
                ti = perf_counter()
                with spectrum_ctx, smoothcache_ctx:
                    pred = native_unet(latents, timestep, context).sample
                _sync(device)
                result.step_ms.append(round((perf_counter() - ti) * 1000.0, 3))
                # Flow-matching Euler update: x += (σ_next - σ) · v_pred.
                latents = latents + (sigma_next - sigma) * pred.to(latents.dtype)
        result.total_wall_seconds = round(perf_counter() - t0, 3)
        result.peak_vram_mb = _peak_vram_mb(device)
        if result.step_ms:
            result.mean_step_ms = round(statistics.fmean(result.step_ms), 3)
            result.median_step_ms = round(statistics.median(result.step_ms), 3)
        if baseline_final is not None and latents.shape == baseline_final.shape:
            diff = (latents.float() - baseline_final.float())
            result.output_mse_vs_baseline = round(float(torch.mean(diff * diff)), 8)
            result.output_max_abs_vs_baseline = round(float(torch.max(torch.abs(diff))), 6)
        result.success = bool(torch.isfinite(latents).all())
        if not result.success:
            result.failed_reason = "non-finite latents"
    except Exception as exc:  # noqa: BLE001 — record, keep the matrix going
        result.failed_reason = f"{type(exc).__name__}: {exc}"
        return result, None
    return result, latents.detach()


def run_benchmark(
    *,
    arms: tuple,
    steps: int,
    seed: int,
    latent_hw: int,
    latent_channels: int,
    context_seq: int,
    context_dim: int,
    block_count: int,
    num_train_timesteps: int,
    discrete_flow_shift: float,
    window_size: float,
    model_path: str,
    output_root: str,
) -> dict:
    repo_root = _resolve_repo_root()
    if model_path:
        checkpoint = Path(model_path)
    else:
        model_dir = repo_root / "models" / "anima" / "diffusion_models"
        checkpoint = model_dir / "anima-preview2.safetensors"
        if not checkpoint.exists():
            checkpoint = model_dir / "anima-base-v1.0.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima DiT checkpoint not found: {checkpoint}")

    device, dtype, _mixed_precision = _resolve_runtime()
    print(f"[inference-ab] loading native DiT ({block_count} blocks) device={device} dtype={dtype} from {checkpoint}", flush=True)
    native_unet, report = load_anima_native_executable_subset(
        checkpoint, block_indices=tuple(range(block_count)), device=device, dtype=dtype,
    )
    # The loader's load_state_dict(assign=True) replaces params with the CPU
    # tensors read from safetensors, so the constructor's device is lost. Move
    # the assembled module onto the target device (dtype was already applied).
    native_unet = native_unet.to(device=device)
    for param in native_unet.parameters():
        param.requires_grad_(False)

    # Fixed synthetic inputs shared verbatim across arms -> trajectories are
    # comparable; only the seam's skipping introduces divergence.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    latents0 = torch.randn(1, latent_channels, latent_hw, latent_hw, generator=gen).to(device=device, dtype=dtype)
    context = torch.randn(1, context_seq, context_dim, generator=gen).to(device=device, dtype=dtype)

    # Global warm-up: absorb first-call CUDA kernel/cuDNN autotune cost.
    try:
        with torch.no_grad():
            _ = native_unet(latents0, torch.tensor([250], device=device), context).sample
        _sync(device)
    except Exception as exc:  # noqa: BLE001
        print(f"[inference-ab] warmup skipped: {type(exc).__name__}: {exc}", flush=True)

    ordered = [a for a in ARMS if a in arms] or list(arms)
    if "baseline" in ordered:
        ordered = ["baseline"] + [a for a in ordered if a != "baseline"]

    results: list = []
    baseline_final: "torch.Tensor | None" = None
    for arm in ordered:
        print(f"[inference-ab] arm={arm} scheme={SCHEME_BY_ARM[arm]} steps={steps} latent={latent_hw} -> running", flush=True)
        result, final = _run_arm(
            arm,
            native_unet=native_unet,
            device=device,
            dtype=dtype,
            latents0=latents0,
            context=context,
            num_steps=steps,
            num_train_timesteps=num_train_timesteps,
            discrete_flow_shift=discrete_flow_shift,
            window_size=window_size,
            baseline_final=baseline_final,
        )
        if arm == "baseline" and result.success:
            baseline_final = final
        status = "OK" if result.success else f"FAIL ({result.failed_reason})"
        print(
            f"[inference-ab] {arm}: {status} mean_step_ms={result.mean_step_ms:.2f} "
            f"vram_mb={result.peak_vram_mb:.0f} mse={result.output_mse_vs_baseline:.6f}",
            flush=True,
        )
        results.append(result)

    baseline = next((r for r in results if r.arm == "baseline" and r.success), None)
    comparison: dict = {}
    if baseline is not None:
        for r in results:
            if r.arm == "baseline" or not r.success:
                continue
            comparison[r.arm] = {
                "step_speedup_vs_baseline": round(baseline.mean_step_ms / r.mean_step_ms, 4) if r.mean_step_ms else None,
                "median_step_speedup_vs_baseline": round(baseline.median_step_ms / r.median_step_ms, 4) if r.median_step_ms else None,
                "peak_vram_delta_mb": round(r.peak_vram_mb - baseline.peak_vram_mb, 1),
                "output_mse_vs_baseline": r.output_mse_vs_baseline,
                "output_max_abs_vs_baseline": r.output_max_abs_vs_baseline,
            }

    return {
        "benchmark": "inference_accel_ab_benchmark",
        "level": "dit_block_forward",
        "device": device,
        "dtype": str(dtype),
        "checkpoint": str(checkpoint),
        "loaded_key_count": int(getattr(report, "loaded_key_count", 0) or 0),
        "steps": int(steps),
        "seed": int(seed),
        "latent_shape": [1, int(latent_channels), int(latent_hw), int(latent_hw)],
        "context_shape": [1, int(context_seq), int(context_dim)],
        "block_count": int(block_count),
        "num_train_timesteps": int(num_train_timesteps),
        "discrete_flow_shift": float(discrete_flow_shift),
        "results": [asdict(r) for r in results],
        "comparison_vs_baseline": comparison,
        "interpretation": (
            "baseline = scheme 'none', every DiT block computed (exact denoising "
            "trajectory anchor). spectrum / smoothcache flip the per-step decision "
            "probe AND the block seam together for a real skip. Read: "
            "step_speedup_vs_baseline>1 = faster per denoising step; "
            "peak_vram_delta_mb = the cache's memory cost; output_mse_vs_baseline = "
            "denoised-latent drift from the exact trajectory (0 = identical, larger "
            "= more deviation; this is the upstream proxy for decoded-image drift). "
            "High MSE means the scheme may mislead training-preview judgement -> "
            "promote it as a finished-product fast path, not a preview default."
        ),
    }


def main(argv: "list | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None, choices=list(ARMS))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--latent-hw", type=int, default=32, help="latent spatial size (32 ~= 256px / 8)")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--context-seq", type=int, default=77)
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=28, help="DiT block count to load/run")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--flow-shift", type=float, default=1.0)
    parser.add_argument("--window-size", type=float, default=3.0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    arms = tuple(args.arms) if args.arms else ARMS
    payload = run_benchmark(
        arms=arms,
        steps=max(int(args.steps), 1),
        seed=int(args.seed),
        latent_hw=max(int(args.latent_hw), 1),
        latent_channels=max(int(args.latent_channels), 1),
        context_seq=max(int(args.context_seq), 1),
        context_dim=max(int(args.context_dim), 1),
        block_count=max(int(args.blocks), 1),
        num_train_timesteps=max(int(args.num_train_timesteps), 1),
        discrete_flow_shift=float(args.flow_shift),
        window_size=float(args.window_size),
        model_path=str(args.model_path or ""),
        output_root=str(args.output_root or ""),
    )

    if args.json:
        out_path = Path(args.json)
    else:
        repo_root = _resolve_repo_root()
        session_parent, _reason = _resolve_session_parent(repo_root, args.output_root)
        session_root = _create_session_root(session_parent)
        out_path = session_root / "inference_accel_ab.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[inference-ab] report -> {out_path}", flush=True)
    print(json.dumps(payload["comparison_vs_baseline"], indent=2, sort_keys=True), flush=True)
    return 0 if any(r["success"] for r in payload["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
