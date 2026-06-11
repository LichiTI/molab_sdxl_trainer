"""Run SDXL native UNet runtime comparisons.

This is a thin orchestration wrapper around ``real_model_training_smoke.py``.
It exists so performance-first, balanced, and extreme low-VRAM checks can be
repeated without hand-copying long commands for every residency/swap mode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESIDENCY_MODES = ("resident", "linear_cpu_pinned", "linear_conv_cpu_pinned")
BENCHMARK_PROFILES = ("comparison", "performance", "balanced", "low_vram")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    residency: str
    unet_backend: str = "lulynx_native"
    precision_swap: bool = False
    precision_swap_strategy: str = "balanced"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    return Path("H:/tmp/lulynx_native_unet_benchmark")


def _case_specs(profile: str, include_precision_swap: bool) -> list[CaseSpec]:
    normalized = str(profile or "comparison").strip().lower()
    if normalized == "performance":
        specs = [
            CaseSpec(name="performance_diffusers", residency="resident", unet_backend="diffusers"),
            CaseSpec(name="performance_lulynx_native", residency="resident", unet_backend="lulynx_native"),
        ]
    elif normalized == "balanced":
        specs = [
            CaseSpec(name="balanced_diffusers", residency="resident", unet_backend="diffusers"),
            CaseSpec(name="balanced_lulynx_native", residency="resident", unet_backend="lulynx_native"),
            CaseSpec(name="balanced_lulynx_native_linear_conv_cpu_pinned", residency="linear_conv_cpu_pinned", unet_backend="lulynx_native"),
        ]
    elif normalized == "low_vram":
        specs = [
            CaseSpec(name="low_vram_linear_cpu_pinned", residency="linear_cpu_pinned"),
            CaseSpec(name="low_vram_linear_conv_cpu_pinned", residency="linear_conv_cpu_pinned"),
        ]
    else:
        specs = [CaseSpec(name=mode, residency=mode) for mode in RESIDENCY_MODES]
    if include_precision_swap:
        specs.append(
            CaseSpec(
                name=f"{normalized}_linear_conv_cpu_pinned_precision_swap",
                residency="linear_conv_cpu_pinned",
                unet_backend="lulynx_native",
                precision_swap=True,
                precision_swap_strategy="balanced",
            )
        )
    return specs


def _resolve_auto_defaults(args: argparse.Namespace) -> None:
    profile = str(args.benchmark_profile or "comparison").strip().lower()
    if str(args.te_vae_offload_strategy or "auto").strip().lower() == "auto":
        args.te_vae_offload_strategy = "phase" if profile in {"performance", "balanced"} else "aggressive"
    if str(args.cuda_cache_release_strategy or "auto").strip().lower() == "auto":
        args.cuda_cache_release_strategy = "off" if profile == "performance" else "after_optimizer" if profile == "low_vram" else "off"


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    events = result.get("runtime_event_tail")
    if not isinstance(events, list):
        return []
    return [event for event in events if event.get("event_type") == "step"]


def _summarize_report(path: Path, case: CaseSpec) -> dict[str, Any]:
    report = _load_report(path)
    result = (report.get("results") or [{}])[0]
    events = _step_events(result)
    peaks = [
        event.get("data", {}).get("peak_vram_stages")
        for event in events
        if isinstance(event.get("data"), dict) and event.get("data", {}).get("peak_vram_stages")
    ]
    max_peak = 0.0
    for peak in peaks:
        if isinstance(peak, dict):
            max_peak = max(max_peak, float(peak.get("optimizer_mb") or 0.0), float(peak.get("backward_mb") or 0.0), float(peak.get("forward_mb") or 0.0))
    diagnostics = [
        event.get("data", {}).get("peak_vram_diagnostics")
        for event in events
        if isinstance(event.get("data"), dict) and isinstance(event.get("data", {}).get("peak_vram_diagnostics"), dict)
    ]
    latest_diagnostics = diagnostics[-1] if diagnostics else {}
    cache_releases = [
        event.get("data", {}).get("cuda_cache_release")
        for event in events
        if isinstance(event.get("data"), dict) and isinstance(event.get("data", {}).get("cuda_cache_release"), dict)
    ]
    latest_cache_release = cache_releases[-1] if cache_releases else {}
    precision_swap_offloads = [
        event.get("data", {}).get("precision_swap_offload")
        for event in events
        if isinstance(event.get("data"), dict) and isinstance(event.get("data", {}).get("precision_swap_offload"), dict)
    ]
    latest_precision_swap_offload = precision_swap_offloads[-1] if precision_swap_offloads else {}
    step_wall_seconds = [
        round(float(event.get("data", {}).get("step_wall_seconds") or 0.0), 4)
        for event in events
        if isinstance(event.get("data"), dict)
    ]
    steady_steps = step_wall_seconds[1:] if len(step_wall_seconds) > 1 else step_wall_seconds
    avg_step_wall = round(sum(step_wall_seconds) / len(step_wall_seconds), 4) if step_wall_seconds else 0.0
    steady_avg_step_wall = round(sum(steady_steps) / len(steady_steps), 4) if steady_steps else 0.0
    return {
        "name": case.name,
        "ok": bool(result.get("ok")),
        "duration_seconds": result.get("duration_seconds"),
        "global_step": result.get("global_step"),
        "step_wall_seconds": step_wall_seconds,
        "avg_step_wall_seconds": avg_step_wall,
        "steady_avg_step_wall_seconds": steady_avg_step_wall,
        "max_peak_vram_mb": round(max_peak, 1),
        "peak_vram_diagnostics": latest_diagnostics,
        "cuda_cache_release": latest_cache_release,
        "precision_swap_offload": latest_precision_swap_offload,
        "unet_backend": case.unet_backend,
        "weight_residency": (result.get("native_unet") or {}).get("weight_residency"),
        "memory_optimization": result.get("memory_optimization"),
        "report": str(path),
        "error": result.get("error"),
    }


def _run_case(args: argparse.Namespace, case: CaseSpec, output_root: Path) -> dict[str, Any]:
    real_smoke = Path(__file__).with_name("real_model_training_smoke.py")
    report_path = output_root / f"{case.name}_report.json"
    cmd = [
        sys.executable,
        str(real_smoke),
        "--family",
        "sdxl",
        "--adapter",
        "lora",
        "--steps",
        str(args.steps),
        "--epochs",
        "0",
        "--sample-limit",
        str(args.sample_limit),
        "--resolution",
        str(args.resolution),
        "--rank",
        str(args.rank),
        "--learning-rate",
        str(args.learning_rate),
        "--optimizer",
        str(args.optimizer),
        "--source-data",
        str(args.source_data),
        "--output-root",
        str(output_root / case.name),
        "--json",
        str(report_path),
        "--allow-short-steps",
        "--attention-backend",
        str(args.attention_backend),
        "--te-vae-offload-strategy",
        str(args.te_vae_offload_strategy),
        "--preview-device",
        "off",
        "--sdxl-unet-backend",
        case.unet_backend,
        "--lulynx-weight-residency",
        case.residency,
        "--lulynx-weight-residency-min-params",
        str(args.residency_min_params),
        "--peak-vram-diagnostics",
        "--cuda-cache-release-strategy",
        str(args.cuda_cache_release_strategy),
        "--cuda-cache-release-interval",
        str(args.cuda_cache_release_interval),
        "--stop-on-failure",
    ]
    if case.precision_swap:
        cmd.extend(["--precision-swap", "--precision-swap-strategy", case.precision_swap_strategy])
    if args.dry_run:
        return {"name": case.name, "dry_run": True, "command": cmd}
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_repo_root()))
    summary = _summarize_report(report_path, case) if report_path.exists() else {"name": case.name, "ok": False, "error": "report not written"}
    summary["wall_seconds"] = round(time.perf_counter() - started, 3)
    summary["returncode"] = int(proc.returncode)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SDXL native UNet residency/swap runtime modes.")
    parser.add_argument("--benchmark-profile", default="comparison", choices=BENCHMARK_PROFILES)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "AdamW8bit", "Automagic++"])
    parser.add_argument("--source-data", default="sucai/6_lulu")
    parser.add_argument("--output-root", default=str(_default_output_root()))
    parser.add_argument("--attention-backend", default="flash2")
    parser.add_argument("--te-vae-offload-strategy", default="auto", choices=["auto", "resident", "phase", "aggressive"])
    parser.add_argument("--residency-min-params", type=int, default=0)
    parser.add_argument("--cuda-cache-release-strategy", default="auto", choices=["auto", "off", "after_optimizer", "every_step"])
    parser.add_argument("--cuda-cache-release-interval", type=int, default=1)
    parser.add_argument("--include-precision-swap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _resolve_auto_defaults(args)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        _run_case(args, case, output_root)
        for case in _case_specs(str(args.benchmark_profile), bool(args.include_precision_swap))
    ]
    aggregate = {
        "ok": all(bool(item.get("ok", item.get("dry_run"))) for item in summaries),
        "benchmark_profile": str(args.benchmark_profile),
        "resolution": int(args.resolution),
        "steps": int(args.steps),
        "sample_limit": int(args.sample_limit),
        "attention_backend": str(args.attention_backend),
        "te_vae_offload_strategy": str(args.te_vae_offload_strategy),
        "residency_min_params": max(int(args.residency_min_params or 0), 0),
        "cuda_cache_release_strategy": str(args.cuda_cache_release_strategy or "off"),
        "cuda_cache_release_interval": max(int(args.cuda_cache_release_interval or 1), 1),
        "results": summaries,
    }
    aggregate_path = output_root / f"native_unet_benchmark_{int(args.resolution)}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"[native-benchmark] report={aggregate_path}")
    return 0 if aggregate["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
