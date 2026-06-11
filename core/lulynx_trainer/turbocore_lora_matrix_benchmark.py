"""Benchmark matrix for TurboCore LoRA fused delta candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_lora_candidate_policy import decide_lora_candidate_for_shape  # noqa: E402
from core.lulynx_trainer.turbocore_lora_fused_benchmark import SHAPE_PRESETS, run_benchmark  # noqa: E402


def _benchmark_quality(*, iters: int, warmup: int, case_count: int, speedups: list[float]) -> dict[str, Any]:
    smoke_only = int(iters) < 5 or int(warmup) < 1 or int(case_count) < 4
    outlier_count = _count_speedup_outliers(speedups)
    warnings: list[str] = []
    if smoke_only:
        warnings.append("smoke_only_low_iteration_result")
    if outlier_count:
        warnings.append("speedup_outliers_present")
    return {
        "evidence_level": "smoke" if smoke_only else "benchmark",
        "smoke_only": bool(smoke_only),
        "speedup_outlier_count": int(outlier_count),
        "warnings": warnings,
    }


def _count_speedup_outliers(values: list[float]) -> int:
    if len(values) < 4:
        return 0
    sorted_values = sorted(float(value) for value in values)
    median = sorted_values[len(sorted_values) // 2]
    if median <= 0:
        return 0
    return sum(1 for value in sorted_values if value > median * 2.5 or value < median / 2.5)


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = str(value or "float32").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"} and device.type != "cpu":
        return torch.float16
    return torch.float32


def _parse_csv(value: str, default: list[str]) -> list[str]:
    items = [part.strip() for part in str(value or "").split(",") if part.strip()]
    return items or default


def _parse_int_csv(value: str, default: list[int]) -> list[int]:
    items: list[int] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if not text:
            continue
        items.append(max(int(text), 1))
    return items or default


def _safe_benchmark(
    *,
    preset: str,
    candidate: str,
    ranks: list[int],
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    shape_policy: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    effective_ranks, skipped_cases, shape_filter = _apply_shape_policy(
        preset=preset,
        candidate=candidate,
        ranks=ranks,
        shape_policy=shape_policy,
    )
    if not effective_ranks:
        return {
            "ok": True,
            "skipped": True,
            "preset": preset,
            "candidate": candidate,
            "results": [],
            "skipped_cases": skipped_cases,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    try:
        payload = run_benchmark(
            preset=preset,
            ranks=effective_ranks,
            dtype=dtype,
            device=device,
            iters=iters,
            warmup=warmup,
            candidate_name=candidate,
            shape_filter=shape_filter,
        )
        payload["ok"] = True
        payload["shape_policy"] = shape_policy
        payload["requested_ranks"] = ranks
        payload["effective_ranks"] = effective_ranks
        payload["skipped_cases"] = skipped_cases
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "preset": preset,
            "candidate": candidate,
            "error": f"{type(exc).__name__}: {exc}",
            "shape_policy": shape_policy,
            "requested_ranks": ranks,
            "effective_ranks": effective_ranks,
            "skipped_cases": skipped_cases,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _apply_shape_policy(
    *,
    preset: str,
    candidate: str,
    ranks: list[int],
    shape_policy: str,
) -> tuple[list[int], list[dict[str, Any]], Any]:
    shapes = SHAPE_PRESETS[preset]
    effective: list[int] = []
    skipped: list[dict[str, Any]] = []
    allowed: set[tuple[int, int, int, int]] = set()
    for rank in ranks:
        rank_allowed = False
        for batch, tokens, width in shapes:
            decision = decide_lora_candidate_for_shape(
                candidate=candidate,
                preset=preset,
                batch=batch,
                tokens=tokens,
                width=width,
                rank=rank,
                shape_policy=shape_policy,
            )
            if decision.should_run:
                rank_allowed = True
                allowed.add((int(batch), int(tokens), int(width), int(rank)))
            else:
                skipped.append(decision.as_dict())
        if rank_allowed:
            effective.append(rank)
    def shape_filter(batch: int, tokens: int, width: int, rank: int) -> bool:
        return (int(batch), int(tokens), int(width), int(rank)) in allowed

    return effective, skipped, shape_filter


def _summarize_candidate(candidate: str, runs: list[dict[str, Any]], *, iters: int, warmup: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped_cases: list[dict[str, Any]] = []
    for run in runs:
        skipped_cases.extend(run.get("skipped_cases", []) or [])
        if not bool(run.get("ok", False)):
            errors.append(str(run.get("error", "unknown error")))
            continue
        for row in run.get("results", []) or []:
            reference_ms = float(row.get("reference_ms") or 0.0)
            candidate_ms = float(row.get("candidate_ms") or 0.0)
            speedup = reference_ms / candidate_ms if candidate_ms > 0 else 0.0
            rows.append({**row, "speedup_vs_reference": round(speedup, 4)})
    speedups = [float(row["speedup_vs_reference"]) for row in rows]
    wins = [value for value in speedups if value > 1.05]
    losses = [value for value in speedups if value < 0.95]
    quality = _benchmark_quality(iters=iters, warmup=warmup, case_count=len(rows), speedups=speedups)
    return {
        "candidate": candidate,
        "ok": bool(rows) and not errors,
        "case_count": len(rows),
        "skipped_case_count": len(skipped_cases),
        "win_count": len(wins),
        "loss_count": len(losses),
        "avg_speedup_vs_reference": round(sum(speedups) / len(speedups), 4) if speedups else None,
        "best_speedup_vs_reference": round(max(speedups), 4) if speedups else None,
        "worst_speedup_vs_reference": round(min(speedups), 4) if speedups else None,
        "errors": errors[:5],
        "skip_reasons": _count_reasons(skipped_cases),
        "quality": quality,
    }


def _count_reasons(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reason = str(item.get("reason", "unknown") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def build_lora_matrix_benchmark(
    *,
    presets: list[str],
    candidates: list[str],
    ranks: list[int],
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    shape_policy: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    invalid_presets = [preset for preset in presets if preset not in SHAPE_PRESETS]
    if invalid_presets:
        raise ValueError(f"unknown presets {invalid_presets}; available={sorted(SHAPE_PRESETS)}")

    runs: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in candidates}
    for preset in presets:
        for candidate in candidates:
            run = _safe_benchmark(
                preset=preset,
                candidate=candidate,
                ranks=ranks,
                dtype=dtype,
                device=device,
                iters=max(int(iters), 1),
                warmup=max(int(warmup), 0),
                shape_policy=shape_policy,
            )
            runs.append(run)
            by_candidate.setdefault(candidate, []).append(run)

    summaries = [
        _summarize_candidate(candidate, by_candidate.get(candidate, []), iters=max(int(iters), 1), warmup=max(int(warmup), 0))
        for candidate in candidates
    ]
    ranked = sorted(
        summaries,
        key=lambda item: float(item.get("avg_speedup_vs_reference") or 0.0),
        reverse=True,
    )
    case_count = sum(int(item.get("case_count", 0)) for item in summaries)
    matrix_quality = _benchmark_quality(
        iters=max(int(iters), 1),
        warmup=max(int(warmup), 0),
        case_count=case_count,
        speedups=[],
    )
    outliers = sum(int((item.get("quality") or {}).get("speedup_outlier_count", 0)) for item in summaries)
    candidate_warnings = sorted({
        str(warning)
        for item in summaries
        for warning in ((item.get("quality") or {}).get("warnings") or [])
    })
    matrix_quality["speedup_outlier_count"] = outliers
    for warning in candidate_warnings:
        if warning not in matrix_quality["warnings"]:
            matrix_quality["warnings"].append(warning)
    return {
        "schema_version": 1,
        "benchmark": "turbocore_lora_candidate_matrix",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "presets": presets,
        "candidates": candidates,
        "ranks": ranks,
        "iters": int(iters),
        "warmup": int(warmup),
        "shape_policy": shape_policy,
        "summary": {
            "run_count": len(runs),
            "case_count": case_count,
            "skipped_case_count": sum(int(item.get("skipped_case_count", 0)) for item in summaries),
            "best_candidate": ranked[0]["candidate"] if ranked else "",
            "candidate_summaries": summaries,
            "quality": matrix_quality,
            "ready_for_training_activation": False,
            "recommended_next_step": "repeat promising candidates on real route shapes before any training integration",
        },
        "runs": runs,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TurboCore LoRA candidate benchmark matrix")
    parser.add_argument("--presets", default="tiny,sdxl_short,dit_short")
    parser.add_argument("--candidates", default="pytorch_explicit,triton_lora_delta_v0,triton_lora_delta_v1,triton_lora_delta_v2,triton_lora_delta_v2_tc,triton_lora_delta_v3_dispatch")
    parser.add_argument("--ranks", default="4,8,16,32")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--shape-policy", default="auto", choices=["auto", "off", "disabled"])
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    payload = build_lora_matrix_benchmark(
        presets=_parse_csv(args.presets, ["tiny"]),
        candidates=_parse_csv(args.candidates, ["pytorch_explicit", "triton_lora_delta_v1", "triton_lora_delta_v2", "triton_lora_delta_v2_tc", "triton_lora_delta_v3_dispatch"]),
        ranks=_parse_int_csv(args.ranks, [4]),
        dtype=dtype,
        device=device,
        iters=max(int(args.iters), 1),
        warmup=max(int(args.warmup), 0),
        shape_policy=str(args.shape_policy or "auto"),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
