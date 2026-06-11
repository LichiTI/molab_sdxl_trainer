# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Synthetic matrix benchmark for launcher toolbox device policy.

The benchmark is intentionally model-free.  It measures LoRA-like matrix
reconstruction and optional SVD re-projection across CPU/CUDA dtype candidates,
then emits a compact matrix that can guide toolbox auto-device decisions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    out_dim: int
    in_dim: int
    rank: int


PRESETS: dict[str, list[MatrixCase]] = {
    "smoke": [MatrixCase("smoke_64_r4", 64, 64, 4)],
    "quick": [
        MatrixCase("lora_small_320_r16", 320, 320, 16),
        MatrixCase("attention_mid_768_r32", 768, 768, 32),
    ],
    "standard": [
        MatrixCase("lora_small_320_r16", 320, 320, 16),
        MatrixCase("attention_mid_768_r32", 768, 768, 32),
        MatrixCase("unet_wide_1280_r64", 1280, 1280, 64),
        MatrixCase("transformer_wide_1536_r128", 1536, 1536, 128),
    ],
}

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}


def _normalize_preset(value: str) -> str:
    preset = str(value or "quick").strip().lower()
    return preset if preset in PRESETS else "quick"


def _normalize_device(value: str) -> str:
    requested = str(value or "auto").strip().lower()
    return requested if requested in {"auto", "cuda", "cpu"} else "auto"


def _sync(torch: Any, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup(torch: Any, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _time_operation(torch: Any, fn: Any, *, device: str, warmup: int, iterations: int) -> dict[str, Any]:
    for _ in range(max(int(warmup), 0)):
        fn()
    _sync(torch, device)

    samples: list[float] = []
    for _ in range(max(int(iterations), 1)):
        started = time.perf_counter()
        fn()
        _sync(torch, device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "elapsed_ms": round(statistics.median(samples), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
        "samples_ms": [round(value, 4) for value in samples],
    }


def _iteration_plan(preset: str, operation: str) -> tuple[int, int]:
    if operation == "svd_reproject":
        return (1, 1) if preset in {"smoke", "quick"} else (1, 2)
    return (1, 5) if preset in {"smoke", "quick"} else (2, 8)


def _dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def _shape_payload(case: MatrixCase, dtype_name: str) -> dict[str, Any]:
    dense_elements = int(case.out_dim * case.in_dim)
    factor_elements = int(case.out_dim * case.rank + case.rank * case.in_dim)
    dtype_bytes = DTYPE_BYTES.get(dtype_name, 4)
    return {
        "case_id": case.case_id,
        "out_dim": int(case.out_dim),
        "in_dim": int(case.in_dim),
        "rank": int(case.rank),
        "dense_elements": dense_elements,
        "factor_elements": factor_elements,
        "dense_mb": round(dense_elements * dtype_bytes / 1024 / 1024, 4),
        "factor_mb": round(factor_elements * dtype_bytes / 1024 / 1024, 4),
    }


def _candidate_specs(torch: Any, requested_device: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    specs = [{"device": "cpu", "dtype": "fp32", "torch_dtype": torch.float32}]
    skipped: list[dict[str, str]] = []
    if requested_device == "cpu":
        return specs, skipped

    if not torch.cuda.is_available():
        skipped.append({"device": "cuda", "dtype": "fp32", "reason": "cuda_unavailable"})
        skipped.append({"device": "cuda", "dtype": "fp16", "reason": "cuda_unavailable"})
        skipped.append({"device": "cuda", "dtype": "bf16", "reason": "cuda_unavailable"})
        return specs, skipped

    specs.extend([
        {"device": "cuda", "dtype": "fp32", "torch_dtype": torch.float32},
        {"device": "cuda", "dtype": "fp16", "torch_dtype": torch.float16},
    ])
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16_supported:
        specs.append({"device": "cuda", "dtype": "bf16", "torch_dtype": torch.bfloat16})
    else:
        skipped.append({"device": "cuda", "dtype": "bf16", "reason": "cuda_bf16_unavailable"})
    return specs, skipped


def _run_case(
    torch: Any,
    *,
    case: MatrixCase,
    operation: str,
    device: str,
    dtype_name: str,
    torch_dtype: Any,
    preset: str,
) -> dict[str, Any]:
    warmup, iterations = _iteration_plan(preset, operation)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "operation": operation,
        "device": device,
        "dtype": dtype_name,
        "warmup": warmup,
        "iterations": iterations,
        "shape": _shape_payload(case, dtype_name),
    }
    try:
        up = torch.randn((case.out_dim, case.rank), device=device, dtype=torch_dtype)
        down = torch.randn((case.rank, case.in_dim), device=device, dtype=torch_dtype)
        if operation == "lora_reconstruct":
            timing = _time_operation(torch, lambda: up @ down, device=device, warmup=warmup, iterations=iterations)
        else:
            timing = _time_operation(
                torch,
                lambda: torch.linalg.svd((up @ down).float(), full_matrices=False),
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
        row.update({"ok": True, **timing})
    except Exception as exc:  # keep the matrix useful even when one dtype fails
        row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        _cleanup(torch, device)
    return row


def _annotate_speedups(rows: list[dict[str, Any]]) -> None:
    baseline: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("ok") and row.get("device") == "cpu" and row.get("dtype") == "fp32":
            baseline[(str(row.get("case_id")), str(row.get("operation")))] = float(row.get("elapsed_ms") or 0.0)
    for row in rows:
        base = baseline.get((str(row.get("case_id")), str(row.get("operation"))))
        elapsed = float(row.get("elapsed_ms") or 0.0)
        if row.get("ok") and base and elapsed > 0:
            row["speedup_vs_cpu_fp32"] = round(base / elapsed, 4)


def _build_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("operation")))
        item = grouped.setdefault(key, {"case_id": key[0], "operation": key[1]})
        label = f"{row.get('device')}_{row.get('dtype')}_ms"
        item[label] = row.get("elapsed_ms") if row.get("ok") else None
        if row.get("ok"):
            best_ms = item.get("best_ms")
            elapsed = float(row.get("elapsed_ms") or 0.0)
            if best_ms is None or elapsed < float(best_ms):
                item["best_ms"] = row.get("elapsed_ms")
                item["best_candidate"] = f"{row.get('device')}:{row.get('dtype')}"
                item["best_speedup_vs_cpu_fp32"] = row.get("speedup_vs_cpu_fp32", 1.0)
    return list(grouped.values())


def _precision_policy(rows: list[dict[str, Any]], dtype_name: str) -> str:
    gains: list[float] = []
    for row in rows:
        if row.get("ok") and row.get("device") == "cuda" and row.get("dtype") == dtype_name:
            speedup = float(row.get("speedup_vs_cpu_fp32") or 0.0)
            if speedup > 0:
                gains.append(speedup)
    if not gains:
        return "unavailable_or_failed"
    median = statistics.median(gains)
    if median >= 1.25:
        return "promising"
    if median >= 1.05:
        return "marginal"
    return "not_better_than_cpu_baseline"


def _recommendation(rows: list[dict[str, Any]], *, requested_device: str, cuda_available: bool, include_svd: bool) -> dict[str, Any]:
    cuda_wins: list[dict[str, Any]] = []
    for item in _build_matrix(rows):
        best = str(item.get("best_candidate") or "")
        speedup = float(item.get("best_speedup_vs_cpu_fp32") or 0.0)
        if best.startswith("cuda:") and speedup >= 1.1:
            cuda_wins.append(item)

    threshold = None
    if cuda_wins:
        case_ids = {str(item.get("case_id")) for item in cuda_wins}
        dense_values = [
            int((row.get("shape") or {}).get("dense_elements") or 0)
            for row in rows
            if str(row.get("case_id")) in case_ids
        ]
        threshold = min(value for value in dense_values if value > 0) if dense_values else None

    preferred = "cuda" if cuda_wins and cuda_available and requested_device != "cpu" else "cpu"
    notes = []
    if not cuda_available and requested_device != "cpu":
        notes.append("cuda_unavailable_auto_uses_cpu")
    if include_svd:
        notes.append("svd_uses_float32_factorization; fp16/bf16 mainly affect reconstruction cost")
    if preferred == "cuda":
        notes.append("keep_training_guard_enabled_before_using_cuda_for_toolbox_jobs")

    return {
        "preferred_device": preferred,
        "cuda_dense_elements_threshold": threshold,
        "auto_policy": (
            f"use cuda when dense_elements >= {threshold} and no training job is active"
            if threshold
            else "use cpu until local CUDA benchmark shows a stable win"
        ),
        "cuda_fp16_policy": _precision_policy(rows, "fp16"),
        "cuda_bf16_policy": _precision_policy(rows, "bf16"),
        "cuda_win_count": len(cuda_wins),
        "notes": notes,
    }


def run_matrix_task_benchmark(
    *,
    preset: str = "quick",
    device: str = "auto",
    include_svd: bool = True,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    preset = _normalize_preset(preset)
    requested_device = _normalize_device(device)
    specs, skipped_candidates = _candidate_specs(torch, requested_device)
    operations = ["lora_reconstruct", "svd_reproject"] if include_svd else ["lora_reconstruct"]

    rows: list[dict[str, Any]] = []
    for case in PRESETS[preset]:
        for operation in operations:
            for spec in specs:
                rows.append(_run_case(
                    torch,
                    case=case,
                    operation=operation,
                    device=str(spec["device"]),
                    dtype_name=str(spec["dtype"]),
                    torch_dtype=spec["torch_dtype"],
                    preset=preset,
                ))
    _annotate_speedups(rows)

    cuda_available = bool(torch.cuda.is_available())
    gpu_info: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        gpu_info.update({
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_version": getattr(torch.version, "cuda", None),
            "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        })

    recommendation = _recommendation(
        rows,
        requested_device=requested_device,
        cuda_available=cuda_available,
        include_svd=include_svd,
    )
    fallback_reason = "" if cuda_available or requested_device == "cpu" else "CUDA is not available; benchmark used CPU baseline only"
    return {
        "schema_version": 1,
        "benchmark": "toolbox_matrix_task_benchmark",
        "preset": preset,
        "include_svd": bool(include_svd),
        "torch_version": str(torch.__version__),
        "requested_device": requested_device,
        "device": recommendation["preferred_device"],
        "device_fallback_reason": fallback_reason,
        "cuda_available": cuda_available,
        "gpu": gpu_info,
        "skipped_candidates": skipped_candidates,
        "matrix": _build_matrix(rows),
        "cases": rows,
        "recommendation": recommendation,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic toolbox matrix benchmark")
    parser.add_argument("--preset", default="quick", choices=sorted(PRESETS))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--include-svd", action="store_true", default=False)
    args = parser.parse_args(argv)
    payload = run_matrix_task_benchmark(
        preset=args.preset,
        device=args.device,
        include_svd=bool(args.include_svd),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
