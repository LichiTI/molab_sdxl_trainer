"""Benchmark experimental PCIe transfer formats for frozen tensor residency.

This script is intentionally standalone. It does not patch the trainer. The
goal is to measure whether a tensor-friendly "training transfer format" can
reduce H2D wait time enough to pay for CPU packing and GPU unpacking.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.transfer_format import (  # type: ignore[no-redef]
        TransferFormatPolicy,
        available_transfer_formats,
        decode_transfer_tensor,
        normalize_transfer_format,
        pack_tensor_for_transfer,
        recommend_transfer_formats,
    )
else:
    from .transfer_format import (
        TransferFormatPolicy,
        available_transfer_formats,
        decode_transfer_tensor,
        normalize_transfer_format,
        pack_tensor_for_transfer,
        recommend_transfer_formats,
    )


def _dtype(name: str) -> torch.dtype:
    value = str(name or "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_cpu_ms(fn: Callable[[], object], *, iters: int) -> tuple[float, object]:
    samples: list[float] = []
    result = None
    for _ in range(max(int(iters), 1)):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), result


def _time_cuda_ms(fn: Callable[[], object], *, device: torch.device, iters: int, warmup: int) -> tuple[float, object]:
    result = None
    for _ in range(max(int(warmup), 0)):
        result = fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(max(int(iters), 1)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples), result


def _parse_shapes(value: str, *, rows: int, cols: int) -> list[tuple[int, int]]:
    raw = str(value or "").strip()
    if not raw:
        return [(int(rows), int(cols))]
    shapes: list[tuple[int, int]] = []
    for item in raw.split(","):
        token = item.strip().lower().replace("*", "x")
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"shape must look like ROWSxCOLS: {item!r}")
        left, right = token.split("x", 1)
        shapes.append((int(left), int(right)))
    return shapes or [(int(rows), int(cols))]


def _parse_formats(value: str) -> list[str]:
    raw = str(value or "raw_fp16,raw_bf16,fp8_e4m3,int8_rowwise,uint4_rowwise")
    formats = [normalize_transfer_format(item.strip()) for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(formats))


def _run_case(
    fmt: str,
    *,
    weight: torch.Tensor,
    activation: torch.Tensor,
    device: torch.device,
    compute_dtype: torch.dtype,
    iters: int,
    warmup: int,
    pack_iters: int,
    include_matmul: bool,
) -> dict[str, object]:
    policy = TransferFormatPolicy(format=fmt, experimental=True)
    cpu_ms, packed = _time_cpu_ms(lambda: pack_tensor_for_transfer(weight, policy), iters=pack_iters)

    def _decode_only() -> torch.Tensor:
        return decode_transfer_tensor(packed, device=device, compute_dtype=compute_dtype)

    decode_ms, decoded = _time_cuda_ms(_decode_only, device=device, iters=iters, warmup=warmup)
    matmul_ms = None
    total_ms = decode_ms
    if include_matmul:
        act = activation.to(device=device, dtype=compute_dtype, non_blocking=True)

        def _decode_matmul() -> torch.Tensor:
            decoded_weight = decode_transfer_tensor(packed, device=device, compute_dtype=compute_dtype)
            return F.linear(act, decoded_weight)

        total_ms, _ = _time_cuda_ms(_decode_matmul, device=device, iters=iters, warmup=warmup)

        def _matmul_only() -> torch.Tensor:
            return F.linear(act, decoded)

        matmul_ms, _ = _time_cuda_ms(_matmul_only, device=device, iters=iters, warmup=warmup)

    return {
        "format": fmt,
        "cpu_pack_ms": round(float(cpu_ms), 4),
        "transfer_mb": round(float(packed.transfer_mb), 4),
        "decode_h2d_ms": round(float(decode_ms), 4),
        "matmul_only_ms": None if matmul_ms is None else round(float(matmul_ms), 4),
        "decode_h2d_matmul_ms": None if not include_matmul else round(float(total_ms), 4),
        "metadata": packed.metadata,
    }


def _recommend(results: list[dict[str, object]], *, include_matmul: bool) -> list[dict[str, object]]:
    ranked = recommend_transfer_formats(results)
    return [
        {
            "format": item["format"],
            "score": item["recommendation_score"],
            "source": item["recommendation_source"],
            "total_ms": item.get("total_ms"),
            "pack_ms": item.get("pack_ms"),
            "amortized_pack_ms": item.get("amortized_pack_ms"),
            "reuse_factor": item.get("reuse_factor"),
            "transfer_mb": item.get("transfer_mb"),
        }
        for item in ranked
        if item.get("recommendation_source") == "benchmark"
    ][:3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--shapes", default="", help="Comma-separated ROWSxCOLS list, e.g. 4096x4096,8192x4096")
    parser.add_argument("--formats", default="raw_fp16,raw_bf16,fp8_e4m3,int8_rowwise,uint4_rowwise")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--compute-dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pack-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-matmul", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    compute_dtype = _dtype(args.compute_dtype)
    torch.manual_seed(int(args.seed))

    availability = available_transfer_formats()
    requested_formats = _parse_formats(args.formats)
    formats = [fmt for fmt in requested_formats if availability.get(fmt, {}).get("available", False)]
    skipped_formats = [fmt for fmt in requested_formats if fmt not in formats]
    shapes = _parse_shapes(args.shapes, rows=int(args.rows), cols=int(args.cols))
    include_matmul = not bool(args.no_matmul)

    cases: list[dict[str, object]] = []
    for rows, cols in shapes:
        weight = torch.randn((rows, cols), dtype=torch.float32) * 0.02
        activation = torch.randn((int(args.batch), cols), dtype=torch.float32)
        results = [
            _run_case(
                fmt,
                weight=weight,
                activation=activation,
                device=device,
                compute_dtype=compute_dtype,
                iters=max(int(args.iters), 1),
                warmup=max(int(args.warmup), 0),
                pack_iters=max(int(args.pack_iters), 1),
                include_matmul=include_matmul,
            )
            for fmt in formats
        ]
        cases.append(
            {
                "shape": {"rows": int(rows), "cols": int(cols), "batch": int(args.batch)},
                "results": results,
                "recommendation": _recommend(results, include_matmul=include_matmul),
            }
        )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "formats": formats,
        "skipped_formats": skipped_formats,
        "availability": availability,
        "cases": cases,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
