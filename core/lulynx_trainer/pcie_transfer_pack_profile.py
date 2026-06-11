"""Profile CPU pack stages for experimental PCIe transfer formats."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.transfer_format import TransferFormatPolicy, pack_tensor_for_transfer  # type: ignore[no-redef]
else:
    from .transfer_format import TransferFormatPolicy, pack_tensor_for_transfer


def _parse_shapes(value: str, *, rows: int, cols: int) -> list[tuple[int, int]]:
    raw = str(value or "").strip()
    if not raw:
        return [(int(rows), int(cols))]
    shapes: list[tuple[int, int]] = []
    for item in raw.split(","):
        token = item.strip().lower().replace("*", "x")
        if not token:
            continue
        left, right = token.split("x", 1)
        shapes.append((int(left), int(right)))
    return shapes or [(int(rows), int(cols))]


def _time_ms(fn: Callable[[], Any], *, iters: int) -> tuple[float, Any]:
    samples: list[float] = []
    value: Any = None
    for _ in range(max(int(iters), 1)):
        start = time.perf_counter()
        value = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples)), value


def _dtype(name: str) -> torch.dtype:
    normalized = str(name or "float32").strip().lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _pin(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def _profile_fp8(weight: torch.Tensor, *, iters: int) -> dict[str, Any]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return {"available": False, "reason": "torch.float8_e4m3fn_unavailable"}
    cast_ms, fp8 = _time_ms(lambda: weight.to(dtype=fp8_dtype), iters=iters)
    contiguous_ms, contiguous = _time_ms(lambda: fp8.contiguous(), iters=iters)
    pin_ms, pinned = _time_ms(lambda: _pin(contiguous), iters=iters)
    full_ms, packed = _time_ms(
        lambda: pack_tensor_for_transfer(weight, TransferFormatPolicy(format="fp8_e4m3", experimental=True)),
        iters=iters,
    )
    return {
        "available": True,
        "cast_ms": round(cast_ms, 4),
        "contiguous_ms": round(contiguous_ms, 4),
        "pin_ms": round(pin_ms, 4),
        "full_pack_ms": round(full_ms, 4),
        "payload_mb": round(float(_bytes(pinned)) / (1024.0 * 1024.0), 4),
        "packed_transfer_mb": round(float(packed.transfer_mb), 4),
        "pin_returned_pinned": bool(getattr(pinned, "is_pinned", lambda: False)()),
    }


def _profile_raw(weight: torch.Tensor, *, fmt: str, dtype: torch.dtype, iters: int) -> dict[str, Any]:
    cast_ms, cast = _time_ms(lambda: weight.to(dtype=dtype), iters=iters)
    contiguous_ms, contiguous = _time_ms(lambda: cast.contiguous(), iters=iters)
    pin_ms, pinned = _time_ms(lambda: _pin(contiguous), iters=iters)
    full_ms, packed = _time_ms(
        lambda: pack_tensor_for_transfer(weight, TransferFormatPolicy(format=fmt, experimental=True)),
        iters=iters,
    )
    return {
        "available": True,
        "cast_ms": round(cast_ms, 4),
        "contiguous_ms": round(contiguous_ms, 4),
        "pin_ms": round(pin_ms, 4),
        "full_pack_ms": round(full_ms, 4),
        "payload_mb": round(float(_bytes(pinned)) / (1024.0 * 1024.0), 4),
        "packed_transfer_mb": round(float(packed.transfer_mb), 4),
        "pin_returned_pinned": bool(getattr(pinned, "is_pinned", lambda: False)()),
    }


def _profile_case(rows: int, cols: int, *, source_dtype: torch.dtype, iters: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(rows) + int(cols))
    weight = torch.randn((int(rows), int(cols)), dtype=source_dtype) * 0.02
    return {
        "shape": {"rows": int(rows), "cols": int(cols)},
        "source_dtype": str(source_dtype).replace("torch.", ""),
        "source_mb": round(float(_bytes(weight)) / (1024.0 * 1024.0), 4),
        "formats": {
            "raw_fp16": _profile_raw(weight, fmt="raw_fp16", dtype=torch.float16, iters=iters),
            "raw_bf16": _profile_raw(weight, fmt="raw_bf16", dtype=torch.bfloat16, iters=iters),
            "fp8_e4m3": _profile_fp8(weight, iters=iters),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile CPU pack stages for PCIe transfer formats")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--shapes", default="")
    parser.add_argument("--source-dtype", default="float32", choices=["float32", "fp16", "bf16"])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    source_dtype = _dtype(args.source_dtype)
    cases = [
        _profile_case(rows, cols, source_dtype=source_dtype, iters=max(int(args.iters), 1), seed=int(args.seed))
        for rows, cols in _parse_shapes(args.shapes, rows=int(args.rows), cols=int(args.cols))
    ]
    payload = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "source_dtype": str(source_dtype).replace("torch.", ""),
        "iters": max(int(args.iters), 1),
        "cases": cases,
        "summary": _summarize(cases),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        shape = case.get("shape", {})
        for fmt, metrics in (case.get("formats") or {}).items():
            if not isinstance(metrics, dict) or not metrics.get("available", False):
                continue
            rows.append(
                {
                    "shape": f"{shape.get('rows')}x{shape.get('cols')}",
                    "format": fmt,
                    "full_pack_ms": metrics.get("full_pack_ms"),
                    "cast_ms": metrics.get("cast_ms"),
                    "pin_ms": metrics.get("pin_ms"),
                    "packed_transfer_mb": metrics.get("packed_transfer_mb"),
                }
            )
    return {
        "rows": rows,
        "note": "full_pack_ms is median wall-clock CPU time and includes cast, contiguous, and pin attempts",
    }


if __name__ == "__main__":
    raise SystemExit(main())
