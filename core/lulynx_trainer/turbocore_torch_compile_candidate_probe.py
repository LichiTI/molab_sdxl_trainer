"""Optional probe for the TurboCore torch.compile LoRA candidate.

This is intentionally non-fatal: local PyTorch/Windows/Inductor setups vary a
lot, so failures are reported as candidate evidence instead of process errors.
"""

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

from core.turbocore_candidates import get_turbocore_candidate  # noqa: E402
from core.turbocore_parity import check_lora_delta_parity  # noqa: E402
from core.lulynx_trainer.turbocore_lora_fused_benchmark import run_benchmark  # noqa: E402


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


def run_probe(
    *,
    device: torch.device,
    dtype: torch.dtype,
    preset: str = "tiny",
    rank: int = 4,
    iters: int = 1,
    warmup: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    candidate = get_turbocore_candidate("lora_fused", "torch_compile")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "probe": "turbocore_torch_compile_lora_delta",
        "candidate": "torch_compile",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "available": False,
        "ok": False,
        "non_fatal": True,
    }
    if candidate is None:
        payload.update(
            {
                "reason": "candidate_unavailable",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        )
        return payload

    payload["candidate_metadata"] = candidate.as_dict()
    try:
        parity = check_lora_delta_parity(
            batch=1,
            tokens=8,
            in_features=16,
            out_features=16,
            rank=max(int(rank), 1),
            dtype=dtype,
            device=device,
            candidate_name="torch_compile",
            atol=1e-4 if dtype in {torch.float16, torch.bfloat16} else 1e-5,
            rtol=1e-3 if dtype in {torch.float16, torch.bfloat16} else 1e-4,
        )
        benchmark = run_benchmark(
            preset=preset,
            ranks=[max(int(rank), 1)],
            dtype=dtype,
            device=device,
            iters=max(int(iters), 1),
            warmup=max(int(warmup), 0),
            candidate_name="torch_compile",
        )
        payload.update(
            {
                "available": bool(parity.ok),
                "ok": bool(parity.ok),
                "reason": "ok" if parity.ok else "parity_failed",
                "parity": parity.as_dict(),
                "benchmark_summary": benchmark.get("summary", {}),
                "benchmark_results": benchmark.get("results", []),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "available": False,
                "ok": False,
                "reason": "probe_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional TurboCore torch.compile candidate probe")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    payload = run_probe(
        device=device,
        dtype=dtype,
        preset=str(args.preset or "tiny"),
        rank=int(args.rank),
        iters=int(args.iters),
        warmup=int(args.warmup),
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
