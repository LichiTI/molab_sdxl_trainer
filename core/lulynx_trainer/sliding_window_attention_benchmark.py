# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Benchmark/probe for sliding-window attention backends.

This script is intentionally outside the training hot path.  It gives us a
small CUDA-backed A/B probe for the experimental attention profile without
claiming training quality validation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    import sys

    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.runtime_optimizations import (  # noqa: E402
    _flex_attention_available,
    resolve_sliding_window_backend,
    sliding_window_attention,
)


@dataclass
class SlidingWindowCaseResult:
    backend: str
    resolved_backend: str
    supported: bool
    success: bool
    skipped_reason: str = ""
    failed_reason: str = ""
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    peak_allocated_mb: float = 0.0
    max_abs_diff_vs_sdpa: float | None = None
    output_dtype: str = ""
    output_shape: tuple[int, ...] = ()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _make_qkv(
    *,
    batch: int,
    heads: int,
    tokens: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device.type if device.type != "cuda" else device)
    generator.manual_seed(seed)
    q = torch.randn(batch, heads, tokens, head_dim, device=device, dtype=dtype, generator=generator)
    k = torch.randn(batch, heads, tokens, head_dim, device=device, dtype=dtype, generator=generator)
    v = torch.randn(batch, heads, tokens, head_dim, device=device, dtype=dtype, generator=generator)
    return q.requires_grad_(True), k.requires_grad_(True), v.requires_grad_(True)


def _build_flex_runner(
    *,
    backend: str,
    q_len: int,
    kv_len: int,
    window: int,
    device: torch.device,
):
    if backend not in {"flex", "flex_compiled"}:
        return None
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def sliding_mask(_batch, _head, q_idx, kv_idx):
        distance = q_idx - kv_idx
        return (distance >= 0) & (distance < window)

    block_mask = create_block_mask(
        sliding_mask,
        B=None,
        H=None,
        Q_LEN=int(q_len),
        KV_LEN=int(kv_len),
        device=device,
    )
    attention_fn = flex_attention
    if backend == "flex_compiled":
        attention_fn = torch.compile(flex_attention, mode="reduce-overhead")

    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = attention_fn(q, k, v, block_mask=block_mask, scale=None)
        return out[0] if isinstance(out, tuple) else out

    return run

def _run_backend(
    backend: str,
    *,
    reference: torch.Tensor | None,
    batch: int,
    heads: int,
    tokens: int,
    head_dim: int,
    window: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    seed: int,
    torch_limit: int,
    flex_runtime_active: bool,
) -> tuple[SlidingWindowCaseResult, torch.Tensor | None]:
    q, k, v = _make_qkv(
        batch=batch,
        heads=heads,
        tokens=tokens,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    if backend == "flex_compiled":
        resolved = "flex_compiled"
    else:
        resolved = resolve_sliding_window_backend(
            q,
            backend,
            launcher_attention_backend="auto",
            flex_runtime_active=flex_runtime_active,
        )
    result = SlidingWindowCaseResult(
        backend=backend,
        resolved_backend=resolved,
        supported=True,
        success=False,
    )
    if backend in {"flex", "flex_compiled"} and not _flex_attention_available():
        result.supported = False
        result.skipped_reason = "torch.nn.attention.flex_attention is unavailable"
        return result, None
    if backend in {"flex", "flex_compiled"} and device.type != "cuda":
        result.supported = False
        result.skipped_reason = "FlexAttention benchmark is only meaningful on CUDA"
        return result, None
    if backend == "torch_fallback" and tokens > torch_limit:
        result.supported = False
        result.skipped_reason = f"torch_fallback guard: tokens={tokens} > limit={torch_limit}"
        return result, None

    try:
        flex_runner = _build_flex_runner(
            backend=backend,
            q_len=tokens,
            kv_len=tokens,
            window=window,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(max(int(warmup), 0)):
            if flex_runner is not None:
                out = flex_runner(q, k, v)
            else:
                out = sliding_window_attention(
                    q,
                    k,
                    v,
                    window_size=window,
                    backend=backend,
                    torch_fallback_max_tokens=torch_limit,
                    flex_runtime_active=flex_runtime_active,
                )
            out.sum().backward()
            q.grad = k.grad = v.grad = None

        _sync_if_cuda(device)
        forward_total = 0.0
        backward_total = 0.0
        last_out: torch.Tensor | None = None
        for _ in range(max(int(repeats), 1)):
            q.grad = k.grad = v.grad = None
            _sync_if_cuda(device)
            start = time.perf_counter()
            if flex_runner is not None:
                out = flex_runner(q, k, v)
            else:
                out = sliding_window_attention(
                    q,
                    k,
                    v,
                    window_size=window,
                    backend=backend,
                    torch_fallback_max_tokens=torch_limit,
                    flex_runtime_active=flex_runtime_active,
                )
            _sync_if_cuda(device)
            forward_total += time.perf_counter() - start
            loss = out.float().square().mean()
            start = time.perf_counter()
            loss.backward()
            _sync_if_cuda(device)
            backward_total += time.perf_counter() - start
            last_out = out.detach().float().cpu()

        result.success = True
        result.forward_ms = (forward_total / max(int(repeats), 1)) * 1000.0
        result.backward_ms = (backward_total / max(int(repeats), 1)) * 1000.0
        result.peak_allocated_mb = _peak_mb(device)
        if last_out is not None:
            result.output_dtype = str(out.dtype)
            result.output_shape = tuple(int(dim) for dim in last_out.shape)
            if reference is not None:
                result.max_abs_diff_vs_sdpa = float((last_out - reference).abs().max().item())
        return result, last_out
    except Exception as exc:
        result.failed_reason = str(exc)
        return result, None


def run_sliding_window_attention_benchmark(
    *,
    device: str = "cuda",
    dtype: str = "bf16",
    batch: int = 1,
    heads: int = 8,
    tokens: int = 1024,
    head_dim: int = 64,
    window: int = 128,
    warmup: int = 1,
    repeats: int = 3,
    seed: int = 1234,
    include_torch_fallback: bool = False,
    include_compiled_flex: bool = False,
    torch_limit: int = 2048,
) -> dict[str, Any]:
    requested_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)
    if requested_device.type == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
        torch_dtype = torch.float32

    backends = ["sdpa_masked", "flex"]
    if include_compiled_flex:
        backends.append("flex_compiled")
    if include_torch_fallback:
        backends.append("torch_fallback")
    flex_runtime_active = bool(requested_device.type == "cuda")
    reference = None
    results: list[SlidingWindowCaseResult] = []
    for backend in backends:
        result, out = _run_backend(
            backend,
            reference=reference,
            batch=batch,
            heads=heads,
            tokens=tokens,
            head_dim=head_dim,
            window=window,
            device=requested_device,
            dtype=torch_dtype,
            warmup=warmup,
            repeats=repeats,
            seed=seed,
            torch_limit=torch_limit,
            flex_runtime_active=flex_runtime_active,
        )
        if backend == "sdpa_masked" and out is not None:
            reference = out
        results.append(result)

    return {
        "benchmark": "sliding_window_attention",
        "device": str(requested_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "flex_attention_available": bool(_flex_attention_available()),
        "dtype": str(torch_dtype),
        "shape": {
            "batch": int(batch),
            "heads": int(heads),
            "tokens": int(tokens),
            "head_dim": int(head_dim),
            "window": int(window),
        },
        "warmup": int(warmup),
        "repeats": int(repeats),
        "results": [asdict(item) for item in results],
        "interpretation": "Smoke/profile only; short CUDA success is not a training-quality guarantee.",
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe sliding-window attention backends.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-torch-fallback", action="store_true")
    parser.add_argument("--include-compiled-flex", action="store_true")
    parser.add_argument("--torch-limit", type=int, default=2048)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    payload = run_sliding_window_attention_benchmark(
        device=args.device,
        dtype=args.dtype,
        batch=args.batch,
        heads=args.heads,
        tokens=args.tokens,
        head_dim=args.head_dim,
        window=args.window,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        include_torch_fallback=args.include_torch_fallback,
        torch_limit=args.torch_limit,
        include_compiled_flex=args.include_compiled_flex,
    )
    output = Path(args.out) if args.out else _repo_root() / "temp" / "sliding_window_attention_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    print(f"[sliding-window-benchmark] report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

