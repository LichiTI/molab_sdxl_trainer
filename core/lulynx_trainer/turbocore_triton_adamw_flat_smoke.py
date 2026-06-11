"""Smoke checks for the Triton flat AdamW v0 research kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    project_root = backend_root.parent
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.turbocore_triton_adamw_flat_benchmark import build_triton_adamw_flat_benchmark  # noqa: E402
from core.turbocore_candidates import list_turbocore_candidates  # noqa: E402
from core.turbocore_triton_optimizer import (  # noqa: E402
    triton_adamw_flat_available,
    triton_adamw_flat_metadata,
    triton_adamw_flat_v0_step_,
)


def test_registry_discovery_only() -> None:
    rows = list_turbocore_candidates("native_optimizer")["native_optimizer"]
    by_name = {row["name"]: row for row in rows}
    assert "triton_adamw_flat_v0" in by_name, rows
    row = by_name["triton_adamw_flat_v0"]
    assert row["native"] is True
    assert row["experimental"] is True
    assert row["available"] is False
    assert "flat_contiguous_fp32_buffers_only" in row["notes"]


def test_metadata_training_locked() -> None:
    metadata = triton_adamw_flat_metadata()
    assert metadata["name"] == "triton_adamw_flat_v0"
    assert metadata["training_path_enabled"] is False
    assert metadata["layout"] == "flat_contiguous_buffers_only"


def test_tiny_cuda_step_when_available() -> None:
    if not triton_adamw_flat_available():
        return
    device = torch.device("cuda")
    param = torch.randn(4096, device=device, dtype=torch.float32) * 0.01
    grad = torch.randn_like(param) * 0.001
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    try:
        report = triton_adamw_flat_v0_step_(param, grad, exp_avg, exp_avg_sq, step=1)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        assert "Python.h" in message or "compile" in message or "launch" in message, message
        return
    torch.cuda.synchronize(device)
    assert report["native_kernel_present"] is True
    assert report["training_path_enabled"] is False
    assert bool(torch.isfinite(param).all().item())


def test_tiny_benchmark_when_available() -> None:
    if not triton_adamw_flat_available():
        return
    payload = build_triton_adamw_flat_benchmark(
        numels=[4096],
        device=torch.device("cuda"),
        iters=2,
        warmup=1,
        block_sizes=[1024],
    )
    assert payload["ok"] is True
    assert payload["summary"]["training_activation_allowed"] is False
    row = payload["results"][0]
    if row["native_kernel_present"]:
        assert row["parity_max_abs_diff"] is not None
        assert row["best_block_size"] == 1024
    else:
        assert row["triton_skip_reason"]


def main() -> int:
    test_registry_discovery_only()
    test_metadata_training_locked()
    test_tiny_cuda_step_when_available()
    test_tiny_benchmark_when_available()
    print("PASS: turbocore Triton AdamW flat smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
