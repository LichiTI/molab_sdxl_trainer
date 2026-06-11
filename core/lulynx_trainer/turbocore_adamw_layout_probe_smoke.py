"""Smoke checks for TurboCore AdamW layout-cost probe."""

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

from core.lulynx_trainer.turbocore_adamw_layout_probe import run_adamw_layout_probe  # noqa: E402


def test_cpu_reports_requires_cuda() -> None:
    payload = run_adamw_layout_probe(preset="micro", device="cpu", iters=1, warmup=0)
    assert payload["ok"] is False
    assert payload["error"] == "layout_probe_requires_cuda"
    assert payload["candidate_metadata"]["training_path_enabled"] is False


def test_cuda_layout_probe_when_available() -> None:
    if not torch.cuda.is_available():
        return
    payload = run_adamw_layout_probe(preset="micro", device="cuda", iters=2, warmup=1, block_size=1024)
    assert payload["ok"] is True
    assert payload["summary"]["training_activation_allowed"] is False
    assert "flat_kernel_gate_ok" in payload["summary"]
    assert "layout_including_gather_scatter_gate_ok" in payload["summary"]
    row = payload["results"][0]
    assert row["torch_adamw_fused_ms"] > 0
    if row["triton_flat_persistent_ms"] is not None:
        assert row["triton_flat_persistent_ms_parity_max_abs_diff"] is not None
    if row["triton_with_gather_scatter_ms"] is not None:
        assert row["gather_grad_ms"] is not None
        assert row["scatter_param_ms"] is not None


def main() -> int:
    test_cpu_reports_requires_cuda()
    test_cuda_layout_probe_when_available()
    print("PASS: turbocore AdamW layout probe smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
