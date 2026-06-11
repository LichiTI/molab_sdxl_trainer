# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for the sliding-window attention benchmark/probe."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.sliding_window_attention_benchmark import (  # noqa: E402
    run_sliding_window_attention_benchmark,
)


def main() -> int:
    payload = run_sliding_window_attention_benchmark(
        device="cpu",
        dtype="fp32",
        batch=1,
        heads=2,
        tokens=16,
        head_dim=8,
        window=4,
        warmup=0,
        repeats=1,
        include_torch_fallback=True,
        torch_limit=32,
    )
    assert payload["benchmark"] == "sliding_window_attention"
    assert payload["device"] == "cpu"
    results = {item["backend"]: item for item in payload["results"]}
    assert results["sdpa_masked"]["success"] is True
    assert results["torch_fallback"]["success"] is True
    assert results["torch_fallback"]["max_abs_diff_vs_sdpa"] is not None
    assert results["torch_fallback"]["max_abs_diff_vs_sdpa"] < 1e-5
    assert "flex" in results
    print("Sliding-window attention benchmark smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
