# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for benchmark_pipeline.py (Phase 9.8 / #123)."""

from __future__ import annotations

import os
import sys
import json
import importlib.util
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.benchmark_pipeline",
    os.path.join(_HERE, "benchmark_pipeline.py"),
)
_bp = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.benchmark_pipeline"] = _bp
_spec.loader.exec_module(_bp)


def _make_step_fn(model, x):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def step(_idx: int):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    return step


def test_runner_records_step_latencies():
    model = nn.Linear(16, 16)
    x = torch.randn(2, 16)
    cfg = _bp.BenchmarkConfig(warmup_steps=2, measure_steps=5, sync_cuda=False)
    runner = _bp.BenchmarkRunner(cfg)
    result = runner.run(_make_step_fn(model, x))
    assert len(result.step_latencies_ms) == 5
    assert result.measure_steps == 5
    assert result.mean_step_ms > 0.0
    print("PASS: runner records latencies for measure_steps")


def test_result_summary_returns_string():
    result = _bp.BenchmarkResult(label="test", step_latencies_ms=[1.0, 2.0, 3.0], measure_steps=3)
    s = result.summary()
    assert "test" in s
    assert "[benchmark]" in s
    assert "avg_step_time=" in s
    assert "samples_per_sec=" in s
    print("PASS: BenchmarkResult.summary returns formatted string")


def test_percentile_and_throughput_math():
    result = _bp.BenchmarkResult(
        label="t",
        step_latencies_ms=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        measure_steps=10,
    )
    assert result.median_step_ms == 5.5
    # p95 of 10 sorted values: idx = int(0.95 * 9) = 8 -> value 9.0
    assert result.p95_step_ms == 9.0
    assert abs(result.steps_per_second - 1000.0 / 5.5) < 1e-3
    print("PASS: percentile and throughput math are correct")


def test_compare_results_speedup():
    baseline = _bp.BenchmarkResult(
        label="base", step_latencies_ms=[10.0] * 5, measure_steps=5,
    )
    candidate = _bp.BenchmarkResult(
        label="cand", step_latencies_ms=[5.0] * 5, measure_steps=5,
    )
    cmp = _bp.compare_results(baseline, candidate)
    assert abs(cmp["latency_delta_pct"] - (-50.0)) < 1e-3
    assert abs(cmp["throughput_speedup"] - 2.0) < 1e-3
    print("PASS: compare_results computes speedup correctly")


def test_warmup_does_not_count_toward_measurements():
    counter = {"n": 0}

    def step(_idx):
        counter["n"] += 1

    cfg = _bp.BenchmarkConfig(warmup_steps=4, measure_steps=3, sync_cuda=False)
    result = _bp.BenchmarkRunner(cfg).run(step)
    assert counter["n"] == 7  # warmup + measure
    assert len(result.step_latencies_ms) == 3
    print("PASS: warmup steps run but don't count in measurements")


def test_write_benchmark_json_round_trip():
    results = [
        _bp.BenchmarkResult(label="a", step_latencies_ms=[1.0, 2.0], measure_steps=2),
        _bp.BenchmarkResult(label="b", step_latencies_ms=[3.0, 4.0], measure_steps=2),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.json"
        _bp.write_benchmark_json(results, str(path))
        data = json.loads(path.read_text())
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["label"] == "a"
    print("PASS: write_benchmark_json produces valid JSON payload")


def test_torch_profile_block_disabled_yields_none():
    with _bp.torch_profile_block(enabled=False) as prof:
        assert prof is None
    print("PASS: torch_profile_block(enabled=False) yields None")


if __name__ == "__main__":
    test_runner_records_step_latencies()
    test_result_summary_returns_string()
    test_percentile_and_throughput_math()
    test_compare_results_speedup()
    test_warmup_does_not_count_toward_measurements()
    test_write_benchmark_json_round_trip()
    test_torch_profile_block_disabled_yields_none()
    print("\nAll benchmark_pipeline smoke tests passed!")
