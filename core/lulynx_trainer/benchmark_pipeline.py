# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Benchmark / profiling pipeline (Phase 9.8 / #123).

A reusable harness for measuring training step latency, GPU memory peak,
and throughput.  Used to verify that optimizations (compile, blockswap,
fp8, fused optimizers) actually deliver wins on real workloads.

Usage::

    from .benchmark_pipeline import BenchmarkConfig, BenchmarkRunner

    cfg = BenchmarkConfig(warmup_steps=3, measure_steps=20)
    runner = BenchmarkRunner(cfg)
    result = runner.run(step_fn)
    print(result.summary())

The harness operates on any callable that performs one optimization
step.  It is decoupled from the trainer to make A/B comparisons easy.
"""

from __future__ import annotations

import logging
import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Tuning knobs for a benchmark run."""

    warmup_steps: int = 3
    measure_steps: int = 20
    sync_cuda: bool = True
    record_memory: bool = True
    label: str = ""


@dataclass
class BenchmarkResult:
    label: str
    step_latencies_ms: List[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    total_time_ms: float = 0.0
    measure_steps: int = 0
    warmup_steps: int = 0
    samples_per_step: float = 1.0

    @property
    def mean_step_ms(self) -> float:
        return statistics.mean(self.step_latencies_ms) if self.step_latencies_ms else 0.0

    @property
    def median_step_ms(self) -> float:
        return statistics.median(self.step_latencies_ms) if self.step_latencies_ms else 0.0

    @property
    def p95_step_ms(self) -> float:
        if not self.step_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.step_latencies_ms)
        idx = int(0.95 * (len(sorted_lat) - 1))
        return sorted_lat[idx]

    @property
    def std_step_ms(self) -> float:
        if len(self.step_latencies_ms) < 2:
            return 0.0
        return statistics.stdev(self.step_latencies_ms)

    @property
    def steps_per_second(self) -> float:
        if self.mean_step_ms <= 0:
            return 0.0
        return 1000.0 / self.mean_step_ms

    @property
    def samples_per_second(self) -> float:
        return self.steps_per_second * max(float(self.samples_per_step or 1.0), 0.0)

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_bytes / (1024 * 1024)

    def summary(self) -> str:
        return (
            f"[benchmark] label={self.label or 'training'} "
            f"peak_vram={self.peak_memory_mb:.1f}MB "
            f"avg_step_time={self.mean_step_ms:.2f}ms "
            f"median_step_time={self.median_step_ms:.2f}ms "
            f"p95_step_time={self.p95_step_ms:.2f}ms "
            f"samples_per_sec={self.samples_per_second:.2f} "
            f"steps_per_sec={self.steps_per_second:.2f} "
            f"warmup_steps={self.warmup_steps} measured_steps={self.measure_steps}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "mean_step_ms": self.mean_step_ms,
            "median_step_ms": self.median_step_ms,
            "p95_step_ms": self.p95_step_ms,
            "std_step_ms": self.std_step_ms,
            "steps_per_second": self.steps_per_second,
            "samples_per_second": self.samples_per_second,
            "peak_memory_mb": self.peak_memory_mb,
            "warmup_steps": self.warmup_steps,
            "measure_steps": self.measure_steps,
            "samples_per_step": self.samples_per_step,
            "step_latencies_ms": list(self.step_latencies_ms),
        }


class BenchmarkRunner:
    """Run a step function repeatedly with timing + memory tracking."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self, step_fn: Callable[[int], Any]) -> BenchmarkResult:
        """Run ``step_fn(step_idx)`` for warmup + measure_steps and return stats.

        ``step_fn`` should perform one full optimization step (forward,
        loss, backward, optimizer.step, optimizer.zero_grad).  Its
        return value is ignored.
        """
        cfg = self.config
        result = BenchmarkResult(label=cfg.label, measure_steps=cfg.measure_steps, warmup_steps=cfg.warmup_steps)

        cuda_available = torch.cuda.is_available()
        if cuda_available and cfg.record_memory:
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        for i in range(cfg.warmup_steps):
            step_fn(i)
            if cuda_available and cfg.sync_cuda:
                torch.cuda.synchronize()

        # Measure
        total_start = time.perf_counter()
        for i in range(cfg.measure_steps):
            step_start = time.perf_counter()
            step_fn(cfg.warmup_steps + i)
            if cuda_available and cfg.sync_cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - step_start) * 1000.0
            result.step_latencies_ms.append(elapsed_ms)
        total_elapsed = (time.perf_counter() - total_start) * 1000.0
        result.total_time_ms = total_elapsed

        if cuda_available and cfg.record_memory:
            result.peak_memory_bytes = torch.cuda.max_memory_allocated()

        return result


def compare_results(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> Dict[str, float]:
    """Compare two benchmark results and return percentage deltas.

    Positive ``latency_delta_pct`` means candidate is slower than baseline.
    Positive ``memory_delta_pct`` means candidate uses more memory.
    """
    base_lat = baseline.mean_step_ms or 1.0
    cand_lat = candidate.mean_step_ms or 1.0
    base_mem = baseline.peak_memory_bytes or 1
    cand_mem = candidate.peak_memory_bytes or 1

    return {
        "latency_delta_pct": ((cand_lat - base_lat) / base_lat) * 100.0,
        "throughput_speedup": (base_lat / cand_lat),
        "memory_delta_pct": ((cand_mem - base_mem) / base_mem) * 100.0,
        "baseline_step_ms": base_lat,
        "candidate_step_ms": cand_lat,
    }


@contextmanager
def torch_profile_block(
    *,
    enabled: bool = True,
    activities: Optional[List[Any]] = None,
    record_shapes: bool = False,
    profile_memory: bool = False,
):
    """Wrap a code block with ``torch.profiler.profile``.

    Yields the profiler instance (or None when disabled).  Useful for
    ad-hoc deep-dive runs without rewriting the inner loop::

        with torch_profile_block() as prof:
            for i in range(20):
                step_fn(i)
        if prof:
            print(prof.key_averages().table(sort_by="cuda_time_total"))
    """
    if not enabled:
        yield None
        return

    try:
        from torch.profiler import profile as _profile, ProfilerActivity
    except ImportError:
        logger.warning("torch.profiler unavailable — running without profiling")
        yield None
        return

    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

    with _profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
    ) as prof:
        yield prof


def write_benchmark_json(results: List[BenchmarkResult], output_path: str) -> None:
    """Write a list of benchmark results to a JSON file for later comparison."""
    import json
    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [r.to_dict() for r in results],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Benchmark results written to %s", out)
