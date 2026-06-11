"""Smoke tests for PCIe cache observe recommendations."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.pcie_cache_profiler import PcieCacheCandidate, PcieCacheProfile  # noqa: E402


def _candidate(*, missed: int) -> PcieCacheCandidate:
    return PcieCacheCandidate(
        name="net.blocks.0.mlp.layer1.original",
        block_index=0,
        parameter_count=16_777_216,
        transfer_mb=16.0,
        format="fp8_e4m3",
        packed=True,
        reason="cpu_pinned",
        sparse_decision="",
        prefetch_submitted=40,
        prefetch_consumed=40,
        prefetch_missed=missed,
        prefetch_errors=0,
        pack_errors=0,
        decode_errors=0,
        score=64.0,
        recommendation="high_value_cache_candidate",
    )


def test_prefetch_covered_is_ab_only() -> None:
    profile = PcieCacheProfile(
        enabled=True,
        family="anima",
        mode="streaming_offload",
        scope="modules",
        candidate_count=1,
        total_transfer_mb=256.0,
        estimated_cache_mb=256.0,
        high_value_count=1,
        medium_value_count=0,
        low_value_count=0,
        candidates=(_candidate(missed=0),),
        notes=(),
    ).as_dict()
    assert profile["next_action"] == "prefetch_covered_cache_v0_ab_only", profile


def test_missed_prefetch_can_recommend_cache_v0() -> None:
    profile = PcieCacheProfile(
        enabled=True,
        family="anima",
        mode="block_cpu_pinned",
        scope="modules",
        candidate_count=8,
        total_transfer_mb=256.0,
        estimated_cache_mb=256.0,
        high_value_count=8,
        medium_value_count=0,
        low_value_count=0,
        candidates=tuple(_candidate(missed=40) for _ in range(8)),
        notes=(),
    ).as_dict()
    assert profile["next_action"] == "cache_v0_manual_candidate", profile


if __name__ == "__main__":
    test_prefetch_covered_is_ab_only()
    test_missed_prefetch_can_recommend_cache_v0()
    print("pcie_cache_profiler_smoke: ok")
