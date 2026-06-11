"""Smoke tests for PCIe transfer-format benchmark recommendations."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE_ROOT = HERE.parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from lulynx_trainer.transfer_format import (  # noqa: E402
    recommend_transfer_formats,
    transfer_format_experiment_plan,
)


def _formats(rows: list[dict[str, object]]) -> list[str]:
    return [str(row["format"]) for row in rows]


def test_static_fallback() -> None:
    ranked = recommend_transfer_formats()
    formats = _formats(ranked)
    assert formats[:5] == ["fp8_e4m3", "raw_bf16", "raw_fp16", "int8_rowwise", "uint4_rowwise"]
    assert all(row["recommendation_source"] == "static" for row in ranked)
    print("  [PASS] static fallback ranking")


def test_benchmark_ranking_and_quality_penalty() -> None:
    ranked = recommend_transfer_formats(
        [
            {"format": "raw_fp16", "pack_ms": 0.05, "h2d_ms": 1.0, "decode_ms": 0.2, "error_mae": 0.0},
            {"format": "raw_bf16", "pack_ms": 0.05, "h2d_ms": 0.9, "decode_ms": 0.2, "error_mae": 0.0},
            {"format": "int8_rowwise", "pack_ms": 0.5, "h2d_ms": 0.35, "decode_ms": 0.25, "error_mae": 0.01},
            {"format": "uint4_rowwise", "pack_ms": 0.5, "h2d_ms": 0.2, "decode_ms": 0.2, "error_mae": 0.08},
        ],
        max_error_mae=0.02,
    )
    formats = _formats(ranked)
    assert formats[:3] == ["int8_rowwise", "raw_bf16", "raw_fp16"]
    assert formats.index("uint4_rowwise") > formats.index("raw_fp16")
    assert ranked[0]["recommendation_source"] == "benchmark"
    print("  [PASS] benchmark latency ranking + quality penalty")


def test_standalone_benchmark_payload_and_path_read() -> None:
    payload = {
        "cases": [
            {
                "shape": {"rows": 16, "cols": 16, "batch": 2},
                "results": [
                    {"format": "raw_fp16", "cpu_pack_ms": 0.1, "decode_h2d_ms": 0.8, "decode_h2d_matmul_ms": 1.0, "transfer_mb": 0.0005},
                    {"format": "fp8_e4m3", "cpu_pack_ms": 0.2, "decode_h2d_ms": 0.3, "decode_h2d_matmul_ms": 0.55, "transfer_mb": 0.0003, "error_mae": 0.01},
                ],
            }
        ]
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "pcie_transfer_format_benchmark.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        ranked = recommend_transfer_formats(benchmark_path=path)
    assert _formats(ranked)[:2] == ["fp8_e4m3", "raw_fp16"]
    assert ranked[0]["total_ms"] == 0.55
    print("  [PASS] standalone payload path read")


def test_experiment_plan_uses_recommendations() -> None:
    plan = transfer_format_experiment_plan([{"format": "raw_bf16", "total_ms": 0.4, "error_mae": 0.0}])
    assert plan["recommended_first"] == "raw_bf16"
    assert plan["ranked_formats"][0]["recommendation_source"] == "benchmark"
    print("  [PASS] manifest experiment plan accepts benchmark")


def test_reuse_factor_amortizes_pack_cost() -> None:
    rows = [
        {"format": "raw_bf16", "cpu_pack_ms": 0.5, "decode_h2d_matmul_ms": 0.9, "error_mae": 0.0},
        {"format": "fp8_e4m3", "cpu_pack_ms": 12.0, "decode_h2d_matmul_ms": 0.35, "error_mae": 0.0},
    ]
    one_shot = recommend_transfer_formats(rows, reuse_factor=1.0)
    reused = recommend_transfer_formats(rows, reuse_factor=40.0)
    assert _formats(one_shot)[0] == "raw_bf16"
    assert _formats(reused)[0] == "fp8_e4m3"
    assert reused[0]["amortized_pack_ms"] == 0.3
    print("  [PASS] reuse factor amortizes one-time pack cost")


def main() -> int:
    tests = [
        test_static_fallback,
        test_benchmark_ranking_and_quality_penalty,
        test_standalone_benchmark_payload_and_path_read,
        test_experiment_plan_uses_recommendations,
        test_reuse_factor_amortizes_pack_cost,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {exc}")
    print(f"\ntransfer format recommendation smoke: {len(tests) - failed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
