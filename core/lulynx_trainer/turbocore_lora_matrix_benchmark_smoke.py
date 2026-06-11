"""Smoke test for the TurboCore LoRA candidate benchmark matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_lora_matrix_benchmark import build_lora_matrix_benchmark  # noqa: E402
from core.lulynx_trainer.turbocore_lora_fused_benchmark import run_benchmark  # noqa: E402


def test_matrix_smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = build_lora_matrix_benchmark(
        presets=["tiny"],
        candidates=["pytorch_explicit", "triton_lora_delta_v1"],
        ranks=[4],
        dtype=torch.float32,
        device=device,
        iters=1,
        warmup=0,
        shape_policy="auto",
    )
    assert payload["benchmark"] == "turbocore_lora_candidate_matrix"
    assert payload["summary"]["run_count"] == 2
    assert len(payload["summary"]["candidate_summaries"]) == 2
    names = {item["candidate"] for item in payload["summary"]["candidate_summaries"]}
    assert {"pytorch_explicit", "triton_lora_delta_v1"}.issubset(names)
    assert payload["summary"]["ready_for_training_activation"] is False
    assert payload["summary"]["quality"]["smoke_only"] is True
    assert "smoke_only_low_iteration_result" in payload["summary"]["quality"]["warnings"]
    for summary in payload["summary"]["candidate_summaries"]:
        assert summary["quality"]["evidence_level"] == "smoke"


def test_shape_policy_skips_dit_v1() -> None:
    payload = build_lora_matrix_benchmark(
        presets=["dit_short"],
        candidates=["triton_lora_delta_v1"],
        ranks=[4],
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        iters=1,
        warmup=0,
        shape_policy="auto",
    )
    assert payload["summary"]["case_count"] == 0
    assert payload["summary"]["skipped_case_count"] > 0
    summary = payload["summary"]["candidate_summaries"][0]
    assert summary["skip_reasons"].get("dit_large_width_matrix_loss", 0) > 0


def test_shape_policy_skips_tiny_v2() -> None:
    payload = build_lora_matrix_benchmark(
        presets=["tiny"],
        candidates=["triton_lora_delta_v2"],
        ranks=[4],
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        iters=1,
        warmup=0,
        shape_policy="auto",
    )
    assert payload["summary"]["case_count"] == 0
    assert payload["summary"]["skipped_case_count"] > 0
    summary = payload["summary"]["candidate_summaries"][0]
    assert summary["skip_reasons"].get("v2_not_target_small_width", 0) > 0


def test_v2_benchmark_reports_config_when_available() -> None:
    if not torch.cuda.is_available():
        return
    payload = run_benchmark(
        preset="sdxl_short",
        ranks=[4],
        dtype=torch.float32,
        device=torch.device("cuda"),
        iters=1,
        warmup=0,
        candidate_name="triton_lora_delta_v2",
        shape_filter=lambda batch, tokens, width, rank: width == 1280,
    )
    row = payload["results"][0]
    assert row["candidate"] == "triton_lora_delta_v2"
    assert row["candidate_config"]["name"] == "sdxl_1280_midwide"


def test_v2_tc_benchmark_reports_config_when_available() -> None:
    if not torch.cuda.is_available():
        return
    payload = run_benchmark(
        preset="sdxl_short",
        ranks=[4],
        dtype=torch.float16,
        device=torch.device("cuda"),
        iters=1,
        warmup=0,
        candidate_name="triton_lora_delta_v2_tc",
        shape_filter=lambda batch, tokens, width, rank: width == 1280,
    )
    row = payload["results"][0]
    assert row["candidate"] == "triton_lora_delta_v2_tc"
    assert row["candidate_config"]["name"] == "sdxl_1280_midwide"


def test_v3_benchmark_reports_route_when_available() -> None:
    if not torch.cuda.is_available():
        return
    payload = run_benchmark(
        preset="dit_short",
        ranks=[8],
        dtype=torch.float16,
        device=torch.device("cuda"),
        iters=1,
        warmup=0,
        candidate_name="triton_lora_delta_v3_dispatch",
        shape_filter=lambda batch, tokens, width, rank: width == 1152,
    )
    row = payload["results"][0]
    assert row["candidate"] == "triton_lora_delta_v3_dispatch"
    assert row["candidate_route"]["path"] == "pytorch_explicit"
    assert row["candidate_route"]["reason"] == "v2_tc_route_disabled_by_paired_benchmark"


if __name__ == "__main__":
    test_matrix_smoke()
    test_shape_policy_skips_dit_v1()
    test_shape_policy_skips_tiny_v2()
    test_v2_benchmark_reports_config_when_available()
    test_v2_tc_benchmark_reports_config_when_available()
    test_v3_benchmark_reports_route_when_available()
    print("turbocore_lora_matrix_benchmark_smoke: ok")
