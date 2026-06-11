"""Smoke tests for TurboCore LoRA shape-aware candidate policy."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_lora_candidate_policy import decide_lora_candidate_for_shape  # noqa: E402


def test_policy_allows_middle_width() -> None:
    decision = decide_lora_candidate_for_shape(
        candidate="triton_lora_delta_v1",
        preset="sdxl_short",
        batch=2,
        tokens=128,
        width=640,
        rank=8,
    )
    assert decision.should_run is True
    assert decision.reason == "matrix_positive_width_le_768"


def test_policy_blocks_dit() -> None:
    decision = decide_lora_candidate_for_shape(
        candidate="triton_lora_delta_v1",
        preset="dit_short",
        batch=1,
        tokens=512,
        width=1536,
        rank=8,
    )
    assert decision.should_run is False
    assert decision.reason == "dit_large_width_matrix_loss"


def test_policy_can_disable_filter() -> None:
    decision = decide_lora_candidate_for_shape(
        candidate="triton_lora_delta_v1",
        preset="dit_short",
        batch=1,
        tokens=512,
        width=1536,
        rank=8,
        shape_policy="off",
    )
    assert decision.should_run is True
    assert decision.reason == "policy_disabled"


def test_v2_policy_targets_large_width() -> None:
    decision = decide_lora_candidate_for_shape(
        candidate="triton_lora_delta_v2",
        preset="dit_short",
        batch=1,
        tokens=512,
        width=1536,
        rank=8,
    )
    assert decision.should_run is True
    assert decision.reason == "v2_large_width_target"


def test_v2_policy_skips_small_width() -> None:
    decision = decide_lora_candidate_for_shape(
        candidate="triton_lora_delta_v2",
        preset="tiny",
        batch=2,
        tokens=128,
        width=768,
        rank=8,
    )
    assert decision.should_run is False
    assert decision.reason == "v2_not_target_small_width"


def test_v2_tc_policy_targets_large_width() -> None:
    decision = decide_lora_candidate_for_shape(
        candidate="triton_lora_delta_v2_tc",
        preset="dit_short",
        batch=1,
        tokens=512,
        width=1536,
        rank=8,
    )
    assert decision.should_run is True
    assert decision.reason == "v2_tc_large_width_target"


def test_v3_policy_runs_supported_rank() -> None:
    decision = decide_lora_candidate_for_shape(
        candidate="triton_lora_delta_v3_dispatch",
        preset="dit_short",
        batch=1,
        tokens=512,
        width=1536,
        rank=8,
    )
    assert decision.should_run is True
    assert decision.reason == "v3_dispatcher_routes_or_fallbacks"


if __name__ == "__main__":
    test_policy_allows_middle_width()
    test_policy_blocks_dit()
    test_policy_can_disable_filter()
    test_v2_policy_targets_large_width()
    test_v2_policy_skips_small_width()
    test_v2_tc_policy_targets_large_width()
    test_v3_policy_runs_supported_rank()
    print("turbocore_lora_candidate_policy_smoke: ok")
