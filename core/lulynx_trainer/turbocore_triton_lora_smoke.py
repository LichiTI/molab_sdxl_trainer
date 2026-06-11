"""Smoke test for Triton LoRA delta research candidates."""

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

from core.turbocore_candidates import get_turbocore_candidate, list_turbocore_candidates  # noqa: E402
from core.turbocore_parity import check_lora_delta_parity  # noqa: E402
from core.turbocore_triton_lora import (  # noqa: E402
    triton_lora_delta_available,
    triton_lora_delta_metadata,
    triton_lora_delta_v1_metadata,
    triton_lora_delta_v2_config_for_shape,
    triton_lora_delta_v2_metadata,
    triton_lora_delta_v2_tc_config_candidates_for_shape,
    triton_lora_delta_v2_tc_metadata,
    triton_lora_delta_v3_decision_for_shape,
    triton_lora_delta_v3_metadata,
)


def test_registry_state() -> None:
    rows = list_turbocore_candidates("lora_fused")["lora_fused"]
    by_name = {row["name"]: row for row in rows}
    assert "triton_lora_delta_v0" in by_name, rows
    assert "triton_lora_delta_v1" in by_name, rows
    assert "triton_lora_delta_v2" in by_name, rows
    assert "triton_lora_delta_v2_tc" in by_name, rows
    assert "triton_lora_delta_v3_dispatch" in by_name, rows
    assert by_name["triton_lora_delta_v0"]["experimental"] is True
    assert by_name["triton_lora_delta_v1"]["experimental"] is True
    assert by_name["triton_lora_delta_v2"]["experimental"] is True
    assert by_name["triton_lora_delta_v2_tc"]["experimental"] is True
    assert by_name["triton_lora_delta_v3_dispatch"]["experimental"] is True
    assert by_name["triton_lora_delta_v0"]["available"] is triton_lora_delta_available()
    assert by_name["triton_lora_delta_v1"]["available"] is triton_lora_delta_available()
    assert by_name["triton_lora_delta_v2"]["available"] is triton_lora_delta_available()
    assert by_name["triton_lora_delta_v2_tc"]["available"] is triton_lora_delta_available()
    assert by_name["triton_lora_delta_v3_dispatch"]["available"] is triton_lora_delta_available()
    assert by_name["triton_lora_delta_v0"]["reason"] == triton_lora_delta_metadata()["reason"]
    assert by_name["triton_lora_delta_v1"]["reason"] == triton_lora_delta_v1_metadata()["reason"]
    assert by_name["triton_lora_delta_v2"]["reason"] == triton_lora_delta_v2_metadata()["reason"]
    assert by_name["triton_lora_delta_v2_tc"]["reason"] == triton_lora_delta_v2_tc_metadata()["reason"]
    assert by_name["triton_lora_delta_v3_dispatch"]["reason"] == triton_lora_delta_v3_metadata()["reason"]


def test_triton_parity_when_available() -> None:
    candidate = get_turbocore_candidate("lora_fused", "triton_lora_delta_v0")
    if not triton_lora_delta_available():
        assert candidate is None
        return
    assert candidate is not None
    result = check_lora_delta_parity(
        batch=1,
        tokens=32,
        in_features=64,
        out_features=64,
        rank=4,
        dtype=torch.float16,
        device=torch.device("cuda"),
        candidate_name="triton_lora_delta_v0",
        atol=2e-2,
        rtol=2e-2,
    )
    assert result.ok, result.as_dict()
    assert result.details and result.details["candidate"] == "triton_lora_delta_v0"

    result_v1 = check_lora_delta_parity(
        batch=1,
        tokens=32,
        in_features=64,
        out_features=64,
        rank=4,
        dtype=torch.float16,
        device=torch.device("cuda"),
        candidate_name="triton_lora_delta_v1",
        atol=3e-2,
        rtol=3e-2,
    )
    assert result_v1.ok, result_v1.as_dict()
    assert result_v1.details and result_v1.details["candidate"] == "triton_lora_delta_v1"

    result_v2 = check_lora_delta_parity(
        batch=1,
        tokens=32,
        in_features=64,
        out_features=64,
        rank=4,
        dtype=torch.float16,
        device=torch.device("cuda"),
        candidate_name="triton_lora_delta_v2",
        atol=2e-2,
        rtol=2e-2,
    )
    assert result_v2.ok, result_v2.as_dict()
    assert result_v2.details and result_v2.details["candidate"] == "triton_lora_delta_v2"

    result_v2_tc = check_lora_delta_parity(
        batch=1,
        tokens=32,
        in_features=64,
        out_features=64,
        rank=4,
        dtype=torch.float16,
        device=torch.device("cuda"),
        candidate_name="triton_lora_delta_v2_tc",
        atol=3e-2,
        rtol=3e-2,
    )
    assert result_v2_tc.ok, result_v2_tc.as_dict()
    assert result_v2_tc.details and result_v2_tc.details["candidate"] == "triton_lora_delta_v2_tc"

    result_v3 = check_lora_delta_parity(
        batch=1,
        tokens=32,
        in_features=64,
        out_features=64,
        rank=4,
        dtype=torch.float16,
        device=torch.device("cuda"),
        candidate_name="triton_lora_delta_v3_dispatch",
        atol=3e-2,
        rtol=3e-2,
    )
    assert result_v3.ok, result_v3.as_dict()
    assert result_v3.details and result_v3.details["candidate"] == "triton_lora_delta_v3_dispatch"


def test_v2_config_table() -> None:
    config_1280 = triton_lora_delta_v2_config_for_shape(out_features=1280, rank=8)
    assert config_1280["name"] == "sdxl_1280_midwide"
    assert config_1280["block_n"] == 64
    assert config_1280["block_r"] == 16

    config_3072 = triton_lora_delta_v2_config_for_shape(out_features=3072, rank=32)
    assert config_3072["name"] == "dit_3072_wide"
    assert config_3072["block_n"] == 128
    assert config_3072["block_r"] == 32

    sweep = triton_lora_delta_v2_tc_config_candidates_for_shape(out_features=1280, rank=8)
    names = {item["name"] for item in sweep}
    assert "sdxl_1280_midwide" in names
    assert "tc_m16_n32_w4_s2" in names
    assert all(item["out_features"] == 1280 for item in sweep)

    decision_small = triton_lora_delta_v3_decision_for_shape(dtype=torch.float16, out_features=640, rank=8)
    assert decision_small["path"] == "triton_lora_delta_v1"
    decision_large = triton_lora_delta_v3_decision_for_shape(dtype=torch.float16, out_features=1152, rank=8)
    assert decision_large["path"] == "pytorch_explicit"
    assert decision_large["reason"] == "v2_tc_route_disabled_by_paired_benchmark"
    decision_disabled = triton_lora_delta_v3_decision_for_shape(dtype=torch.float16, out_features=1536, rank=8)
    assert decision_disabled["path"] == "pytorch_explicit"
    assert decision_disabled["reason"] == "v2_tc_route_disabled_by_paired_benchmark"


if __name__ == "__main__":
    test_registry_state()
    test_triton_parity_when_available()
    test_v2_config_table()
    print("turbocore_triton_lora_smoke: ok")
