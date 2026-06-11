"""Smoke tests for TurboCore parity anchors."""

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

from core.turbocore_parity import (  # noqa: E402
    build_turbocore_parity_report,
    check_lora_delta_parity,
    check_native_optimizer_parity,
    check_stateful_native_optimizer_parity,
)


def test_lora_delta_parity_anchor() -> None:
    result = check_lora_delta_parity(
        batch=1,
        tokens=8,
        in_features=16,
        out_features=16,
        rank=4,
        dtype=torch.float32,
        device="cpu",
    )
    assert result.ok, result.as_dict()
    assert result.max_abs_error <= result.tolerance_abs


def test_native_optimizer_parity_anchor() -> None:
    result = check_native_optimizer_parity(
        layers=2,
        in_features=16,
        out_features=16,
        rank=4,
        dtype=torch.float32,
        device="cpu",
    )
    assert result.ok, result.as_dict()
    assert result.details and result.details["parameter_tensors"] == 4


def test_stateful_native_optimizer_parity_anchor() -> None:
    result = check_stateful_native_optimizer_parity(
        layers=2,
        in_features=16,
        out_features=16,
        rank=4,
        steps=3,
        dtype=torch.float32,
        device="cpu",
    )
    assert result.ok, result.as_dict()
    assert result.details and result.details["restore_ok"] is True
    assert result.details["nonfinite_skip_ok"] is True
    assert result.details["nonfinite_params_unchanged"] is True


def test_default_parity_report() -> None:
    report = build_turbocore_parity_report(device="cpu", dtype=torch.float32)
    assert report["summary"]["ok"] is True
    assert report["summary"]["native_kernel_present"] is False
    assert {row["name"] for row in report["results"]} == {
        "lora_fused_delta",
        "native_optimizer_adamw",
        "native_optimizer_adamw_stateful",
    }


if __name__ == "__main__":
    test_lora_delta_parity_anchor()
    test_native_optimizer_parity_anchor()
    test_stateful_native_optimizer_parity_anchor()
    test_default_parity_report()
    print("turbocore_parity_smoke: ok")
