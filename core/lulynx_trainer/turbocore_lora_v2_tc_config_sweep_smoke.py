"""Smoke test for the TurboCore LoRA v2_tc config sweep."""

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

from core.lulynx_trainer.turbocore_lora_v2_tc_config_sweep import build_v2_tc_config_sweep  # noqa: E402


def test_sweep_smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = build_v2_tc_config_sweep(
        presets=["sdxl_short"],
        ranks=[4],
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device=device,
        iters=1,
        warmup=0,
        min_width=1024,
        max_configs=2,
    )
    assert payload["benchmark"] == "turbocore_lora_v2_tc_config_sweep"
    assert payload["ok"] is True
    if device.type != "cuda":
        assert payload["skipped"] is True
        return
    assert payload["skipped"] is False
    assert payload["summary"]["measurement_count"] >= 1
    assert payload["summary"]["best_cases"]


if __name__ == "__main__":
    test_sweep_smoke()
    print("turbocore_lora_v2_tc_config_sweep_smoke: ok")
