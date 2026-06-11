# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test base LoRA weight parsing, multiplier merge, and injector load."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.core.lulynx_trainer.base_lora_weights import (
    load_base_lora_weights,
    merge_base_lora_state_dicts,
    parse_base_lora_weight_request,
)
from backend.core.lulynx_trainer.lora_injector import LoRAInjector


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "target.lora_down.weight": torch.full((2, 4), value),
        "target.lora_up.weight": torch.full((3, 2), value + 1),
    }


def test_parse_base_lora_weights_pads_multipliers() -> None:
    cfg = SimpleNamespace(
        base_weight_path="a.safetensors,b.safetensors,c.safetensors",
        base_weights_multiplier="0.5,1.25",
    )
    paths, multipliers = parse_base_lora_weight_request(cfg)

    assert paths == ["a.safetensors", "b.safetensors", "c.safetensors"]
    assert multipliers == [0.5, 1.25, 1.0]


def test_merge_base_lora_state_dicts_applies_multipliers() -> None:
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "first.safetensors"
        second = Path(tmp) / "second.safetensors"
        save_file(_state(2.0), str(first))
        save_file(_state(4.0), str(second))
        merged, loaded = merge_base_lora_state_dicts(
            [str(first), str(second)],
            [0.5, 0.25],
        )

    assert loaded == (str(first), str(second))
    torch.testing.assert_close(merged["target.lora_down.weight"], torch.full((2, 4), 2.0))
    torch.testing.assert_close(merged["target.lora_up.weight"], torch.full((3, 2), 2.75))


def test_load_base_lora_weights_merges_into_injector() -> None:
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "first.safetensors"
        second = Path(tmp) / "second.safetensors"
        save_file(_state(2.0), str(first))
        save_file(_state(4.0), str(second))

        model = torch.nn.Sequential()
        model.add_module("target", torch.nn.Linear(4, 3, bias=False))
        injector = LoRAInjector(rank=2, alpha=2)
        injector.inject(model, ["target"])

        cfg = SimpleNamespace(
            base_weight_path=f"{first},{second}",
            base_weights_multiplier="0.5,0.25",
        )
        resolution = load_base_lora_weights(cfg, injector)
        wrapper = injector.injected_layers["target"]

    assert resolution.applied is True
    assert resolution.mode == "merged"
    assert resolution.loaded_keys == 2
    assert resolution.expected_keys == 2
    torch.testing.assert_close(wrapper.lora.lora_down.weight, torch.full((2, 4), 2.0))
    torch.testing.assert_close(wrapper.lora.lora_up.weight, torch.full((3, 2), 2.75))


def main() -> int:
    test_parse_base_lora_weights_pads_multipliers()
    test_merge_base_lora_state_dicts_applies_multipliers()
    test_load_base_lora_weights_merges_into_injector()
    print("base_lora_weights_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
