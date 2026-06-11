# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test mem_eff_save / mem_efficient_save save-path behavior."""

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

from backend.core.lulynx_trainer.config_adapter import ConfigAdapter
from backend.core.lulynx_trainer.trainer import LulynxTrainer


def test_config_alias_maps_to_mem_efficient_save() -> None:
    cfg = ConfigAdapter.from_frontend_dict({"mem_eff_save": "true"})
    assert cfg.mem_efficient_save is True


def test_save_state_dict_moves_through_mem_efficient_path() -> None:
    try:
        from safetensors import safe_open
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmp:
        trainer = LulynxTrainer.__new__(LulynxTrainer)
        trainer.config = SimpleNamespace(mem_efficient_save=True)
        trainer._log = lambda _msg: None
        trainer._released_cuda_cache_tags = []
        trainer._maybe_release_tool_cuda_cache = trainer._released_cuda_cache_tags.append

        save_path = Path(tmp) / "adapter.safetensors"
        trainer._save_state_dict_to_path(
            {"lora_down.weight": torch.ones((2, 2), dtype=torch.float32)},
            save_path,
            {"test": "mem_eff"},
        )

        with safe_open(str(save_path), framework="pt") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()

    assert keys == ["lora_down.weight"]
    assert metadata == {"test": "mem_eff"}
    assert "mem_efficient_state_dict_save" in trainer._released_cuda_cache_tags


def main() -> int:
    test_config_alias_maps_to_mem_efficient_save()
    test_save_state_dict_moves_through_mem_efficient_path()
    print("mem_efficient_save_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
