# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test save_to alternate checkpoint directory support."""

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

from backend.core.lulynx_trainer.trainer import LulynxTrainer


class _Injector:
    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_down.weight": torch.ones((1, 1), dtype=torch.float32)}


def _trainer(output_dir: Path, save_to: Path) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = SimpleNamespace(
        output_dir=str(output_dir),
        save_to=str(save_to),
        output_name="save_to_smoke",
        model_arch="sdxl",
        network_dim=4,
        network_alpha=2,
        learning_rate=1e-4,
        training_comment="save_to smoke",
        no_metadata=False,
        save_model_as="safetensors",
        save_state=False,
        semantic_tuner_enabled=False,
        mem_efficient_save=False,
        rs_lora_enabled=False,
        reft_enabled=False,
        prefix_tuning_length=0,
        postfix_tuning_length=0,
        anima_merge_export=False,
        merge_export=False,
        save_state_to_huggingface=False,
    )
    trainer.lora_injector = _Injector()
    trainer.model = None
    trainer._ema_tracker = None
    trainer._ti_trainer = None
    trainer._easy_control = None
    trainer._ip_adapter = None
    trainer._repa_projector = None
    trainer._log = lambda _msg: None
    trainer._maybe_release_tool_cuda_cache = lambda _tag: None
    trainer._emit_runtime_event = lambda _event: None
    trainer._write_run_manifest = lambda *_args, **_kwargs: None
    trainer._run_manifest_extra = lambda: {}
    trainer._prune_saved_artifacts = lambda *_args, **_kwargs: None
    return trainer


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output_dir = root / "output_dir"
        save_to = root / "save_to"
        trainer = _trainer(output_dir, save_to)
        trainer._save_model(epoch=1, final=True)

        assert (save_to / "save_to_smoke.safetensors").is_file()
        assert not (output_dir / "save_to_smoke.safetensors").exists()

    print("save_to_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
