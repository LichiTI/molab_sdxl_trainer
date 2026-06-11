# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test shared checkpoint/state retention behavior on the native trainer."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.trainer import LulynxTrainer


class _Injector:
    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_down.weight": torch.ones((1, 1), dtype=torch.float32)}


class _Loop:
    def __init__(self) -> None:
        self.global_step = 0
        self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
        self.lr_scheduler = None


def _trainer(root: Path) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = SimpleNamespace(
        output_dir=str(root),
        output_name="retention_smoke",
        model_arch="sdxl",
        network_dim=8,
        network_alpha=4,
        learning_rate=1e-4,
        semantic_tuner_enabled=False,
        training_comment="",
        no_metadata=False,
        save_state=True,
        save_state_on_train_end=True,
        save_model_as="pt",
        checkpoint_keep_last=2,
        save_last_n_epochs=2,
        save_last_n_steps=30,
        save_last_n_epochs_state=1,
        save_last_n_steps_state=20,
        save_every_n_epochs=1,
        save_n_epoch_ratio=3,
        max_train_epochs=6,
        save_state_to_huggingface=False,
    )
    trainer.lora_injector = _Injector()
    trainer.training_loop = _Loop()
    trainer._ema_tracker = None
    trainer._ti_trainer = None
    trainer._resource_manager = None
    trainer._log = lambda _msg: None
    return trainer


def _names(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_file())


def main() -> int:
    root = Path("H:/tmp/lulynx_checkpoint_retention_smoke")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    trainer = _trainer(root)

    assert trainer._resolve_epoch_save_interval(6) == 2

    trainer._save_model(epoch=1)
    trainer._save_model(epoch=2)
    trainer._save_model(epoch=3)

    names = _names(root)
    assert "retention_smoke-000001.pt" not in names, names
    assert "retention_smoke-000002.pt" in names, names
    assert "retention_smoke-000003.pt" in names, names
    assert "retention_smoke-000001-state.pt" not in names, names
    assert "retention_smoke-000002-state.pt" not in names, names
    assert "retention_smoke-000003-state.pt" in names, names

    trainer.training_loop.global_step = 10
    trainer._save_model(epoch=3, step=10)
    trainer.training_loop.global_step = 20
    trainer._save_model(epoch=3, step=20)
    trainer.training_loop.global_step = 40
    trainer._save_model(epoch=3, step=40)

    names = _names(root)
    assert "retention_smoke-step000010.pt" not in names, names
    assert "retention_smoke-step000020.pt" in names, names
    assert "retention_smoke-step000040.pt" in names, names
    assert "retention_smoke-step000010-state.pt" not in names, names
    assert "retention_smoke-step000020-state.pt" not in names, names
    assert "retention_smoke-step000040-state.pt" in names, names

    trainer._maybe_save_final_training_state(6)
    names = _names(root)
    assert "retention_smoke-last-state.pt" in names, names

    print("Checkpoint retention smoke passed: ratio save interval, model retention, state retention, and final state save are honored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
