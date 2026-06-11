# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima full-finetune 16G research prototypes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.configs import ModelArch, OptimizerType, UnifiedTrainingConfig
from core.lulynx_trainer.activation_compression import ActivationCompressionContext
from core.lulynx_trainer.anima_factored_optimizer import AnimaFactoredAdamW
from core.lulynx_trainer.anima_native_dit import AnimaNativeDiTTinyTrainable
from core.lulynx_trainer.optimizer_state_paging import maybe_wrap_optimizer_state_paging
from core.lulynx_trainer.trainer import LulynxTrainer


def _make_config(**overrides):
    values = dict(
        model_type=ModelArch.ANIMA,
        training_type="full_finetune",
        pretrained_model_name_or_path="H:/tmp/anima-16g-smoke-model",
        train_data_dir="H:/tmp/anima-16g-smoke-data",
        output_dir="H:/tmp/anima-16g-smoke-out",
        output_name="anima_16g_smoke",
        optimizer_type=OptimizerType.ANIMA_FACTORED_ADAMW,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer_args="min_dim=2 min_numel=16",
        max_train_steps=2,
        max_train_epochs=1,
    )
    values.update(overrides)
    return UnifiedTrainingConfig(**values)


def test_factored_optimizer() -> None:
    model = nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 4, bias=False))
    optimizer = AnimaFactoredAdamW(model.parameters(), lr=1e-3, min_dim=2, min_numel=16)
    x = torch.randn(4, 16)
    loss = model(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    profile = optimizer.get_profile()
    assert profile["factored_param_tensors"] >= 1, profile
    print("  [PASS] factored optimizer")


def test_optimizer_state_paging() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Linear(32, 32, bias=False).to(device)
    base = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = maybe_wrap_optimizer_state_paging(base, enabled=True, min_tensor_mb=0.0)
    loss = model(torch.randn(2, 32, device=device)).pow(2).mean()
    loss.backward()
    optimizer.step()
    profile = optimizer.get_profile()
    assert profile["enabled"] is True, profile
    if device == "cuda":
        assert profile["paged_tensors"] >= 1, profile
    else:
        assert profile["status"] == "active", profile
    print("  [PASS] optimizer-state paging")


def test_activation_compression() -> None:
    ctx = ActivationCompressionContext(enabled=True, storage_dtype="fp16", min_tensor_bytes=0)
    model = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 8))
    with ctx.context():
        loss = model(torch.randn(4, 16)).pow(2).mean()
    loss.backward()
    profile = ctx.as_dict()
    assert profile["packed_tensors"] >= 1, profile
    assert profile["restored_tensors"] >= 1, profile
    print("  [PASS] activation compression")


def test_progressive_profile_and_manifest() -> None:
    cfg = _make_config(
        optimizer_type=OptimizerType.ADAMW,
        anima_progressive_full_finetune_enabled=True,
        anima_progressive_full_finetune_schedule="0:1,2:0-1",
        activation_compression_enabled=True,
        activation_compression_min_tensor_mb=0.0,
        anima_rematerializable_block_enabled=True,
    )
    trainer = LulynxTrainer(cfg)
    trainer._log = lambda _msg: None
    trainer.model = SimpleNamespace(
        unet=AnimaNativeDiTTinyTrainable(hidden_dim=8, block_count=2),
    )
    trainer.training_loop = SimpleNamespace(
        optimizer=torch.optim.AdamW(trainer.model.unet.parameters(), lr=1e-3),
        get_memory_experiment_profile=lambda: {"activation_compression": {"enabled": True}},
    )

    trainer._apply_anima_progressive_full_finetune(global_step=0, reason="smoke")
    profile = trainer._refresh_anima_full_finetune_experiments_profile()
    progressive = profile["progressive_full_finetune"]
    assert progressive["active_blocks"] == [1], progressive
    assert profile["rematerializable_block"]["status"] == "profile_only", profile
    assert profile["activation_compression"]["enabled"] is True, profile

    trainer._apply_anima_progressive_full_finetune(global_step=2, reason="smoke")
    profile = trainer._refresh_anima_full_finetune_experiments_profile()
    assert profile["progressive_full_finetune"]["active_blocks"] == [0, 1], profile
    print("  [PASS] progressive/rematerial manifest profile")


def test_optimizer_arg_parser_space_separated() -> None:
    cfg = _make_config(optimizer_args="min_dim=2 min_numel=16 factored_eps=1e-12")
    trainer = LulynxTrainer(cfg)
    parsed = trainer._filtered_custom_args(
        cfg.optimizer_args,
        trainer._optimizer_allowed_args(),
        "optimizer_args",
    )
    assert parsed["min_dim"] == 2, parsed
    assert parsed["min_numel"] == 16, parsed
    assert abs(float(parsed["factored_eps"]) - 1e-12) < 1e-20, parsed
    print("  [PASS] optimizer arg parser")


def main() -> int:
    print("Anima full-finetune 16G experiment smoke")
    print("=" * 52)
    test_factored_optimizer()
    test_optimizer_state_paging()
    test_activation_compression()
    test_progressive_profile_and_manifest()
    test_optimizer_arg_parser_space_separated()
    print("=" * 52)
    print("All Anima 16G experiment smokes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
