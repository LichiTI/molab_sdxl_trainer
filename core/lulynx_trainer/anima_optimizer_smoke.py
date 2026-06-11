# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Anima grouped-LR optimizer construction on native-style names."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.configs import ModelArch, MixedPrecision, OptimizerType, UnifiedTrainingConfig
from core.lulynx_trainer.anima_targets import get_anima_dit_targets
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.trainer import LulynxTrainer


class _TinyAnimaBlock(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_attn.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_attn.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_attn.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_attn.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.mlp.layer2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False),
        )
        self.adaln_modulation_cross_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False),
        )
        self.adaln_modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False),
        )


class _TinyAnimaNet(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyAnimaBlock(hidden_dim)])
        self.final_layer = nn.Module()
        self.final_layer.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
        )
        self.llm_adapter = nn.Module()
        self.llm_adapter.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)


class _TinyAnimaRoot(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = _TinyAnimaNet(hidden_dim)


def main() -> int:
    cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        pretrained_model_name_or_path="H:/tmp/anima-optimizer-smoke",
        train_data_dir="H:/tmp/anima-optimizer-smoke-data",
        output_dir="H:/tmp/anima-optimizer-smoke-out",
        output_name="anima_optimizer_smoke",
        mixed_precision=MixedPrecision.NO,
        optimizer_type=OptimizerType.ADAMW,
        learning_rate=1e-4,
        weight_decay=0.125,
        network_dim=1,
        network_alpha=1,
        anima_self_attn_lr=1e-4,
        anima_cross_attn_lr=2e-4,
        anima_mlp_lr=3e-4,
        anima_mod_lr=4e-4,
        anima_llm_adapter_lr=5e-4,
    )

    trainer = LulynxTrainer(cfg)
    trainer._log = lambda _msg: None

    model = _TinyAnimaRoot()
    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(model, get_anima_dit_targets(include_llm_adapter=True), prefix="net")
    if not injected:
        raise RuntimeError("Anima optimizer smoke failed to inject native-style tiny DiT LoRA targets")

    trainer.device = "cpu"
    trainer.dtype = torch.float32
    trainer.model = type("Loaded", (), {"unet": model})()
    trainer.lora_injector = injector

    param_groups = trainer._build_anima_grouped_param_groups()
    if param_groups is None:
        raise AssertionError("Expected Anima grouped LR param groups to be built")
    if len(param_groups) != 5:
        raise AssertionError(f"Expected 5 Anima grouped LR param groups, got {len(param_groups)}")

    lr_to_count = {float(group["lr"]): len(group["params"]) for group in param_groups}
    expected_lrs = {1e-4, 2e-4, 3e-4, 4e-4, 5e-4}
    if set(lr_to_count) != expected_lrs:
        raise AssertionError(f"Unexpected grouped LR values: {lr_to_count}")
    if any(count <= 0 for count in lr_to_count.values()):
        raise AssertionError(f"Expected each Anima LR group to receive parameters, got {lr_to_count}")
    if any(float(group["weight_decay"]) != 0.125 for group in param_groups):
        raise AssertionError(f"Expected Anima grouped LR weight_decay=0.125, got {param_groups}")

    optimizer = trainer._create_optimizer()
    if type(optimizer).__name__ != "AdamW":
        raise AssertionError(f"Expected AdamW optimizer, got {type(optimizer).__name__}")
    built_lrs = {float(group["lr"]): len(group["params"]) for group in optimizer.param_groups}
    if built_lrs != lr_to_count:
        raise AssertionError(f"Optimizer param groups do not match Anima grouped LR plan: {built_lrs} vs {lr_to_count}")

    print(
        "Anima optimizer smoke passed: grouped LR fields built 5 AdamW param groups "
        f"with expected learning rates {sorted(built_lrs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
