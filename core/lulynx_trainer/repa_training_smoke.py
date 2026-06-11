# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for REPA training-loop wiring."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_load_module("core.safe_pickle", CORE_ROOT / "safe_pickle.py")
_load_module("core.configs", CORE_ROOT / "configs.py")
_load_module("core.lulynx_trainer.config", TRAINER_ROOT / "config.py")
config_adapter_mod = _load_module("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
repa_mod = _load_module("core.lulynx_trainer.repa", TRAINER_ROOT / "repa.py")
training_loop_mod = _load_module("core.lulynx_trainer.training_loop", TRAINER_ROOT / "training_loop.py")

ConfigAdapter = config_adapter_mod.ConfigAdapter
REPAFeatureProjector = repa_mod.REPAFeatureProjector
TrainingLoop = training_loop_mod.TrainingLoop


class _UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, sample, timestep, encoder_hidden_states, **_kwargs):
        hidden = self.proj(encoder_hidden_states)
        value = hidden.mean(dim=(1, 2)).view(sample.shape[0], 1, 1, 1)
        return SimpleNamespace(sample=sample + value)


class _Loop(TrainingLoop):
    def __init__(self, projector: nn.Module, *, softrepa: bool = False) -> None:
        unet = _UNet()
        super().__init__(
            unet=unet,
            text_encoder_1=None,
            text_encoder_2=None,
            vae=None,
            tokenizer_1=None,
            tokenizer_2=None,
            noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
            lora_injector=None,
            optimizer=torch.optim.SGD(list(unet.parameters()) + list(projector.parameters()), lr=0.1),
            lr_scheduler=None,
            device="cpu",
            dtype=torch.float32,
            model_arch="anima",
            repa_enabled=True,
            repa_target_modules="proj",
            repa_loss_type="l2",
            repa_loss_weight=0.5,
            repa_projection_dim=4,
            repa_projector=projector,
            softrepa_enabled=softrepa,
            softrepa_schedule="linear",
            softrepa_min_weight=0.0,
            softrepa_max_weight=0.5,
            softrepa_sigma_min=0.2,
            softrepa_sigma_max=0.8,
        )
        self.total_steps = 2


def test_repa_config_aliases() -> bool:
    cfg = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "anima-lora",
            "repa_enabled": "true",
            "repa_layers": "blocks.0,blocks.1",
            "repa_loss_weight": 0.25,
            "repa_projection_dim": 4,
            "repa_stop_grad_target": "false",
        }
    )
    assert cfg.repa_enabled is True
    assert cfg.repa_target_modules == "blocks.0,blocks.1"
    assert cfg.repa_loss_weight == 0.25
    assert cfg.repa_projection_dim == 4
    assert cfg.repa_stop_grad_target is False

    soft = ConfigAdapter.from_frontend_dict(
        {
            "schema_id": "anima-lora",
            "soft_repa_enabled": "true",
            "softrepa_layers": "blocks.2",
            "softrepa_weight": 0.75,
        }
    )
    assert soft.softrepa_enabled is True
    assert soft.repa_enabled is True
    assert soft.repa_target_modules == "blocks.2"
    assert soft.softrepa_max_weight == 0.75
    print("PASS: test_repa_config_aliases")
    return True


def test_repa_training_loss_backward() -> bool:
    projector = REPAFeatureProjector(hidden_dim=8, projection_dim=4)
    loop = _Loop(projector)
    batch = {
        "captions": ["a", "b"],
        "latents": torch.randn(2, 1, 2, 2),
        "encoder_hidden_states": torch.randn(2, 3, 8),
        "repa_target_features": torch.randn(2, 3, 4),
    }
    loss = loop.train_step(batch, accumulation_steps=1)
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())
    assert loop.repa_capture is not None
    assert loop.repa_capture.features == {}
    print("PASS: test_repa_training_loss_backward")
    return True


def test_softrepa_training_window() -> bool:
    projector = REPAFeatureProjector(hidden_dim=8, projection_dim=4)
    loop = _Loop(projector, softrepa=True)
    loop.global_step = 1
    batch = {
        "captions": ["a", "b"],
        "latents": torch.randn(2, 1, 2, 2),
        "encoder_hidden_states": torch.randn(2, 3, 8),
        "repa_target_features": torch.randn(2, 3, 4),
        "repa_sigmas": torch.tensor([0.5, 0.5]),
    }
    loss = loop.train_step(batch, accumulation_steps=1)
    assert isinstance(loss, float)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())
    print("PASS: test_softrepa_training_window")
    return True


def main() -> int:
    tests = [test_repa_config_aliases, test_repa_training_loss_backward, test_softrepa_training_window]
    results = []
    for test_fn in tests:
        try:
            results.append((test_fn.__name__, test_fn()))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {exc}")
            results.append((test_fn.__name__, False))
    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print("REPA Training Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
