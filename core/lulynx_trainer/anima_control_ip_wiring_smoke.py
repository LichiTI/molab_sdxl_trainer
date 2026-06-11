# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Anima EasyControl and IP-Adapter wiring primitives."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

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
easy_mod = _load_module("core.lulynx_trainer.easy_control_dit", TRAINER_ROOT / "easy_control_dit.py")
ip_mod = _load_module("core.lulynx_trainer.anima_ip_adapter", TRAINER_ROOT / "anima_ip_adapter.py")

EasyControl = easy_mod.EasyControl
EasyControlConfig = easy_mod.EasyControlConfig
AnimaIPAdapter = ip_mod.AnimaIPAdapter
IPAdapterConfig = ip_mod.IPAdapterConfig


def test_easy_control_residual_shape_and_grad() -> bool:
    ctrl = EasyControl(EasyControlConfig(in_channels=3, latent_channels=4, hidden_dim=8, downsample_factor=4, scale=0.5, init_zero_out=False))
    control = torch.randn(2, 3, 32, 32)
    residual = ctrl(control, target_size=(8, 8))
    assert residual.shape == (2, 4, 8, 8)
    loss = residual.square().mean()
    loss.backward()
    params = ctrl.get_trainable_params()
    assert params
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    print("PASS: test_easy_control_residual_shape_and_grad")
    return True


def test_ip_adapter_cached_feature_merge_and_grad() -> bool:
    def encode_fn(_image: torch.Tensor) -> torch.Tensor:
        raise AssertionError("cached feature path should not call encode_fn")

    adapter = AnimaIPAdapter(
        encode_fn,
        IPAdapterConfig(encoder_dim=6, cond_dim=8, num_image_tokens=3, num_layers=1, scale=0.75, cond_mode="concat"),
    )
    features = torch.randn(2, 5, 6)
    image_tokens = adapter.projector(features) * adapter.config.scale
    text_tokens = torch.randn(2, 4, 8)
    text_mask = torch.ones(2, 4, dtype=torch.long)
    merged, mask = adapter.merge_with_text_cond(image_tokens, text_tokens, text_mask)
    assert merged.shape == (2, 7, 8)
    assert mask is not None and mask.shape == (2, 7)
    loss = merged.square().mean()
    loss.backward()
    params = adapter.get_trainable_params()
    assert params
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    print("PASS: test_ip_adapter_cached_feature_merge_and_grad")
    return True


def main() -> int:
    tests = [test_easy_control_residual_shape_and_grad, test_ip_adapter_cached_feature_merge_and_grad]
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
    print("Anima Control/IP Wiring Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
