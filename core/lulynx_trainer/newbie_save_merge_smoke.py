# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for Newbie adapter save/load and merge-export boundary."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)
    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module  # type: ignore[attr-defined]
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


_install_xformers_stub()

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.merge_export import export_merged_model


class _TinyNewbieCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.qkv = nn.Linear(8, 8, bias=False)
        self.feed_forward = nn.Module()
        self.feed_forward.w1 = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward.w1(torch.tanh(self.attention.qkv(x)))


class _TinyNewbieRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = _TinyNewbieCore()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


def _inject(model: nn.Module) -> LoRAInjector:
    injector = LoRAInjector(rank=2, alpha=2, model_arch="newbie")
    injected = injector.inject(model.unet, ["attention.qkv", "feed_forward.w1"], prefix="unet")
    assert len(injected) == 2
    return injector


def test_newbie_adapter_save_load_roundtrip(root: Path) -> None:
    model = _TinyNewbieRoot()
    injector = _inject(model)
    x = torch.randn(2, 3, 8)
    model(x).sum().backward()
    for param in injector.get_trainable_params():
        if param.grad is not None:
            param.data.add_(param.grad)

    path = root / "newbie_adapter.safetensors"
    injector.save_lora(str(path), metadata={"ss_base_model_version": "newbie"})
    assert path.is_file(), path
    expected = {key: value.detach().clone() for key, value in injector.get_lora_state_dict().items()}

    reloaded_model = _TinyNewbieRoot()
    reloaded = _inject(reloaded_model)
    reloaded.load_lora(str(path))
    actual = reloaded.get_lora_state_dict()
    for key, expected_value in expected.items():
        assert key in actual, key
        assert torch.allclose(actual[key], expected_value), key
    print("PASS: Newbie adapter save/load roundtrip")


def test_newbie_merge_export_writes_full_state(root: Path) -> None:
    model = _TinyNewbieRoot()
    injector = _inject(model)
    with torch.no_grad():
        for param in injector.get_trainable_params():
            param.add_(0.01)
    out = root / "newbie_merged.safetensors"
    result = export_merged_model(
        model=model,
        output_path=str(out),
        save_precision="fp32",
        lora_injector=injector,
    )
    result_path = Path(result)
    assert result_path.is_file() or result_path.with_suffix(".pt").is_file(), result
    print("PASS: Newbie merge-export boundary writes full state")


def main() -> int:
    root = Path("H:/tmp/lulynx_newbie_save_merge_smoke")
    root.mkdir(parents=True, exist_ok=True)
    test_newbie_adapter_save_load_roundtrip(root)
    test_newbie_merge_export_writes_full_state(root)
    print("PASS: Newbie save/load/merge smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
