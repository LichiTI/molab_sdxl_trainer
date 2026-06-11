"""Smoke tests for compression companion merge behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_here = Path(__file__).resolve().parent
_backend = _here.parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))


def _import_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_lulynx_pkg = types.ModuleType("core.lulynx_trainer")
_lulynx_pkg.__path__ = [str(_here)]
sys.modules["core.lulynx_trainer"] = _lulynx_pkg
_import_from_file("core.lulynx_trainer.model_family", _here / "model_family.py")
_import_from_file("core.lulynx_trainer.tlora", _here / "tlora.py")
_companion = _import_from_file("core.lulynx_trainer.compression_companion", _here / "compression_companion.py")
_lora = _import_from_file("core.lulynx_trainer.lora_injector", _here / "lora_injector.py")
LoRAInjector = _lora.LoRAInjector


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_merge_into_base_and_reset() -> None:
    model = TinyModel()
    model.proj.weight.data.zero_()
    injector = LoRAInjector(rank=2, alpha=2.0, target_modules=["proj"])
    injector.inject(model, ["proj"], prefix="unet")
    layer = injector.injected_layers["unet.proj"]
    layer.lora.lora_down.weight.data.copy_(torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0]]))
    layer.lora.lora_up.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
    expected = layer.lora.lora_up.weight.detach() @ layer.lora.lora_down.weight.detach()

    result = _companion.apply_compression_companion(
        injector,
        path=str(_here / "missing.safetensors"),
    )
    assert result.merged_layers == 0
    assert result.warnings

    class NoLoadInjector:
        injected_layers = injector.injected_layers
        def load_lora(self, path, **kwargs):
            return None

    result = _companion.apply_compression_companion(NoLoadInjector(), path=__file__, scale=0.5)
    assert result.merged_layers == 1
    assert torch.allclose(layer.original.weight.float(), expected * 0.5, atol=1e-6)
    assert torch.count_nonzero(layer.lora.lora_up.weight).item() == 0
    assert result.reset_layers == 1
    print("  [PASS] merge_into_base_and_reset")


def test_normalizers() -> None:
    assert _companion.normalize_compression_companion_type("lycoris") == "lora"
    assert _companion.normalize_compression_companion_mode("bake") == "merge_into_base"
    print("  [PASS] normalizers")


def main() -> int:
    tests = [test_merge_into_base_and_reset, test_normalizers]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {exc}")
    print(f"\ncompression_companion smoke: {len(tests) - failed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

