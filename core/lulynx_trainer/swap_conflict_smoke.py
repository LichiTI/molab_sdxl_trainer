"""Smoke test for memory swap conflict handling."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

TRAINER_ROOT = Path(__file__).resolve().parent
CORE_ROOT = TRAINER_ROOT.parent
BACKEND_ROOT = CORE_ROOT.parent
for _path in (BACKEND_ROOT, CORE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_training_loop_module():
    module_name = "core.lulynx_trainer.training_loop"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    package_name = "core.lulynx_trainer"
    if package_name not in sys.modules:
        package = type(sys)(package_name)
        package.__path__ = [str(TRAINER_ROOT)]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, TRAINER_ROOT / "training_loop.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load training_loop.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.mid_block = TinyBlock()
        self.up_blocks = nn.ModuleList([TinyBlock(), TinyBlock()])


class TinyVAE(nn.Module):
    config = SimpleNamespace(scaling_factor=1.0)


class TinyScheduler:
    config = SimpleNamespace(num_train_timesteps=1000)
    alphas_cumprod = torch.linspace(1.0, 0.0, 1000)


def _make_loop(**kwargs):
    TrainingLoop = _load_training_loop_module().TrainingLoop
    unet = TinyUNet()
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01)
    params = {
        "blocks_to_swap": 0,
        "swap_granularity": "block",
        "swap_count": 2,
    }
    params.update(kwargs)
    return TrainingLoop(
        unet=unet,
        text_encoder_1=nn.Identity(),
        text_encoder_2=None,
        vae=TinyVAE(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=TinyScheduler(),
        lora_injector=None,
        optimizer=optimizer,
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        **params,
    )


def test_swap_torch_compile_conflict():
    loop = _make_loop(torch_compile=True)
    assert loop.memory_optimization_state.get("enabled") is False
    print("PASS: test_swap_torch_compile_conflict")
    return True


def test_swap_vram_swap_conflict():
    loop = _make_loop(vram_swap_to_ram=True)
    assert loop.memory_optimization_state.get("enabled") is False
    print("PASS: test_swap_vram_swap_conflict")
    return True


def test_swap_safe_fallback_conflict():
    loop = _make_loop(safe_fallback=True)
    assert loop.memory_optimization_state.get("enabled") is False
    print("PASS: test_swap_safe_fallback_conflict")
    return True


def test_layer_gradient_checkpointing_conflict():
    try:
        _make_loop(swap_granularity="layer", gradient_checkpointing=True)
    except ValueError as exc:
        assert "gradient_checkpointing" in str(exc)
        print("PASS: test_layer_gradient_checkpointing_conflict")
        return True
    raise AssertionError("Expected ValueError for layer swap + gradient_checkpointing")


def main():
    tests = [
        test_swap_torch_compile_conflict,
        test_swap_vram_swap_conflict,
        test_swap_safe_fallback_conflict,
        test_layer_gradient_checkpointing_conflict,
    ]
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
    print("Memory Swap Conflict Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
