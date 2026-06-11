"""Smoke-test the real Anima cache dataset contract.

This intentionally imports the dataset module by file path so it can validate
cache discovery/collation without importing the full trainer/diffusers stack.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_dataset_module() -> ModuleType:
    module_path = Path(__file__).with_name("anima_cached_dataset.py")
    spec = importlib.util.spec_from_file_location("anima_cached_dataset_smoke_target", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dataset module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "sucai" / "6_lulu"
    if not data_dir.exists():
        raise FileNotFoundError(f"Anima cached data not found: {data_dir}")

    module = _load_dataset_module()
    dataset = module.AnimaCachedDataset(data_dir)
    dataloader = module.create_anima_cached_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    if batch["latents"].dim() != 4:
        raise AssertionError(f"Expected BCHW latents, got {tuple(batch['latents'].shape)}")
    if batch["latents"].shape[0] != 2:
        raise AssertionError(f"Expected batch size 2, got {batch['latents'].shape[0]}")
    if batch["padding_mask"] is not None and batch["padding_mask"].shape[-2:] != batch["latents"].shape[-2:]:
        raise AssertionError("Padding mask spatial shape must match padded latents")
    if batch["encoder_hidden_states"].dim() != 3:
        raise AssertionError("Expected padded text conditioning shaped [batch, tokens, channels]")

    print(
        "Anima cached dataset smoke passed: "
        f"samples={len(dataset)}, "
        f"latents={tuple(batch['latents'].shape)}, "
        f"text={tuple(batch['encoder_hidden_states'].shape)}, "
        f"padding_mask={None if batch['padding_mask'] is None else tuple(batch['padding_mask'].shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
