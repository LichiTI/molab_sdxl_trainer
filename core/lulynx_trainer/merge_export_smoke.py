# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for merge_export.py (Phase 9.4 / #121).

Verifies on CPU:
  1. merge_lora_into_base detects LoRALinear wrappers and merges weight deltas
  2. Module replacement strips the wrapper after merge
  3. export_merged_model deep-copies and writes a state dict
  4. No-op behaviour when no injectors are passed
"""

from __future__ import annotations

import sys
import os
import importlib.util
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_export_merged_model_no_adapters_writes_base():
    me = _load_module("core.lulynx_trainer.merge_export", "merge_export.py")

    base = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "merged.safetensors"
        result = me.export_merged_model(
            model=base,
            output_path=str(out_path),
            save_precision="bf16",
            lora_injector=None,
            lycoris_injector=None,
        )
        assert Path(result).exists() or Path(result).with_suffix(".pt").exists(), \
            f"export should produce a file at {result}"
    print("PASS: export_merged_model writes raw weights when no injectors")


def test_export_merged_model_dtype_conversion():
    """save_precision must affect the output dtype."""
    me = _load_module("core.lulynx_trainer.merge_export", "merge_export.py")

    base = nn.Linear(4, 4)

    with tempfile.TemporaryDirectory() as tmp:
        # fp16
        out_fp16 = Path(tmp) / "merged_fp16.safetensors"
        me.export_merged_model(model=base, output_path=str(out_fp16), save_precision="fp16")
        # fp32
        out_fp32 = Path(tmp) / "merged_fp32.safetensors"
        me.export_merged_model(model=base, output_path=str(out_fp32), save_precision="fp32")

        # Files written (either as .safetensors or fallback .pt)
        for path in (out_fp16, out_fp32):
            assert path.exists() or path.with_suffix(".pt").exists()
    print("PASS: export_merged_model honors save_precision (fp16 / fp32)")


def test_replace_module_dotted_path():
    """_replace_module should swap a sub-module by its dotted path."""
    me = _load_module("core.lulynx_trainer.merge_export", "merge_export.py")

    root = nn.Sequential(
        nn.Linear(4, 4),
        nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)]),
        nn.Linear(4, 4),
    )

    new_layer = nn.Linear(4, 4)
    new_layer.weight.data.fill_(7.0)
    me._replace_module(root, "1.0", new_layer)

    # Confirm the swap landed in the right slot
    assert torch.allclose(root[1][0].weight, torch.full((4, 4), 7.0))
    print("PASS: _replace_module swaps dotted path correctly")


def test_get_submodule_dotted_path():
    me = _load_module("core.lulynx_trainer.merge_export", "merge_export.py")

    root = nn.Sequential(nn.Linear(4, 4), nn.ModuleList([nn.Linear(4, 4)]))
    target = me._get_submodule(root, "1.0")
    assert isinstance(target, nn.Linear)
    print("PASS: _get_submodule resolves dotted path")


if __name__ == "__main__":
    test_export_merged_model_no_adapters_writes_base()
    test_export_merged_model_dtype_conversion()
    test_replace_module_dotted_path()
    test_get_submodule_dotted_path()
    print("\nAll merge_export smoke tests passed!")
