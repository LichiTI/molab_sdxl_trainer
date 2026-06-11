# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Newbie cache manifest parity validation."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


TRAINER_ROOT = Path(__file__).resolve().parent


def _load_local_module(module_name: str, filename: str):
    full_name = f"_lulynx_{module_name}_smoke_target"
    module = sys.modules.get(full_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(full_name, TRAINER_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


manifest_mod = _load_local_module("cache_manifest", "cache_manifest.py")
sys.modules.setdefault("cache_manifest", manifest_mod)
dataset_mod = _load_local_module("newbie_cached_dataset", "newbie_cached_dataset.py")
sys.modules.setdefault("newbie_cached_dataset", dataset_mod)
parity_mod = _load_local_module("newbie_cache_parity", "newbie_cache_parity.py")

write_cache_manifest = manifest_mod.write_cache_manifest
validate_newbie_cache_parity = parity_mod.validate_newbie_cache_parity


def _prepare_cache(root: Path, *, pooled: bool = True, prompt: str = "Gemma3: {caption}") -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (12, 34, 56)).save(root / "sample.png")
    (root / "sample.txt").write_text("sample caption", encoding="utf-8")
    payload = {
        "newbie_cache_schema_version": np.asarray(2, dtype=np.int32),
        "latents": np.ones((16, 2, 2), dtype=np.float16),
        "encoder_hidden_states": np.ones((4, 8), dtype=np.float16),
        "attention_mask": np.ones((4,), dtype=np.bool_),
        "loss_mask": np.ones((2, 2), dtype=np.float16),
    }
    if pooled:
        payload["pooled_prompt_embeds"] = np.ones((8,), dtype=np.float16)
    np.savez(root / "sample_newbie.npz", **payload)
    write_cache_manifest(
        root,
        family="newbie",
        builder="parity_smoke",
        config={
            "schema_version": 2,
            "gemma3_prompt": prompt,
            "gemma_max_token_length": 512,
            "clip_max_token_length": 2048,
        },
    )


def test_valid_newbie_cache_parity(root: Path) -> None:
    _prepare_cache(root)
    report = validate_newbie_cache_parity(root, expected_gemma3_prompt="Gemma3: {caption}")
    assert report.ok, report
    assert report.sample_count == 1, report
    assert report.samples[0].latent_shape == (16, 2, 2), report
    assert report.samples[0].pooled_shape == (8,), report
    assert report.samples[0].loss_mask_shape == (2, 2), report
    print("PASS: Newbie cache parity validates schema/shape/prompt contract")


def test_newbie_cache_parity_detects_bad_cache(root: Path) -> None:
    _prepare_cache(root, pooled=False)
    report = validate_newbie_cache_parity(root, expected_gemma3_prompt="Gemma3: {caption}")
    assert not report.ok, report
    assert any("pooled" in error.lower() for error in report.errors), report
    print("PASS: Newbie cache parity detects missing pooled features")


def test_newbie_cache_parity_detects_prompt_mismatch(root: Path) -> None:
    _prepare_cache(root, prompt="Other: {caption}")
    report = validate_newbie_cache_parity(root, expected_gemma3_prompt="Gemma3: {caption}")
    assert not report.ok, report
    assert any("prompt" in error.lower() for error in report.errors), report
    print("PASS: Newbie cache parity detects Gemma3 prompt mismatch")


def main() -> int:
    root = Path("H:/tmp/lulynx_newbie_cache_parity_smoke")
    if root.exists():
        shutil.rmtree(root)
    test_valid_newbie_cache_parity(root / "valid")
    test_newbie_cache_parity_detects_bad_cache(root / "bad")
    test_newbie_cache_parity_detects_prompt_mismatch(root / "prompt")
    print("PASS: Newbie cache parity smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
