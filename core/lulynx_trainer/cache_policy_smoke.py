# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Standalone smoke tests for cache policy behavior."""

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
newbie_contract_mod = _load_local_module("newbie_cache_contract", "newbie_cache_contract.py")
sys.modules.setdefault("newbie_cache_contract", newbie_contract_mod)
policy_mod = _load_local_module("cache_policy", "cache_policy.py")

write_cache_manifest = manifest_mod.write_cache_manifest
resolve_cache_policy = policy_mod.resolve_cache_policy


def _prepare_newbie_cache(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (12, 34, 56)).save(root / "sample.png")
    (root / "sample.txt").write_text("sample caption", encoding="utf-8")
    np.savez(
        root / "sample_newbie.npz",
        newbie_cache_schema_version=np.asarray(2, dtype=np.int32),
        latents=np.ones((16, 2, 2), dtype=np.float16),
        encoder_hidden_states=np.ones((4, 8), dtype=np.float16),
        pooled_prompt_embeds=np.ones((1024,), dtype=np.float16),
        attention_mask=np.ones((4,), dtype=np.bool_),
    )


def test_legacy_cache_without_manifest_is_usable(root: Path) -> None:
    _prepare_newbie_cache(root)
    report = resolve_cache_policy(root, family="newbie", mode="cache_first")
    assert report.can_use_cache, report
    assert report.cache_sample_count == 1, report
    assert report.warnings, report
    print("PASS: legacy cache without manifest remains usable with warning")


def test_manifest_ok_is_usable(root: Path) -> None:
    _prepare_newbie_cache(root)
    write_cache_manifest(root, family="newbie", builder="smoke")
    report = resolve_cache_policy(root, family="newbie", mode="cache_first")
    assert report.can_use_cache, report
    assert report.manifest_ok, report
    assert not report.errors, report
    print("PASS: valid manifest cache is usable")


def test_changed_manifest_warns_unless_strict(root: Path) -> None:
    _prepare_newbie_cache(root)
    write_cache_manifest(root, family="newbie", builder="smoke")
    (root / "sample.txt").write_text("changed caption", encoding="utf-8")

    report = resolve_cache_policy(root, family="newbie", mode="cache_first")
    assert report.can_use_cache, report
    assert report.warnings, report

    strict = resolve_cache_policy(root, family="newbie", mode="cache_first", strict_manifest=True)
    assert not strict.can_use_cache, strict
    assert strict.errors, strict

    trusted = resolve_cache_policy(root, family="newbie", mode="cache_first", trust_cache=True, strict_manifest=True)
    assert trusted.can_use_cache, trusted
    assert trusted.warnings, trusted
    print("PASS: changed manifest warns by default, fails strict, and allows trust_cache")


def test_rebuild_and_force_cache_only_flags(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rebuild = resolve_cache_policy(root, family="newbie", mode="rebuild_cache")
    assert rebuild.should_rebuild, rebuild

    force = resolve_cache_policy(root, family="newbie", mode="force_cache_only", force_cache_only=True)
    assert not force.can_use_cache, force
    assert force.errors, force
    print("PASS: rebuild and force_cache_only flags resolve")


def main() -> int:
    root = Path("H:/tmp/lulynx_cache_policy_smoke")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    test_legacy_cache_without_manifest_is_usable(root / "legacy")
    test_manifest_ok_is_usable(root / "valid")
    test_changed_manifest_warns_unless_strict(root / "changed")
    test_rebuild_and_force_cache_only_flags(root / "flags")
    print("PASS: cache policy smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
