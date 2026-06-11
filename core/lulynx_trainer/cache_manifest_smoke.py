# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for cache manifest and fingerprint behavior."""

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
write_cache_manifest = manifest_mod.write_cache_manifest
validate_cache_manifest = manifest_mod.validate_cache_manifest
load_cache_manifest = manifest_mod.load_cache_manifest
build_cache_trust_report = manifest_mod.build_cache_trust_report


def _prepare_newbie_cache(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (12, 34, 56)).save(root / "sample.png")
    (root / "sample.txt").write_text("sample caption", encoding="utf-8")
    np.savez(
        root / "sample_newbie.npz",
        newbie_cache_schema_version=np.asarray(2, dtype=np.int32),
        latents=np.ones((16, 2, 2), dtype=np.float16),
        encoder_hidden_states=np.ones((4, 8), dtype=np.float16),
        pooled_prompt_embeds=np.ones((8,), dtype=np.float16),
        attention_mask=np.ones((4,), dtype=np.bool_),
    )


def _prepare_anima_cache(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (90, 12, 30)).save(root / "anima.png")
    (root / "anima.txt").write_text("anima caption", encoding="utf-8")
    np.savez(
        root / "anima_2x2_anima.npz",
        schema_version=np.asarray(2, dtype=np.int32),
        latents_2x2=np.ones((16, 2, 2), dtype=np.float16),
    )
    np.savez(
        root / "anima_anima_te.npz",
        schema_version=np.asarray(2, dtype=np.int32),
        has_loss_mask=np.asarray(0, dtype=np.int32),
        prompt_embeds=np.ones((4, 8), dtype=np.float16),
        attn_mask=np.ones((4,), dtype=np.bool_),
    )


def test_newbie_manifest_detects_changed_caption(root: Path) -> None:
    _prepare_newbie_cache(root)
    result = write_cache_manifest(root, family="newbie", builder="smoke")
    assert result.sample_count == 1, result
    assert result.cache_file_count == 1, result

    manifest = load_cache_manifest(root, family="newbie")
    assert manifest["family"] == "newbie"
    assert manifest["samples"][0]["cache_files"] == ["sample_newbie.npz"]

    report = validate_cache_manifest(root, family="newbie")
    assert report.ok, report

    (root / "sample.txt").write_text("changed caption", encoding="utf-8")
    changed = validate_cache_manifest(root, family="newbie")
    assert not changed.ok, changed
    assert "sample.txt" in changed.changed_files, changed
    print("PASS: Newbie manifest validates and detects changed caption")


def test_anima_manifest_records_paired_cache(root: Path) -> None:
    _prepare_anima_cache(root)
    result = write_cache_manifest(root, family="anima", builder="smoke")
    assert result.sample_count == 1, result
    assert result.cache_file_count == 2, result

    manifest = load_cache_manifest(root, family="anima")
    cache_files = set(manifest["samples"][0]["cache_files"])
    assert "anima_2x2_anima.npz" in cache_files
    assert "anima_anima_te.npz" in cache_files

    report = validate_cache_manifest(root, family="anima")
    assert report.ok, report
    print("PASS: Anima manifest records paired latent/text caches")


def test_strict_sha256_trust_report_and_trusted_warning(root: Path) -> None:
    _prepare_newbie_cache(root)
    write_cache_manifest(
        root,
        family="newbie",
        builder="smoke",
        include_sha256=True,
        config={"fingerprint_mode": "strict_sha256"},
    )

    manifest = load_cache_manifest(root, family="newbie")
    fingerprints = manifest["samples"][0]["fingerprints"]
    assert fingerprints, manifest
    assert all("sha256" in item for item in fingerprints.values()), fingerprints

    strict = build_cache_trust_report(root, family="newbie", mode="strict")
    assert strict.ok, strict
    assert strict.strict_sha256_required, strict
    assert not strict.hash_validation_skipped, strict
    assert not strict.warnings, strict

    trusted = build_cache_trust_report(root, family="newbie", mode="trusted")
    assert trusted.ok, trusted
    assert not trusted.strict_sha256_required, trusted
    assert trusted.hash_validation_skipped, trusted
    assert "hash_validation_skipped_trusted_cache" in trusted.warnings, trusted
    print("PASS: strict sha256 trust report and trusted warning")


def test_strict_trust_blocks_manifest_without_sha256(root: Path) -> None:
    _prepare_newbie_cache(root)
    write_cache_manifest(root, family="newbie", builder="smoke")

    strict = build_cache_trust_report(root, family="newbie", mode="strict")
    assert not strict.ok, strict
    assert strict.changed_files, strict
    trusted = build_cache_trust_report(root, family="newbie", mode="trusted")
    assert trusted.ok, trusted
    assert trusted.hash_validation_skipped, trusted
    print("PASS: strict trust blocks non-sha manifest while trusted warns")


def main() -> int:
    root = Path("H:/tmp/lulynx_cache_manifest_smoke")
    if root.exists():
        shutil.rmtree(root)
    newbie_root = root / "newbie"
    anima_root = root / "anima"
    newbie_root.mkdir(parents=True)
    anima_root.mkdir(parents=True)

    test_newbie_manifest_detects_changed_caption(newbie_root)
    test_anima_manifest_records_paired_cache(anima_root)
    test_strict_sha256_trust_report_and_trusted_warning(root / "strict")
    test_strict_trust_blocks_manifest_without_sha256(root / "trusted")
    print("PASS: cache manifest smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
