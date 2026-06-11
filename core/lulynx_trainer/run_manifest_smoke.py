# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Standalone smoke tests for run manifest and resume preflight."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


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


manifest_mod = _load_local_module("run_manifest", "run_manifest.py")
write_run_manifest = manifest_mod.write_run_manifest
validate_resume_manifest = manifest_mod.validate_resume_manifest
manifest_path_for = manifest_mod.manifest_path_for


def _config(root: Path, *, output_name: str = "demo") -> SimpleNamespace:
    train_dir = root / "train"
    eval_dir = root / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "sample.txt").write_text("caption", encoding="utf-8")
    (eval_dir / "sample.txt").write_text("caption", encoding="utf-8")
    return SimpleNamespace(
        model_arch="sdxl",
        model_type="sdxl",
        training_type="lora",
        output_name=output_name,
        train_data_dir=str(train_dir),
        eval_data_dir=str(eval_dir),
        pretrained_model_name_or_path="model.safetensors",
        network_module="networks.lora",
        native_cache_mode="",
    )


def test_manifest_roundtrip(root: Path) -> None:
    cfg = _config(root)
    out = root / "output"
    state_path = out / "demo-step000010-state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(b"state")
    path = write_run_manifest(
        out,
        config=cfg,
        status="state_saved",
        epoch=1,
        global_step=10,
        total_steps=100,
        steps_per_epoch=20,
        state_path=str(state_path),
    )
    assert path == manifest_path_for(out)
    report = validate_resume_manifest(state_path, config=cfg)
    assert report.ok, report
    assert report.found, report
    assert report.previous_global_step == 10, report
    print("PASS: run manifest writes and validates matching resume")


def test_manifest_detects_mismatch_and_path_change(root: Path) -> None:
    cfg = _config(root)
    out = root / "output"
    state_path = out / "demo-step000010-state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(b"state")
    write_run_manifest(out, config=cfg, status="state_saved", global_step=10, state_path=str(state_path))
    (Path(cfg.train_data_dir) / "new_file.txt").write_text("changed", encoding="utf-8")

    changed = _config(root, output_name="other")
    report = validate_resume_manifest(state_path, config=changed)
    assert not report.ok, report
    assert any("output_name" in item for item in report.errors), report
    assert any("train_data_dir" in item for item in report.warnings), report
    print("PASS: run manifest detects config mismatch and dataset fingerprint change")


def test_missing_manifest_is_legacy_warning(root: Path) -> None:
    cfg = _config(root)
    state_path = root / "legacy-state.pt"
    state_path.write_bytes(b"state")
    report = validate_resume_manifest(state_path, config=cfg)
    assert report.ok, report
    assert not report.found, report
    assert report.warnings, report
    strict = validate_resume_manifest(state_path, config=cfg, strict=True)
    assert not strict.ok, strict
    assert strict.errors, strict
    print("PASS: missing run manifest warns by default and fails strict")


def main() -> int:
    root = Path("H:/tmp/lulynx_run_manifest_smoke")
    if root.exists():
        shutil.rmtree(root)
    test_manifest_roundtrip(root / "roundtrip")
    test_manifest_detects_mismatch_and_path_change(root / "mismatch")
    test_missing_manifest_is_legacy_warning(root / "missing")
    print("PASS: run manifest smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
