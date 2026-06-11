# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for mask_pipeline.py (Phase 8.12 / #99)."""

from __future__ import annotations

import os
import sys
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.mask_pipeline",
    os.path.join(_HERE, "mask_pipeline.py"),
)
_mp = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.mask_pipeline"] = _mp
_spec.loader.exec_module(_mp)


def _make_rgba_image(path: Path, *, alpha=128):
    arr = np.full((32, 32, 4), 255, dtype="uint8")
    arr[..., 3] = alpha
    Image.fromarray(arr, mode="RGBA").save(str(path))


def _make_threshold_image(path: Path):
    """White background with a coloured square in the middle."""
    arr = np.full((64, 64, 3), 255, dtype="uint8")
    arr[16:48, 16:48] = (200, 50, 50)  # red square — high saturation
    Image.fromarray(arr, mode="RGB").save(str(path))


def test_alpha_channel_strategy():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        _make_rgba_image(path, alpha=200)

        cfg = _mp.MaskPipelineConfig(data_dir=tmp, strategy="alpha_channel")
        result = _mp.generate_masks_for_directory(cfg)

        assert result.written == 1
        assert result.skipped == 0

        mask_path = path.with_name("img_mask.png")
        assert mask_path.exists()
        mask = Image.open(mask_path)
        assert mask.mode == "L"
        # Alpha was 200, mask should reflect that
        arr = np.asarray(mask)
        assert int(arr.mean()) > 150
    print("PASS: alpha_channel strategy extracts alpha")


def test_threshold_strategy_finds_foreground():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        _make_threshold_image(path)

        cfg = _mp.MaskPipelineConfig(data_dir=tmp, strategy="threshold")
        result = _mp.generate_masks_for_directory(cfg)
        assert result.written == 1

        mask = Image.open(path.with_name("img_mask.png"))
        arr = np.asarray(mask)
        center = arr[16:48, 16:48]
        corner = arr[0:8, 0:8]
        # Centre (red square) should be foreground (255), corner (white) background (0)
        assert center.mean() > 200
        assert corner.mean() < 50
    print("PASS: threshold strategy isolates coloured foreground")


def test_skip_existing_mask_without_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        _make_rgba_image(path)
        # Pre-create the mask
        mask_path = path.with_name("img_mask.png")
        Image.fromarray(np.zeros((4, 4), dtype="uint8"), mode="L").save(str(mask_path))

        cfg = _mp.MaskPipelineConfig(data_dir=tmp, strategy="alpha_channel", overwrite=False)
        result = _mp.generate_masks_for_directory(cfg)
        assert result.skipped == 1
        assert result.written == 0
    print("PASS: skip_existing_mask honors overwrite=False")


def test_overwrite_flag_regenerates_mask():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        _make_rgba_image(path, alpha=180)
        mask_path = path.with_name("img_mask.png")
        Image.fromarray(np.zeros((4, 4), dtype="uint8"), mode="L").save(str(mask_path))

        cfg = _mp.MaskPipelineConfig(data_dir=tmp, strategy="alpha_channel", overwrite=True)
        result = _mp.generate_masks_for_directory(cfg)
        assert result.written == 1
        # Mask now reflects alpha=180
        arr = np.asarray(Image.open(mask_path))
        assert arr.mean() > 100
    print("PASS: overwrite=True regenerates existing masks")


def test_invert_flips_foreground_and_background():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        _make_threshold_image(path)
        cfg = _mp.MaskPipelineConfig(data_dir=tmp, strategy="threshold", invert=True)
        _mp.generate_masks_for_directory(cfg)
        mask_arr = np.asarray(Image.open(path.with_name("img_mask.png")))
        # After invert: corner (was white background) should be 255
        assert mask_arr[0:8, 0:8].mean() > 200
        # Centre (was foreground) should be 0
        assert mask_arr[16:48, 16:48].mean() < 50
    print("PASS: invert flips foreground/background")


def test_skips_existing_mask_files_in_iteration():
    with tempfile.TemporaryDirectory() as tmp:
        # Create an image and a stale "_mask" file that should NOT be processed
        path = Path(tmp) / "img.png"
        _make_threshold_image(path)
        stale = Path(tmp) / "stale_mask.png"
        Image.fromarray(np.zeros((4, 4), dtype="uint8"), mode="L").save(str(stale))

        cfg = _mp.MaskPipelineConfig(data_dir=tmp, strategy="threshold")
        result = _mp.generate_masks_for_directory(cfg)
        # Only the non-mask image should be processed
        assert result.written == 1
        # stale_mask.png should NOT have spawned a stale_mask_mask.png
        assert not (Path(tmp) / "stale_mask_mask.png").exists()
    print("PASS: pipeline skips files ending in '_mask'")


def test_unknown_strategy_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        _make_rgba_image(path)
        cfg = _mp.MaskPipelineConfig(data_dir=tmp, strategy="garbage")
        result = _mp.generate_masks_for_directory(cfg)
        # The error is recorded per-image rather than raised globally
        assert result.written == 0
        assert any("Unknown mask strategy" in e for e in result.errors)
    print("PASS: unknown strategy surfaces as per-image error")


def test_missing_data_dir_raises():
    cfg = _mp.MaskPipelineConfig(data_dir="/does/not/exist/xyz", strategy="alpha_channel")
    try:
        _mp.generate_masks_for_directory(cfg)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
    print("PASS: missing data_dir raises FileNotFoundError")


if __name__ == "__main__":
    test_alpha_channel_strategy()
    test_threshold_strategy_finds_foreground()
    test_skip_existing_mask_without_overwrite()
    test_overwrite_flag_regenerates_mask()
    test_invert_flips_foreground_and_background()
    test_skips_existing_mask_files_in_iteration()
    test_unknown_strategy_raises()
    test_missing_data_dir_raises()
    print("\nAll mask_pipeline smoke tests passed!")
