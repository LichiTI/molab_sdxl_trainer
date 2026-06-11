# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for JPEG XL (.jxl) native support."""

from __future__ import annotations
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_dl = _import_module("dataset_loader", os.path.join(_HERE, "dataset_loader.py"))


def test_jxl_in_supported_extensions():
    assert ".jxl" in _dl.CaptionDataset.SUPPORTED_EXTENSIONS, (
        ".jxl must be listed in CaptionDataset.SUPPORTED_EXTENSIONS"
    )


def test_pillow_jxl_import_guard():
    try:
        import pillow_jxl  # noqa: F401
    except ImportError:
        pass  # acceptable — optional dependency not installed
    except Exception as exc:
        raise AssertionError(
            f"pillow_jxl import raised unexpected exception: {exc}"
        ) from exc


def test_other_extensions_preserved():
    required = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    missing = required - set(_dl.CaptionDataset.SUPPORTED_EXTENSIONS)
    assert not missing, (
        f"Standard extensions missing from SUPPORTED_EXTENSIONS: {missing}"
    )


if __name__ == "__main__":
    test_jxl_in_supported_extensions()
    test_pillow_jxl_import_guard()
    test_other_extensions_preserved()
    print("jxl_support_smoke: all tests passed")
