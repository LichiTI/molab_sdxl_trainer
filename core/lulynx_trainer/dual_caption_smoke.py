# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for dual-caption (short/long) caption selection."""

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

_dc = _import_module("dual_caption", os.path.join(_HERE, "dual_caption.py"))
try_dual_caption = _dc.try_dual_caption


def test_both_keys_random():
    text = '{"short": "tags here", "long": "description here"}'
    seen = set()
    for _ in range(100):
        result = try_dual_caption(text)
        if result is not None:
            seen.add(result)
    assert "tags here" in seen, (
        "Expected 'short' value to appear in 100 trials"
    )
    assert "description here" in seen, (
        "Expected 'long' value to appear in 100 trials"
    )


def test_only_short():
    text = '{"short": "tags"}'
    result = try_dual_caption(text)
    assert result == "tags", (
        f"Expected 'tags' when only 'short' key present, got {result!r}"
    )


def test_only_long():
    text = '{"long": "description"}'
    result = try_dual_caption(text)
    assert result == "description", (
        f"Expected 'description' when only 'long' key present, got {result!r}"
    )


def test_non_json():
    text = "hello world"
    result = try_dual_caption(text)
    assert result is None, (
        f"Expected None for plain text input, got {result!r}"
    )


def test_missing_keys():
    text = '{"caption": "something"}'
    result = try_dual_caption(text)
    assert result is None, (
        f"Expected None when neither 'short' nor 'long' key is present, got {result!r}"
    )


if __name__ == "__main__":
    test_both_keys_random()
    test_only_short()
    test_only_long()
    test_non_json()
    test_missing_keys()
    print("dual_caption_smoke: all tests passed")
