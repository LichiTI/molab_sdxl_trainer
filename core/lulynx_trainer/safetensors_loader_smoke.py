# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for safetensors_loader.py (#73)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.lulynx_trainer import safetensors_loader as _sl  # noqa: E402


def _write_test_safetensors(path: str) -> dict:
    from safetensors.torch import save_file

    state = {
        "weight": torch.randn(8, 16),
        "bias": torch.randn(8),
        "ln.weight": torch.ones(16),
    }
    save_file(state, path, metadata={"trainer": "lulynx", "schema": "1"})
    return state


def test_mmap_load_returns_correct_state():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = os.path.join(tmp, "model.safetensors")
        ref = _write_test_safetensors(p)
        loaded = _sl.load_safetensors(p, disable_mmap=False)
        assert set(loaded.keys()) == set(ref.keys())
        for k in ref:
            assert torch.allclose(loaded[k], ref[k])
        del loaded
        print("PASS: mmap load returns identical tensors")


def test_disable_mmap_load_returns_correct_state():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = os.path.join(tmp, "model.safetensors")
        ref = _write_test_safetensors(p)
        loaded = _sl.load_safetensors(p, disable_mmap=True)
        assert set(loaded.keys()) == set(ref.keys())
        for k in ref:
            assert torch.allclose(loaded[k], ref[k])
        print("PASS: disable_mmap load returns identical tensors")


def test_disable_mmap_does_not_hold_file_handle():
    """disable_mmap should release the file handle after read; mmap may keep it."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = os.path.join(tmp, "model.safetensors")
        _write_test_safetensors(p)
        _ = _sl.load_safetensors(p, disable_mmap=True)
        try:
            os.remove(p)
        except PermissionError:
            assert False, "disable_mmap should release the file handle"
        assert not os.path.exists(p)
        print("PASS: disable_mmap releases file handle")


def test_open_safetensors_shim_exposes_api():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = os.path.join(tmp, "model.safetensors")
        ref = _write_test_safetensors(p)
        with _sl.open_safetensors(p, disable_mmap=True) as handle:
            keys = list(handle.keys())
            assert set(keys) == set(ref.keys())
            t = handle.get_tensor("weight")
            assert torch.allclose(t, ref["weight"])
            md = handle.metadata()
            assert md.get("trainer") == "lulynx"
        print("PASS: open_safetensors shim mirrors safe_open API")


def test_open_safetensors_mmap_path_works():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = os.path.join(tmp, "model.safetensors")
        ref = _write_test_safetensors(p)
        with _sl.open_safetensors(p, disable_mmap=False) as handle:
            keys = list(handle.keys())
            assert set(keys) == set(ref.keys())
            t = handle.get_tensor("bias")
            assert torch.allclose(t, ref["bias"])
        print("PASS: open_safetensors mmap path works")


def test_resolve_disable_mmap_reads_config_attr():
    from types import SimpleNamespace

    cfg_on = SimpleNamespace(disable_mmap_load_safetensors=True)
    cfg_off = SimpleNamespace(disable_mmap_load_safetensors=False)
    cfg_missing = SimpleNamespace()

    assert _sl.resolve_disable_mmap(cfg_on) is True
    assert _sl.resolve_disable_mmap(cfg_off) is False
    assert _sl.resolve_disable_mmap(cfg_missing) is False
    assert _sl.resolve_disable_mmap(cfg_missing, default=True) is True
    assert _sl.resolve_disable_mmap(None) is False
    print("PASS: resolve_disable_mmap reads config attribute")


if __name__ == "__main__":
    test_mmap_load_returns_correct_state()
    test_disable_mmap_load_returns_correct_state()
    test_disable_mmap_does_not_hold_file_handle()
    test_open_safetensors_shim_exposes_api()
    test_open_safetensors_mmap_path_works()
    test_resolve_disable_mmap_reads_config_attr()
    print("\nAll safetensors_loader smoke tests passed!")
