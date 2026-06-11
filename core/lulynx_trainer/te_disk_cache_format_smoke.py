# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for TextEncoderDiskCache disk format / dtype (#16)."""

from __future__ import annotations

import os
import sys
import importlib.util
import tempfile
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_ROOT = Path(_HERE)
_CORE_ROOT = _TRAINER_ROOT.parent
_BACKEND_ROOT = _CORE_ROOT.parent
for _path in (str(_BACKEND_ROOT), str(_CORE_ROOT), str(_TRAINER_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"core.lulynx_trainer.{name}",
        os.path.join(_HERE, f"{name}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"core.lulynx_trainer.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_dl = _load("dataset_loader")


def _outputs():
    return {
        "prompt_embeds": torch.randn(1, 77, 768),
        "pooled": torch.randn(1, 768),
    }


def test_npz_round_trip_default_dtype_float16():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = _dl.TextEncoderDiskCache(tmp, disk_format="npz", disk_dtype="float16")
        ref = _outputs()
        cache.put("hello", ref, model_id="sdxl")
        loaded = cache.get("hello", model_id="sdxl")
        assert loaded is not None
        for k in ref:
            # float16 cast is lossy, allow tolerance
            ref_fp16 = ref[k].to(torch.float16).float()
            assert torch.allclose(loaded[k].float(), ref_fp16, atol=1e-3, rtol=1e-3)
        print("PASS: npz round-trip in float16")


def test_safetensors_round_trip_bfloat16():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = _dl.TextEncoderDiskCache(tmp, disk_format="safetensors", disk_dtype="bfloat16")
        ref = _outputs()
        cache.put("hi", ref, model_id="sdxl")
        loaded = cache.get("hi", model_id="sdxl")
        assert loaded is not None
        for k in ref:
            assert loaded[k].dtype == torch.bfloat16
        print("PASS: safetensors round-trip in bfloat16")


def test_pt_round_trip_float32():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = _dl.TextEncoderDiskCache(tmp, disk_format="pt", disk_dtype="float32")
        ref = _outputs()
        cache.put("a caption", ref, model_id="sdxl")
        loaded = cache.get("a caption", model_id="sdxl")
        assert loaded is not None
        for k in ref:
            assert torch.allclose(loaded[k], ref[k], atol=1e-6, rtol=1e-6)
        print("PASS: pt round-trip in float32 is bit-exact")


def test_path_extension_matches_format():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for fmt, ext in [("npz", ".npz"), ("safetensors", ".safetensors"), ("pt", ".pt")]:
            cache = _dl.TextEncoderDiskCache(tmp, disk_format=fmt, disk_dtype="float16")
            cache.put("k", _outputs(), model_id="m")
            files = list(cache.cache_dir.glob(f"te_*{ext}"))
            assert files, f"no file with extension {ext} for format {fmt}"
        print("PASS: cache file extension matches disk_format")


def test_clear_removes_all_format_files():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for fmt in ("npz", "safetensors", "pt"):
            cache = _dl.TextEncoderDiskCache(tmp, disk_format=fmt, disk_dtype="float16")
            cache.put(f"k_{fmt}", _outputs(), model_id="m")
        any_cache = _dl.TextEncoderDiskCache(tmp, disk_format="npz", disk_dtype="float16")
        n = any_cache.clear()
        assert n == 3, f"expected 3 cleared, got {n}"
        print("PASS: clear() removes files of all supported formats")


def test_invalid_format_raises():
    try:
        _dl.TextEncoderDiskCache("/tmp/nope", disk_format="garbage")
    except ValueError:
        pass
    else:
        assert False, "expected ValueError"
    try:
        _dl.TextEncoderDiskCache("/tmp/nope", disk_format="npz", disk_dtype="garbage")
    except ValueError:
        pass
    else:
        assert False, "expected ValueError"
    print("PASS: invalid format / dtype raises ValueError")


if __name__ == "__main__":
    test_npz_round_trip_default_dtype_float16()
    test_safetensors_round_trip_bfloat16()
    test_pt_round_trip_float32()
    test_path_extension_matches_format()
    test_clear_removes_all_format_files()
    test_invalid_format_raises()
    print("\nAll TE disk cache format smoke tests passed!")
