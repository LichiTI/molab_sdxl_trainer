# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for LatentDiskCache disk format / dtype."""

from __future__ import annotations

import importlib.util
import os
import sys
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


def _latents() -> dict[str, torch.Tensor]:
    return {"latents": torch.randn(4, 8, 8)}


def test_npz_round_trip_default_dtype_float16() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = _dl.LatentDiskCache(tmp, disk_format="npz", disk_dtype="float16")
        ref = _latents()
        cache.put("image.png", (1024, 1024), ref)
        loaded = cache.get("image.png", (1024, 1024))
        assert loaded is not None
        assert loaded["latents"].dtype == torch.float16
        assert torch.allclose(loaded["latents"].float(), ref["latents"].to(torch.float16).float(), atol=1e-3, rtol=1e-3)


def test_safetensors_round_trip_bfloat16() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = _dl.LatentDiskCache(tmp, disk_format="safetensors", disk_dtype="bfloat16")
        cache.put("image.png", (768, 768), _latents())
        loaded = cache.get("image.png", (768, 768))
        assert loaded is not None
        assert loaded["latents"].dtype == torch.bfloat16


def test_pt_round_trip_float32() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = _dl.LatentDiskCache(tmp, disk_format="pt", disk_dtype="float32")
        ref = _latents()
        cache.put("image.png", (512, 512), ref)
        loaded = cache.get("image.png", (512, 512))
        assert loaded is not None
        assert loaded["latents"].dtype == torch.float32
        assert torch.allclose(loaded["latents"], ref["latents"], atol=1e-6, rtol=1e-6)


def test_resolution_is_part_of_cache_key() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = _dl.LatentDiskCache(tmp, disk_format="pt", disk_dtype="float32")
        cache.put("image.png", (512, 512), {"latents": torch.ones(1, 2, 2)})
        cache.put("image.png", (768, 768), {"latents": torch.full((1, 2, 2), 2.0)})
        small = cache.get("image.png", (512, 512))
        large = cache.get("image.png", (768, 768))
        assert small is not None and large is not None
        assert torch.allclose(small["latents"], torch.ones(1, 2, 2))
        assert torch.allclose(large["latents"], torch.full((1, 2, 2), 2.0))


def test_path_extension_and_clear_cover_all_formats() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for fmt, ext in [("npz", ".npz"), ("safetensors", ".safetensors"), ("pt", ".pt")]:
            cache = _dl.LatentDiskCache(tmp, disk_format=fmt, disk_dtype="float16")
            cache.put(f"image-{fmt}.png", (512, 512), _latents())
            assert list(cache.cache_dir.glob(f"lat_*{ext}"))
        any_cache = _dl.LatentDiskCache(tmp, disk_format="npz", disk_dtype="float16")
        assert any_cache.clear() == 3


def test_invalid_format_or_dtype_raises() -> None:
    try:
        _dl.LatentDiskCache("/tmp/nope", disk_format="garbage")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid disk_format")

    try:
        _dl.LatentDiskCache("/tmp/nope", disk_format="npz", disk_dtype="garbage")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid disk_dtype")


def main() -> int:
    test_npz_round_trip_default_dtype_float16()
    test_safetensors_round_trip_bfloat16()
    test_pt_round_trip_float32()
    test_resolution_is_part_of_cache_key()
    test_path_extension_and_clear_cover_all_formats()
    test_invalid_format_or_dtype_raises()
    print("latent_disk_cache_format_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
