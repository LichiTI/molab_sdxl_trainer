# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for the flat LXFS sidecar prototype."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_flat_sidecar import (  # type: ignore[no-redef]
        decode_flat_lossless_cache_sidecar_arrays,
        encode_flat_lossless_cache_sidecar,
        write_flat_lossless_cache_sidecar_file,
    )
    from lulynx_trainer.lossless_cache_sidecar import LosslessCacheEntry  # type: ignore[no-redef]
else:
    from .lossless_cache_flat_sidecar import (
        decode_flat_lossless_cache_sidecar_arrays,
        encode_flat_lossless_cache_sidecar,
        write_flat_lossless_cache_sidecar_file,
    )
    from .lossless_cache_sidecar import LosslessCacheEntry


def main() -> int:
    latent = (np.arange(32, dtype=np.float16).reshape(2, 4, 4) / np.float16(7.0)).copy()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:5] = 1
    entries = [
        LosslessCacheEntry(
            "latent",
            latent.tobytes(order="C"),
            element_size=latent.dtype.itemsize,
            metadata={"dtype": str(latent.dtype), "shape": list(latent.shape)},
        ),
        LosslessCacheEntry(
            "mask",
            mask.tobytes(order="C"),
            element_size=mask.dtype.itemsize,
            metadata={"dtype": str(mask.dtype), "shape": list(mask.shape)},
        ),
    ]
    blob = encode_flat_lossless_cache_sidecar(entries, codecs="fast-cache")
    arrays = decode_flat_lossless_cache_sidecar_arrays(blob)
    assert np.array_equal(arrays["latent"], latent)
    assert np.array_equal(arrays["mask"], mask)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "sample_anima.npz"
        sidecar = Path(tmp) / "sample_anima.lxfs"
        np.savez(cache, latent=latent, mask=mask)
        report = write_flat_lossless_cache_sidecar_file(cache, sidecar_path=sidecar)
        assert report["ok"]
        arrays = decode_flat_lossless_cache_sidecar_arrays(sidecar.read_bytes(), copy_arrays=False)
        assert np.array_equal(arrays["latent"], latent)
        assert np.array_equal(arrays["mask"], mask)
    print("lossless_cache_flat_sidecar_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
