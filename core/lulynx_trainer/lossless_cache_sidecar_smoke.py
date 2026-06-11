"""Smoke checks for the LXCS cache sidecar prototype."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_sidecar import (  # type: ignore[no-redef]
        LosslessCacheEntry,
        analyze_lossless_cache_sidecar,
        decode_lossless_cache_sidecar,
        decode_lossless_cache_sidecar_arrays,
        encode_lossless_cache_sidecar,
        load_numpy_cache_entries,
    )
else:
    from .lossless_cache_sidecar import (
        LosslessCacheEntry,
        analyze_lossless_cache_sidecar,
        decode_lossless_cache_sidecar,
        decode_lossless_cache_sidecar_arrays,
        encode_lossless_cache_sidecar,
        load_numpy_cache_entries,
    )


def test_entry_roundtrip() -> None:
    entries = (
        LosslessCacheEntry("zeros", b"\0" * 8192, element_size=2),
        LosslessCacheEntry("runs", (b"latent-cache-" * 1024) + bytes(range(128))),
    )
    blob = encode_lossless_cache_sidecar(entries, chunk_size=4096)
    decoded = decode_lossless_cache_sidecar(blob)
    assert decoded["zeros"] == bytes(entries[0].data)
    assert decoded["runs"] == bytes(entries[1].data)
    stats = analyze_lossless_cache_sidecar(entries, chunk_size=4096)
    assert stats["roundtrip_ok"] is True
    assert stats["entry_count"] == 2
    assert stats["compression_ratio"] < 0.8


def test_numpy_cache_roundtrip() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_smoke_") as tmp:
        path = Path(tmp) / "sample_newbie.npz"
        latents = np.zeros((1, 16, 16, 16), dtype=np.float16)
        hidden = np.arange(512, dtype=np.float32).reshape(32, 16)
        np.savez(path, latents=latents, encoder_hidden_states=hidden)

        entries = load_numpy_cache_entries(path)
        blob = encode_lossless_cache_sidecar(entries, chunk_size=4096)
        arrays = decode_lossless_cache_sidecar_arrays(blob)
        assert np.array_equal(arrays["latents"], latents)
        assert np.array_equal(arrays["encoder_hidden_states"], hidden)


def main() -> int:
    test_entry_roundtrip()
    test_numpy_cache_roundtrip()
    print("lossless_cache_sidecar_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
