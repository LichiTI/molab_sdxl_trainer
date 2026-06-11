"""Smoke checks for LXCS shadow parity probes."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_shadow_parity import (  # type: ignore[no-redef]
        CacheShadowParityOptions,
        run_lossless_cache_shadow_parity,
        run_lossless_cache_shadow_parity_matrix,
    )
else:
    from .lossless_cache_shadow_parity import (
        CacheShadowParityOptions,
        run_lossless_cache_shadow_parity,
        run_lossless_cache_shadow_parity_matrix,
    )


def test_npz_shadow_parity() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_parity_") as tmp:
        path = Path(tmp) / "sample_anima.npz"
        np.savez(
            path,
            latents_64x64=np.arange(16 * 64 * 64, dtype=np.float32).reshape(16, 64, 64),
            prompt_embeds=np.zeros((32, 1024), dtype=np.float32),
            attn_mask=np.ones((32,), dtype=np.int64),
        )
        report = run_lossless_cache_shadow_parity(path, options=CacheShadowParityOptions(chunk_size=4096))
        assert report["ok"] is True, report
        assert report["entry_count"] == 3, report
        assert report["matched_entries"] == 3, report
        assert report["training_path_enabled"] is False, report


def test_matrix_shadow_parity() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_parity_matrix_") as tmp:
        root = Path(tmp)
        paths = []
        for index in range(2):
            path = root / f"sample_{index}_newbie.npz"
            np.savez(
                path,
                latents=np.full((1, 16, 8, 8), index, dtype=np.float16),
                encoder_hidden_states=np.arange(128, dtype=np.float32).reshape(8, 16),
            )
            paths.append(path)
        report = run_lossless_cache_shadow_parity_matrix(paths, options=CacheShadowParityOptions(chunk_size=2048))
        assert report["ok"] is True, report
        assert report["case_count"] == 2, report
        assert report["ok_count"] == 2, report
        assert report["shadow_only"] is True, report


def main() -> int:
    test_npz_shadow_parity()
    test_matrix_shadow_parity()
    print("lossless_cache_shadow_parity_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
