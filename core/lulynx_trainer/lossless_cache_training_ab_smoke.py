"""Smoke check for the LXCS mini training-step A/B probe."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_training_ab import (  # type: ignore[no-redef]
        LosslessCacheTrainingAbConfig,
        run_newbie_lossless_cache_training_ab,
    )
    from lulynx_trainer.newbie_cached_dataset import NewbieCachedDataset
else:
    from .lossless_cache_training_ab import LosslessCacheTrainingAbConfig, run_newbie_lossless_cache_training_ab
    from .newbie_cached_dataset import NewbieCachedDataset


def _write_newbie_cache(path: Path, offset: int) -> None:
    import numpy as np

    np.savez(
        path,
        newbie_cache_schema_version=np.array([2], dtype=np.int64),
        latents=(np.arange(1 * 16 * 8 * 8, dtype=np.float16).reshape(1, 16, 8, 8) + offset),
        encoder_hidden_states=(np.arange(12 * 16, dtype=np.float32).reshape(12, 16) + offset),
        pooled_prompt_embeds=(np.arange(16, dtype=np.float32) + offset),
        attention_mask=np.ones((12,), dtype=np.int64),
    )


def test_newbie_training_ab() -> None:
    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_training_ab_") as tmp:
        root = Path(tmp)
        for index in range(4):
            _write_newbie_cache(root / f"sample_{index:03d}_newbie.npz", index)
        report = run_newbie_lossless_cache_training_ab(
            NewbieCachedDataset(root),
            NewbieCachedDataset(root),
            config=LosslessCacheTrainingAbConfig(
                batch_size=2,
                max_batches=2,
                prefetch_depth=2,
                sidecar_dir=str(root / "sidecars"),
                sidecar_strict=True,
                fallback_to_raw=False,
                chunk_size=4096,
                device="cpu",
                compute_repeat=2,
            ),
        )
        assert report["ok"] is True, report
        assert report["mini_training_step_ab"] is True, report
        assert report["training_path_enabled"] is False, report
        assert report["p3_training_step_ab_probe_ready"] is True, report
        assert report["baseline"]["batch_count"] == 2, report
        assert report["replacement"]["batch_count"] == 2, report


def main() -> int:
    test_newbie_training_ab()
    print("lossless_cache_training_ab_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
