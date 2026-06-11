"""Smoke checks for the LXCS DataLoader shadow probe."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_dataloader_shadow import (  # type: ignore[no-redef]
        LosslessCacheDataloaderShadowConfig,
        run_lossless_cache_dataloader_shadow,
    )
    from lulynx_trainer.newbie_cached_dataset import (  # type: ignore[no-redef]
        NewbieCachedDataset,
        create_newbie_cached_dataloader,
    )
else:
    from .lossless_cache_dataloader_shadow import (
        LosslessCacheDataloaderShadowConfig,
        run_lossless_cache_dataloader_shadow,
    )
    from .newbie_cached_dataset import NewbieCachedDataset, create_newbie_cached_dataloader


def _write_newbie_cache(path: Path, offset: int) -> None:
    import numpy as np

    np.savez(
        path,
        newbie_cache_schema_version=np.array([2], dtype=np.int64),
        latents=(np.arange(1 * 16 * 8 * 8, dtype=np.float16).reshape(1, 16, 8, 8) + offset),
        encoder_hidden_states=(np.arange(32 * 16, dtype=np.float32).reshape(32, 16) + offset),
        pooled_prompt_embeds=(np.arange(16, dtype=np.float32) + offset),
        attention_mask=np.ones((32,), dtype=np.int64),
    )


def test_newbie_dataloader_shadow() -> None:
    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_dataloader_shadow_") as tmp:
        root = Path(tmp)
        for index in range(4):
            _write_newbie_cache(root / f"sample_{index:03d}_newbie.npz", index)

        def dataset_factory() -> NewbieCachedDataset:
            return NewbieCachedDataset(root)

        def dataloader_factory(dataset: NewbieCachedDataset):
            return create_newbie_cached_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

        report = run_lossless_cache_dataloader_shadow(
            dataset_factory,
            dataloader_factory,
            config=LosslessCacheDataloaderShadowConfig(
                batch_size=2,
                max_batches=2,
                prefetch_depth=2,
                sidecar_dir=str(root / "sidecars"),
                chunk_size=4096,
            ),
        )
        assert report["ok"] is True, report
        assert report["real_dataloader_shadow"] is True, report
        assert report["training_path_enabled"] is False, report
        assert report["path_count"] == 4, report
        assert report["consumed_path_count"] == 4, report
        assert report["fallback_count"] == 0, report
        assert report["p3_dataloader_shadow_ready"] is True, report


def main() -> int:
    test_newbie_dataloader_shadow()
    print("lossless_cache_dataloader_shadow_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
