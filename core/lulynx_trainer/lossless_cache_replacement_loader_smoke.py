"""Smoke checks for the experimental LXCS replacement loader."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_prefetch_queue import (  # type: ignore[no-redef]
        LosslessCachePrefetchQueueConfig,
        load_lossless_cache_prefetch_payload,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from lulynx_trainer.lossless_cache_replacement_loader import (  # type: ignore[no-redef]
        LosslessCacheReplacementLoaderConfig,
        _build_newbie_item_from_payload,
        run_newbie_lossless_cache_replacement_loader,
    )
    from lulynx_trainer.newbie_cached_dataset import (  # type: ignore[no-redef]
        NewbieCachedDataset,
        create_newbie_cached_dataloader,
        newbie_cached_collate,
    )
else:
    from .lossless_cache_prefetch_queue import (
        LosslessCachePrefetchQueueConfig,
        load_lossless_cache_prefetch_payload,
        prepare_lossless_cache_prefetch_sidecars,
    )
    from .lossless_cache_replacement_loader import (
        LosslessCacheReplacementLoaderConfig,
        _build_newbie_item_from_payload,
        run_newbie_lossless_cache_replacement_loader,
    )
    from .newbie_cached_dataset import (
        NewbieCachedDataset,
        create_newbie_cached_dataloader,
        newbie_cached_collate,
    )


def _write_newbie_cache(path: Path, offset: int) -> None:
    import numpy as np

    np.savez(
        path,
        newbie_cache_schema_version=np.array([2], dtype=np.int64),
        latents=(np.arange(1 * 16 * 8 * 8, dtype=np.float16).reshape(1, 16, 8, 8) + offset),
        encoder_hidden_states=(np.arange(12 * 16, dtype=np.float32).reshape(12, 16) + offset),
        pooled_prompt_embeds=(np.arange(16, dtype=np.float32) + offset),
        attention_mask=np.ones((12,), dtype=np.int64),
        loss_mask=np.ones((8, 8), dtype=np.float32),
    )


def _assert_batch_equal(left: dict[str, object], right: dict[str, object]) -> None:
    import torch

    for key in ("latents", "encoder_hidden_states", "attention_mask", "pooled_prompt_embeds", "loss_masks"):
        if key not in left and key not in right:
            continue
        assert key in left and key in right, (key, left.keys(), right.keys())
        assert torch.equal(left[key], right[key]), key  # type: ignore[arg-type]
    assert left["sample_ids"] == right["sample_ids"], (left["sample_ids"], right["sample_ids"])


def test_newbie_replacement_loader() -> None:
    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_replacement_loader_") as tmp:
        root = Path(tmp)
        sidecar_dir = root / "sidecars"
        for index in range(4):
            _write_newbie_cache(root / f"sample_{index:03d}_newbie.npz", index)

        dataset = NewbieCachedDataset(root, latent_crop_size=6, text_token_limit=8)
        report = run_newbie_lossless_cache_replacement_loader(
            dataset,
            config=LosslessCacheReplacementLoaderConfig(
                batch_size=2,
                max_batches=2,
                prefetch_depth=2,
                sidecar_dir=str(sidecar_dir),
                fallback_to_raw=False,
                sidecar_strict=True,
                chunk_size=4096,
            ),
        )
        assert report["ok"] is True, report
        assert report["experimental_replacement_path"] is True, report
        assert report["training_path_enabled"] is False, report
        assert report["duplicate_dataset_read"] is False, report
        assert report["fallback_count"] == 0, report
        assert report["p3_replacement_loader_ready"] is True, report

        baseline_loader = create_newbie_cached_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        baseline_batches = list(baseline_loader)

        prepare_lossless_cache_prefetch_sidecars(
            [sample.cache_path for sample in dataset.samples],
            config=LosslessCachePrefetchQueueConfig(
                sidecar_dir=str(sidecar_dir),
                sidecar_strict=True,
                fallback_to_raw=False,
            ),
            chunk_size=4096,
        )
        replacement_batches = []
        for start in range(0, len(dataset.samples), 2):
            items = []
            for offset, sample in enumerate(dataset.samples[start : start + 2]):
                payload = load_lossless_cache_prefetch_payload(
                    sample.cache_path,
                    index=start + offset,
                    config=LosslessCachePrefetchQueueConfig(
                        sidecar_dir=str(sidecar_dir),
                        sidecar_strict=True,
                        fallback_to_raw=False,
                    ),
                )
                assert payload.ok, payload.report
                items.append(_build_newbie_item_from_payload(dataset, sample, payload))
            replacement_batches.append(newbie_cached_collate(items))

        assert len(baseline_batches) == len(replacement_batches)
        for baseline, replacement in zip(baseline_batches, replacement_batches):
            _assert_batch_equal(baseline, replacement)


def main() -> int:
    test_newbie_replacement_loader()
    print("lossless_cache_replacement_loader_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
