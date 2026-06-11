"""Smoke checks for optional LXCS dataset adapter plumbing."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.anima_cached_dataset import AnimaCachedDataset  # type: ignore[no-redef]
    from lulynx_trainer.newbie_cached_dataset import NewbieCachedDataset
    from lulynx_trainer.lossless_cache_sidecar import write_lossless_cache_sidecar_file
else:
    from .anima_cached_dataset import AnimaCachedDataset
    from .newbie_cached_dataset import NewbieCachedDataset
    from .lossless_cache_sidecar import write_lossless_cache_sidecar_file


def _assert_tensors_equal(left, right) -> None:
    import torch

    assert tuple(left.shape) == tuple(right.shape), (left.shape, right.shape)
    assert left.dtype == right.dtype, (left.dtype, right.dtype)
    assert torch.equal(left, right)


def test_newbie_dataset_sidecar_adapter() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_newbie_dataset_") as tmp:
        root = Path(tmp)
        cache = root / "sample_000_newbie.npz"
        np.savez(
            cache,
            newbie_cache_schema_version=np.array([2], dtype=np.int64),
            latents=np.arange(1 * 16 * 8 * 8, dtype=np.float16).reshape(1, 16, 8, 8),
            encoder_hidden_states=np.arange(32 * 16, dtype=np.float32).reshape(32, 16),
            pooled_prompt_embeds=np.arange(16, dtype=np.float32),
            attention_mask=np.ones((32,), dtype=np.int64),
        )
        write_lossless_cache_sidecar_file(cache, chunk_size=4096)

        raw = NewbieCachedDataset(root)[0]
        adapted_dataset = NewbieCachedDataset(root, lossless_cache_sidecar_enabled=True)
        adapted = adapted_dataset[0]
        _assert_tensors_equal(raw["latents"], adapted["latents"])
        _assert_tensors_equal(raw["encoder_hidden_states"], adapted["encoder_hidden_states"])
        assert adapted_dataset.lossless_cache_sidecar_last_report["fallback_to_raw_cache"] is False


def test_anima_dataset_sidecar_adapter() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_anima_dataset_") as tmp:
        root = Path(tmp)
        latent = root / "sample_64x64_anima.npz"
        text = root / "sample_anima_te.npz"
        np.savez(
            latent,
            latents_64x64=np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8),
            original_size_64x64=np.array([64, 64], dtype=np.int64),
            crop_ltrb_64x64=np.array([0, 0, 64, 64], dtype=np.int64),
        )
        np.savez(
            text,
            prompt_embeds=np.arange(16 * 32, dtype=np.float32).reshape(16, 32),
            attn_mask=np.ones((16,), dtype=np.int64),
        )
        write_lossless_cache_sidecar_file(latent, chunk_size=4096)
        write_lossless_cache_sidecar_file(text, chunk_size=4096)

        raw = AnimaCachedDataset(root)[0]
        adapted_dataset = AnimaCachedDataset(root, lossless_cache_sidecar_enabled=True)
        adapted = adapted_dataset[0]
        _assert_tensors_equal(raw["latents"], adapted["latents"])
        _assert_tensors_equal(raw["encoder_hidden_states"], adapted["encoder_hidden_states"])
        _assert_tensors_equal(raw["attention_mask"], adapted["attention_mask"])
        assert adapted_dataset.lossless_cache_sidecar_last_report["fallback_to_raw_cache"] is False


def main() -> int:
    test_newbie_dataset_sidecar_adapter()
    test_anima_dataset_sidecar_adapter()
    print("lossless_cache_dataset_adapter_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
