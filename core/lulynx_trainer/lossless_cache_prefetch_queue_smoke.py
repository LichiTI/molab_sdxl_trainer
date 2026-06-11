"""Smoke checks for the experimental LXCS prefetch queue probe."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_prefetch_queue import (  # type: ignore[no-redef]
        LosslessCachePrefetchPayload,
        LosslessCachePrefetchQueueConfig,
        prepare_lossless_cache_prefetch_sidecars,
        run_lossless_cache_prefetch_queue,
        run_lossless_cache_serial_probe,
    )
else:
    from .lossless_cache_prefetch_queue import (
        LosslessCachePrefetchPayload,
        LosslessCachePrefetchQueueConfig,
        prepare_lossless_cache_prefetch_sidecars,
        run_lossless_cache_prefetch_queue,
        run_lossless_cache_serial_probe,
    )


def _make_cache(path: Path, offset: int) -> dict[str, object]:
    import numpy as np

    latents = (np.arange(2 * 4 * 8 * 8, dtype=np.float16).reshape(2, 4, 8, 8) + offset)
    embeds = (np.arange(12 * 16, dtype=np.float32).reshape(12, 16) + offset)
    mask = np.ones((12,), dtype=np.int64) * (offset + 1)
    np.savez(path, latents=latents, prompt_embeds=embeds, attention_mask=mask)
    return {"latents": latents, "prompt_embeds": embeds, "attention_mask": mask}


def test_prefetch_queue_sidecar_and_fallback() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_prefetch_queue_") as tmp:
        root = Path(tmp)
        sidecar_dir = root / "sidecars"
        expected: dict[str, dict[str, object]] = {}
        paths: list[Path] = []
        for index in range(4):
            path = root / f"sample_{index:03d}.npz"
            expected[str(path)] = _make_cache(path, index)
            paths.append(path)

        config = LosslessCachePrefetchQueueConfig(
            prefetch_depth=2,
            sidecar_dir=str(sidecar_dir),
            consumer_delay_ms=1.0,
        )
        prepare = prepare_lossless_cache_prefetch_sidecars(paths[:3], config=config, chunk_size=4096)
        assert prepare["ok"] is True, prepare

        seen: list[str] = []

        def consumer(payload: LosslessCachePrefetchPayload) -> None:
            seen.append(str(payload.path))
            source = expected[str(payload.path)]
            for name, array in source.items():
                assert name in payload.arrays, payload.report
                assert np.array_equal(payload.arrays[name], array), payload.report

        report = run_lossless_cache_prefetch_queue(paths, config=config, consumer=consumer)
        assert report["ok"] is True, report
        assert report["consumed"] == 4, report
        assert report["fallback_count"] == 1, report
        assert report["reads_tensor_payloads"] is True, report
        assert report["training_path_enabled"] is False, report
        assert report["p3_prefetch_queue_probe_ready"] is True, report
        assert report["p3_prefetch_h2d_ready"] is False, report
        assert seen == [str(path) for path in paths], report

        serial = run_lossless_cache_serial_probe(paths, config=config)
        assert serial["ok"] is True, serial
        assert serial["fallback_count"] == 1, serial


def test_prefetch_queue_disabled() -> None:
    with tempfile.TemporaryDirectory(prefix="lulynx_lxcs_prefetch_queue_disabled_") as tmp:
        path = Path(tmp) / "sample.npz"
        _make_cache(path, 0)
        report = run_lossless_cache_prefetch_queue(
            [path],
            config=LosslessCachePrefetchQueueConfig(enabled=False),
        )
        assert report["ok"] is True, report
        assert report["skipped"] is True, report
        assert report["p3_prefetch_queue_probe_ready"] is False, report


def main() -> int:
    test_prefetch_queue_sidecar_and_fallback()
    test_prefetch_queue_disabled()
    print("lossless_cache_prefetch_queue_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
