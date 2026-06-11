"""Smoke probe for debug-only cached dataset native prefetch adapter."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.lulynx_trainer.anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader  # noqa: E402
from core.lulynx_trainer.newbie_cached_dataset import (  # noqa: E402
    NewbieCacheSchema,
    NewbieCachedDataset,
    create_newbie_cached_dataloader,
)
from core.turbocore_cached_dataset_prefetch import (  # noqa: E402
    ENABLE_ENV,
    EPOCH_ENV,
    SEED_ENV,
    build_cached_dataset_prefetch_adapter_policy,
    build_cached_dataset_prefetch_manifest,
    close_cached_dataset_prefetch_session,
    create_cached_dataset_prefetch_session,
    run_cached_dataset_prefetch_shadow_adapter,
)


def _set_env(values: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        os.environ[key] = value
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_newbie_cache(root: Path, stem: str, *, size: int = 4) -> None:
    np.savez(
        root / f"{stem}_newbie.npz",
        newbie_cache_schema_version=np.array([2], dtype=np.int64),
        latents=np.zeros((1, 4, size, size), dtype=np.float32),
        encoder_hidden_states=np.zeros((1, 3, 8), dtype=np.float32),
        pooled_prompt_embeds=np.zeros((1, 6), dtype=np.float32),
        attention_mask=np.ones((1, 3), dtype=np.int64),
    )


def _write_anima_cache(root: Path, stem: str, *, size: int = 4) -> None:
    np.savez(root / f"{stem}_0001_anima.npz", latents_4=np.zeros((4, size, size), dtype=np.float32))
    np.savez(
        root / f"{stem}_anima_te.npz",
        prompt_embeds=np.zeros((3, 8), dtype=np.float32),
        attn_mask=np.ones((3,), dtype=np.int64),
    )


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lulynx_cache_prefetch_") as tmp:
        root = Path(tmp)
        _write_newbie_cache(root, "sample_a", size=4)
        _write_newbie_cache(root, "sample_b", size=4)
        _write_newbie_cache(root, "sample_c", size=6)
        newbie = NewbieCachedDataset(
            root,
            schema=NewbieCacheSchema(require_schema_version=True),
            cache_mmap=True,
            cache_lazy=True,
        )
        disabled_loader = create_newbie_cached_dataloader(
            newbie,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        assert not hasattr(disabled_loader, "native_cache_prefetch_shadow_adapter"), disabled_loader

        old_env = _set_env({ENABLE_ENV: "1", SEED_ENV: "19", EPOCH_ENV: "2"})
        try:
            newbie_loader = create_newbie_cached_dataloader(
                newbie,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        finally:
            _restore_env(old_env)
        newbie_report = getattr(newbie_loader, "native_cache_prefetch_shadow_adapter")
        assert newbie_report["ok"] is True, newbie_report
        assert newbie_report["shadow_run"] is False, newbie_report
        assert newbie_report["skipped"] is True, newbie_report
        assert newbie_report["bucket_sampler_detected"] is True, newbie_report
        assert newbie_report["fallback_reason"] == "bucket_sampler_order_parity_not_ready", newbie_report
        assert newbie_report["training_path_enabled"] is False, newbie_report

        flat_root = root / "newbie_flat"
        flat_root.mkdir()
        _write_newbie_cache(flat_root, "flat_a", size=4)
        _write_newbie_cache(flat_root, "flat_b", size=4)
        flat_newbie = NewbieCachedDataset(
            flat_root,
            schema=NewbieCacheSchema(require_schema_version=True),
            cache_mmap=True,
            cache_lazy=True,
        )
        flat_policy = build_cached_dataset_prefetch_adapter_policy(
            flat_newbie,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=2,
            prefetch_factor=3,
            seed=19,
            epoch=2,
        )
        assert flat_policy["effective_seed"] == 21, flat_policy
        assert flat_policy["worker_shard_policy"] == "main_process_sampler_order_only", flat_policy
        assert flat_policy["training_path_enabled"] is False, flat_policy

        flat_probe = run_cached_dataset_prefetch_shadow_adapter(
            flat_newbie,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            prefetch_factor=2,
            seed=19,
            epoch=2,
            prefer_native=False,
        )
        assert flat_probe["ok"] is True, flat_probe
        assert flat_probe["shadow_run"] is True, flat_probe
        assert flat_probe["prefetch_probe"]["metadata_only"] is True, flat_probe
        assert flat_probe["prefetch_probe"]["training_path_enabled"] is False, flat_probe

        session_report: dict[str, Any]
        try:
            manifest = build_cached_dataset_prefetch_manifest(flat_newbie)
            with create_cached_dataset_prefetch_session(flat_newbie) as session:
                first = session.run(
                    batch_size=1,
                    shuffle=True,
                    drop_last=False,
                    seed=21,
                    prefetch_depth=2,
                    manifest=manifest,
                )
                second = session.run(
                    batch_size=1,
                    shuffle=True,
                    drop_last=False,
                    seed=22,
                    prefetch_depth=2,
                    manifest=manifest,
                )
                fast = session.run_fast(
                    batch_size=1,
                    shuffle=True,
                    drop_last=False,
                    seed=23,
                    prefetch_depth=2,
                    manifest=manifest,
                )
                stats = session.stats()
            assert first["ok"] is True, first
            assert second["ok"] is True, second
            assert fast["ok"] is True, fast
            assert second["session_reused"] is True, second
            assert fast["fast_summary"] is True, fast
            assert fast["plan"]["native_index_materialized"] is False, fast
            assert fast["plan"]["sampler_order_exact"] is False, fast
            assert stats["run_count"] == 3, stats
            assert first["manifest_owned"] is True, first
            assert first["training_path_enabled"] is False, first
            session_report = {"first": first, "second": second, "fast": fast, "stats": stats}
        except RuntimeError as exc:
            session_report = {"ok": False, "skipped": True, "reason": str(exc), "training_path_enabled": False}

        persistent_report: dict[str, Any]
        old_env = _set_env({ENABLE_ENV: "1", SEED_ENV: "31", EPOCH_ENV: "0"})
        try:
            first_loader = create_newbie_cached_dataloader(
                flat_newbie,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            second_loader = create_newbie_cached_dataloader(
                flat_newbie,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            with np.load(flat_root / "flat_a_newbie.npz") as data:
                arrays = {key: data[key] for key in data.files}
            arrays["latents"] = np.zeros((1, 4, 5, 5), dtype=np.float32)
            np.savez(flat_root / "flat_a_newbie.npz", **arrays)
            third_loader = create_newbie_cached_dataloader(
                flat_newbie,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        finally:
            _restore_env(old_env)
        try:
            first_persistent = getattr(first_loader, "native_cache_prefetch_shadow_adapter")
            second_persistent = getattr(second_loader, "native_cache_prefetch_shadow_adapter")
            third_persistent = getattr(third_loader, "native_cache_prefetch_shadow_adapter")
            assert first_persistent["persistent_session"] is True, first_persistent
            assert second_persistent["persistent_session"] is True, second_persistent
            assert third_persistent["persistent_session"] is True, third_persistent
            assert first_persistent["session_reused_by_adapter"] is False, first_persistent
            assert second_persistent["session_reused_by_adapter"] is True, second_persistent
            assert third_persistent["session_reused_by_adapter"] is False, third_persistent
            assert third_persistent["requires_rebuild"] is True, third_persistent
            assert second_persistent["prefetch_probe"]["session_reused"] is True, second_persistent
            assert third_persistent["prefetch_probe"]["session_reused"] is False, third_persistent
            assert second_persistent["training_path_enabled"] is False, second_persistent
            assert third_persistent["training_path_enabled"] is False, third_persistent
            persistent_report = {"first": first_persistent, "second": second_persistent, "third": third_persistent}
        except RuntimeError as exc:
            persistent_report = {"ok": False, "skipped": True, "reason": str(exc), "training_path_enabled": False}
        finally:
            close_cached_dataset_prefetch_session(flat_newbie)

        anima_root = root / "anima"
        anima_root.mkdir()
        _write_anima_cache(anima_root, "anime_a", size=4)
        _write_anima_cache(anima_root, "anime_b", size=4)
        anima = AnimaCachedDataset(anima_root, enable_bucket=False, cache_mmap=True, cache_lazy=True)
        old_env = _set_env({ENABLE_ENV: "1", SEED_ENV: "5", EPOCH_ENV: "1"})
        try:
            anima_loader = create_anima_cached_dataloader(
                anima,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        finally:
            _restore_env(old_env)
        anima_report = getattr(anima_loader, "native_cache_prefetch_shadow_adapter")
        assert anima_report["ok"] is True, anima_report
        assert anima_report["shadow_run"] is True, anima_report
        assert anima_report["native_shadow_supported"] is True, anima_report
        assert anima_report["prefetch_probe"]["manifest"]["sample_count"] == 2, anima_report
        assert anima_report["prefetch_probe"]["reads_tensor_payloads"] is False, anima_report
        assert anima_report["training_path_enabled"] is False, anima_report

        return {
            "schema_version": 1,
            "probe": "turbocore_cached_dataset_prefetch_smoke",
            "ok": True,
            "training_path_enabled": False,
            "newbie_report": newbie_report,
            "flat_policy": flat_policy,
            "flat_probe": flat_probe,
            "session_report": session_report,
            "persistent_report": persistent_report,
            "anima_report": anima_report,
        }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
