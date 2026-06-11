"""Smoke probe for debug-only TurboCore dataset shadow adapter."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.lulynx_trainer.dataset_loader import CaptionDataset, create_dataloader  # noqa: E402
from core.turbocore_dataset_shadow_adapter import (  # noqa: E402
    ENABLE_ENV,
    EPOCH_ENV,
    SEED_ENV,
    build_caption_dataset_shadow_adapter_policy,
)


def _write_sample(root: Path, name: str, size: tuple[int, int], caption: str) -> None:
    Image.new("RGB", size, color=(48, 112, 176)).save(root / f"{name}.png")
    (root / f"{name}.txt").write_text(caption, encoding="utf-8")


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


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lulynx_shadow_adapter_") as tmp:
        root = Path(tmp)
        _write_sample(root, "img_0001", (512, 768), "one")
        _write_sample(root, "img_0002", (768, 512), "two")
        _write_sample(root, "img_0003", (640, 640), "three")

        flat_dataset = CaptionDataset(
            str(root),
            resolution=512,
            caption_extension=".txt",
            enable_bucket=False,
            shuffle_caption=False,
        )
        disabled_loader = create_dataloader(
            flat_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        assert not hasattr(disabled_loader, "native_dataset_shadow_adapter"), disabled_loader

        old_env = _set_env({ENABLE_ENV: "1", SEED_ENV: "100", EPOCH_ENV: "7"})
        try:
            flat_loader = create_dataloader(
                flat_dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        finally:
            _restore_env(old_env)
        flat_report = getattr(flat_loader, "native_dataset_shadow_adapter")
        assert flat_report["ok"] is True, flat_report
        assert flat_report["shadow_run"] is True, flat_report
        assert flat_report["skipped"] is False, flat_report
        assert flat_report["seed"] == 100, flat_report
        assert flat_report["epoch"] == 7, flat_report
        assert flat_report["effective_seed"] == 107, flat_report
        assert flat_report["epoch_reseed_policy"] == "base_seed_plus_epoch_v1", flat_report
        assert flat_report["worker_shard_policy"] == "single_process_order", flat_report
        assert flat_report["sampler_order_live_equivalent"] is True, flat_report
        assert flat_report["shadow_order_scope"] == "live_equivalent", flat_report
        assert flat_report["lifecycle"]["ok"] is True, flat_report
        assert flat_report["training_path_enabled"] is False, flat_report

        worker_policy = build_caption_dataset_shadow_adapter_policy(
            flat_dataset,
            batch_size=2,
            shuffle=True,
            drop_last=False,
            num_workers=2,
            seed=100,
            epoch=7,
        )
        assert worker_policy["worker_shard_policy"] == "main_process_sampler_order_only", worker_policy
        assert worker_policy["worker_fetch_timing_equivalent"] is False, worker_policy
        assert worker_policy["effective_seed"] == 107, worker_policy
        assert worker_policy["shadow_order_scope"] == "live_equivalent", worker_policy
        assert worker_policy["training_path_enabled"] is False, worker_policy

        bucket_dataset = CaptionDataset(
            str(root),
            resolution=512,
            caption_extension=".txt",
            enable_bucket=True,
            min_bucket_reso=512,
            max_bucket_reso=768,
            bucket_reso_steps=128,
            bucket_custom_resos="512x768,768x512,640x640",
            shuffle_caption=False,
        )
        old_env = _set_env({ENABLE_ENV: "1", SEED_ENV: "100", EPOCH_ENV: "7"})
        try:
            bucket_loader = create_dataloader(
                bucket_dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        finally:
            _restore_env(old_env)
        bucket_report = getattr(bucket_loader, "native_dataset_shadow_adapter")
        assert bucket_report["ok"] is True, bucket_report
        assert bucket_report["skipped"] is True, bucket_report
        assert bucket_report["shadow_run"] is False, bucket_report
        assert bucket_report["bucket_sampler_detected"] is True, bucket_report
        assert bucket_report["native_shadow_supported"] is False, bucket_report
        assert bucket_report["shadow_order_scope"] == "diagnostic_reference_only", bucket_report
        assert bucket_report["fallback_reason"] == "bucket_sampler_order_parity_not_ready", bucket_report
        assert bucket_report["training_path_enabled"] is False, bucket_report

        return {
            "schema_version": 1,
            "probe": "turbocore_dataset_shadow_adapter_smoke",
            "ok": True,
            "training_path_enabled": False,
            "flat_report": flat_report,
            "worker_policy": worker_policy,
            "bucket_report": bucket_report,
        }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
