"""Smoke probe for native cached tensor shape metadata."""

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

from core.lulynx_trainer.anima_cached_dataset import AnimaCachedDataset  # noqa: E402
from core.lulynx_trainer.newbie_cached_dataset import NewbieCacheSchema, NewbieCachedDataset  # noqa: E402
from core.turbocore_cache_shape_metadata import build_cache_shape_index, scan_native_cache_shape_metadata  # noqa: E402


def _write_newbie_cache(root: Path, stem: str, size: int) -> Path:
    path = root / f"{stem}_newbie.npz"
    np.savez(
        path,
        newbie_cache_schema_version=np.array([2], dtype=np.int64),
        latents=np.zeros((1, 4, size, size + 2), dtype=np.float32),
        encoder_hidden_states=np.zeros((1, 3, 8), dtype=np.float32),
        pooled_prompt_embeds=np.zeros((1, 6), dtype=np.float32),
        attention_mask=np.ones((1, 3), dtype=np.int64),
    )
    return path


def _write_anima_cache(root: Path, stem: str, size: int) -> Path:
    latent_path = root / f"{stem}_0001_anima.npz"
    np.savez(latent_path, latents_4=np.zeros((4, size, size + 4), dtype=np.float32))
    np.savez(
        root / f"{stem}_anima_te.npz",
        prompt_embeds=np.zeros((3, 8), dtype=np.float32),
        attn_mask=np.ones((3,), dtype=np.int64),
    )
    return latent_path


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lulynx_cache_shape_") as tmp:
        root = Path(tmp)
        newbie_path = _write_newbie_cache(root, "sample_a", 5)
        anima_root = root / "anima"
        anima_root.mkdir()
        anima_path = _write_anima_cache(anima_root, "anime_a", 7)

        report = scan_native_cache_shape_metadata([newbie_path, anima_path])
        assert report["ok"] is True, report
        assert report["native_runtime"] is True, report
        records = {Path(str(item["path"])).name: item for item in report["records"]}
        assert records[newbie_path.name]["selected_latent_shape"] == [1, 4, 5, 7], records
        assert records[anima_path.name]["selected_latent_shape"] == [4, 7, 11], records
        assert report["training_path_enabled"] is False, report

        index = build_cache_shape_index([newbie_path, anima_path])
        assert index, index

        newbie = NewbieCachedDataset(
            root,
            schema=NewbieCacheSchema(require_schema_version=True),
            cache_mmap=True,
            cache_lazy=True,
        )
        newbie_summary = newbie.get_cache_metadata_summary()
        assert newbie_summary["native_shape_metadata_records"] > 0, newbie_summary
        assert newbie_summary["metadata_shape_hits"] >= 1, newbie_summary
        assert newbie_summary["fallback_shape_loads"] == 0, newbie_summary

        anima = AnimaCachedDataset(anima_root, enable_bucket=True, cache_mmap=True, cache_lazy=True)
        anima_summary = anima.get_cache_metadata_summary()
        assert anima_summary["native_shape_metadata_records"] > 0, anima_summary
        assert anima_summary["metadata_shape_hits"] >= 1, anima_summary
        assert anima_summary["fallback_shape_loads"] == 0, anima_summary

        return {
            "schema_version": 1,
            "probe": "turbocore_cache_shape_metadata_smoke",
            "ok": True,
            "native_report": report,
            "newbie_summary": newbie_summary,
            "anima_summary": anima_summary,
            "training_path_enabled": False,
        }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
