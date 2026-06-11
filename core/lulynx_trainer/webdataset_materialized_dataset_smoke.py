# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test materialized WebDataset adapter."""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.dataset_loader import collate_fn  # noqa: E402
from core.lulynx_trainer.webdataset_materialized_dataset import MaterializedWebDataset  # noqa: E402


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (16, 16), color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def main() -> int:
    old_enabled = os.environ.get("LULYNX_ENABLE_NATIVE_WEBDATASET")
    old_validate = os.environ.get("LULYNX_NATIVE_WEBDATASET_VALIDATE_IMAGES")
    os.environ["LULYNX_ENABLE_NATIVE_WEBDATASET"] = "1"
    os.environ["LULYNX_NATIVE_WEBDATASET_VALIDATE_IMAGES"] = "header"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shard = root / "train-000.tar"
        with tarfile.open(shard, "w") as tar:
            _add_bytes(tar, "000001.png", _png_bytes((255, 0, 0)))
            _add_bytes(tar, "000001.txt", b"red dress, test caption")
            _add_bytes(tar, "nested/000002.png", _png_bytes((0, 255, 0)))
            _add_bytes(tar, "nested/000002.caption", b"green hair")

        dataset = MaterializedWebDataset(
            source_data_dir=str(root),
            resolution=16,
            enable_bucket=False,
            caption_extension=".txt",
            shuffle_caption=False,
            image_decode_backend="pil",
        )
        try:
            assert len(dataset) == 2
            summary = dataset.webdataset_materialization_summary
            assert summary["image_count"] == 2
            assert summary["caption_count"] == 2
            shard_summary = summary["shards"][0]
            assert shard_summary["provider"] in {"native", "python_fallback"}
            if shard_summary["provider"] == "native":
                assert shard_summary["native_tar_passes"] == 1
                assert shard_summary["image_validation"] == "native_header"
                assert shard_summary["native_image_header_validated"] is True
            else:
                assert shard_summary["native_tar_passes"] == 0
                assert shard_summary["image_validation"] in {"pil", "header", "none"}
                assert shard_summary["native_image_header_validated"] is False
            items = [dataset[0], dataset[1]]
            captions = sorted(item["caption"] for item in items)
            assert captions == ["green, hair", "red dress, test caption"]
            batch = collate_fn(items)
            assert batch["images"].shape == (2, 3, 16, 16)
            assert sorted(batch["captions"]) == captions
        finally:
            dataset.cleanup()
    if old_enabled is None:
        os.environ.pop("LULYNX_ENABLE_NATIVE_WEBDATASET", None)
    else:
        os.environ["LULYNX_ENABLE_NATIVE_WEBDATASET"] = old_enabled
    if old_validate is None:
        os.environ.pop("LULYNX_NATIVE_WEBDATASET_VALIDATE_IMAGES", None)
    else:
        os.environ["LULYNX_NATIVE_WEBDATASET_VALIDATE_IMAGES"] = old_validate
    print("webdataset materialized dataset smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



