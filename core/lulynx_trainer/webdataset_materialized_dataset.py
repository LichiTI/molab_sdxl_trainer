# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Materialized WebDataset adapter for CaptionDataset-compatible training.

This is a conservative first training integration: explicit WebDataset shards
are extracted to a temporary directory and then consumed by the existing
CaptionDataset path.  It validates archive member names and keeps the default
``data_backend=auto`` behavior unchanged.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from .data_backend_resolver import discover_webdataset_shards
from .dataset_loader import CaptionDataset

logger = logging.getLogger(__name__)


class WebDatasetMaterializationError(RuntimeError):
    pass


def _native_webdataset_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_WEBDATASET", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_webdataset_enabled() -> bool:
    value = str(os.environ.get("LULYNX_ENABLE_NATIVE_WEBDATASET", "1") or "1").strip().lower()
    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_webdataset_image_validation_mode() -> str:
    return str(os.environ.get("LULYNX_NATIVE_WEBDATASET_VALIDATE_IMAGES", "pil") or "pil").strip().lower()


@lru_cache(maxsize=1)
def _load_native_webdataset_api() -> Any:
    artifact_dir = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if artifact_dir and artifact_dir not in sys.path and Path(artifact_dir).is_dir():
        sys.path.insert(0, artifact_dir)
    try:
        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def _native_webdataset_api() -> Any:
    if _native_webdataset_disabled() or not _native_webdataset_enabled():
        return None
    native = _load_native_webdataset_api()
    if not hasattr(native, "materialize_webdataset_tar"):
        return None
    return native


class MaterializedWebDataset(CaptionDataset):
    """CaptionDataset wrapper backed by extracted WebDataset shards."""

    def __init__(self, *, source_data_dir: str, **caption_kwargs: Any) -> None:
        self.source_data_dir = str(source_data_dir)
        self._materialized_tempdir = tempfile.TemporaryDirectory(prefix="lulynx_wds_", ignore_cleanup_errors=True)
        self._materialized_root = Path(self._materialized_tempdir.name)
        summary = materialize_webdataset_shards(self.source_data_dir, self._materialized_root)
        self.webdataset_materialization_summary = summary
        super().__init__(data_dir=str(self._materialized_root), **caption_kwargs)

    def cleanup(self) -> None:
        tempdir = getattr(self, "_materialized_tempdir", None)
        if tempdir is not None:
            tempdir.cleanup()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.cleanup()
        except Exception:
            pass


def materialize_webdataset_shards(data_dir: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    shard_plan = discover_webdataset_shards(data_dir, max_preview=1_000_000)
    if shard_plan.shard_count <= 0:
        raise WebDatasetMaterializationError("no WebDataset shards detected")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_count = 0
    caption_count = 0
    skipped_members: List[str] = []
    shard_summaries: List[Dict[str, Any]] = []

    for shard in shard_plan.shards:
        shard_path = Path(shard)
        shard_summary = _extract_shard_auto(shard_path, output)
        local_images = int(shard_summary.get("images", 0) or 0)
        local_captions = int(shard_summary.get("captions", 0) or 0)
        local_skipped = [str(item) for item in shard_summary.get("skipped_members", [])]
        skipped_count = int(shard_summary.get("skipped_count", len(local_skipped)) or 0)
        image_count += local_images
        caption_count += local_captions
        skipped_members.extend(local_skipped[:20])
        shard_summaries.append({"path": str(shard_path), **shard_summary, "skipped": skipped_count})

    if image_count <= 0:
        shutil.rmtree(output, ignore_errors=True)
        raise WebDatasetMaterializationError("WebDataset shards contained no supported image samples")

    manifest = {
        "source_data_dir": str(data_dir),
        "output_dir": str(output),
        "shard_count": int(shard_plan.shard_count),
        "image_count": int(image_count),
        "caption_count": int(caption_count),
        "skipped_member_preview": skipped_members[:20],
        "shards": shard_summaries,
        "mode": "materialized_captiondataset",
    }
    (output / "_lulynx_webdataset_materialization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _extract_shard_auto(shard_path: Path, output: Path) -> Dict[str, Any]:
    native = _native_webdataset_api()
    if native is None or not _native_supports_shard(shard_path):
        images, captions, skipped = _extract_shard(shard_path, output)
        return _shard_summary(
            images=images,
            captions=captions,
            skipped=skipped,
            provider="python_fallback",
            image_validation="pil",
            native_tar_passes=0,
        )
    try:
        validation_mode = _native_webdataset_image_validation_mode()
        validate_in_native = validation_mode in {"header", "native_header", "native"}
        payload = native.materialize_webdataset_tar(str(shard_path), str(output), validate_in_native)
        image_validation = "native_header" if validate_in_native else "pil_after_native_extract"
        if not validate_in_native:
            image_paths = [Path(path) for path in payload.get("image_paths", []) if path]
            for path in image_paths:
                _validate_image(path)
        return _shard_summary(
            images=int(payload.get("images", 0) or 0),
            captions=int(payload.get("captions", 0) or 0),
            skipped=[str(item) for item in payload.get("skipped", [])],
            provider="native",
            image_validation=image_validation,
            native_tar_passes=int(payload.get("native_tar_passes", 1) or 1),
            native_image_header_validated=bool(payload.get("native_image_header_validated", False)),
        )
    except Exception as exc:
        logger.info("Native WebDataset materialization failed for %s; falling back to Python: %s", shard_path, exc)
        images, captions, skipped = _extract_shard(shard_path, output)
        return _shard_summary(
            images=images,
            captions=captions,
            skipped=skipped,
            provider="python_fallback_after_native_error",
            image_validation="pil",
            native_tar_passes=0,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def _shard_summary(
    *,
    images: int,
    captions: int,
    skipped: List[str],
    provider: str,
    image_validation: str,
    native_tar_passes: int,
    native_image_header_validated: bool = False,
    fallback_reason: str = "",
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "images": int(images),
        "captions": int(captions),
        "skipped_members": skipped[:20],
        "skipped_count": len(skipped),
        "provider": provider,
        "image_validation": image_validation,
        "native_tar_passes": int(native_tar_passes),
        "native_image_header_validated": bool(native_image_header_validated),
    }
    if fallback_reason:
        summary["fallback_reason"] = fallback_reason
    return summary


def _native_supports_shard(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar") and not name.endswith(".tar.gz") and not name.endswith(".tgz")


def _extract_shard(shard_path: Path, output: Path) -> Tuple[int, int, List[str]]:
    image_count = 0
    caption_count = 0
    skipped: List[str] = []
    prefix = _safe_stem(shard_path.name)
    with tarfile.open(shard_path, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = str(member.name)
            suffix = Path(name).suffix.lower()
            if suffix not in CaptionDataset.SUPPORTED_EXTENSIONS and suffix not in {".txt", ".caption", ".json"}:
                skipped.append(name)
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                skipped.append(name)
                continue
            raw = extracted.read()
            rel_stem = _safe_sample_key(name)
            base = f"{prefix}_{rel_stem}"
            try:
                if suffix in CaptionDataset.SUPPORTED_EXTENSIONS:
                    target = output / f"{base}{suffix}"
                    target.write_bytes(raw)
                    _validate_image(target)
                    image_count += 1
                else:
                    target = output / f"{base}.txt"
                    target.write_text(_decode_caption(raw), encoding="utf-8")
                    caption_count += 1
            except Exception as exc:
                skipped.append(f"{name}: {exc}")
    return image_count, caption_count, skipped


def _validate_image(path: Path) -> None:
    with Image.open(path) as img:
        img.verify()


def _decode_caption(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def _safe_stem(name: str) -> str:
    stem = Path(name).name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return _sanitize(stem or "shard")


def _safe_sample_key(name: str) -> str:
    path = Path(name)
    parts = [part for part in path.with_suffix("").parts if part not in {"", ".", ".."}]
    return _sanitize("_".join(parts) or path.stem or "sample")


def _sanitize(value: str) -> str:
    cleaned = []
    for ch in str(value):
        cleaned.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    text = "".join(cleaned).strip("_")
    return text[:160] or "sample"


