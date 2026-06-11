# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Common tag-editor caption/path helpers and data models."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

try:
    from backend.core.services.native_module_loader import load_lulynx_native
except ModuleNotFoundError:
    from core.services.native_module_loader import load_lulynx_native


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
DEFAULT_CAPTION_EXTENSION = ".txt"
SUPPORTED_CAPTION_EXTENSIONS = (".txt", ".caption")
TAGEDITOR_HISTORY_MANIFEST = ".tageditor_manifest.json"
LEADING_FILENAME_INDEX_RE = re.compile(r"^\s*\d+[\s._-]*")
APPROX_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def native_tag_editor_scan_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_TAGEDITOR_SCAN", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def native_tag_editor_batch_action_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_TAGEDITOR_BATCH_ACTION", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def native_tag_editor_history_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_TAGEDITOR_HISTORY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_native_tag_editor_scan_api() -> Any:
    return load_lulynx_native()


load_native_tag_editor_scan_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_tag_editor_scan_api() -> Any:
    if native_tag_editor_scan_disabled():
        return None
    return _native_with_tag_editor_entrypoint("scan_tag_editor_dataset")


def native_tag_editor_batch_action_api() -> Any:
    if native_tag_editor_batch_action_disabled():
        return None
    return _native_with_tag_editor_entrypoint("apply_tag_editor_batch_action")


def native_tag_editor_history_api() -> Any:
    if native_tag_editor_history_disabled():
        return None
    return _native_with_tag_editor_entrypoint("list_tag_editor_history_snapshots")


def _native_with_tag_editor_entrypoint(entrypoint: str) -> Any:
    native = load_native_tag_editor_scan_api()
    if native is None or not hasattr(native, entrypoint):
        return None
    return native


def split_tags(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part and part.strip()]


def join_tags(tags: Sequence[str]) -> str:
    return ", ".join(tag for tag in tags if tag)


def dedupe_tags(tags: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = tag.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


def normalize_caption_extension(raw: str = "") -> str:
    value = str(raw or "").strip()
    if not value:
        return DEFAULT_CAPTION_EXTENSION
    if not value.startswith("."):
        value = "." + value
    return value


def relative_to(path: Path, parent: Path) -> str:
    return str(path.relative_to(parent)).replace("\\", "/")


def find_existing_caption_path(image_path: Path) -> tuple[Path, bool]:
    candidates = (
        image_path.parent / f"{image_path.name}.txt",
        image_path.with_suffix(".txt"),
        image_path.with_suffix(".caption"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, True
    return image_path.with_suffix(DEFAULT_CAPTION_EXTENSION), False


def resolve_caption_path(image_path: Path, preferred_extension: str = "") -> tuple[Path, bool]:
    existing_path, exists = find_existing_caption_path(image_path)
    if exists:
        return existing_path, True
    return image_path.with_suffix(normalize_caption_extension(preferred_extension)), False


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


@dataclass
class DatasetCaptionItem:
    image_path: Path
    relative_path: str
    caption_path: Path
    caption_exists: bool
    caption_text: str
    mtime: float
    caption_source: str = "file"

    @property
    def tags(self) -> List[str]:
        return split_tags(self.caption_text)


# Backwards-compatible private names used by older tests/patch points.
_APPROX_TOKEN_RE = APPROX_TOKEN_RE
_LEADING_FILENAME_INDEX_RE = LEADING_FILENAME_INDEX_RE
_find_existing_caption_path = find_existing_caption_path
_load_native_tag_editor_scan_api = load_native_tag_editor_scan_api
_native_tag_editor_batch_action_api = native_tag_editor_batch_action_api
_native_tag_editor_batch_action_disabled = native_tag_editor_batch_action_disabled
_native_tag_editor_history_api = native_tag_editor_history_api
_native_tag_editor_history_disabled = native_tag_editor_history_disabled
_native_tag_editor_scan_api = native_tag_editor_scan_api
_native_tag_editor_scan_disabled = native_tag_editor_scan_disabled
_read_text = read_text
_relative_to = relative_to
_resolve_caption_path = resolve_caption_path
