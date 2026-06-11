"""Dataset scanning helpers for TagEditorService."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List

from core.services.tag_editor_common import (
    IMAGE_EXTENSIONS,
    LEADING_FILENAME_INDEX_RE,
    DatasetCaptionItem,
    native_tag_editor_scan_api,
    read_text,
    relative_to,
    resolve_caption_path,
)


CaptionDeriver = Callable[[Path], str]


def derive_caption_from_filename(
    image_path: Path,
    *,
    filename_regex: str = "",
    filename_joiner: str = ", ",
) -> str:
    stem = LEADING_FILENAME_INDEX_RE.sub("", image_path.stem).strip()
    if not stem:
        return ""
    regex = str(filename_regex or "").strip()
    if regex:
        try:
            matches = re.findall(regex, stem)
        except re.error:
            matches = []
        tokens: List[str] = []
        for match in matches:
            if isinstance(match, tuple):
                tokens.extend(str(part).strip() for part in match if str(part).strip())
            else:
                value = str(match).strip()
                if value:
                    tokens.append(value)
        if tokens:
            return str(filename_joiner or ", ").join(tokens)
    return re.sub(r"_+", " ", stem).strip()


def scan_tag_editor_dataset(
    dataset_dir: Path,
    *,
    recursive: bool,
    caption_extension: str,
    load_caption_from_filename: bool = False,
    filename_regex: str = "",
    filename_joiner: str = ", ",
) -> List[DatasetCaptionItem]:
    native = None if load_caption_from_filename else native_tag_editor_scan_api()
    if native is not None:
        try:
            records = native.scan_tag_editor_dataset(str(dataset_dir), bool(recursive), str(caption_extension or ""))
            return [
                DatasetCaptionItem(
                    image_path=Path(str(record.get("image_path") or "")),
                    relative_path=str(record.get("relative_path") or ""),
                    caption_path=Path(str(record.get("caption_path") or "")),
                    caption_exists=bool(record.get("caption_exists", False)),
                    caption_text=str(record.get("caption_text") or ""),
                    mtime=float(record.get("mtime") or 0.0),
                    caption_source=str(record.get("caption_source") or "file"),
                )
                for record in records
                if isinstance(record, dict) and str(record.get("image_path") or "")
            ]
        except Exception:
            pass

    iterator = dataset_dir.rglob("*") if recursive else dataset_dir.glob("*")
    items: List[DatasetCaptionItem] = []
    for candidate in iterator:
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path, exists = resolve_caption_path(candidate, caption_extension)
        caption_text = read_text(caption_path) if exists else ""
        caption_source = "file" if exists else "none"
        if not exists and load_caption_from_filename:
            caption_text = derive_caption_from_filename(
                candidate,
                filename_regex=filename_regex,
                filename_joiner=filename_joiner,
            )
            caption_source = "filename" if caption_text else "none"
        items.append(
            DatasetCaptionItem(
                image_path=candidate,
                relative_path=relative_to(candidate, dataset_dir),
                caption_path=caption_path,
                caption_exists=exists,
                caption_text=caption_text,
                mtime=candidate.stat().st_mtime,
                caption_source=caption_source,
            )
        )
    return items
