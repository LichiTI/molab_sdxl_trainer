# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Cross-dataset tag migration / synchronization utils."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def _split_tags(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part and part.strip()]


def migrate_tags_between_datasets(
    source_dir: str,
    target_dir: str,
    *,
    mapping: Dict[str, str],
    dry_run: bool = True,
    tag_editor: Any = None,
) -> Dict[str, Any]:
    """Migrate/synchronize tags from source to target dataset.

    Args:
        source_dir: source dataset root.
        target_dir: target dataset root.
        mapping: dict mapping source tags to target tags.
        dry_run: if True, only preview changes without writing.
        tag_editor: optional injected tag editor service.

    Returns:
        dict with ``affected_files``, ``changes_preview``, ``dry_run``.
    """
    if tag_editor is None:
        from core.services.tageditor_service_locator import tag_editor_service
        tag_editor = tag_editor_service()

    target_items = tag_editor._scan_dataset(Path(target_dir), recursive=True, load_caption_from_filename=False)

    changes: List[Dict[str, Any]] = []
    affected = 0

    for tgt in target_items:
        original_text = tgt.caption_text
        tags = [t.strip().lower() for t in _split_tags(original_text) if t.strip()]
        new_tags = list(tags)
        changed = False

        for src_tag, dst_tag in mapping.items():
            src_normalized = src_tag.strip().lower()
            dst_normalized = dst_tag.strip()
            if src_normalized in new_tags:
                new_tags[new_tags.index(src_normalized)] = dst_normalized
                changed = True

        if not changed:
            continue

        affected += 1
        new_text = ", ".join(new_tags)
        preview = {
            "file": tgt.relative_path,
            "original": original_text,
            "new": new_text,
        }
        changes.append(preview)

        if not dry_run:
            tgt.caption_path.write_text(new_text, encoding="utf-8")

    return {
        "dry_run": dry_run,
        "affected_files": affected,
        "changes_preview": changes[:50],
    }
