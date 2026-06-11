# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Dataset manifest and diff contracts for tag workstation backends."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from backend.core.services.native_module_loader import load_lulynx_native
from core.services.tag_editor_service import TagEditorService, split_tags
from core.services.tag_analysis_service import TagAnalysisService


def native_dataset_manifest_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_DATASET_MANIFEST", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_native_dataset_manifest_api() -> Any:
    return load_lulynx_native()


load_native_dataset_manifest_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_dataset_manifest_api() -> Any:
    if native_dataset_manifest_disabled():
        return None
    native = load_native_dataset_manifest_api()
    if native is None or not hasattr(native, "diff_dataset_manifests"):
        return None
    return native


class DatasetManifestService:
    def __init__(self, tag_editor: TagEditorService | None = None, analysis_service: TagAnalysisService | None = None) -> None:
        self.tag_editor = tag_editor or TagEditorService()
        self.analysis_service = analysis_service or TagAnalysisService(self.tag_editor)

    def build_manifest(
        self,
        directory: str,
        *,
        recursive: bool = True,
        caption_extension: str = "",
        route_family: str = "",
        trigger_words: List[str] | None = None,
        max_token_count: int = 75,
    ) -> Dict[str, Any]:
        dataset_dir = Path(directory).resolve()
        items = self.tag_editor._scan_dataset(
            dataset_dir,
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=False,
        )
        tag_counts: Counter[str] = Counter()
        rows: List[Dict[str, Any]] = []
        captioned = 0
        token_counts: List[int] = []
        for item in items:
            caption_hash = hashlib.sha1(item.caption_text.encode("utf-8")).hexdigest() if item.caption_text else ""
            image_stat = item.image_path.stat()
            caption_stat = item.caption_path.stat() if item.caption_path.exists() else None
            tags = split_tags(item.caption_text)
            token_info = self.tag_editor.analyze_caption_tokens(item.caption_text, max_token_count=max_token_count)
            tag_counts.update(tags)
            if item.caption_text:
                captioned += 1
            token_counts.append(int(token_info["token_count"]))
            rows.append({
                "image_relative_path": item.relative_path,
                "image_size_bytes": image_stat.st_size,
                "image_mtime": image_stat.st_mtime,
                "caption_relative_path": self._safe_relative_path(item.caption_path, dataset_dir),
                "caption_exists": item.caption_exists,
                "caption_hash": caption_hash,
                "caption_length": len(item.caption_text),
                "caption_mtime": caption_stat.st_mtime if caption_stat else 0,
                "caption_source": item.caption_source,
                "tag_count": len(tags),
                "token_count": int(token_info["token_count"]),
            })
        signature = self._manifest_signature(rows)
        consistency = self._trigger_caption_consistency(
            items=rows,
            tag_counts=tag_counts,
            trigger_words=trigger_words or [],
            token_counts=token_counts,
            max_token_count=max_token_count,
        )
        return {
            "schema_version": 1,
            "dataset_path": str(dataset_dir),
            "dataset_signature": signature,
            "recursive": recursive,
            "caption_extension": caption_extension,
            "route_family": route_family or "generic",
            "summary": {
                "image_count": len(items),
                "captioned_count": captioned,
                "missing_caption_count": len(items) - sum(1 for item in items if item.caption_exists),
                "caption_coverage": round((captioned / len(items)) if items else 0.0, 4),
                "unique_tag_count": len(tag_counts),
                "avg_token_count": round((sum(token_counts) / len(token_counts)) if token_counts else 0.0, 2),
            },
            "tag_distribution": {
                "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(50)],
            },
            "bucket_preview": self._bucket_preview(rows),
            "trigger_caption_consistency": consistency,
            "items": rows,
        }

    def diff_manifests(self, old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> Dict[str, Any]:
        native_diff = self._diff_manifests_native(old_manifest, new_manifest)
        if native_diff is not None:
            return native_diff
        old_rows = {row.get("image_relative_path"): row for row in old_manifest.get("items", []) if row.get("image_relative_path")}
        new_rows = {row.get("image_relative_path"): row for row in new_manifest.get("items", []) if row.get("image_relative_path")}
        added = sorted(path for path in new_rows if path not in old_rows)
        removed = sorted(path for path in old_rows if path not in new_rows)
        changed: List[Dict[str, Any]] = []
        for path in sorted(set(old_rows) & set(new_rows)):
            old = old_rows[path]
            new = new_rows[path]
            fields = [
                field for field in ("caption_hash", "caption_exists", "caption_length", "tag_count", "token_count")
                if old.get(field) != new.get(field)
            ]
            if fields:
                changed.append({"image_relative_path": path, "changed_fields": fields, "old": old, "new": new})
        return {
            "schema_version": 1,
            "old_signature": old_manifest.get("dataset_signature", ""),
            "new_signature": new_manifest.get("dataset_signature", ""),
            "summary": {
                "added_count": len(added),
                "removed_count": len(removed),
                "changed_count": len(changed),
            },
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def _diff_manifests_native(self, old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> Dict[str, Any] | None:
        native = native_dataset_manifest_api()
        if native is None:
            return None
        try:
            result = native.diff_dataset_manifests(
                json.dumps(old_manifest, ensure_ascii=False),
                json.dumps(new_manifest, ensure_ascii=False),
            )
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    def _safe_relative_path(self, path: Path, parent: Path) -> str:
        try:
            return str(path.relative_to(parent)).replace("\\", "/")
        except ValueError:
            return path.name

    def _manifest_signature(self, rows: List[Dict[str, Any]]) -> str:
        material = "\n".join(
            f"{row['image_relative_path']}|{row['image_size_bytes']}|{int(row['image_mtime'])}|{row['caption_hash']}|{row['caption_length']}"
            for row in sorted(rows, key=lambda item: item["image_relative_path"])
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()

    def _bucket_preview(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        buckets: Counter[str] = Counter()
        for row in rows:
            tags = int(row.get("tag_count", 0) or 0)
            if tags <= 0:
                buckets["empty"] += 1
            elif tags <= 8:
                buckets["short"] += 1
            elif tags <= 30:
                buckets["medium"] += 1
            else:
                buckets["dense"] += 1
        return {"caption_tag_count_buckets": dict(buckets)}

    def _trigger_caption_consistency(
        self,
        *,
        items: List[Dict[str, Any]],
        tag_counts: Counter[str],
        trigger_words: List[str],
        token_counts: List[int],
        max_token_count: int,
    ) -> Dict[str, Any]:
        trigger_set = {str(word).strip().lower() for word in trigger_words if str(word).strip()}
        trigger_counts = {tag: count for tag, count in tag_counts.items() if tag.lower() in trigger_set}
        image_count = len(items)
        dense = sum(1 for row in items if int(row.get("token_count", 0) or 0) > max_token_count)
        sparse = sum(1 for row in items if row.get("caption_exists") and int(row.get("token_count", 0) or 0) <= 2)
        return {
            "trigger_words": sorted(trigger_set),
            "trigger_counts": trigger_counts,
            "trigger_coverage": round((sum(trigger_counts.values()) / image_count) if image_count else 0.0, 4),
            "dense_caption_count": dense,
            "sparse_caption_count": sparse,
            "max_token_count": max_token_count,
            "token_count_range": {
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
            },
        }
