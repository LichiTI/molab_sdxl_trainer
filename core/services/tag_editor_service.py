# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Clean-room dataset tag editor backend service."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from core.security import validate_path
from core.services.tag_editor_actions import (
    DESTRUCTIVE_ACTIONS,
    apply_caption_action,
    search_replace_tags,
)
from core.services.tag_editor_common import (
    APPROX_TOKEN_RE,
    SUPPORTED_CAPTION_EXTENSIONS,
    TAGEDITOR_HISTORY_MANIFEST,
    LEADING_FILENAME_INDEX_RE,
    DatasetCaptionItem,
    find_existing_caption_path,
    join_tags,
    native_tag_editor_scan_api,
    normalize_caption_extension,
    read_text,
    relative_to,
    resolve_caption_path,
    split_tags,
)
from core.services.tag_editor_dataset_view import (
    build_sidebar_stats,
    common_tags,
    filter_items,
    match_tag_logic,
    serialize_item,
    sort_items,
    tag_counts,
)
from core.services.tag_editor_history import (
    backup_manifest_path,
    create_backup_for_items,
    list_history_snapshots as list_tag_editor_history_snapshots,
    make_backup_name,
    normalize_selected_paths,
    read_backup_manifest,
    restore_history_snapshot as restore_tag_editor_history_snapshot,
)
from core.services.tag_editor_native_batch import apply_batch_action_native, preview_batch_action_for_paths_native
from core.services.tag_editor_scan import derive_caption_from_filename, scan_tag_editor_dataset


# Backwards-compatible private names used by older tests/patch points.
_APPROX_TOKEN_RE = APPROX_TOKEN_RE
_LEADING_FILENAME_INDEX_RE = LEADING_FILENAME_INDEX_RE
_find_existing_caption_path = find_existing_caption_path
_native_tag_editor_scan_api = native_tag_editor_scan_api
_read_text = read_text
_relative_to = relative_to
_resolve_caption_path = resolve_caption_path


class TagEditorService:
    """Stateless clean-room service for tag-editor style dataset operations."""

    _clip_tokenizer = None
    _clip_tokenizer_checked = False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "mode": "cleanroom",
            "filter_logic": ["AND", "OR"],
            "selection_modes": ["all", "inclusive", "exclusive"],
            "has_caption_modes": ["any", "yes", "no"],
            "sort_by": ["name", "mtime", "tag_count", "caption_length"],
            "batch_actions": [
                "append_tags",
                "prepend_tags",
                "replace_tags",
                "remove_tags",
                "search_replace_tags",
                "search_replace_caption",
                "sort_tags",
                "dedupe_tags",
                "set_caption",
                "truncate_tags_by_token_count",
            ],
            "caption_extensions": list(SUPPORTED_CAPTION_EXTENSIONS),
            "safe_delete_mode": "trash",
            "token_count_modes": ["clip_local", "approximate"],
            "history": {
                "create": True,
                "list": True,
                "restore": True,
                "auto_backup": True,
            },
        }

    @classmethod
    def _get_clip_tokenizer(cls):
        if cls._clip_tokenizer_checked:
            return cls._clip_tokenizer
        cls._clip_tokenizer_checked = True
        try:
            from transformers import CLIPTokenizer

            cls._clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                local_files_only=True,
            )
        except Exception:
            cls._clip_tokenizer = None
        return cls._clip_tokenizer

    @classmethod
    def _count_tokens(cls, text: str) -> tuple[int, str]:
        normalized = str(text or "").strip()
        tokenizer = cls._get_clip_tokenizer()
        if tokenizer is not None:
            try:
                token_ids = tokenizer(
                    normalized,
                    truncation=False,
                    add_special_tokens=False,
                )["input_ids"]
                return len(token_ids), "clip_local"
            except Exception:
                pass
        if not normalized:
            return 0, "approximate"
        return len(_APPROX_TOKEN_RE.findall(normalized)), "approximate"

    def analyze_caption_tokens(self, caption: str, *, max_token_count: int = 75) -> Dict[str, Any]:
        normalized = str(caption or "").strip()
        token_count, mode = self._count_tokens(normalized)
        tags = split_tags(normalized)
        breakdown: List[Dict[str, Any]] = []
        running_tags: List[str] = []
        for tag in tags:
            running_tags.append(tag)
            single_count, _ = self._count_tokens(tag)
            cumulative_count, _ = self._count_tokens(join_tags(running_tags))
            breakdown.append(
                {
                    "tag": tag,
                    "token_count": single_count,
                    "cumulative_token_count": cumulative_count,
                }
            )
        return {
            "caption": normalized,
            "token_count": token_count,
            "max_token_count": int(max_token_count),
            "over_limit": token_count > int(max_token_count),
            "tokenizer_mode": mode,
            "tag_breakdown": breakdown,
        }

    def _derive_caption_from_filename(
        self,
        image_path: Path,
        *,
        filename_regex: str = "",
        filename_joiner: str = ", ",
    ) -> str:
        return derive_caption_from_filename(
            image_path,
            filename_regex=filename_regex,
            filename_joiner=filename_joiner,
        )

    def load_dataset(
        self,
        directory: str,
        *,
        recursive: bool = True,
        caption_extension: str = "",
        load_caption_from_filename: bool = False,
        filename_regex: str = "",
        filename_joiner: str = ", ",
        limit: int = 200,
        offset: int = 0,
        sort_by: str = "name",
        sort_order: str = "asc",
        filename_query: str = "",
        caption_query: str = "",
        positive_tags: Optional[Sequence[str]] = None,
        positive_logic: str = "OR",
        negative_tags: Optional[Sequence[str]] = None,
        negative_logic: str = "OR",
        selected_paths: Optional[Sequence[str]] = None,
        selection_mode: str = "all",
        has_caption: str = "any",
        top_tags_limit: int = 50,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        items = self._scan_dataset(
            dataset_dir,
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=load_caption_from_filename,
            filename_regex=filename_regex,
            filename_joiner=filename_joiner,
        )
        filtered = self._filter_items(
            items,
            filename_query=filename_query,
            caption_query=caption_query,
            positive_tags=positive_tags or [],
            positive_logic=positive_logic,
            negative_tags=negative_tags or [],
            negative_logic=negative_logic,
            selected_paths=selected_paths or [],
            selection_mode=selection_mode,
            has_caption=has_caption,
        )
        sorted_items = self._sort_items(filtered, sort_by=sort_by, sort_order=sort_order)
        page = sorted_items[offset : offset + limit]
        tag_counts = self._tag_counts(filtered)
        tokenizer_mode = self._count_tokens("")[1]
        return {
            "directory": str(dataset_dir),
            "recursive": recursive,
            "total": len(items),
            "filtered_total": len(filtered),
            "offset": offset,
            "limit": limit,
            "items": [self._serialize_item(item) for item in page],
            "common_tags": self._common_tags(filtered),
            "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(max(0, top_tags_limit))],
            "summary": {
                "captioned_count": sum(1 for item in filtered if item.caption_text),
                "uncaptioned_count": sum(1 for item in filtered if not item.caption_text),
                "selected_count": len(filtered),
            },
            "tokenizer_mode": tokenizer_mode,
            "capabilities": self.capabilities(),
        }

    def save_caption(
        self,
        *,
        image_path: str,
        caption: str,
        caption_extension: str = "",
        dataset_dir: str = "",
        create_backup: bool = False,
        backup_label: str = "save_caption",
    ) -> Dict[str, Any]:
        image = validate_path(image_path, must_exist=True, allow_files=True, allow_dirs=False)
        caption_path, _ = _resolve_caption_path(image, caption_extension)
        dataset_root = validate_path(dataset_dir, must_exist=True, allow_dirs=True, allow_files=False) if dataset_dir else image.parent
        backup_name = ""
        if create_backup:
            try:
                relative = _relative_to(image, dataset_root)
            except ValueError:
                relative = image.name
            backup_name = self._create_backup_for_items(
                dataset_root,
                [
                    DatasetCaptionItem(
                        image_path=image,
                        relative_path=relative,
                        caption_path=caption_path,
                        caption_exists=caption_path.exists(),
                        caption_text=_read_text(caption_path) if caption_path.exists() else "",
                        mtime=image.stat().st_mtime,
                        caption_source="file" if caption_path.exists() else "none",
                    )
                ],
                label=backup_label,
                operation="save_caption",
            )
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(str(caption or "").strip(), encoding="utf-8")
        return {
            "image_path": str(image),
            "caption_path": str(caption_path),
            "caption_length": len(str(caption or "").strip()),
            "backup_name": backup_name,
        }

    def save_captions_batch(
        self,
        updates: Sequence[Dict[str, Any]],
        *,
        caption_extension: str = "",
        dataset_dir: str = "",
        create_backup: bool = False,
        backup_label: str = "save_batch",
    ) -> Dict[str, Any]:
        dataset_root = validate_path(dataset_dir, must_exist=True, allow_dirs=True, allow_files=False) if dataset_dir else None
        backup_items: List[DatasetCaptionItem] = []
        if create_backup:
            for update in updates:
                try:
                    image = validate_path(
                        str(update.get("image_path", "") or ""),
                        must_exist=True,
                        allow_files=True,
                        allow_dirs=False,
                    )
                    caption_path, _ = _resolve_caption_path(
                        image,
                        str(update.get("caption_extension", "") or caption_extension),
                    )
                    root = dataset_root or image.parent
                    try:
                        relative = _relative_to(image, root)
                    except ValueError:
                        relative = image.name
                    backup_items.append(
                        DatasetCaptionItem(
                            image_path=image,
                            relative_path=relative,
                            caption_path=caption_path,
                            caption_exists=caption_path.exists(),
                            caption_text=_read_text(caption_path) if caption_path.exists() else "",
                            mtime=image.stat().st_mtime,
                            caption_source="file" if caption_path.exists() else "none",
                        )
                    )
                except Exception:
                    continue
        backup_name = ""
        if backup_items:
            backup_name = self._create_backup_for_items(
                dataset_root or backup_items[0].image_path.parent,
                backup_items,
                label=backup_label,
                operation="save_batch",
            )
        success = 0
        failed = 0
        errors: List[str] = []
        for update in updates:
            try:
                self.save_caption(
                    image_path=str(update.get("image_path", "") or ""),
                    caption=str(update.get("caption", "") or ""),
                    caption_extension=str(update.get("caption_extension", "") or caption_extension),
                    dataset_dir=str(dataset_root or ""),
                    create_backup=False,
                )
                success += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{update.get('image_path', '')}: {exc}")
        return {"success": success, "failed": failed, "errors": errors, "backup_name": backup_name}

    def get_sidebar_stats(
        self,
        directory: str,
        *,
        recursive: bool = True,
        caption_extension: str = "",
        load_caption_from_filename: bool = False,
        filename_regex: str = "",
        filename_joiner: str = ", ",
        filename_query: str = "",
        caption_query: str = "",
        positive_tags: Optional[Sequence[str]] = None,
        positive_logic: str = "OR",
        negative_tags: Optional[Sequence[str]] = None,
        negative_logic: str = "OR",
        selected_paths: Optional[Sequence[str]] = None,
        selection_mode: str = "all",
        has_caption: str = "any",
        top_tags_limit: int = 50,
        rare_tags_limit: int = 20,
        token_limit: int = 75,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        items = self._scan_dataset(
            dataset_dir,
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=load_caption_from_filename,
            filename_regex=filename_regex,
            filename_joiner=filename_joiner,
        )
        filtered = self._filter_items(
            items,
            filename_query=filename_query,
            caption_query=caption_query,
            positive_tags=positive_tags or [],
            positive_logic=positive_logic,
            negative_tags=negative_tags or [],
            negative_logic=negative_logic,
            selected_paths=selected_paths or [],
            selection_mode=selection_mode,
            has_caption=has_caption,
        )
        return self._build_sidebar_stats(
            filtered,
            top_tags_limit=top_tags_limit,
            rare_tags_limit=rare_tags_limit,
            token_limit=token_limit,
            directory=str(dataset_dir),
        )

    def create_history_snapshot(
        self,
        directory: str,
        *,
        image_paths: Optional[Sequence[str]] = None,
        recursive: bool = True,
        caption_extension: str = "",
        filter_payload: Optional[Dict[str, Any]] = None,
        label: str = "",
        operation: str = "manual_snapshot",
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        items = self._resolve_target_items(
            dataset_dir,
            image_paths=image_paths or [],
            recursive=recursive,
            caption_extension=caption_extension,
            filter_payload=filter_payload or {},
        )
        backup_name = self._create_backup_for_items(dataset_dir, items, label=label, operation=operation) if items else ""
        return {"backup_name": backup_name, "target_count": len(items)}

    def list_history_snapshots(self, directory: str, *, limit: int = 50) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        return list_tag_editor_history_snapshots(dataset_dir, limit=limit)

    def restore_history_snapshot(
        self,
        directory: str,
        *,
        backup_name: str,
        image_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        return restore_tag_editor_history_snapshot(
            dataset_dir,
            backup_name=backup_name,
            image_paths=image_paths or [],
        )

    def apply_batch_action(
        self,
        directory: str,
        *,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        image_paths: Optional[Sequence[str]] = None,
        recursive: bool = True,
        caption_extension: str = "",
        create_backup: bool = False,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        items = self._resolve_target_items(
            dataset_dir,
            image_paths=image_paths or [],
            recursive=recursive,
            caption_extension=caption_extension,
            filter_payload=filter_payload or {},
        )
        params = dict(params or {})
        backup_name = (
            self._create_backup_for_items(
                dataset_dir,
                items,
                label=str(params.get("history_label", "") or action),
                operation=f"batch_{str(action or '').strip().lower() or 'edit'}",
            )
            if create_backup and items
            else ""
        )
        modified = 0
        unchanged = 0
        samples: List[Dict[str, Any]] = []
        tag_counts = self._tag_counts(items)
        native_result = apply_batch_action_native(
            items,
            action=action,
            params=params,
            tag_counts=tag_counts,
            sample_limit=20,
            include_changes=True,
        )
        if native_result is not None:
            for change in native_result.get("changes", []):
                caption_path = Path(str(change.get("caption_path", "") or ""))
                caption_path.parent.mkdir(parents=True, exist_ok=True)
                caption_path.write_text(str(change.get("after", "") or ""), encoding="utf-8")
            return {
                "target_count": int(native_result.get("target_count", len(items)) or len(items)),
                "modified_count": int(native_result.get("modified_count", 0) or 0),
                "unchanged_count": int(native_result.get("unchanged_count", 0) or 0),
                "backup_name": backup_name,
                "samples": list(native_result.get("samples", []) or []),
                "native_provider": str(native_result.get("native_provider", "") or ""),
            }
        for item in items:
            before = item.caption_text
            after = self._apply_action(before, action=action, params=params, tag_counts=tag_counts)
            if before == after:
                unchanged += 1
                continue
            item.caption_path.parent.mkdir(parents=True, exist_ok=True)
            item.caption_path.write_text(after, encoding="utf-8")
            modified += 1
            if len(samples) < 20:
                samples.append({"image_path": str(item.image_path), "caption_path": str(item.caption_path), "before": before, "after": after})
        return {"target_count": len(items), "modified_count": modified, "unchanged_count": unchanged, "backup_name": backup_name, "samples": samples}

    def preview_batch_action(
        self,
        directory: str,
        *,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        image_paths: Optional[Sequence[str]] = None,
        recursive: bool = True,
        caption_extension: str = "",
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        raw_image_paths = list(image_paths or [])
        params = dict(params or {})
        if raw_image_paths and not (filter_payload or {}):
            validated_image_paths = [
                str(validate_path(raw_path, must_exist=True, allow_files=True, allow_dirs=False))
                for raw_path in raw_image_paths
            ]
            native_paths_preview = preview_batch_action_for_paths_native(
                dataset_dir=str(dataset_dir),
                image_paths=validated_image_paths,
                caption_extension=caption_extension,
                action=action,
                params=params,
                sample_limit=50,
            )
            if native_paths_preview is not None:
                operation_id = hashlib.sha1(
                    f"{dataset_dir}|{action}|{params}|{validated_image_paths}".encode("utf-8")
                ).hexdigest()[:16]
                modified = int(native_paths_preview.get("modified_count", 0) or 0)
                return {
                    "target_count": int(native_paths_preview.get("target_count", len(raw_image_paths)) or len(raw_image_paths)),
                    "modified_count": modified,
                    "unchanged_count": int(native_paths_preview.get("unchanged_count", 0) or 0),
                    "samples": list(native_paths_preview.get("samples", []) or []),
                    "operation_id": operation_id,
                    "requires_backup": bool(modified and str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS),
                    "native_provider": str(native_paths_preview.get("native_provider", "") or ""),
                }
        items = self._resolve_target_items(
            dataset_dir,
            image_paths=raw_image_paths,
            recursive=recursive,
            caption_extension=caption_extension,
            filter_payload=filter_payload or {},
        )
        modified = 0
        unchanged = 0
        samples: List[Dict[str, Any]] = []
        tag_counts = self._tag_counts(items)
        native_result = apply_batch_action_native(
            items,
            action=action,
            params=params,
            tag_counts=tag_counts,
            sample_limit=50,
            include_changes=False,
        )
        if native_result is not None:
            modified = int(native_result.get("modified_count", 0) or 0)
            unchanged = int(native_result.get("unchanged_count", 0) or 0)
            samples = list(native_result.get("samples", []) or [])
            operation_id = hashlib.sha1(f"{dataset_dir}|{action}|{params}|{[item.relative_path for item in items]}".encode("utf-8")).hexdigest()[:16]
            return {
                "target_count": len(items),
                "modified_count": modified,
                "unchanged_count": unchanged,
                "samples": samples,
                "operation_id": operation_id,
                "requires_backup": bool(modified and str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS),
                "risk_summary": {
                    "risk_level": "medium" if modified and str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS else ("low" if modified else "none"),
                    "destructive": str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS,
                    "preview_only": True,
                },
                "native_provider": str(native_result.get("native_provider", "") or ""),
            }
        for item in items:
            before = item.caption_text
            after = self._apply_action(before, action=action, params=params, tag_counts=tag_counts)
            if before == after:
                unchanged += 1
                continue
            modified += 1
            if len(samples) < 50:
                samples.append({"image_path": str(item.image_path), "caption_path": str(item.caption_path), "before": before, "after": after})
        operation_id = hashlib.sha1(f"{dataset_dir}|{action}|{params}|{[item.relative_path for item in items]}".encode("utf-8")).hexdigest()[:16]
        return {
            "target_count": len(items),
            "modified_count": modified,
            "unchanged_count": unchanged,
            "samples": samples,
            "operation_id": operation_id,
            "requires_backup": bool(modified and str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS),
            "risk_summary": {
                "risk_level": "medium" if modified and str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS else ("low" if modified else "none"),
                "destructive": str(action or "").strip().lower() in DESTRUCTIVE_ACTIONS,
                "preview_only": True,
            },
        }

    def interrogate_image(self, *, image_path: str, method: str = "wd14", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        image = validate_path(image_path, must_exist=True, allow_files=True, allow_dirs=False)
        config = dict(config or {})
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")
            pil_image.load()
        if method == "gemini":
            from core.gemini_tagger import GeminiTagger

            api_key = str(config.get("api_key", "") or "")
            if not api_key:
                raise ValueError("gemini api_key is required")
            tagger = GeminiTagger(
                api_key=api_key,
                base_url=str(config.get("base_url", "") or None) or None,
                proxy=str(config.get("proxy", "") or None) or None,
                model=str(config.get("model", "gemini-1.5-flash") or "gemini-1.5-flash"),
                safety_none=bool(config.get("safety_none", True)),
            )
            if config.get("prompt"):
                tagger.set_prompt(str(config.get("prompt", "")))
            if config.get("prefix_tags"):
                tagger.set_prefix_tags(str(config.get("prefix_tags", "")))
            caption = tagger.tag_image(pil_image)
            return {"method": method, "caption": caption or "", "tags": split_tags(caption or ""), "scores": {}}

        from core.wd14_tagger import WD14Tagger

        tagger = WD14Tagger(model_name=str(config.get("model", "wd-convnext-v3") or "wd-convnext-v3"))
        ratings, tags = tagger.tag_image(
            pil_image,
            threshold=float(config.get("threshold", 0.35) or 0.35),
            character_threshold=float(config.get("character_threshold", 0.85) or 0.85),
            exclude_tags=list(config.get("exclude_tags", []) or []),
            replace_underscore=bool(config.get("replace_underscore", True)),
        )
        try:
            tagger.unload()
        except Exception:
            pass
        return {"method": method, "caption": join_tags(list(tags.keys())), "tags": list(tags.keys()), "scores": tags, "ratings": ratings}

    def move_files(self, *, directory: str, image_paths: Sequence[str], target_dir: str, keep_relative_structure: bool = True) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        target_root = validate_path(target_dir, must_exist=False, allow_dirs=True, allow_files=False)
        target_root.mkdir(parents=True, exist_ok=True)
        moved = 0
        for raw_path in image_paths:
            image = validate_path(raw_path, must_exist=True, allow_files=True, allow_dirs=False)
            if keep_relative_structure:
                try:
                    relative_parent = image.parent.relative_to(dataset_dir)
                except ValueError:
                    relative_parent = Path()
            else:
                relative_parent = Path()
            destination_dir = target_root / relative_parent
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_image = destination_dir / image.name
            caption_path, caption_exists = _find_existing_caption_path(image)
            shutil.move(str(image), str(destination_image))
            if caption_exists and caption_path.exists():
                shutil.move(str(caption_path), str(destination_dir / caption_path.name))
            moved += 1
        return {"moved_count": moved, "skipped_count": 0, "target_dir": str(target_root)}

    def delete_files(self, *, directory: str, image_paths: Sequence[str]) -> Dict[str, Any]:
        dataset_dir = validate_path(directory, must_exist=True, allow_dirs=True, allow_files=False)
        trash_root = dataset_dir / ".trash" / datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_root.mkdir(parents=True, exist_ok=True)
        return self.move_files(directory=str(dataset_dir), image_paths=image_paths, target_dir=str(trash_root), keep_relative_structure=True) | {"trash_dir": str(trash_root)}

    def _scan_dataset(
        self,
        dataset_dir: Path,
        *,
        recursive: bool,
        caption_extension: str,
        load_caption_from_filename: bool = False,
        filename_regex: str = "",
        filename_joiner: str = ", ",
    ) -> List[DatasetCaptionItem]:
        return scan_tag_editor_dataset(
            dataset_dir,
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=load_caption_from_filename,
            filename_regex=filename_regex,
            filename_joiner=filename_joiner,
        )

    def _filter_items(
        self,
        items: Sequence[DatasetCaptionItem],
        *,
        filename_query: str,
        caption_query: str,
        positive_tags: Sequence[str],
        positive_logic: str,
        negative_tags: Sequence[str],
        negative_logic: str,
        selected_paths: Sequence[str],
        selection_mode: str,
        has_caption: str,
    ) -> List[DatasetCaptionItem]:
        return filter_items(
            items,
            filename_query=filename_query,
            caption_query=caption_query,
            positive_tags=positive_tags,
            positive_logic=positive_logic,
            negative_tags=negative_tags,
            negative_logic=negative_logic,
            selected_paths=selected_paths,
            selection_mode=selection_mode,
            has_caption=has_caption,
        )

    def _match_tag_logic(self, tags_lower: set[str], filter_tags: set[str], logic: str) -> bool:
        return match_tag_logic(tags_lower, filter_tags, logic)

    def _sort_items(self, items: Sequence[DatasetCaptionItem], *, sort_by: str, sort_order: str) -> List[DatasetCaptionItem]:
        return sort_items(items, sort_by=sort_by, sort_order=sort_order)

    def _serialize_item(self, item: DatasetCaptionItem) -> Dict[str, Any]:
        return serialize_item(item, self._count_tokens)

    def _tag_counts(self, items: Sequence[DatasetCaptionItem]) -> Counter[str]:
        return tag_counts(items)

    def _common_tags(self, items: Sequence[DatasetCaptionItem]) -> List[str]:
        return common_tags(items)

    def _resolve_target_items(
        self,
        dataset_dir: Path,
        *,
        image_paths: Sequence[str],
        recursive: bool,
        caption_extension: str,
        filter_payload: Dict[str, Any],
    ) -> List[DatasetCaptionItem]:
        if image_paths:
            results: List[DatasetCaptionItem] = []
            for raw_path in image_paths:
                image = validate_path(raw_path, must_exist=True, allow_files=True, allow_dirs=False)
                caption_path, exists = _resolve_caption_path(image, caption_extension)
                try:
                    relative = _relative_to(image, dataset_dir)
                except ValueError:
                    relative = image.name
                results.append(
                    DatasetCaptionItem(
                        image_path=image,
                        relative_path=relative,
                        caption_path=caption_path,
                        caption_exists=exists,
                        caption_text=_read_text(caption_path) if exists else "",
                        mtime=image.stat().st_mtime,
                    )
                )
            return results
        listing = self.load_dataset(
            str(dataset_dir),
            recursive=recursive,
            caption_extension=caption_extension,
            load_caption_from_filename=bool(filter_payload.get("load_caption_from_filename", False)),
            filename_regex=str(filter_payload.get("filename_regex", "") or ""),
            filename_joiner=str(filter_payload.get("filename_joiner", ", ") or ", "),
            limit=1_000_000,
            offset=0,
            sort_by=str(filter_payload.get("sort_by", "name") or "name"),
            sort_order=str(filter_payload.get("sort_order", "asc") or "asc"),
            filename_query=str(filter_payload.get("filename_query", "") or ""),
            caption_query=str(filter_payload.get("caption_query", "") or ""),
            positive_tags=list(filter_payload.get("positive_tags", []) or []),
            positive_logic=str(filter_payload.get("positive_logic", "OR") or "OR"),
            negative_tags=list(filter_payload.get("negative_tags", []) or []),
            negative_logic=str(filter_payload.get("negative_logic", "OR") or "OR"),
            selected_paths=list(filter_payload.get("selected_paths", []) or []),
            selection_mode=str(filter_payload.get("selection_mode", "all") or "all"),
            has_caption=str(filter_payload.get("has_caption", "any") or "any"),
        )
        return [
            DatasetCaptionItem(
                image_path=Path(item["image_path"]),
                relative_path=str(item["relative_path"]),
                caption_path=Path(item["caption_path"]),
                caption_exists=bool(item["caption_exists"]),
                caption_text=str(item["caption"] or ""),
                mtime=float(item["mtime"]),
                caption_source=str(item.get("caption_source", "file") or "file"),
            )
            for item in listing["items"]
        ]

    def _make_backup_name(self, operation: str = "") -> str:
        return make_backup_name(operation)

    def _backup_manifest_path(self, backup_root: Path) -> Path:
        return backup_manifest_path(backup_root)

    def _read_backup_manifest(self, backup_root: Path) -> Dict[str, Any]:
        return read_backup_manifest(backup_root)

    def _normalize_selected_paths(self, dataset_dir: Path, image_paths: Sequence[str]) -> set[str]:
        return normalize_selected_paths(dataset_dir, image_paths)

    def _create_backup_for_items(
        self,
        dataset_dir: Path,
        items: Sequence[DatasetCaptionItem],
        *,
        label: str = "",
        operation: str = "",
    ) -> str:
        return create_backup_for_items(dataset_dir, items, label=label, operation=operation)

    def _build_sidebar_stats(
        self,
        items: Sequence[DatasetCaptionItem],
        *,
        top_tags_limit: int,
        rare_tags_limit: int,
        token_limit: int,
        directory: str,
    ) -> Dict[str, Any]:
        return build_sidebar_stats(
            items,
            top_tags_limit=top_tags_limit,
            rare_tags_limit=rare_tags_limit,
            token_limit=token_limit,
            directory=directory,
            token_counter=self._count_tokens,
        )

    def _apply_action(self, caption: str, *, action: str, params: Dict[str, Any], tag_counts: Counter[str]) -> str:
        return apply_caption_action(
            caption,
            action=action,
            params=params,
            tag_counts=tag_counts,
            token_counter=self._count_tokens,
        )

    def _search_replace_tags(self, tags: Sequence[str], *, search_text: str, replace_text: str, use_regex: bool) -> List[str]:
        return search_replace_tags(tags, search_text=search_text, replace_text=replace_text, use_regex=use_regex)

