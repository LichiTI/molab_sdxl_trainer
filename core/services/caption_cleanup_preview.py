"""Caption cleanup preview helpers shared by legacy routes."""

from __future__ import annotations

import re
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.services.native_module_loader import load_lulynx_native


def split_tags(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part and part.strip()]


def join_tags(tags: list[str]) -> str:
    return ", ".join(tag for tag in tags if tag)


def dedupe_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = str(tag or "").strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


def normalize_cleanup_params(params: dict[str, Any], *, for_tag_manager: bool = False) -> dict[str, Any]:
    caption_ext = str(params.get("caption_extension", ".txt") or ".txt")
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext
    max_tag_len_default = 0 if for_tag_manager else 100
    remove_parens_default = False if for_tag_manager else True
    return {
        "remove_parens": bool(params.get("remove_parens", remove_parens_default)),
        "dedup": bool(params.get("dedup", True) or params.get("dedupe_tags", False)),
        "max_tag_len": int(params.get("max_tag_len", max_tag_len_default) or 0),
        "sort_tags": bool(params.get("sort_tags", False)),
        "collapse_whitespace": bool(params.get("collapse_whitespace", False)),
        "replace_underscore": bool(params.get("replace_underscore", False)),
        "prepend_tags": str(params.get("prepend_tags", "") or ""),
        "append_tags": str(params.get("append_tags", "") or ""),
        "remove_tags": str(params.get("remove_tags", "") or ""),
        "search_text": str(params.get("search_text", "") or ""),
        "replace_text": str(params.get("replace_text", "") or ""),
        "use_regex": bool(params.get("use_regex", False)),
        "create_backup": bool(params.get("create_backup_before_apply", False)),
        "caption_ext": caption_ext,
        "recursive": bool(params.get("recursive", True)),
        "blacklist_tags": parse_caption_tag_list(params.get("blacklist_tags", "")),
        "alias_map": build_alias_map(params.get("alias_map", "")),
        "bulk_replace_rules": parse_rewrite_rules(params.get("bulk_replace_rules", "")),
        "stats_top_limit": max(1, min(200, int(params.get("stats_top_limit", 15) or 15))),
    }


def native_caption_cleanup_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_CAPTION_CLEANUP", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_native_caption_cleanup_api() -> Any:
    return load_lulynx_native()


load_native_caption_cleanup_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_caption_cleanup_api() -> Any:
    if native_caption_cleanup_disabled():
        return None
    native = load_native_caption_cleanup_api()
    if not hasattr(native, "preview_caption_cleanup"):
        return None
    return native


def native_caption_cleanup_supported(cleanup: dict[str, Any]) -> bool:
    return not bool(cleanup.get("use_regex", False))


def collect_cleanup_preview(dataset_dir: Path, cleanup: dict[str, Any], *, include_stats: bool = False) -> dict[str, Any]:
    native_preview = preview_cleanup_dataset_native(dataset_dir, cleanup, include_stats=include_stats, sample_limit=20)
    if native_preview is not None:
        native_preview.pop("changes", None)
        return native_preview
    files = collect_caption_files(dataset_dir, cleanup)
    records = read_caption_cleanup_records(files)
    native_preview = preview_cleanup_records_native(records, cleanup, include_stats=include_stats, sample_limit=20)
    if native_preview is not None:
        native_preview.pop("changes", None)
        return native_preview
    total = 0
    changed = 0
    unchanged = 0
    samples = []
    before_captions: list[str] = []
    after_captions: list[str] = []
    for record in records:
        txt_file = Path(str(record.get("path", "") or ""))
        total += 1
        before = str(record.get("caption", "") or "")
        after = cleanup_caption_from_config(before, cleanup)
        before_captions.append(before)
        after_captions.append(after)
        if before != after:
            changed += 1
            if len(samples) < 20:
                samples.append({"file": txt_file.name, "before": before, "after": after, "path": str(txt_file)})
        else:
            unchanged += 1
    payload: dict[str, Any] = {
        "summary": {
            "total_file_count": total,
            "changed_file_count": changed,
            "unchanged_file_count": unchanged,
        },
        "samples": samples,
    }
    if include_stats:
        payload["rules"] = {
            "alias_count": len(cleanup.get("alias_map") or {}),
            "blacklist_count": len(cleanup.get("blacklist_tags") or []),
            "bulk_replace_count": len(cleanup.get("bulk_replace_rules") or []),
        }
        top_limit = int(cleanup.get("stats_top_limit", 15) or 15)
        payload["stats"] = {
            "top_limit": cleanup.get("stats_top_limit", 15),
            "before": build_caption_frequency_stats(before_captions, top_limit=top_limit),
            "after": build_caption_frequency_stats(after_captions, top_limit=top_limit),
        }
    return payload


def apply_cleanup_to_dataset(
    dataset_dir: Path,
    cleanup: dict[str, Any],
    *,
    progress_callback: Callable[..., None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    invalidate_cache: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    files = collect_caption_files(dataset_dir, cleanup)
    backup_dir = get_backup_dir(dataset_dir)
    backup_name = ""
    if cleanup["create_backup"]:
        backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_sub = backup_dir / backup_name
        backup_sub.mkdir(parents=True, exist_ok=True)
        for txt_file in files:
            rel = txt_file.relative_to(dataset_dir)
            dest = backup_sub / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt_file, dest)

    changed = 0
    samples = []
    total = max(1, len(files))
    progress = progress_callback or (lambda *_args, **_kwargs: None)
    records = read_caption_cleanup_records(files)
    native_preview = preview_cleanup_records_native(records, cleanup, include_stats=False, sample_limit=20)
    native_changes = None
    if native_preview is not None:
        native_changes = {
            str(change.get("path", "") or ""): change
            for change in native_preview.get("changes", [])
            if str(change.get("path", "") or "")
        }
    for idx, txt_file in enumerate(files, start=1):
        if cancel_check and cancel_check():
            break
        if native_changes is not None:
            change = native_changes.get(str(txt_file))
            if change is None:
                progress(idx, total)
                continue
            before = str(change.get("before", "") or "")
            after = str(change.get("after", "") or "")
        else:
            before = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            after = cleanup_caption_from_config(before, cleanup)
        if before != after:
            txt_file.write_text(after, encoding="utf-8")
            changed += 1
            if len(samples) < 20:
                samples.append({"file": txt_file.name, "before": before, "after": after, "path": str(txt_file)})
        progress(idx, total)

    if invalidate_cache is not None:
        invalidate_cache(str(dataset_dir))
    return {
        "changed_count": changed,
        "processed_count": len(files),
        "backup_name": backup_name,
        "sample_changes": samples,
    }


def collect_caption_files(dataset_dir: Path, cleanup: dict[str, Any]) -> list[Path]:
    backup_dir = get_backup_dir(dataset_dir)
    glob_fn = dataset_dir.rglob if cleanup["recursive"] else dataset_dir.glob
    return [
        txt_file
        for txt_file in sorted(glob_fn(f"*{cleanup['caption_ext']}"))
        if txt_file.is_file() and not txt_file.name.startswith(".") and backup_dir not in txt_file.parents
    ]


def read_caption_cleanup_records(files: list[Path]) -> list[dict[str, str]]:
    return [
        {
            "file": txt_file.name,
            "path": str(txt_file),
            "caption": txt_file.read_text(encoding="utf-8", errors="replace").strip(),
        }
        for txt_file in files
    ]


def preview_cleanup_records_native(
    records: list[dict[str, str]],
    cleanup: dict[str, Any],
    *,
    include_stats: bool,
    sample_limit: int,
) -> dict[str, Any] | None:
    if not native_caption_cleanup_supported(cleanup):
        return None
    native = native_caption_cleanup_api()
    if native is None:
        return None
    try:
        result = native.preview_caption_cleanup(
            json.dumps(records, ensure_ascii=False),
            json.dumps(cleanup, ensure_ascii=False),
            bool(include_stats),
            int(sample_limit),
        )
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def preview_cleanup_dataset_native(
    dataset_dir: Path,
    cleanup: dict[str, Any],
    *,
    include_stats: bool,
    sample_limit: int,
) -> dict[str, Any] | None:
    if not native_caption_cleanup_supported(cleanup):
        return None
    native = native_caption_cleanup_api()
    if native is None or not hasattr(native, "preview_caption_cleanup_dataset"):
        return None
    try:
        result = native.preview_caption_cleanup_dataset(
            str(dataset_dir),
            json.dumps(cleanup, ensure_ascii=False),
            bool(include_stats),
            int(sample_limit),
        )
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def submit_caption_cleanup_job(
    *,
    params: dict[str, Any],
    dataset_dir: Path,
    cleanup: dict[str, Any],
    preview: dict[str, Any],
    job_manager: Any,
    job_store: Any,
    kind: str,
    job_name: str,
    route_family_default: str = "generic",
    invalidate_cache: Callable[[str], None] | None = None,
) -> str:
    from backend.core.job_manager import Job, JobType

    job = Job(
        type=JobType.TAG_BATCH_EDIT,
        name=job_name,
        total_items=int(preview.get("summary", {}).get("total_file_count", 0) or 0),
        metadata={"dataset_path": str(dataset_dir), "preview": preview, "config": dict(params)},
    )

    def _worker(progress_callback=None, cancel_check=None):
        result = apply_cleanup_to_dataset(
            dataset_dir,
            cleanup,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            invalidate_cache=invalidate_cache,
        )
        return job_store.save_result(
            kind=kind,
            job_id=job.id,
            dataset_path=str(dataset_dir),
            route_family=str(params.get("route_family", "") or route_family_default),
            submitted_config=dict(params),
            payload={**result, "preview": preview},
            started_at=job.started_at.isoformat() if job.started_at else None,
        )

    return str(job_manager.submit(job, _worker))


def cleanup_caption_from_config(text: str, cleanup: dict[str, Any]) -> str:
    return cleanup_tags(
        text,
        remove_parens=cleanup["remove_parens"],
        dedup=cleanup["dedup"],
        max_tag_len=cleanup["max_tag_len"],
        sort_tags=cleanup["sort_tags"],
        collapse_whitespace=cleanup["collapse_whitespace"],
        replace_underscore=cleanup["replace_underscore"],
        prepend_tags=cleanup["prepend_tags"],
        append_tags=cleanup["append_tags"],
        remove_tags=cleanup["remove_tags"],
        search_text=cleanup["search_text"],
        replace_text=cleanup["replace_text"],
        use_regex=cleanup["use_regex"],
        blacklist_tags=cleanup.get("blacklist_tags") or [],
        alias_map=cleanup.get("alias_map") or {},
        bulk_replace_rules=cleanup.get("bulk_replace_rules") or [],
    )


def cleanup_tags(
    text: str,
    remove_parens: bool = True,
    dedup: bool = True,
    max_tag_len: int = 100,
    sort_tags: bool = False,
    collapse_whitespace: bool = False,
    replace_underscore: bool = False,
    prepend_tags: str = "",
    append_tags: str = "",
    remove_tags: str = "",
    search_text: str = "",
    replace_text: str = "",
    use_regex: bool = False,
    blacklist_tags: list[str] | None = None,
    alias_map: dict[str, str] | None = None,
    bulk_replace_rules: list[tuple[str, str]] | None = None,
) -> str:
    normalized_text = str(text or "")
    for search_value, replacement_value in bulk_replace_rules or []:
        if search_value:
            normalized_text = normalized_text.replace(search_value, replacement_value)
    tags = split_tags(normalized_text)
    if remove_parens:
        tags = [re.sub(r"[()（）\[\]【】{}]", "", tag).strip() for tag in tags]
        tags = [tag for tag in tags if tag]
    if replace_underscore:
        tags = [tag.replace("_", " ") for tag in tags]
    if collapse_whitespace:
        tags = [re.sub(r"\s+", " ", tag).strip() for tag in tags]
    if search_text:
        if use_regex:
            try:
                tags = [re.sub(search_text, replace_text, tag) for tag in tags]
            except re.error:
                pass
        else:
            tags = [tag.replace(search_text, replace_text) for tag in tags]
    if alias_map:
        remapped: list[str] = []
        for tag in tags:
            replacement = alias_map.get(tag.lower())
            if replacement is None:
                remapped.append(tag)
                continue
            replacement_text = str(replacement or "").strip()
            if replacement_text:
                remapped.extend(split_tags(replacement_text))
        tags = remapped
    remove_set = {tag.strip().lower() for tag in split_tags(remove_tags)}
    remove_set.update(tag.lower() for tag in (blacklist_tags or []) if str(tag or "").strip())
    if remove_set:
        tags = [tag for tag in tags if tag.lower() not in remove_set]
    if dedup:
        tags = dedupe_tags(tags)
    if sort_tags:
        tags.sort(key=lambda tag: tag.lower())
    if max_tag_len > 0:
        tags = [tag[:max_tag_len] for tag in tags]
    result_tags = list(tags)
    if prepend_tags:
        result_tags = split_tags(prepend_tags) + result_tags
    if append_tags:
        result_tags.extend(split_tags(append_tags))
    return join_tags(result_tags)


def get_backup_dir(dataset_dir: Path) -> Path:
    return dataset_dir / ".backups"


def parse_caption_tag_list(raw: Any) -> list[str]:
    values: list[str] = []
    for chunk in str(raw or "").replace("\r", "\n").split("\n"):
        text = chunk.strip()
        if not text:
            continue
        values.extend(split_tags(text))
    return dedupe_tags(values)


def parse_rewrite_rules(raw: Any) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for line in str(raw or "").replace("\r", "\n").split("\n"):
        text = line.strip()
        if not text:
            continue
        left = ""
        right = ""
        for separator in ("=>", "->", "\t"):
            if separator in text:
                left, right = text.split(separator, 1)
                break
        else:
            if "=" in text:
                left, right = text.split("=", 1)
            else:
                continue
        left = left.strip()
        right = right.strip()
        if left:
            rules.append((left, right))
    return rules


def build_alias_map(raw: Any) -> dict[str, str]:
    return {source.lower(): target for source, target in parse_rewrite_rules(raw)}


def build_caption_frequency_stats(captions: list[str], *, top_limit: int) -> dict[str, Any]:
    tag_counts: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    total_tag_count = 0
    for caption in captions:
        text = str(caption or "").strip()
        if not text:
            continue
        normalized_tags = dedupe_tags(split_tags(text))
        total_tag_count += len(normalized_tags)
        tag_counts.update(normalized_tags)
        caption_counts.update([text])
    captioned_count = sum(1 for caption in captions if str(caption or "").strip())
    avg_tags_per_caption = (total_tag_count / captioned_count) if captioned_count else 0.0
    return {
        "captioned_count": captioned_count,
        "empty_count": max(0, len(captions) - captioned_count),
        "total_tag_count": total_tag_count,
        "unique_tag_count": len(tag_counts),
        "avg_tags_per_caption": round(avg_tags_per_caption, 3),
        "repeated_caption_count": sum(1 for count in caption_counts.values() if count > 1),
        "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(max(1, top_limit))],
        "top_captions": [{"caption": caption, "count": count} for caption, count in caption_counts.most_common(max(1, top_limit))],
    }
