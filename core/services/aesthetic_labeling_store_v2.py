"""Versioned JSON store for Lulynx-native aesthetic labeling state."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, TypeVar

from backend.core.aesthetic_labeling import AestheticLabelStore
from backend.core.services.aesthetic_labeling_events import make_review_event, now_iso


SCHEMA_VERSION = 2
_T = TypeVar("_T")


class AestheticLabelingStoreV2:
    """Small versioned state store with v1 JSON migration support."""

    def __init__(self, legacy_store: AestheticLabelStore | None = None) -> None:
        self.legacy_store = legacy_store or AestheticLabelStore()
        self.state_path = self.legacy_store.state_path
        self._lock = threading.RLock()

    def read_state(self) -> dict[str, Any]:
        with self._lock:
            raw = self._read_raw()
            state, migrated = self._normalize_state(raw)
            if migrated:
                self._backup_v1(raw)
                self._write_state(state)
            return state

    def update_state(self, mutator: Callable[[dict[str, Any]], _T]) -> _T:
        with self._lock:
            state = self.read_state()
            result = mutator(state)
            state["updated_at"] = now_iso()
            self._write_state(state)
            return result

    def get_annotation(self, state: dict[str, Any], sample_id: int | str) -> dict[str, Any] | None:
        value = state.get("annotations", {}).get(str(sample_id))
        return deepcopy(value) if isinstance(value, dict) else None

    def set_annotation(
        self,
        state: dict[str, Any],
        sample_id: int | str,
        annotation: dict[str, Any],
        *,
        local_path: str = "",
        event_type: str = "annotated",
    ) -> dict[str, Any]:
        key = str(sample_id)
        before = self.get_annotation(state, key)
        state.setdefault("annotations", {})[key] = deepcopy(annotation)
        if local_path:
            self._sync_label_entry(state, local_path, annotation)
        event = make_review_event(sample_id=int(sample_id), event_type=event_type, before=before, after=annotation)
        state.setdefault("review_events", []).append(event)
        return event

    def upsert_sample(self, state: dict[str, Any], sample: dict[str, Any]) -> None:
        sample_id = str(int(sample.get("sample_id") or 0))
        if sample_id == "0":
            return
        existing = state.setdefault("samples", {}).get(sample_id, {})
        merged = {**existing, **deepcopy(sample)}
        merged.setdefault("storage_state", "available")
        state["samples"][sample_id] = merged

    def get_sample_by_sha(self, state: dict[str, Any], sha256: str) -> dict[str, Any] | None:
        if not sha256:
            return None
        for sample in state.get("samples", {}).values():
            if not isinstance(sample, dict):
                continue
            if sample.get("sha256") == sha256 and sample.get("storage_state", "available") == "available":
                return deepcopy(sample)
        return None

    def get_sample_by_source(self, state: dict[str, Any], source: str, post_id: str) -> dict[str, Any] | None:
        if not source or not post_id:
            return None
        alias = state.get("sample_source_aliases", {}).get(_source_alias_key(source, post_id))
        if alias:
            sample = state.get("samples", {}).get(str(alias))
            if isinstance(sample, dict) and sample.get("storage_state", "available") == "available":
                return deepcopy(sample)
        for sample in state.get("samples", {}).values():
            if not isinstance(sample, dict):
                continue
            if sample.get("source") == source and str(sample.get("source_post_id") or "") == str(post_id):
                if sample.get("storage_state", "available") == "available":
                    return deepcopy(sample)
        return None

    def remember_source_alias(self, state: dict[str, Any], source: str, post_id: str, sample_id: int | str) -> None:
        if source and post_id and sample_id:
            state.setdefault("sample_source_aliases", {})[_source_alias_key(source, post_id)] = int(sample_id)

    def mark_deleted(self, state: dict[str, Any], sample: dict[str, Any], record: dict[str, Any]) -> None:
        sample_id = str(int(sample.get("sample_id") or 0))
        local_path = str(sample.get("local_path") or "")
        if sample_id in state.setdefault("samples", {}):
            state["samples"][sample_id]["storage_state"] = "quarantined" if record.get("image_quarantined") else "deleted"
        if local_path:
            state.setdefault("labels", {}).pop(local_path, None)
            state.setdefault("deleted_samples", {})[local_path] = deepcopy(record)
        state.setdefault("annotations", {}).pop(sample_id, None)
        self.remove_from_index(state, local_path)
        event = make_review_event(
            sample_id=int(sample_id or 0),
            event_type="quarantined" if record.get("image_quarantined") else "deleted",
            before=None,
            after=None,
            changes={"deleted_sample": record},
        )
        state.setdefault("review_events", []).append(event)

    def remove_from_index(self, state: dict[str, Any], local_path: str) -> None:
        index = state.get("local_index")
        if not isinstance(index, dict) or not isinstance(index.get("items"), list):
            return
        index["items"] = [item for item in index["items"] if item.get("local_path") != local_path]

    def _normalize_state(self, raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        migrated = int(raw.get("schema_version") or 0) < SCHEMA_VERSION
        state = self._empty_state()
        state.update({key: deepcopy(raw.get(key)) for key in ("settings", "local_index", "deleted_samples", "labels", "source_failures") if key in raw})
        state["sources"] = _normalize_sources(raw.get("sources"))
        state["source_snapshots"] = deepcopy(raw.get("source_snapshots") or {})
        state["sample_source_aliases"] = deepcopy(raw.get("sample_source_aliases") or {})
        state["samples"] = _normalize_mapping(raw.get("samples"))
        state["annotations"] = _normalize_mapping(raw.get("annotations"))
        state["review_events"] = list(raw.get("review_events") or [])
        state["schema_version"] = SCHEMA_VERSION
        state["store_id"] = "aesthetic_labeling"
        state["updated_at"] = str(raw.get("updated_at") or now_iso())
        self._migrate_local_index_items(state)
        self._migrate_labels(state)
        return state, migrated

    def _migrate_local_index_items(self, state: dict[str, Any]) -> None:
        index = state.get("local_index")
        if not isinstance(index, dict) or not isinstance(index.get("items"), list):
            return
        index["version"] = max(2, int(index.get("version") or 1))
        for item in index.get("items") or []:
            sample = _sample_from_index_item(item)
            if sample:
                self.upsert_sample(state, sample)

    def _migrate_labels(self, state: dict[str, Any]) -> None:
        labels = state.get("labels") if isinstance(state.get("labels"), dict) else {}
        for local_path, record in labels.items():
            if not isinstance(record, dict):
                continue
            sample_id = _sample_id(str(local_path))
            annotation = _annotation_from_legacy_record(record, sample_id)
            if annotation:
                state.setdefault("annotations", {}).setdefault(str(sample_id), annotation)
                if not _has_event(state, sample_id):
                    event_type = "skipped" if annotation.get("status") == "skipped" else "annotated"
                    state.setdefault("review_events", []).append(
                        make_review_event(sample_id=sample_id, event_type=event_type, after=annotation, actor="legacy_json_migration")
                    )
            sample = _sample_from_local_path(str(local_path))
            if sample:
                self.upsert_sample(state, sample)

    def _sync_label_entry(self, state: dict[str, Any], local_path: str, annotation: dict[str, Any]) -> None:
        scores = annotation.get("scores") if isinstance(annotation.get("scores"), dict) else {}
        state.setdefault("labels", {})[local_path] = {
            "label": annotation.get("status") or "labeled",
            "score": scores.get("aesthetic"),
            "notes": annotation.get("note") or "",
            "annotation": _annotation_to_legacy_shape(annotation),
            "updated_at": time.time(),
        }

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "store_id": "aesthetic_labeling",
            "updated_at": now_iso(),
            "sources": [],
            "source_snapshots": {},
            "sample_source_aliases": {},
            "samples": {},
            "annotations": {},
            "review_events": [],
            "local_index": {},
            "deleted_samples": {},
            "source_failures": {},
            "labels": {},
            "settings": {},
        }

    def _read_raw(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _backup_v1(self, raw: dict[str, Any]) -> None:
        if not raw or not self.state_path.exists():
            return
        backup = self.state_path.with_suffix(self.state_path.suffix + ".v1.bak")
        if backup.exists():
            return
        try:
            shutil.copy2(self.state_path, backup)
        except Exception:
            return


def _normalize_sources(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [dict(item) for item in value.values() if isinstance(item, dict)]
    return []


def _normalize_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {str(key): dict(item) for key, item in value.items() if isinstance(item, dict)}


def _source_alias_key(source: str, post_id: str) -> str:
    return f"{source}:{post_id}"


def _sample_from_index_item(item: dict[str, Any]) -> dict[str, Any] | None:
    local_path = str(item.get("local_path") or "")
    if not local_path:
        return None
    sample_id = int(item.get("sample_id") or _sample_id(local_path))
    return {
        "sample_id": sample_id,
        "sample_key": f"local:{local_path}",
        "source": item.get("source") or (Path(item.get("source_path") or "").name or "local"),
        "source_id": item.get("source_id") or "",
        "source_path": item.get("source_path") or "",
        "source_post_id": None,
        "source_page_url": "",
        "original_url": "",
        "local_path": local_path,
        "path": local_path,
        "name": item.get("name") or Path(local_path).name,
        "storage_state": "available",
        "width": int(item.get("width") or 0),
        "height": int(item.get("height") or 0),
        "sha256": str(item.get("sha256") or ""),
        "created_at": item.get("created_at") or "",
        "imported_at": item.get("imported_at") or now_iso(),
        "mtime": item.get("mtime") or 0,
        "size": item.get("size") or 0,
    }


def _sample_from_local_path(local_path: str) -> dict[str, Any] | None:
    path = Path(local_path)
    if not local_path:
        return None
    stat = None
    try:
        stat = path.stat() if path.exists() else None
    except Exception:
        stat = None
    sample_id = _sample_id(local_path)
    return {
        "sample_id": sample_id,
        "sample_key": f"local:{local_path}",
        "source": path.parent.name or "local",
        "source_path": str(path.parent),
        "source_post_id": None,
        "source_page_url": "",
        "original_url": "",
        "local_path": local_path,
        "path": local_path,
        "name": path.name,
        "storage_state": "available" if path.exists() else "missing",
        "width": 0,
        "height": 0,
        "sha256": "",
        "created_at": _timestamp_to_iso(stat.st_mtime) if stat else "",
        "imported_at": now_iso(),
        "mtime": stat.st_mtime if stat else 0,
        "size": stat.st_size if stat else 0,
    }


def _annotation_from_legacy_record(record: dict[str, Any], sample_id: int) -> dict[str, Any] | None:
    legacy = record.get("annotation") if isinstance(record.get("annotation"), dict) else {}
    if legacy:
        return {
            "sample_id": int(sample_id),
            "status": str(legacy.get("status") or "labeled"),
            "scores": {
                "aesthetic": legacy.get("aesthetic"),
                "composition": legacy.get("composition"),
                "color": legacy.get("color"),
                "sexual": legacy.get("sexual"),
            },
            "in_domain": bool(int(legacy.get("in_domain", 1) or 0)),
            "content_type": str(legacy.get("content_type") or "anime_illust"),
            "exclude_from_score_train": bool(int(legacy.get("exclude_from_score_train", 0) or 0)),
            "exclude_from_cls_train": bool(int(legacy.get("exclude_from_cls_train", 0) or 0)),
            "exclude_reason": legacy.get("exclude_reason"),
            "note": legacy.get("note"),
            "updated_at": legacy.get("updated_at") or _coerce_updated_at(record.get("updated_at")),
        }
    if not record.get("label") and record.get("score") is None:
        return None
    label = str(record.get("label") or "labeled")
    return {
        "sample_id": int(sample_id),
        "status": "skipped" if label == "skipped" else "labeled",
        "scores": {"aesthetic": record.get("score"), "composition": None, "color": None, "sexual": None},
        "in_domain": True,
        "content_type": "anime_illust",
        "exclude_from_score_train": False,
        "exclude_from_cls_train": False,
        "exclude_reason": None,
        "note": record.get("notes") or "",
        "updated_at": _coerce_updated_at(record.get("updated_at")),
    }


def _annotation_to_legacy_shape(annotation: dict[str, Any]) -> dict[str, Any]:
    scores = annotation.get("scores") if isinstance(annotation.get("scores"), dict) else {}
    return {
        "status": annotation.get("status") or "labeled",
        "aesthetic": scores.get("aesthetic"),
        "composition": scores.get("composition"),
        "color": scores.get("color"),
        "sexual": scores.get("sexual"),
        "in_domain": 1 if bool(annotation.get("in_domain", True)) else 0,
        "content_type": annotation.get("content_type") or "anime_illust",
        "exclude_from_score_train": 1 if bool(annotation.get("exclude_from_score_train", False)) else 0,
        "exclude_from_cls_train": 1 if bool(annotation.get("exclude_from_cls_train", False)) else 0,
        "exclude_reason": annotation.get("exclude_reason"),
        "note": annotation.get("note"),
        "updated_at": annotation.get("updated_at"),
    }


def _has_event(state: dict[str, Any], sample_id: int) -> bool:
    return any(int(item.get("sample_id") or 0) == int(sample_id) for item in state.get("review_events") or [])


def _sample_id(path: str) -> int:
    return int(hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)


def _coerce_updated_at(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    try:
        return _timestamp_to_iso(float(value))
    except Exception:
        return now_iso()


def _timestamp_to_iso(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(value))
