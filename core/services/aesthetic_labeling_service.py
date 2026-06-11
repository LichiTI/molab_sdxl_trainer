"""Lulynx-native aesthetic labeling service orchestration."""

from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from backend.core.aesthetic_labeling import AestheticLabelStore, IMAGE_EXTS
from backend.core.services.aesthetic_labeling_events import (
    annotation_to_compat,
    compat_to_annotation,
    merge_dimension_annotation,
    now_compact,
    now_iso,
    skipped_annotation,
)
from backend.core.services.aesthetic_labeling_queue import list_visible_samples, paginate_samples, pick_last_reviewed, pick_next_sample
from backend.core.services.aesthetic_labeling_remote_importer import AestheticRemoteImporter
from backend.core.services.aesthetic_labeling_remote_sources import RemoteSourceProvider
from backend.core.services.aesthetic_labeling_store_v2 import AestheticLabelingStoreV2
from backend.core.services.native_module_loader import native_with_entrypoints


DEFAULT_SETTINGS: dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 7860},
    "ui": {"language": "zh-CN"},
    "sampling": {"max_attempts": 30, "request_timeout_sec": 20, "min_side": 256, "request_retries": 2},
    "sources": {
        "weights": {"danbooru": 0, "e621": 0, "local": 1},
        "danbooru": {"enabled": False, "base_url": "", "tags": "", "limit": 100, "user_agent": "VibeCodeLabeler/1.0", "username_env": "DANBOORU_USERNAME", "api_key_env": "DANBOORU_API_KEY"},
        "e621": {"enabled": False, "base_url": "", "tags": "", "limit": 100, "user_agent": "", "login_env": "E621_LOGIN", "api_key_env": "E621_API_KEY"},
        "local": {"enabled": True, "paths": [], "recursive": False, "extensions": sorted(IMAGE_EXTS)},
    },
    "storage": {"root_dir": "dataset", "images_dir": "dataset/images", "db_path": "dataset/labels.db", "webp_quality": 95},
    "rating_guide": {
        "dimensions": {
            "aesthetic": {"title": "美学", "description": "整体视觉质量与吸引力。", "examples": {"1": "明显粗糙，细节缺失或画面脏乱。", "3": "中等水平，主体清晰，缺少亮点。", "5": "非常优秀，细节、氛围与完成度都很强。"}},
            "composition": {"title": "构图", "description": "主体安排、视觉引导与画面平衡。", "examples": {"1": "构图混乱，主体不明确。", "3": "构图基本合理，信息表达清楚。", "5": "构图非常出色，叙事与视觉层次强。"}},
            "color": {"title": "色彩", "description": "配色、对比、色调统一性。", "examples": {"1": "黑白、线稿或颜色明显失衡。", "3": "颜色基本协调，无明显问题。", "5": "配色光影高级且有记忆点，氛围极佳。"}},
            "sexual": {"title": "色情", "description": "色情程度强弱，不等同于质量好坏。", "examples": {"1": "几乎无性暗示，安全向。", "3": "中度性暗示，边缘内容。", "5": "高度露骨，强成人内容。"}},
        }
    },
}


class AestheticLabelingService:
    def __init__(self, store: AestheticLabelStore | None = None, remote_providers: Iterable[RemoteSourceProvider] | None = None) -> None:
        self.store = AestheticLabelingStoreV2(store)
        self.remote_importer = AestheticRemoteImporter(self.store, self.settings, remote_providers)

    def settings(self) -> dict[str, Any]:
        state = self.store.read_state()
        cfg = _settings_from_state(state, self.store.state_path)
        cfg["status"] = "success"
        cfg["data"] = {"config": deepcopy(cfg)}
        return cfg

    def save_settings(self, config: dict[str, Any]) -> dict[str, Any]:
        def mutate(state: dict[str, Any]) -> None:
            state["settings"] = _deep_merge(DEFAULT_SETTINGS, config or {})

        self.store.update_state(mutate)
        return self.settings()

    def source_names(self) -> list[str]:
        names: list[str] = []
        state = self.store.read_state()
        for source in state.get("sources", []):
            name = str(source.get("name") or source.get("path") or "").strip()
            if name and name not in names:
                names.append(name)
        for path in _settings_from_state(state, self.store.state_path)["sources"]["local"].get("paths") or []:
            name = Path(path).name or str(path)
            if name and name not in names:
                names.append(name)
        return names

    def source_records(self) -> list[dict[str, Any]]:
        return list(self.store.read_state().get("sources", []))

    def add_source(self, path: str, name: str = "") -> dict[str, Any]:
        source = {"id": f"src_{int(time.time() * 1000)}", "name": name or Path(path).name, "path": path, "created_at": time.time()}

        def mutate(state: dict[str, Any]) -> None:
            state.setdefault("sources", []).append(source)
            cfg = _settings_from_state(state, self.store.state_path)
            paths = cfg["sources"]["local"].setdefault("paths", [])
            if path and path not in paths:
                paths.append(path)
            state["settings"] = _strip_response_fields(cfg)

        self.store.update_state(mutate)
        return source

    def list_samples(self, *, page: int = 1, size: int = 24, status: str = "all", source: str = "", order: str = "desc", after_id: int = 0) -> dict[str, Any]:
        state = self.store.read_state()
        samples = self._samples(state)
        items = list_visible_samples(samples, state.get("annotations", {}), status=status, source=source, order=order, after_id=after_id)
        return paginate_samples([self._compat_sample(item, state) for item in items], page=page, size=size)

    def next_sample(self, *, avoid_sample_ids: list[int] | None = None, after_sample_id: int = 0) -> dict[str, Any] | None:
        state = self.store.read_state()
        avoid = {int(item) for item in avoid_sample_ids or [] if _can_int(item)}
        sample = pick_next_sample(self._samples(state), state.get("annotations", {}), avoid_ids=avoid, after_id=after_sample_id)
        if sample:
            return self._compat_sample(sample, state)
        imported = self.import_remote_sample()
        if imported and int(imported.get("sample_id") or 0) not in avoid:
            return imported
        state = self.store.read_state()
        sample = pick_next_sample(self._samples(state), state.get("annotations", {}), avoid_ids=avoid, after_id=after_sample_id)
        return self._compat_sample(sample, state) if sample else None

    def sample(self, sample_id: int | str) -> dict[str, Any] | None:
        token = str(sample_id)
        state = self.store.read_state()
        for sample in self._samples(state):
            if str(sample.get("sample_id")) == token or Path(str(sample.get("local_path") or "")).name == token:
                return self._compat_sample(sample, state)
        return None

    def image_path(self, sample_id: int | str, *, thumb: bool = False, thumb_size: int = 480) -> Path | None:
        sample = self.sample(sample_id)
        if not sample:
            return None
        path = Path(str(sample.get("local_path") or "")).resolve()
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            return None
        if not thumb:
            return path
        return self._thumbnail_path(path, int(sample["sample_id"]), thumb_size) or path

    def annotate(self, body: dict[str, Any]) -> dict[str, Any] | None:
        sample = self.sample(body.get("sample_id", ""))
        if not sample:
            return None
        annotation = compat_to_annotation(body, sample_id=int(sample["sample_id"]), status="labeled")

        def mutate(state: dict[str, Any]) -> None:
            self.store.upsert_sample(state, _sample_for_store(sample))
            self.store.set_annotation(state, sample["sample_id"], annotation, local_path=sample["local_path"], event_type="annotated")

        self.store.update_state(mutate)
        sample["annotation"] = annotation_to_compat(annotation)
        return sample

    def set_label(self, image_path: str, label: str = "", score: Any = None, notes: str = "") -> dict[str, Any]:
        key = _normalize_image_key(image_path)
        sample_id = _sample_id(key)
        annotation = {
            "sample_id": sample_id,
            "status": "skipped" if label in {"skip", "skipped"} else "labeled",
            "scores": {"aesthetic": score, "composition": None, "color": None, "sexual": None},
            "in_domain": True,
            "content_type": "anime_illust",
            "exclude_from_score_train": False,
            "exclude_from_cls_train": False,
            "exclude_reason": None,
            "note": notes,
            "updated_at": now_iso(),
        }
        entry = {"label": label, "score": score, "notes": notes, "annotation": annotation_to_compat(annotation), "updated_at": time.time()}

        def mutate(state: dict[str, Any]) -> None:
            sample = _sample_from_path(key)
            if sample:
                self.store.upsert_sample(state, sample)
            self.store.set_annotation(state, sample_id, annotation, local_path=key, event_type="annotated")
            state.setdefault("labels", {})[key] = entry

        self.store.update_state(mutate)
        return entry

    def annotate_dim(self, body: dict[str, Any]) -> dict[str, Any] | None:
        sample = self.sample(body.get("sample_id", ""))
        if not sample:
            return None

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            current = self.store.get_annotation(state, sample["sample_id"])
            annotation = merge_dimension_annotation(
                current,
                sample_id=int(sample["sample_id"]),
                dim=str(body.get("dim") or ""),
                score=body.get("score"),
                body=body,
            )
            self.store.upsert_sample(state, _sample_for_store(sample))
            self.store.set_annotation(state, sample["sample_id"], annotation, local_path=sample["local_path"], event_type="dimension_annotated")
            return annotation

        annotation = self.store.update_state(mutate)
        sample["annotation"] = annotation_to_compat(annotation)
        return sample

    def skip(self, body: dict[str, Any]) -> dict[str, Any] | None:
        sample = self.sample(body.get("sample_id", ""))
        if not sample:
            return None

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            current = self.store.get_annotation(state, sample["sample_id"])
            annotation = skipped_annotation(current, sample_id=int(sample["sample_id"]), body=body)
            self.store.upsert_sample(state, _sample_for_store(sample))
            self.store.set_annotation(state, sample["sample_id"], annotation, local_path=sample["local_path"], event_type="skipped")
            return annotation

        annotation = self.store.update_state(mutate)
        sample["annotation"] = annotation_to_compat(annotation)
        return sample

    def last_reviewed(self, status: str = "labeled") -> dict[str, Any]:
        state = self.store.read_state()
        sample = pick_last_reviewed(self._samples(state), state.get("annotations", {}), status=status)
        return {"sample": self._compat_sample(sample, state) if sample else None}

    def stats(self) -> dict[str, Any]:
        state = self.store.read_state()
        samples = [self._compat_sample(item, state) for item in self._samples(state) if item.get("storage_state", "available") == "available"]
        labeled = sum(1 for item in samples if _sample_status(item) == "labeled")
        skipped = sum(1 for item in samples if _sample_status(item) == "skipped")
        data = {
            "source_count": len(state.get("sources", [])),
            "label_count": len(state.get("annotations", {})),
            "positive_count": sum(1 for item in state.get("labels", {}).values() if item.get("label") in {"good", "positive", "keep"}),
            "negative_count": sum(1 for item in state.get("labels", {}).values() if item.get("label") in {"bad", "negative", "reject"}),
            "total_samples": len(samples),
            "labeled_samples": labeled,
            "skipped_samples": skipped,
            "unreviewed_samples": max(0, len(samples) - labeled - skipped),
            "review_event_count": len(state.get("review_events", [])),
            "status": "success",
        }
        data["data"] = dict(data)
        return data

    def source_health(self) -> dict[str, Any]:
        cfg = self.settings()
        items = []
        local = cfg["sources"]["local"]
        extensions = {str(ext).lower() for ext in local.get("extensions") or IMAGE_EXTS}
        for path in local.get("paths") or []:
            p = Path(path)
            ok = p.exists() and (p.is_dir() or p.is_file())
            count = len(list(_iter_images(p, bool(local.get("recursive")), extensions))) if ok else 0
            items.append({"source": Path(path).name or str(path), "path": str(path), "enabled": True, "ok": ok, "count": count, "message": "" if ok else "Path not found"})
        items.extend(self.remote_importer.health_items(cfg))
        return {"enabled_count": sum(1 for item in items if item["enabled"]), "ok_count": sum(1 for item in items if item["ok"]), "items": items, "checked_at": now_iso()}

    def import_remote_sample(self) -> dict[str, Any] | None:
        sample = self.remote_importer.import_sample()
        if not sample:
            return None
        return self._compat_sample(sample, self.store.read_state())

    def reindex_local(self, *, hash_local: bool = False, progress_callback: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        def mutate(state: dict[str, Any]) -> int:
            cfg = _settings_from_state(state, self.store.state_path)
            samples = self._scan_samples(state, cfg, hash_local=hash_local, progress_callback=progress_callback)
            samples, duplicate_count = _dedup_local_samples(samples, hash_local=hash_local)
            items = [_index_item(item) for item in samples]
            state["local_index"] = {"version": 2, "signature": _local_source_signature(cfg), "updated_at": now_iso(), "items": items}
            for sample in samples:
                self.store.upsert_sample(state, sample)
            state.setdefault("local_index", {})["hash_local"] = bool(hash_local)
            state.setdefault("local_index", {})["duplicate_sha_skipped"] = duplicate_count
            return len(items)

        count = self.store.update_state(mutate)
        state = self.store.read_state()
        duplicate_count = int(state.get("local_index", {}).get("duplicate_sha_skipped") or 0)
        return {
            "local_indexed_files": count,
            "indexed_files": count,
            "removed_missing_files": 0,
            "hash_local": bool(hash_local),
            "duplicate_sha_skipped": duplicate_count,
            "status": "success",
        }

    def delete_sample(self, sample_id: int | str, delete_image: bool = False) -> dict[str, Any]:
        sample = self.sample(sample_id)
        if not sample:
            return {"deleted": False, "image_deleted": False, "message": "Sample not found."}
        image_quarantined = False
        quarantine_path = ""
        error = ""
        if delete_image:
            try:
                source = Path(sample["local_path"]).resolve()
                quarantine_path = str(self._quarantine_path(source))
                Path(quarantine_path).parent.mkdir(parents=True, exist_ok=True)
                source.replace(quarantine_path)
                image_quarantined = True
            except Exception as exc:
                error = str(exc)
        record = {
            "sample_id": sample["sample_id"],
            "deleted_at": now_iso(),
            "delete_image_requested": bool(delete_image),
            "image_quarantined": image_quarantined,
            "quarantine_path": quarantine_path,
            "error": error,
        }

        def mutate(state: dict[str, Any]) -> None:
            self.store.mark_deleted(state, sample, record)

        self.store.update_state(mutate)
        return {"deleted": True, "image_deleted": image_quarantined, "image_quarantined": image_quarantined, "delete_image_requested": bool(delete_image), "quarantine_path": quarantine_path, "error": error, "sample_id": sample["sample_id"]}

    def _samples(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        cfg = _settings_from_state(state, self.store.state_path)
        index = state.get("local_index") if isinstance(state.get("local_index"), dict) else {}
        if index.get("signature") == _local_source_signature(cfg) and isinstance(index.get("items"), list):
            local_samples = self._samples_from_index(state, index["items"])
        else:
            local_samples = self._scan_samples(state, cfg)
        return _merge_persisted_samples(local_samples, state)

    def _scan_samples(
        self,
        state: dict[str, Any],
        cfg: dict[str, Any],
        *,
        hash_local: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        deleted = state.get("deleted_samples", {})
        samples: list[dict[str, Any]] = []
        seen: set[str] = set()
        local = cfg["sources"]["local"]
        extensions = {str(ext).lower() for ext in local.get("extensions") or IMAGE_EXTS}
        images: list[tuple[str, Path]] = []
        for root in local.get("paths") or []:
            images.extend((str(root), image) for image in _iter_images(Path(root), bool(local.get("recursive")), extensions))
        total = len(images)
        if progress_callback:
            progress_callback(0, total)
        for index, (root, image) in enumerate(images, start=1):
            key = str(image.resolve())
            if key in seen or key in deleted:
                if progress_callback:
                    progress_callback(index, total)
                continue
            seen.add(key)
            stat = image.stat()
            sample_id = _sample_id(key)
            width, height = _image_size(image)
            sha256 = _file_sha256(image) if hash_local else ""
            samples.append({"sample_id": sample_id, "sample_key": f"local:{key}", "source": Path(root).name or str(root), "source_path": str(root), "source_post_id": None, "source_page_url": "", "original_url": "", "local_path": key, "path": key, "name": image.name, "storage_state": "available", "width": width, "height": height, "sha256": sha256, "created_at": _timestamp_to_iso(stat.st_mtime), "imported_at": now_iso(), "mtime": stat.st_mtime, "size": stat.st_size})
            if progress_callback:
                progress_callback(index, total)
        return samples

    def _samples_from_index(self, state: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deleted = state.get("deleted_samples", {})
        samples = []
        for item in items:
            key = str(item.get("local_path") or "")
            path = Path(key)
            if not key or key in deleted or not path.is_file():
                continue
            sample = {"sample_id": int(item.get("sample_id") or _sample_id(key)), "sample_key": f"local:{key}", "source": item.get("source") or (Path(item.get("source_path") or "").name or "local"), "source_path": item.get("source_path") or "", "source_post_id": None, "source_page_url": "", "original_url": "", "local_path": key, "path": key, "name": item.get("name") or path.name, "storage_state": "available", "width": int(item.get("width") or 0), "height": int(item.get("height") or 0), "sha256": str(item.get("sha256") or ""), "created_at": item.get("created_at") or _timestamp_to_iso(path.stat().st_mtime), "imported_at": item.get("imported_at") or now_iso(), "mtime": item.get("mtime") or path.stat().st_mtime, "size": item.get("size") or path.stat().st_size}
            samples.append(sample)
        return samples

    def _compat_sample(self, sample: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any]:
        if not sample:
            return {}
        item = deepcopy(sample)
        sample_id = int(item.get("sample_id") or 0)
        item["image_url"] = f"/api/aesthetic_labeling/image/{sample_id}"
        item["annotation"] = annotation_to_compat(state.get("annotations", {}).get(str(sample_id)))
        return item

    def _thumbnail_path(self, source: Path, sample_id: int, thumb_size: int) -> Path | None:
        size = max(64, min(int(thumb_size or 480), 2048))
        thumb_dir = self.store.state_path.parent / "aesthetic_thumbnails"
        thumb = thumb_dir / f"{sample_id}_{size}.webp"
        try:
            if thumb.is_file() and thumb.stat().st_mtime >= source.stat().st_mtime:
                return thumb
            from PIL import Image

            thumb_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.thumbnail((size, size))
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGB")
                image.save(thumb, format="WEBP", quality=85)
            return thumb
        except Exception:
            return None

    def _quarantine_path(self, source: Path) -> Path:
        root = self.store.state_path.parent / "aesthetic_deleted"
        target = root / f"{now_compact()}_{source.name}"
        index = 1
        while target.exists():
            target = root / f"{now_compact()}_{index}_{source.name}"
            index += 1
        return target


def _settings_from_state(state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    cfg = _deep_merge(DEFAULT_SETTINGS, state.get("settings") or {})
    source_paths = [str(item.get("path") or "") for item in state.get("sources", []) if isinstance(item, dict)]
    paths = cfg["sources"]["local"].setdefault("paths", [])
    for path in source_paths:
        if path and path not in paths:
            paths.append(path)
    cfg.setdefault("_meta", {})["config_path"] = str(state_path)
    return cfg


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (overlay or {}).items():
        if key in {"status", "data"}:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _strip_response_fields(config: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(config)
    clean.pop("status", None)
    clean.pop("data", None)
    clean.pop("_meta", None)
    return clean


def native_aesthetic_image_listing_api() -> Any:
    return native_with_entrypoints("list_image_files")


def _iter_images(root: Path, recursive: bool, extensions: set[str]):
    if root.is_file() and root.suffix.lower() in extensions:
        yield root
        return
    if not root.is_dir():
        return
    native = native_aesthetic_image_listing_api()
    if native is not None:
        try:
            for value in native.list_image_files(str(root), bool(recursive)):
                path = Path(str(value))
                if path.suffix.lower() in extensions:
                    yield path
            return
        except Exception:
            pass
    iterator = root.rglob("*") if recursive else root.iterdir()
    for item in sorted(iterator):
        if item.is_file() and item.suffix.lower() in extensions:
            yield item


def _sample_id(path: str) -> int:
    return int(hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)


def _local_source_signature(config: dict[str, Any]) -> str:
    local = config.get("sources", {}).get("local", {}) if isinstance(config, dict) else {}
    payload = {"paths": [str(path) for path in local.get("paths") or []], "recursive": bool(local.get("recursive")), "extensions": [str(ext).lower() for ext in local.get("extensions") or []]}
    return hashlib.sha1(str(payload).encode("utf-8", errors="ignore")).hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def _timestamp_to_iso(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(value))


def _index_item(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: sample.get(key) for key in ("sample_id", "local_path", "source", "source_path", "name", "width", "height", "created_at", "mtime", "size", "sha256")}


def _dedup_local_samples(samples: list[dict[str, Any]], *, hash_local: bool) -> tuple[list[dict[str, Any]], int]:
    if not hash_local:
        return samples, 0
    seen_sha: set[str] = set()
    deduped = []
    skipped = 0
    for sample in samples:
        sha256 = str(sample.get("sha256") or "")
        if sha256 and sha256 in seen_sha:
            skipped += 1
            continue
        if sha256:
            seen_sha.add(sha256)
        deduped.append(sample)
    return deduped, skipped


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return ""


def _sample_from_path(image_path: str) -> dict[str, Any] | None:
    if not image_path:
        return None
    path = Path(image_path)
    stat = None
    try:
        stat = path.stat() if path.exists() else None
    except Exception:
        stat = None
    return {
        "sample_id": _sample_id(image_path),
        "sample_key": f"local:{image_path}",
        "source": path.parent.name or "local",
        "source_path": str(path.parent),
        "source_post_id": None,
        "source_page_url": "",
        "original_url": "",
        "local_path": image_path,
        "path": image_path,
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


def _merge_persisted_samples(local_samples: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    merged = {str(int(item.get("sample_id") or 0)): item for item in local_samples if int(item.get("sample_id") or 0)}
    deleted = state.get("deleted_samples", {}) if isinstance(state.get("deleted_samples"), dict) else {}
    for sample_id, sample in state.get("samples", {}).items():
        if not isinstance(sample, dict) or str(sample_id) in merged:
            continue
        local_path = str(sample.get("local_path") or "")
        if local_path in deleted or sample.get("storage_state", "available") != "available":
            continue
        if local_path and not Path(local_path).is_file():
            continue
        merged[str(sample_id)] = deepcopy(sample)
    return list(merged.values())


def _sample_for_store(sample: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(sample)
    clean.pop("annotation", None)
    clean.pop("sample_seq", None)
    clean.pop("sample_total", None)
    clean.pop("image_url", None)
    return clean


def _sample_status(sample: dict[str, Any]) -> str:
    annotation = sample.get("annotation") or {}
    status = annotation.get("status")
    return status if status in {"labeled", "skipped"} else "unreviewed"


def _can_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def _normalize_image_key(image_path: str) -> str:
    path = Path(str(image_path or ""))
    try:
        return str(path.resolve()) if path.exists() else str(path)
    except Exception:
        return str(path)
