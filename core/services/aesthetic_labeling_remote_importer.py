"""Remote candidate import pipeline for aesthetic labeling."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from backend.core.services.aesthetic_labeling_events import now_iso
from backend.core.services.aesthetic_labeling_remote_sources import RemoteSourceCandidate, RemoteSourceProvider, build_remote_providers, remote_source_health
from backend.core.services.aesthetic_labeling_store_v2 import AestheticLabelingStoreV2


class AestheticRemoteImporter:
    def __init__(
        self,
        store: AestheticLabelingStoreV2,
        settings_loader: Callable[[], dict[str, Any]],
        providers: Iterable[RemoteSourceProvider] | None = None,
    ) -> None:
        self.store = store
        self._settings_loader = settings_loader
        self._providers = list(providers) if providers is not None else None

    def health_items(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        failures = self.store.read_state().get("source_failures", {})
        items = remote_source_health(self.providers(cfg))
        for item in items:
            failure = failures.get(item.get("source")) if isinstance(failures, dict) else None
            if not isinstance(failure, dict):
                continue
            cooldown_until = float(failure.get("cooldown_until") or 0)
            cooldown_active = cooldown_until > time.time()
            item.update({
                "last_error": failure.get("message") or "",
                "retry_count": int(failure.get("retry_count") or 0),
                "cooldown_until": failure.get("cooldown_until_iso") or "",
                "cooldown_active": cooldown_active,
            })
            if cooldown_active:
                item["ok"] = False
                item["message"] = f"cooldown until {item['cooldown_until']}: {item['last_error']}"
        return items

    def providers(self, cfg: dict[str, Any]) -> list[RemoteSourceProvider]:
        if self._providers is not None:
            return self._providers
        return build_remote_providers(cfg)

    def import_sample(self) -> dict[str, Any] | None:
        cfg = self._settings_loader()
        timeout = float(cfg.get("sampling", {}).get("request_timeout_sec") or 8.0)
        for provider in self.providers(cfg):
            if not provider.enabled:
                continue
            if self._cooldown_active(provider.source_name):
                continue
            try:
                candidate = provider.fetch_candidate(timeout_sec=timeout)
                if not candidate:
                    continue
                existing = self.sample_by_source(candidate.source, candidate.source_post_id)
                if existing:
                    self.clear_source_failure(provider.source_name)
                    return existing
                image_bytes = provider.load_image(candidate, timeout_sec=timeout)
                sample = self.import_candidate(candidate, image_bytes, provider=provider)
                if sample:
                    self.clear_source_failure(provider.source_name)
                return sample
            except Exception as exc:
                self.record_source_failure(provider.source_name, str(exc), cfg=cfg)
                continue
        return None

    def sample_by_source(self, source: str, post_id: str) -> dict[str, Any] | None:
        state = self.store.read_state()
        return self.store.get_sample_by_source(state, source, post_id)

    def import_candidate(
        self,
        candidate: RemoteSourceCandidate,
        image_bytes: bytes,
        *,
        provider: RemoteSourceProvider | None = None,
    ) -> dict[str, Any] | None:
        if not image_bytes:
            return None
        sha256 = hashlib.sha256(image_bytes).hexdigest()
        snapshot_id, snapshot = self._source_snapshot(provider, candidate)
        existing = self.sample_by_sha(sha256)
        if existing:
            def remember(state: dict[str, Any]) -> None:
                if snapshot_id and snapshot:
                    state.setdefault("source_snapshots", {}).setdefault(snapshot_id, snapshot)
                self.store.remember_source_alias(state, candidate.source, candidate.source_post_id, existing["sample_id"])

            self.store.update_state(remember)
            return existing
        sample = self._candidate_sample(candidate, sha256, image_bytes, source_snapshot_id=snapshot_id)

        def mutate(state: dict[str, Any]) -> None:
            duplicate = self.store.get_sample_by_sha(state, sha256)
            if duplicate:
                if snapshot_id and snapshot:
                    state.setdefault("source_snapshots", {}).setdefault(snapshot_id, snapshot)
                self.store.remember_source_alias(state, candidate.source, candidate.source_post_id, duplicate["sample_id"])
                return
            if snapshot_id and snapshot:
                state.setdefault("source_snapshots", {}).setdefault(snapshot_id, snapshot)
            self.store.upsert_sample(state, sample)
            self.store.remember_source_alias(state, candidate.source, candidate.source_post_id, sample["sample_id"])

        self.store.update_state(mutate)
        state = self.store.read_state()
        return state.get("samples", {}).get(str(sample["sample_id"]), sample)

    def sample_by_sha(self, sha256: str) -> dict[str, Any] | None:
        state = self.store.read_state()
        return self.store.get_sample_by_sha(state, sha256)

    def record_source_failure(self, source: str, message: str, *, cfg: dict[str, Any] | None = None) -> None:
        cooldown_sec = float((cfg or {}).get("sampling", {}).get("source_failure_cooldown_sec") or 300)
        cooldown_until = time.time() + max(0.0, cooldown_sec)

        def mutate(state: dict[str, Any]) -> None:
            previous = state.setdefault("source_failures", {}).get(source, {})
            state["source_failures"][source] = {
                "message": message,
                "failed_at": now_iso(),
                "retry_count": int(previous.get("retry_count") or 0) + 1 if isinstance(previous, dict) else 1,
                "cooldown_until": cooldown_until,
                "cooldown_until_iso": _timestamp_to_iso(cooldown_until),
            }

        self.store.update_state(mutate)

    def clear_source_failure(self, source: str) -> None:
        def mutate(state: dict[str, Any]) -> None:
            state.setdefault("source_failures", {}).pop(source, None)

        self.store.update_state(mutate)

    def _cooldown_active(self, source: str) -> bool:
        failure = self.store.read_state().get("source_failures", {}).get(source)
        if not isinstance(failure, dict):
            return False
        return float(failure.get("cooldown_until") or 0) > time.time()

    def _candidate_sample(
        self,
        candidate: RemoteSourceCandidate,
        sha256: str,
        image_bytes: bytes,
        *,
        source_snapshot_id: str = "",
    ) -> dict[str, Any]:
        sample_id = int(sha256[:12], 16)
        local_path = self._remote_cache_path(candidate, sha256)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            local_path.write_bytes(image_bytes)
        width, height = _image_size(local_path)
        resolved = str(local_path.resolve())
        return {
            "sample_id": sample_id,
            "sample_key": f"sha256:{sha256}",
            "source": candidate.source,
            "source_path": "remote",
            "source_post_id": candidate.source_post_id,
            "source_page_url": candidate.source_page_url,
            "original_url": candidate.original_url,
            "local_path": resolved,
            "path": resolved,
            "name": local_path.name,
            "storage_state": "available",
            "width": width,
            "height": height,
            "sha256": sha256,
            "created_at": now_iso(),
            "imported_at": now_iso(),
            "mtime": local_path.stat().st_mtime,
            "size": local_path.stat().st_size,
            "source_metadata": dict(candidate.metadata),
            "source_snapshot_id": source_snapshot_id,
        }

    def _source_snapshot(self, provider: RemoteSourceProvider | None, candidate: RemoteSourceCandidate) -> tuple[str, dict[str, Any]]:
        source = provider.source_name if provider else candidate.source
        config = provider.config if provider else {}
        safe_config = _snapshot_config(config)
        payload = {"source": source, "config": safe_config}
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        snapshot_id = f"srcsnap_{source}_{digest[:12]}"
        return snapshot_id, {
            "snapshot_id": snapshot_id,
            "source": source,
            "base_url": getattr(provider, "base_url", "") if provider else "",
            "tags": str(config.get("tags") or ""),
            "limit": int(config.get("limit") or 0),
            "query_digest": f"sha1:{digest}",
            "credential_refs": _credential_refs(config),
            "created_at": now_iso(),
        }

    def _remote_cache_path(self, candidate: RemoteSourceCandidate, sha256: str) -> Path:
        safe_source = _safe_token(candidate.source, fallback="remote")
        safe_post_id = _safe_token(candidate.source_post_id, fallback="post")
        ext = candidate.file_ext if candidate.file_ext.startswith(".") else f".{candidate.file_ext}"
        return self.store.state_path.parent / "aesthetic_remote_cache" / safe_source / f"{sha256[:16]}_{safe_post_id}{ext}"


def _safe_token(value: str, *, fallback: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value) or fallback


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def _snapshot_config(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {"base_url", "tags", "limit", "user_agent", "username_env", "api_key_env", "login_env"}
    return {key: config.get(key) for key in sorted(allowed) if key in config}


def _credential_refs(config: dict[str, Any]) -> dict[str, str]:
    refs = {}
    for key in ("username_env", "api_key_env", "login_env"):
        if config.get(key):
            refs[key] = str(config.get(key))
    return refs


def _timestamp_to_iso(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(value))
