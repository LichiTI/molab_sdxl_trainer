"""Remote source providers for aesthetic labeling sample import."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


JsonFetcher = Callable[[str, dict[str, str], float], Any]
BytesFetcher = Callable[[str, dict[str, str], float], bytes]


@dataclass(frozen=True)
class RemoteSourceCandidate:
    source: str
    source_post_id: str
    source_page_url: str
    original_url: str
    file_ext: str = ".jpg"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteSourceHealth:
    source: str
    enabled: bool
    ok: bool
    count: int = 0
    message: str = ""


class RemoteSourceProvider:
    source_name = "remote"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        fetch_json: JsonFetcher | None = None,
        fetch_bytes: BytesFetcher | None = None,
    ) -> None:
        self.config = config or {}
        self._fetch_json = fetch_json or _fetch_json
        self._fetch_bytes = fetch_bytes or _fetch_bytes

    def fetch_candidate(self, *, timeout_sec: float = 8.0) -> RemoteSourceCandidate | None:
        raise NotImplementedError

    def load_image(self, candidate: RemoteSourceCandidate, *, timeout_sec: float = 8.0) -> bytes:
        return self._fetch_bytes(candidate.original_url, self._headers(), timeout_sec)

    def check_health(self) -> RemoteSourceHealth:
        if not self.enabled:
            return RemoteSourceHealth(self.source_name, enabled=False, ok=False, message="disabled")
        if not self.base_url:
            return RemoteSourceHealth(self.source_name, enabled=True, ok=False, message="base_url is not configured")
        return RemoteSourceHealth(self.source_name, enabled=True, ok=True, message="configured")

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled"))

    @property
    def base_url(self) -> str:
        return _normalize_base_url(str(self.config.get("base_url") or ""))

    def _headers(self) -> dict[str, str]:
        user_agent = str(self.config.get("user_agent") or "LulynxAestheticLabeler/1.0")
        return {"User-Agent": user_agent}


class DanbooruSourceProvider(RemoteSourceProvider):
    source_name = "danbooru"

    def fetch_candidate(self, *, timeout_sec: float = 8.0) -> RemoteSourceCandidate | None:
        if not self.enabled or not self.base_url:
            return None
        query: dict[str, Any] = {"limit": int(self.config.get("limit") or 20), "random": "true"}
        tags = str(self.config.get("tags") or "").strip()
        if tags:
            query["tags"] = tags
        username = os.getenv(str(self.config.get("username_env") or ""), "")
        api_key = os.getenv(str(self.config.get("api_key_env") or ""), "")
        if username and api_key:
            query["login"] = username
            query["api_key"] = api_key
        payload = self._fetch_json(_with_query(f"{self.base_url}/posts.json", query), self._headers(), timeout_sec)
        posts = payload if isinstance(payload, list) else payload.get("posts", []) if isinstance(payload, dict) else []
        for post in posts:
            candidate = _danbooru_candidate(self.base_url, post)
            if candidate and _candidate_allowed(candidate, self.config):
                return candidate
        return None


class E621SourceProvider(RemoteSourceProvider):
    source_name = "e621"

    def fetch_candidate(self, *, timeout_sec: float = 8.0) -> RemoteSourceCandidate | None:
        if not self.enabled or not self.base_url:
            return None
        query: dict[str, Any] = {"limit": int(self.config.get("limit") or 20)}
        tags = str(self.config.get("tags") or "").strip()
        if tags:
            query["tags"] = tags
        payload = self._fetch_json(_with_query(f"{self.base_url}/posts.json", query), self._headers(), timeout_sec)
        posts = payload.get("posts", []) if isinstance(payload, dict) else []
        for post in posts:
            candidate = _e621_candidate(self.base_url, post)
            if candidate and _candidate_allowed(candidate, self.config):
                return candidate
        return None

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        login = os.getenv(str(self.config.get("login_env") or ""), "")
        api_key = os.getenv(str(self.config.get("api_key_env") or ""), "")
        if login and api_key:
            token = base64.b64encode(f"{login}:{api_key}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return headers


def build_remote_providers(
    settings: dict[str, Any],
    *,
    fetch_json: JsonFetcher | None = None,
    fetch_bytes: BytesFetcher | None = None,
) -> list[RemoteSourceProvider]:
    sources = settings.get("sources", {}) if isinstance(settings, dict) else {}
    return [
        DanbooruSourceProvider(sources.get("danbooru") or {}, fetch_json=fetch_json, fetch_bytes=fetch_bytes),
        E621SourceProvider(sources.get("e621") or {}, fetch_json=fetch_json, fetch_bytes=fetch_bytes),
    ]


def remote_source_health(providers: list[RemoteSourceProvider]) -> list[dict[str, Any]]:
    return [provider.check_health().__dict__ for provider in providers if provider.enabled]


def _danbooru_candidate(base_url: str, post: Any) -> RemoteSourceCandidate | None:
    if not isinstance(post, dict) or not post.get("id"):
        return None
    image_url = str(post.get("file_url") or post.get("large_file_url") or post.get("preview_file_url") or "")
    if not image_url:
        return None
    post_id = str(post.get("id"))
    return RemoteSourceCandidate(
        source="danbooru",
        source_post_id=post_id,
        source_page_url=str(post.get("post_url") or f"{base_url}/posts/{post_id}"),
        original_url=image_url,
        file_ext=_url_ext(image_url),
        metadata={
            "rating": post.get("rating"),
            "tag_string": post.get("tag_string"),
            "width": post.get("image_width"),
            "height": post.get("image_height"),
        },
    )


def _e621_candidate(base_url: str, post: Any) -> RemoteSourceCandidate | None:
    if not isinstance(post, dict) or not post.get("id"):
        return None
    file_info = post.get("file") if isinstance(post.get("file"), dict) else {}
    image_url = str(file_info.get("url") or "")
    if not image_url:
        return None
    post_id = str(post.get("id"))
    tags = post.get("tags") if isinstance(post.get("tags"), dict) else {}
    return RemoteSourceCandidate(
        source="e621",
        source_post_id=post_id,
        source_page_url=f"{base_url}/posts/{post_id}",
        original_url=image_url,
        file_ext=_url_ext(image_url),
        metadata={
            "rating": post.get("rating"),
            "tags": tags,
            "width": file_info.get("width"),
            "height": file_info.get("height"),
        },
    )


def _candidate_allowed(candidate: RemoteSourceCandidate, config: dict[str, Any]) -> bool:
    filters = config.get("filters") if isinstance(config.get("filters"), dict) else {}
    allowed_exts = _list_value(filters.get("allowed_extensions") or config.get("allowed_extensions"))
    if allowed_exts:
        normalized_exts = {ext if ext.startswith(".") else f".{ext}" for ext in allowed_exts}
        if candidate.file_ext.lower() not in normalized_exts:
            return False
    rating = str(candidate.metadata.get("rating") or "").lower()
    allowed_ratings = set(_list_value(filters.get("allowed_ratings") or config.get("allowed_ratings")))
    if allowed_ratings and rating not in allowed_ratings:
        return False
    blocked_ratings = set(_list_value(filters.get("blocked_ratings") or config.get("blocked_ratings")))
    if blocked_ratings and rating in blocked_ratings:
        return False
    if not _passes_min_size(candidate, filters):
        return False
    tags = _candidate_tags(candidate)
    blacklist = set(_list_value(filters.get("tag_blacklist") or config.get("tag_blacklist")))
    if blacklist and tags.intersection(blacklist):
        return False
    required = set(_list_value(filters.get("required_tags") or config.get("required_tags")))
    if required and not required.issubset(tags):
        return False
    return True


def _passes_min_size(candidate: RemoteSourceCandidate, filters: dict[str, Any]) -> bool:
    min_width = int(filters.get("min_width") or 0)
    min_height = int(filters.get("min_height") or 0)
    if min_width <= 0 and min_height <= 0:
        return True
    width = _int_value(candidate.metadata.get("width"))
    height = _int_value(candidate.metadata.get("height"))
    if min_width > 0 and width < min_width:
        return False
    if min_height > 0 and height < min_height:
        return False
    return True


def _candidate_tags(candidate: RemoteSourceCandidate) -> set[str]:
    tags: set[str] = set()
    tag_string = candidate.metadata.get("tag_string")
    if isinstance(tag_string, str):
        tags.update(part.strip().lower() for part in tag_string.split() if part.strip())
    tag_map = candidate.metadata.get("tags")
    if isinstance(tag_map, dict):
        for value in tag_map.values():
            if isinstance(value, list):
                tags.update(str(item).strip().lower() for item in value if str(item).strip())
    return tags


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        return [part.strip().lower() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _fetch_json(url: str, headers: dict[str, str], timeout: float) -> Any:
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_bytes(url: str, headers: dict[str, str], timeout: float) -> bytes:
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def _with_query(url: str, query: dict[str, Any]) -> str:
    return f"{url}?{urlencode({key: value for key, value in query.items() if value not in {None, ''}})}"


def _url_ext(url: str) -> str:
    suffix = os.path.splitext(urlparse(url).path)[1].lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"
