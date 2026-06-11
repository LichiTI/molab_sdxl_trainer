"""Caption sidecar parsing helpers shared by prep, cache, and dataset readers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


_TAG_KEYS = ("tags", "tag", "caption", "prompt")
_NL_KEYS = ("nl", "natural_language", "description", "text")
_CONCEPT_KEYS = ("concept", "concept_group", "identity", "character")
_TRIGGER_KEYS = ("trigger", "triggers", "activation_tag", "activation_tags", "instance_prompt")
_PATH_KEYS = ("concept_path", "path", "hierarchy")
_CATEGORY_KEYS = ("categories", "tag_buckets", "buckets")


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _first_string(payload: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        values = _flatten_strings(value)
        if values:
            return values[0]
    return ""


def _parse_json(raw: str) -> Any | None:
    text = str(raw or "").strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def json_caption_to_training_text(
    raw: str,
    *,
    dual_caption_enabled: bool = False,
    dual_caption_short_key: str = "short",
    dual_caption_long_key: str = "long",
) -> str:
    """Return a trainer prompt string from txt or structured JSON sidecars."""

    if dual_caption_enabled:
        from .dual_caption import try_dual_caption
        result = try_dual_caption(raw, dual_caption_short_key, dual_caption_long_key)
        if result is not None:
            return result.strip()

    parsed = _parse_json(raw)
    if parsed is None:
        return str(raw or "").strip()
    if isinstance(parsed, list):
        return ", ".join(_flatten_strings(parsed)).strip()
    if not isinstance(parsed, Mapping):
        return str(raw or "").strip()

    parts: list[str] = []
    concept = _first_string(parsed, _CONCEPT_KEYS)
    if concept:
        parts.append(concept)
    for key in _TAG_KEYS:
        parts.extend(_flatten_strings(parsed.get(key)))
    for key in _CATEGORY_KEYS:
        categories = parsed.get(key)
        if isinstance(categories, Mapping):
            for values in categories.values():
                parts.extend(_flatten_strings(values))
    for key in _NL_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    deduped: list[str] = []
    for item in parts:
        text = str(item or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return ", ".join(deduped).strip()


def json_caption_to_training_parts(
    raw: str,
    *,
    dual_caption_enabled: bool = False,
    dual_caption_short_key: str = "short",
    dual_caption_long_key: str = "long",
) -> dict[str, Any]:
    """Return structured training parts when the sidecar is JSON.

    The result keeps tags and natural-language text separate so caption policy
    can shuffle tags without disturbing NL order. Plain-text captions fall back
    to a single ``text`` field and ``structured=False``.
    """

    if dual_caption_enabled:
        from .dual_caption import try_dual_caption
        result = try_dual_caption(raw, dual_caption_short_key, dual_caption_long_key)
        if result is not None:
            text = result.strip()
            return {
                "structured": False,
                "text": text,
                "tags": [text] if text else [],
                "nl": [],
            }

    parsed = _parse_json(raw)
    if parsed is None:
        text = str(raw or "").strip()
        return {
            "structured": False,
            "text": text,
            "tags": [text] if text else [],
            "nl": [],
        }

    if isinstance(parsed, list):
        tags = _flatten_strings(parsed)
        text = ", ".join(tags).strip()
        return {
            "structured": True,
            "text": text,
            "tags": tags,
            "nl": [],
        }

    if not isinstance(parsed, Mapping):
        text = str(raw or "").strip()
        return {
            "structured": False,
            "text": text,
            "tags": [text] if text else [],
            "nl": [],
        }

    parts: list[str] = []
    tags: list[str] = []
    trigger_parts: list[str] = []
    nl_parts: list[str] = []

    concept = _first_string(parsed, _CONCEPT_KEYS)
    if concept:
        trigger_parts.append(concept)
        tags.append(concept)
        parts.append(concept)
    for key in _TRIGGER_KEYS:
        values = _flatten_strings(parsed.get(key))
        trigger_parts.extend(values)
        parts.extend(values)
    for key in _TAG_KEYS:
        values = _flatten_strings(parsed.get(key))
        tags.extend(values)
        parts.extend(values)
    for key in _CATEGORY_KEYS:
        categories = parsed.get(key)
        if isinstance(categories, Mapping):
            for values in categories.values():
                flat_values = _flatten_strings(values)
                tags.extend(flat_values)
                parts.extend(flat_values)
    for key in _NL_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            nl_parts.append(value.strip())
            parts.append(value.strip())

    deduped_tags: list[str] = []
    for item in tags:
        text = str(item or "").strip()
        if text and text not in deduped_tags:
            deduped_tags.append(text)

    deduped_nl: list[str] = []
    for item in nl_parts:
        text = str(item or "").strip()
        if text and text not in deduped_nl:
            deduped_nl.append(text)

    deduped_triggers: list[str] = []
    for item in trigger_parts:
        text = str(item or "").strip()
        if text and text not in deduped_triggers:
            deduped_triggers.append(text)

    deduped_parts: list[str] = []
    for item in parts:
        text = str(item or "").strip()
        if text and text not in deduped_parts:
            deduped_parts.append(text)

    return {
        "structured": True,
        "text": ", ".join(deduped_parts).strip(),
        "tags": deduped_tags,
        "nl": deduped_nl,
        "triggers": deduped_triggers,
    }


def json_caption_to_concept_text(raw: str) -> str:
    """Return parse-friendly text for Concept Geometry from txt or JSON sidecars."""

    parsed = _parse_json(raw)
    if parsed is None:
        return str(raw or "").strip()
    if isinstance(parsed, list):
        return ", ".join(_flatten_strings(parsed)).strip()
    if not isinstance(parsed, Mapping):
        return str(raw or "").strip()

    lines: list[str] = []
    concept = _first_string(parsed, _CONCEPT_KEYS)
    if concept:
        lines.append(f"concept: {concept}")
    path_values = _flatten_strings(parsed.get("concept_path")) or _flatten_strings(parsed.get("path"))
    if path_values:
        lines.append("concept_path: " + " > ".join(path_values))
    for key in _TAG_KEYS:
        values = _flatten_strings(parsed.get(key))
        if values:
            lines.append(", ".join(values))
    for key in _CATEGORY_KEYS:
        categories = parsed.get(key)
        if isinstance(categories, Mapping):
            for bucket, values in categories.items():
                bucket_values = _flatten_strings(values)
                if bucket_values:
                    lines.append(f"{bucket}: " + ", ".join(bucket_values))
    for key in _NL_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"nl: {value.strip()}")
    return "\n".join(line for line in lines if line.strip()).strip()
