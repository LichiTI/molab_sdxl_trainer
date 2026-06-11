"""Structured caption source mixing helpers.

Keeps trigger tokens separate from tag/NL source selection so DiT-style
caption policies can mix:

- trigger + NL
- trigger + tags
- trigger only
- empty

The helpers are intentionally side-effect free so datasets can reuse them
without growing more ad hoc prompt logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import random
import re


@dataclass(frozen=True)
class CaptionSourceMixConfig:
    enabled: bool = False
    nl_ratio: float = 65.0
    tag_ratio: float = 20.0
    trigger_only_ratio: float = 10.0
    empty_ratio: float = 5.0
    trigger_tokens: tuple[str, ...] = ()


def parse_trigger_tokens(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [part.strip() for part in re.split(r"[\r\n,]+", raw) if part.strip()]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = [str(part).strip() for part in raw if str(part).strip()]
    else:
        text = str(raw).strip()
        items = [text] if text else []
    return _dedupe(items)


def normalize_caption_source_mix_config(
    *,
    enabled: object = False,
    nl_ratio: object = 65.0,
    tag_ratio: object = 20.0,
    trigger_only_ratio: object = 10.0,
    empty_ratio: object = 5.0,
    trigger_tokens: object = None,
) -> CaptionSourceMixConfig:
    return CaptionSourceMixConfig(
        enabled=_to_bool(enabled),
        nl_ratio=_non_negative_float(nl_ratio, 65.0),
        tag_ratio=_non_negative_float(tag_ratio, 20.0),
        trigger_only_ratio=_non_negative_float(trigger_only_ratio, 10.0),
        empty_ratio=_non_negative_float(empty_ratio, 5.0),
        trigger_tokens=parse_trigger_tokens(trigger_tokens),
    )


def merge_trigger_tokens(
    configured_tokens: Sequence[str] | None,
    structured_tokens: Sequence[str] | None,
) -> list[str]:
    merged: list[str] = []
    merged.extend(configured_tokens or ())
    merged.extend(structured_tokens or ())
    return list(_dedupe(merged))


def remove_trigger_tokens(tags: Sequence[str], triggers: Sequence[str]) -> list[str]:
    trigger_set = {str(token).strip().lower() for token in triggers if str(token).strip()}
    if not trigger_set:
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    cleaned: list[str] = []
    for tag in tags:
        text = str(tag).strip()
        if text and text.lower() not in trigger_set:
            cleaned.append(text)
    return cleaned


def select_caption_source(
    config: CaptionSourceMixConfig,
    *,
    has_tags: bool,
    has_nl: bool,
    has_triggers: bool,
) -> str | None:
    if not config.enabled:
        return None

    weighted_sources: list[tuple[str, float]] = []
    if has_nl and config.nl_ratio > 0:
        weighted_sources.append(("nl", float(config.nl_ratio)))
    if has_tags and config.tag_ratio > 0:
        weighted_sources.append(("tag", float(config.tag_ratio)))
    if has_triggers and config.trigger_only_ratio > 0:
        weighted_sources.append(("trigger_only", float(config.trigger_only_ratio)))
    if config.empty_ratio > 0:
        weighted_sources.append(("empty", float(config.empty_ratio)))

    total = sum(weight for _name, weight in weighted_sources)
    if total <= 0:
        return None

    sample = random.random() * total
    cursor = 0.0
    for name, weight in weighted_sources:
        cursor += weight
        if sample < cursor:
            return name
    return weighted_sources[-1][0]


def compose_caption_from_source(
    source: str | None,
    *,
    trigger_tokens: Sequence[str],
    tags: Sequence[str],
    nl_parts: Sequence[str],
) -> str:
    if source is None:
        return ", ".join(_dedupe(tags)).strip()
    if source == "empty":
        return ""
    if source == "trigger_only":
        return ", ".join(_dedupe(trigger_tokens)).strip()

    parts: list[str] = []
    parts.extend(trigger_tokens)
    if source == "nl":
        parts.extend(nl_parts)
    elif source == "tag":
        parts.extend(tags)
    return ", ".join(_dedupe(parts)).strip()


def caption_source_variant_texts(
    config: CaptionSourceMixConfig,
    structured_parts: Mapping[str, object],
) -> dict[str, str]:
    """Build deterministic prompt variants for cache-first text encoding."""
    if not config.enabled or not structured_parts.get("structured"):
        return {}

    tags = [str(item).strip() for item in structured_parts.get("tags", []) if str(item).strip()]  # type: ignore[arg-type]
    nl_parts = [str(item).strip() for item in structured_parts.get("nl", []) if str(item).strip()]  # type: ignore[arg-type]
    structured_triggers = [
        str(item).strip()
        for item in structured_parts.get("triggers", [])  # type: ignore[arg-type]
        if str(item).strip()
    ]
    triggers = merge_trigger_tokens(config.trigger_tokens, structured_triggers)
    body_tags = remove_trigger_tokens(tags, triggers)

    variants: dict[str, str] = {}
    if nl_parts and config.nl_ratio > 0:
        variants["nl"] = compose_caption_from_source(
            "nl",
            trigger_tokens=triggers,
            tags=body_tags,
            nl_parts=nl_parts,
        )
    if body_tags and config.tag_ratio > 0:
        variants["tag"] = compose_caption_from_source(
            "tag",
            trigger_tokens=triggers,
            tags=body_tags,
            nl_parts=nl_parts,
        )
    if triggers and config.trigger_only_ratio > 0:
        variants["trigger_only"] = compose_caption_from_source(
            "trigger_only",
            trigger_tokens=triggers,
            tags=body_tags,
            nl_parts=nl_parts,
        )
    if config.empty_ratio > 0:
        variants["empty"] = ""
    return variants


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return tuple(out)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on", "enable", "enabled"}


def _non_negative_float(value: object, default: float) -> float:
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return max(float(default), 0.0)
