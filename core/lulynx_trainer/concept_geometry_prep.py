"""Prepare local Concept Geometry metadata for trainer-side sampling.

Concept Geometry Sampling remains a trainer-side experiment: it does not alter the diffusion
loss or LoRA adapter math.  This prep step builds compact concept geometry
from local sources so the dataset can sample and weight examples with more
concept awareness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .dataset_discovery import (
        assign_stable_sample_ids,
        discover_smart_subsets,
        iter_images_for_subset,
        resolve_caption_path,
    )
    from .caption_sidecar import json_caption_to_concept_text
except ImportError:  # pragma: no cover - direct script smoke loading
    from dataset_discovery import (
        assign_stable_sample_ids,
        discover_smart_subsets,
        iter_images_for_subset,
        resolve_caption_path,
    )
    from caption_sidecar import json_caption_to_concept_text


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_LATENT_CACHE_PATTERN = re.compile(r"^(?P<stem>.+)_\d+x\d+_anima$")
_TEXT_CACHE_SUFFIX = "_anima_te"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_WEIGHT_SUFFIX_RE = re.compile(r"(?:^|[\s,])weight\s*:\s*[-+]?\d+(?:\.\d+)?\s*$", re.IGNORECASE)
_DEFAULT_SOURCE_WEIGHTS = {"dino": 0.30, "clip": 0.25, "text_embedding": 0.30, "latent": 0.20, "tags": 0.10}
_BACKENDS = {"auto", "lexical", "latent_tags", "clip", "dino", "hybrid"}
_TEXT_EMBEDDING_SOURCE = "text_embedding"
_DEFAULT_BGE_M3_REPO = "BAAI/bge-m3"
_DEFAULT_BGE_M3_MODEL_URL = "https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin?download=true"
_DEFAULT_BGE_M3_REPO_URL = "https://huggingface.co/BAAI/bge-m3/tree/main"
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_GENERIC_CONCEPT_TAGS = {
    "1girl",
    "1boy",
    "solo",
    "portrait",
    "full body",
    "upper body",
    "furry",
    "furry female",
    "furry male",
    "best quality",
    "masterpiece",
    "high quality",
    "reference sheet",
}


@dataclass
class SampleRef:
    stem: str
    sample_id: str
    rel_parts: Tuple[str, ...]
    caption_path: Optional[Path] = None
    image_path: Optional[Path] = None
    latent_path: Optional[Path] = None


@dataclass
class SourceFeature:
    name: str
    features: np.ndarray
    summaries: Dict[str, Dict[str, Any]]
    fallback_reasons: List[str]


@dataclass
class ParsedCaption:
    text: str
    tags: List[str]
    tag_buckets: Dict[str, List[str]]
    concept_group: str = ""
    concept_group_source: str = ""
    concept_path: List[str] = None
    co_concepts: List[str] = None
    parse_confidence: float = 0.5
    parse_warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.concept_path is None:
            self.concept_path = []
        if self.co_concepts is None:
            self.co_concepts = []
        if self.parse_warnings is None:
            self.parse_warnings = []


_CATEGORY_ALIASES = {
    "id": "identity",
    "identity": "identity",
    "character": "identity",
    "角色": "identity",
    "人物": "identity",
    "主体": "identity",
    "角色名": "identity",
    "char": "identity",
    "person": "identity",
    "subject": "identity",
    "species": "identity",
    "face": "identity",
    "hair": "identity",
    "eyes": "identity",
    "clothes": "appearance",
    "服装": "appearance",
    "衣服": "appearance",
    "穿着": "appearance",
    "外观": "appearance",
    "配饰": "appearance",
    "饰品": "appearance",
    "clothing": "appearance",
    "outfit": "appearance",
    "costume": "appearance",
    "wardrobe": "appearance",
    "appearance": "appearance",
    "accessory": "appearance",
    "accessories": "appearance",
    "pose": "pose",
    "姿势": "pose",
    "动作": "pose",
    "表情": "pose",
    "情绪": "pose",
    "action": "pose",
    "gesture": "pose",
    "expression": "pose",
    "mood": "pose",
    "setting": "setting",
    "场景": "setting",
    "背景": "setting",
    "地点": "setting",
    "环境": "setting",
    "background": "setting",
    "location": "setting",
    "scene": "setting",
    "environment": "setting",
    "style": "style",
    "风格": "style",
    "画风": "style",
    "媒介": "style",
    "质量": "style",
    "medium": "style",
    "artist": "style",
    "render": "style",
    "quality": "style",
    "other": "other",
    "misc": "other",
}
_CONCEPT_KEYS = {"concept", "concept_group", "group", "main", "primary", "概念", "主概念", "分组", "组"}
_CONCEPT_PATH_KEYS = {"concept_path", "path", "hierarchy", "概念路径", "路径", "层级"}
_NL_KEYS = {"nl", "natural_language", "description", "text", "自然语言", "描述"}
_TAG_KEYS = {"tag", "tags", "caption", "prompt", "标签"}
_NL_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "while", "wearing", "wears", "is", "are", "in", "on", "at",
    "of", "to", "from", "by", "for", "as", "portrait", "image", "picture", "photo", "artwork",
}
_APPEARANCE_NOUNS = {
    "dress", "skirt", "shirt", "hoodie", "jacket", "coat", "armor", "uniform", "suit", "cape", "hat", "boots",
    "shoes", "gloves", "scarf", "ribbon", "tie", "sweater", "kimono", "robe", "pants", "shorts", "stockings",
    "hair", "eyes", "horns", "ears", "tail", "wings", "glasses", "mask", "sword", "staff", "bag",
}
_SETTING_NOUNS = {
    "garden", "city", "street", "room", "bedroom", "studio", "forest", "beach", "sky", "window", "castle",
    "school", "classroom", "cafe", "train", "station", "bridge", "river", "snow", "rain", "night", "sunset",
    "moonlight", "background", "indoors", "outdoors",
}
_STYLE_TERMS = {
    "anime", "manga", "watercolor", "sketch", "painting", "photo", "photorealistic", "cinematic", "render",
    "3d", "chibi", "pixel", "lineart", "monochrome", "noir", "cyberpunk", "fantasy",
}
_POSE_TERMS = {
    "standing", "sitting", "kneeling", "walking", "running", "smiling", "looking back", "looking away",
    "looking at camera", "looking at viewer", "arms crossed", "holding", "waving", "jumping", "lying down",
}
_NL_TRAILING_RE = re.compile(
    r"\b(?:while|with|and|standing|sitting|looking|posing|smiling|walking|running|holding|in front of|inside|near|in|at|on)\b.*$",
    re.IGNORECASE,
)
_NL_PATTERNS = (
    ("appearance", re.compile(r"\b(?:wearing|wears|dressed in|with)\s+(?P<value>[^.,;\n]+)", re.IGNORECASE)),
    ("setting", re.compile(r"\b(?:in front of|inside|in|at|near|on)\s+(?P<value>[^.,;\n]+)", re.IGNORECASE)),
    ("style", re.compile(r"\b(?:in|as|with)\s+(?P<value>(?:anime|manga|watercolor|sketch|painting|cinematic|photorealistic|3d render|chibi|pixel|cyberpunk|fantasy)[^.,;\n]*)", re.IGNORECASE)),
    ("pose", re.compile(r"\b(?P<value>standing|sitting|kneeling|looking at camera|looking at viewer|looking away|looking back|arms crossed|smiling|walking|running|waving|jumping|lying down|holding [^.,;\n]+)", re.IGNORECASE)),
    ("identity", re.compile(r"^(?:portrait of |an image of |a photo of |a drawing of |a painting of )?(?P<value>[A-Za-z0-9_ -]{2,48}?)\s+(?:is|are|wears|wearing|standing|sitting|looking|smiling|with|in)\b", re.IGNORECASE)),
)
_OF_IDENTITY_RE = re.compile(r"\b(?:portrait|image|photo|drawing|painting|version)\s+of\s+(?P<value>[A-Za-z0-9_ -]{2,48})(?:\b|,)", re.IGNORECASE)
_CO_CONCEPT_RE = re.compile(r"\b(?:with|and|alongside)\s+(?P<value>[A-Za-z][A-Za-z0-9_ -]{1,32})(?:\b|,)", re.IGNORECASE)
_WITHOUT_RE = re.compile(r"\b(?:without|no)\s+(?P<value>[A-Za-z0-9_ -]{2,48})(?:\b|,)", re.IGNORECASE)
_ZH_TRAILING_RE = re.compile(r"(?:，|。|；|、|并且|同时|然后|站在|坐在|看着|拿着|位于|在).*$")
_ZH_PATTERNS = (
    ("identity", re.compile(r"^\s*(?P<value>[\u3400-\u9fffA-Za-z0-9_ -]{1,32}?)(?:穿着|身穿|戴着|站在|坐在|在|是)")),
    ("appearance", re.compile(r"(?:穿着|身穿|戴着|拿着)(?P<value>[^，。；、\n]+)")),
    ("setting", re.compile(r"(?:在|位于|背景是|场景是)(?P<value>[^，。；、\n]+?)(?:里|中|内|前|旁|附近|$)")),
    ("pose", re.compile(r"(?P<value>站立|站着|站在|坐着|坐在|微笑|看向镜头|看着镜头|回头|行走|奔跑|拿着[^，。；、\n]+)")),
)


def _normalize_phrase(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = normalized.replace("_", " ").replace("/", " ").replace("\\", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip(" ,")


def _strip_caption_weight(text: str) -> str:
    return _WEIGHT_SUFFIX_RE.sub("", text or "").strip(" ,\n\r\t")


def _split_tags(text: str) -> List[str]:
    return _parse_caption(text).tags


def _bucket_name(raw_key: str) -> str:
    key = _normalize_phrase(raw_key)
    key = key.replace(" ", "_")
    return _CATEGORY_ALIASES.get(key, "")


def _clean_nl_value(raw: str) -> str:
    value = _NL_TRAILING_RE.sub("", str(raw or ""))
    value = re.sub(r"\b(?:a|an|the)\b", " ", value, flags=re.IGNORECASE)
    value = _normalize_phrase(value)
    pieces = [piece for piece in value.split() if piece not in _NL_STOPWORDS]
    return " ".join(pieces).strip()


def _clean_pose_value(raw: str) -> str:
    value = re.sub(r"\b(?:near|inside|in front of|in|at|on|with|while)\b.*$", "", str(raw or ""), flags=re.IGNORECASE)
    value = re.sub(r"\b(?:a|an|the)\b", " ", value, flags=re.IGNORECASE)
    return _normalize_phrase(value)


def _strip_without_phrases(text: str) -> str:
    return _WITHOUT_RE.sub("", str(text or ""))


def _extract_dictionary_phrases(text: str, nouns: set[str], *, max_words: int = 4) -> List[str]:
    cleaned = _strip_without_phrases(text)
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned.lower())
    phrases: List[str] = []
    for idx, token in enumerate(tokens):
        if token not in nouns:
            continue
        if idx > 0 and tokens[idx - 1] not in _NL_STOPWORDS:
            compact = " ".join(piece for piece in (tokens[idx - 1], token) if piece not in _NL_STOPWORDS)
            if compact and compact not in phrases:
                phrases.append(compact)
        start = max(0, idx - max_words + 1)
        phrase_tokens = [piece for piece in tokens[start : idx + 1] if piece not in _NL_STOPWORDS]
        if not phrase_tokens:
            continue
        phrase = " ".join(phrase_tokens)
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _extract_style_terms(text: str) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    for term in sorted(_STYLE_TERMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", lowered) and term not in found:
            found.append(term)
    return found


def _extract_pose_terms(text: str) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    for term in sorted(_POSE_TERMS, key=len, reverse=True):
        if term == "holding":
            continue
        if re.search(rf"\b{re.escape(term)}\b", lowered) and term not in found:
            found.append(term)
    return found


def _looks_like_co_concept(value: str) -> bool:
    normalized = _normalize_phrase(value)
    if not normalized or _is_generic_concept(normalized):
        return False
    tokens = normalized.split()
    if len(tokens) > 4:
        return False
    blocked = _APPEARANCE_NOUNS | _SETTING_NOUNS | _STYLE_TERMS | {"wearing", "standing", "sitting", "holding"}
    return not any(token in blocked for token in tokens)


def _clean_zh_value(raw: str) -> str:
    value = _ZH_TRAILING_RE.sub("", str(raw or ""))
    value = re.sub(r"^(一个|一位|一名|这张|画面中|图片中)", "", value)
    value = _normalize_phrase(value)
    return value.strip("的地得了在里中内前旁附近")


def _looks_like_natural_language(text: str) -> bool:
    cleaned = _strip_caption_weight(text)
    if not cleaned:
        return False
    if _CJK_RE.search(cleaned):
        if any(sep in cleaned for sep in ("，", "。", "；")):
            return True
        return any(token in cleaned for token in ("穿着", "身穿", "站在", "坐在", "背景", "场景", "看向", "微笑", "戴着"))
    if ":" in cleaned or "\n" in cleaned:
        return False
    words = _TOKEN_RE.findall(cleaned.lower())
    if len(words) < 5:
        return False
    if "," in cleaned:
        strong_markers = {"is", "are", "wearing", "wears", "with", "near", "inside", "portrait", "photo", "image"}
        return any(word in words for word in strong_markers)
    return any(word in words for word in ("is", "are", "wearing", "wears", "standing", "sitting", "looking", "with", "near", "inside", "portrait", "photo", "image"))


def _append_bucket(buckets: Dict[str, List[str]], bucket: str, value: str) -> None:
    normalized = _normalize_phrase(value)
    if not normalized:
        return
    buckets.setdefault(bucket or "other", [])
    if normalized not in buckets[bucket or "other"]:
        buckets[bucket or "other"].append(normalized)


def _caption_pieces(line: str) -> List[str]:
    return [piece.strip() for piece in re.split(r"[,，]+", line) if piece.strip()]


def _split_explicit_key_value(piece: str) -> Tuple[str, str]:
    for sep in (":", "：", "="):
        if sep in piece:
            left, right = piece.split(sep, 1)
            return left, right
    return "", piece


def _parse_caption(text: str) -> ParsedCaption:
    cleaned = _strip_caption_weight(text)
    buckets: Dict[str, List[str]] = {}
    tags: List[str] = []
    concept_group = ""
    concept_group_source = ""
    concept_path: List[str] = []
    co_concepts: List[str] = []
    parse_warnings: List[str] = []
    is_natural_language = _looks_like_natural_language(cleaned)

    current_bucket = ""
    if not is_natural_language:
        def merge_nested(nested: ParsedCaption) -> None:
            nonlocal concept_group, concept_group_source
            for nested_tag in nested.tags:
                if nested_tag not in tags:
                    tags.append(nested_tag)
            for nested_bucket, nested_values in nested.tag_buckets.items():
                for nested_value in nested_values:
                    _append_bucket(buckets, nested_bucket, nested_value)
            for nested_co in nested.co_concepts:
                if nested_co not in co_concepts:
                    co_concepts.append(nested_co)
            for nested_warning in nested.parse_warnings:
                if nested_warning not in parse_warnings:
                    parse_warnings.append(nested_warning)
            if nested.concept_group and not concept_group:
                concept_group = nested.concept_group
                concept_group_source = nested.concept_group_source or "nl"

        for raw_line in re.split(r"[\n;；]+", cleaned):
            line = raw_line.strip()
            if not line:
                continue
            line_key, line_value = _split_explicit_key_value(line)
            if line_key and _normalize_phrase(line_key).replace(" ", "_") in _NL_KEYS:
                merge_nested(_parse_caption(line_value))
                continue
            for raw_piece in _caption_pieces(line):
                piece = raw_piece.strip()
                if not piece:
                    continue
                key = ""
                value = piece
                maybe_key, maybe_value = _split_explicit_key_value(piece)
                if maybe_key:
                    normalized_key = _normalize_phrase(maybe_key).replace(" ", "_")
                    mapped_bucket = _bucket_name(maybe_key)
                    if normalized_key in _NL_KEYS:
                        merge_nested(_parse_caption(maybe_value))
                        continue
                    if normalized_key in _TAG_KEYS and _looks_like_natural_language(maybe_value):
                        merge_nested(_parse_caption(maybe_value))
                        continue
                    if normalized_key in _CONCEPT_KEYS:
                        concept_group = _normalize_phrase(maybe_value)
                        concept_group_source = normalized_key
                        current_bucket = "identity"
                        if concept_group:
                            _append_bucket(buckets, "identity", concept_group)
                            tags.append(concept_group)
                        continue
                    if normalized_key in _CONCEPT_PATH_KEYS:
                        concept_path = [
                            _normalize_phrase(part)
                            for part in re.split(r"(?:>|/|\|)", maybe_value)
                            if _normalize_phrase(part)
                        ]
                        current_bucket = ""
                        tags.extend([part for part in concept_path if part not in tags])
                        continue
                    if mapped_bucket:
                        key = mapped_bucket
                        value = maybe_value
                        current_bucket = mapped_bucket

                tag = _normalize_phrase(value)
                if not tag:
                    continue
                if _looks_like_natural_language(tag):
                    merge_nested(_parse_caption(value))
                    continue
                if tag not in tags:
                    tags.append(tag)
                if key or current_bucket:
                    _append_bucket(buckets, key or current_bucket, tag)

    if is_natural_language:
        excluded = {_clean_nl_value(match.group("value")) for match in _WITHOUT_RE.finditer(cleaned)}
        for bucket, pattern in _NL_PATTERNS:
            for match in pattern.finditer(cleaned):
                value = _clean_pose_value(match.group("value")) if bucket == "pose" else _clean_nl_value(match.group("value"))
                if not value or _is_generic_concept(value) or value in excluded:
                    continue
                if bucket == "identity" and not concept_group:
                    concept_group = value
                    concept_group_source = "nl"
                if value not in tags:
                    tags.append(value)
                _append_bucket(buckets, bucket, value)
        for match in _OF_IDENTITY_RE.finditer(cleaned):
            value = _clean_nl_value(match.group("value"))
            if value and not _is_generic_concept(value) and not concept_group:
                concept_group = value
                concept_group_source = "nl_of"
                if value not in tags:
                    tags.append(value)
                _append_bucket(buckets, "identity", value)
        for match in _CO_CONCEPT_RE.finditer(cleaned):
            value = _clean_nl_value(match.group("value"))
            if _looks_like_co_concept(value) and value != concept_group and value not in co_concepts:
                co_concepts.append(value)
                if value not in tags:
                    tags.append(value)
                _append_bucket(buckets, "identity", value)
        for value in _extract_dictionary_phrases(cleaned, _APPEARANCE_NOUNS):
            if value not in excluded:
                if value not in tags:
                    tags.append(value)
                _append_bucket(buckets, "appearance", value)
        for value in _extract_dictionary_phrases(cleaned, _SETTING_NOUNS):
            if value not in excluded:
                if value not in tags:
                    tags.append(value)
                _append_bucket(buckets, "setting", value)
        for value in _extract_style_terms(cleaned):
            if value not in tags:
                tags.append(value)
            _append_bucket(buckets, "style", value)
        for value in _extract_pose_terms(cleaned):
            if value not in tags:
                tags.append(value)
            _append_bucket(buckets, "pose", value)
        for bucket, pattern in _ZH_PATTERNS:
            for match in pattern.finditer(cleaned):
                value = _normalize_phrase(match.group("value")) if bucket == "pose" else _clean_zh_value(match.group("value"))
                if not value or _is_generic_concept(value):
                    continue
                if bucket == "identity" and not concept_group:
                    concept_group = value
                    concept_group_source = "zh_nl"
                if value not in tags:
                    tags.append(value)
                _append_bucket(buckets, bucket, value)
        if not buckets:
            nl_tags = [
                token
                for token in _TOKEN_RE.findall(cleaned.lower())
                if token not in _NL_STOPWORDS and len(token) > 2
            ]
            for token in nl_tags[:8]:
                if token not in tags:
                    tags.append(token)
                _append_bucket(buckets, "other", token)

    if not buckets:
        buckets = _tag_buckets(tags)
        if not tags:
            parse_warnings.append("caption produced no tags")
    else:
        inferred = _tag_buckets(tags)
        for bucket, values in inferred.items():
            for value in values:
                _append_bucket(buckets, bucket, value)
    if len(buckets.get("identity", [])) > 1 and co_concepts:
        parse_warnings.append("multiple identity concepts detected")
    parse_confidence = 0.35
    if concept_group:
        parse_confidence += 0.25
    if buckets.get("appearance"):
        parse_confidence += 0.10
    if buckets.get("pose"):
        parse_confidence += 0.10
    if buckets.get("setting"):
        parse_confidence += 0.10
    if buckets.get("style"):
        parse_confidence += 0.05
    if parse_warnings:
        parse_confidence -= 0.10

    return ParsedCaption(
        text=cleaned,
        tags=tags,
        tag_buckets={key: value for key, value in buckets.items() if value},
        concept_group=concept_group,
        concept_group_source=concept_group_source,
        concept_path=concept_path,
        co_concepts=co_concepts,
        parse_confidence=float(np.clip(parse_confidence, 0.0, 1.0)),
        parse_warnings=parse_warnings,
    )


def _tag_buckets(tags: Sequence[str]) -> Dict[str, List[str]]:
    buckets = {"identity": [], "appearance": [], "pose": [], "setting": [], "style": [], "other": []}
    keys = {
        "identity": ("character", "oc", "lulu", "momo", "raven", "face", "hair", "eyes", "horn", "tail", "ear"),
        "appearance": ("dress", "shirt", "outfit", "uniform", "color", "hat", "shoes", "jacket", "hoodie", "armor", "cape"),
        "pose": ("pose", "standing", "sitting", "looking", "smile", "hand", "angle"),
        "setting": ("background", "room", "city", "forest", "beach", "street", "sky", "studio", "garden", "neon"),
        "style": ("anime", "photo", "painting", "sketch", "render", "style", "cinematic"),
    }
    for tag in tags:
        placed = False
        for bucket, needles in keys.items():
            if any(needle in tag for needle in needles):
                buckets[bucket].append(tag)
                placed = True
                break
        if not placed:
            buckets["other"].append(tag)
    return {key: value for key, value in buckets.items() if value}


def _is_generic_concept(value: str) -> bool:
    normalized = _normalize_phrase(value)
    if not normalized:
        return True
    if normalized in _GENERIC_CONCEPT_TAGS:
        return True
    if re.fullmatch(r"(?:sample|img|image|frame)?\s*\d+", normalized):
        return True
    return False


def _load_alias_map(alias_map: str | Mapping[str, Any] | None = None, alias_map_path: str | Path = "") -> Dict[str, str]:
    payload: Any = alias_map or {}
    if alias_map_path:
        path = Path(alias_map_path).expanduser()
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, str) and payload.strip():
        payload = json.loads(payload)
    aliases: Dict[str, str] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_norm = _normalize_phrase(str(key))
            if isinstance(value, (list, tuple)):
                canonical = _normalize_phrase(str(key))
                for item in value:
                    item_norm = _normalize_phrase(str(item))
                    if item_norm and canonical:
                        aliases[item_norm] = canonical
            else:
                canonical = _normalize_phrase(str(value))
                if key_norm and canonical:
                    aliases[key_norm] = canonical
    return aliases


def _parse_source_priority(priority: str | Sequence[str]) -> List[str]:
    if isinstance(priority, str):
        raw_items = re.split(r"[,>\s]+", priority)
    else:
        raw_items = [str(item) for item in priority]
    aliases = {
        "caption": "explicit",
        "explicit_caption": "explicit",
        "natural": "nl",
        "natural_language": "nl",
        "directory": "folder",
        "bucket": "identity",
    }
    parsed: List[str] = []
    for raw in raw_items:
        item = aliases.get(str(raw or "").strip().lower(), str(raw or "").strip().lower())
        if item and item not in parsed:
            parsed.append(item)
    for fallback in ("explicit", "folder", "nl", "identity", "tag", "stem"):
        if fallback not in parsed:
            parsed.append(fallback)
    return parsed


def _canonicalize(value: str, aliases: Mapping[str, str]) -> str:
    normalized = _normalize_phrase(value)
    return str(aliases.get(normalized, normalized))


def _canonicalize_list(values: Sequence[str], aliases: Mapping[str, str]) -> List[str]:
    out: List[str] = []
    for value in values:
        normalized = _canonicalize(str(value), aliases)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _apply_aliases_to_metadata(meta: Dict[str, Any], aliases: Mapping[str, str]) -> None:
    if not aliases:
        return
    meta["tags"] = _canonicalize_list(meta.get("tags", ()), aliases)
    meta["concept_path"] = _canonicalize_list(meta.get("concept_path", ()), aliases)
    meta["co_concepts"] = _canonicalize_list(meta.get("co_concepts", ()), aliases)
    if meta.get("concept_group"):
        meta["concept_group"] = _canonicalize(str(meta["concept_group"]), aliases)
    buckets = meta.get("tag_buckets", {})
    if isinstance(buckets, Mapping):
        meta["tag_buckets"] = {key: _canonicalize_list(values, aliases) for key, values in buckets.items()}


def _choose_concept_group(
    rel_parts: Sequence[str],
    tags: Sequence[str],
    tag_buckets: Mapping[str, Sequence[str]],
    stem: str,
) -> Tuple[str, str]:
    for raw in rel_parts:
        normalized = _normalize_phrase(raw)
        if normalized and not _is_generic_concept(normalized):
            return normalized, "folder"
    for bucket in ("identity", "appearance", "setting", "other"):
        for raw in tag_buckets.get(bucket, ()):
            normalized = _normalize_phrase(raw)
            if normalized and not _is_generic_concept(normalized):
                return normalized, bucket
    for raw in tags:
        normalized = _normalize_phrase(raw)
        if normalized and not _is_generic_concept(normalized):
            return normalized, "tag"
    return f"sample:{_normalize_phrase(stem) or stem}", "stem"


def _choose_concept_group_with_priority(
    *,
    rel_parts: Sequence[str],
    parsed: ParsedCaption,
    stem: str,
    priority: Sequence[str],
) -> Tuple[str, str]:
    candidates: Dict[str, Tuple[str, str]] = {}
    folder_tags = [_normalize_phrase(part) for part in rel_parts if _normalize_phrase(part) and not _is_generic_concept(part)]
    if folder_tags:
        candidates["folder"] = (folder_tags[0], "folder")
    if parsed.concept_group:
        source = str(parsed.concept_group_source or "caption")
        if source in _CONCEPT_KEYS or source in {"concept", "concept_group", "group", "main", "primary"}:
            candidates["explicit"] = (parsed.concept_group, source)
        else:
            candidates["nl"] = (parsed.concept_group, source)
    for bucket in ("identity", "appearance", "setting", "other"):
        for raw in parsed.tag_buckets.get(bucket, ()):
            normalized = _normalize_phrase(raw)
            if normalized and not _is_generic_concept(normalized):
                candidates.setdefault(bucket, (normalized, bucket))
                if bucket == "identity":
                    candidates.setdefault("tag", (normalized, "identity"))
                break
    for raw in parsed.tags:
        normalized = _normalize_phrase(raw)
        if normalized and not _is_generic_concept(normalized):
            candidates.setdefault("tag", (normalized, "tag"))
            break
    candidates["stem"] = (f"sample:{_normalize_phrase(stem) or stem}", "stem")
    for source in priority:
        key = str(source or "").strip().lower()
        if key in candidates:
            return candidates[key]
    return candidates.get("explicit") or candidates.get("folder") or candidates.get("nl") or candidates.get("tag") or candidates["stem"]


def _stable_hash_bucket(token: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % max(dim, 1), (1.0 if (digest[4] & 1) == 0 else -1.0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        out = np.zeros_like(array, dtype=np.float32)
        if out.size:
            out[0] = 1.0
        return out
    return array / norm


def _normalize_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _resize_feature(vector: np.ndarray, dim: int) -> np.ndarray:
    flat = _normalize_vector(vector)
    target = max(int(dim or flat.size or 1), 1)
    if flat.size == target:
        return flat.astype(np.float32)
    out = np.zeros((target,), dtype=np.float32)
    if flat.size == 0:
        out[0] = 1.0
        return out
    if flat.size > target:
        for idx, value in enumerate(flat):
            out[idx % target] += float(value)
    else:
        out[: flat.size] = flat
    return _normalize_vector(out)


def _build_hashed_feature(tokens: Sequence[str], dim: int) -> np.ndarray:
    vector = np.zeros((max(int(dim), 1),), dtype=np.float32)
    for token in tokens:
        for piece in [token, *_TOKEN_RE.findall(token)]:
            if not piece:
                continue
            index, sign = _stable_hash_bucket(piece, vector.size)
            vector[index] += sign
    if not tokens:
        vector[0] = 1.0
    return _normalize_vector(vector)


def _discover_samples(data_dir: Path, caption_extension: str) -> List[SampleRef]:
    caption_ext = str(caption_extension or ".txt")
    sample_map: Dict[Tuple[str, Path], SampleRef] = {}

    def ensure(stem: str, root: Path, rel_parts: Sequence[str]) -> SampleRef:
        key = (stem, root)
        existing = sample_map.get(key)
        if existing is not None:
            return existing
        ref = SampleRef(stem=stem, sample_id=stem, rel_parts=tuple(rel_parts))
        sample_map[key] = ref
        return ref

    for subset in discover_smart_subsets(data_dir):
        for path in iter_images_for_subset(subset):
            if not any(path.stem.endswith(sidecar) for sidecar in ("_mask", "_alpha")):
                ref = ensure(path.stem, subset.root, subset.rel_parts)
                ref.image_path = path
                if ref.caption_path is None:
                    ref.caption_path = resolve_caption_path(subset.root, path.stem, caption_ext)
        for path in sorted(subset.root.glob(f"*{caption_ext}")):
            if path.is_file():
                if Path(path.stem).suffix.lower() in _IMAGE_SUFFIXES:
                    continue
                ref = ensure(path.stem, subset.root, subset.rel_parts)
                if ref.caption_path is None:
                    ref.caption_path = resolve_caption_path(subset.root, path.stem, caption_ext) or path
        for path in sorted(subset.root.glob("*")):
            if not path.is_file():
                continue
            if path.stem.endswith(_TEXT_CACHE_SUFFIX):
                ensure(path.stem[: -len(_TEXT_CACHE_SUFFIX)], subset.root, subset.rel_parts)
                continue
            latent_match = _LATENT_CACHE_PATTERN.match(path.stem)
            if latent_match is not None:
                ref = ensure(latent_match.group("stem"), subset.root, subset.rel_parts)
                ref.latent_path = path
                if ref.caption_path is None:
                    ref.caption_path = resolve_caption_path(subset.root, ref.stem, caption_ext)

    id_map = assign_stable_sample_ids(list(sample_map.keys()), data_dir)
    for key, ref in sample_map.items():
        ref.sample_id = id_map[key]
    return [sample_map[key] for key in sorted(sample_map.keys(), key=lambda item: (str(item[1]), item[0]))]


def _read_caption(sample: SampleRef) -> str:
    if sample.caption_path is None or not sample.caption_path.is_file():
        return ""
    raw = sample.caption_path.read_text(encoding="utf-8", errors="ignore")
    return json_caption_to_concept_text(raw)


def _infer_concept_path(rel_parts: Sequence[str], tags: Sequence[str], *, max_depth: int) -> List[str]:
    concept_path: List[str] = []
    seen = set()
    for raw in list(rel_parts) + list(tags):
        normalized = _normalize_phrase(raw)
        if not normalized or normalized in seen:
            continue
        concept_path.append(normalized)
        seen.add(normalized)
        if len(concept_path) >= max_depth:
            break
    return concept_path


def _base_metadata(
    samples: Sequence[SampleRef],
    concept_depth: int,
    *,
    aliases: Mapping[str, str] | None = None,
    concept_source_priority: str | Sequence[str] = "explicit,folder,nl,identity,tag,stem",
) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    alias_map = aliases or {}
    priority = _parse_source_priority(concept_source_priority)
    for sample in samples:
        caption_text = _read_caption(sample)
        parsed = _parse_caption(caption_text)
        if alias_map:
            parsed.tags = _canonicalize_list(parsed.tags, alias_map)
            parsed.concept_path = _canonicalize_list(parsed.concept_path, alias_map)
            parsed.co_concepts = _canonicalize_list(parsed.co_concepts, alias_map)
            parsed.concept_group = _canonicalize(parsed.concept_group, alias_map) if parsed.concept_group else ""
            parsed.tag_buckets = {
                key: _canonicalize_list(values, alias_map)
                for key, values in parsed.tag_buckets.items()
            }
        tags = parsed.tags
        tag_buckets = parsed.tag_buckets
        concept_group, concept_group_source = _choose_concept_group_with_priority(
            rel_parts=sample.rel_parts,
            parsed=parsed,
            stem=sample.stem,
            priority=priority,
        )
        if alias_map and concept_group:
            concept_group = _canonicalize(concept_group, alias_map)
        concept_path = list(parsed.concept_path) if parsed.concept_path else _infer_concept_path(sample.rel_parts, tags, max_depth=concept_depth)
        if alias_map:
            concept_path = _canonicalize_list(concept_path, alias_map)
        if concept_group and concept_group not in concept_path:
            concept_path = [concept_group, *concept_path]
            concept_path = concept_path[: max(int(concept_depth), 1)]
        meta = {
            "caption": parsed.text,
            "tags": tags,
            "tag_buckets": tag_buckets,
            "concept_path": concept_path,
            "path_depth": len(concept_path),
            "concept_group": concept_group,
            "concept_group_source": concept_group_source,
            "co_concepts": list(parsed.co_concepts),
            "parse_confidence": round(float(parsed.parse_confidence), 6),
            "parse_warnings": list(parsed.parse_warnings),
        }
        _apply_aliases_to_metadata(meta, alias_map)
        metadata.append(meta)
    return metadata


def _build_tags_source(samples: Sequence[SampleRef], metadata: Sequence[Dict[str, Any]], feature_dim: int) -> SourceFeature:
    vectors = []
    summaries: Dict[str, Dict[str, Any]] = {}
    for sample, meta in zip(samples, metadata):
        tokens: List[str] = []
        tokens.extend(meta["concept_path"])
        tokens.extend(meta["tags"])
        vectors.append(_build_hashed_feature(tokens or [sample.stem.lower()], feature_dim))
        summaries[sample.sample_id] = {"tag_count": len(meta["tags"]), "bucket_count": len(meta["tag_buckets"])}
    return SourceFeature("tags", np.stack(vectors, axis=0), summaries, [])


def _text_for_embedding(sample: SampleRef, meta: Mapping[str, Any]) -> str:
    parts: List[str] = []
    if meta.get("translated_caption"):
        parts.append(str(meta["translated_caption"]))
    if meta.get("caption"):
        parts.append(str(meta["caption"]))
    concept_path = meta.get("concept_path", ())
    if concept_path:
        parts.append("concept path: " + " > ".join(str(item) for item in concept_path if str(item).strip()))
    co_concepts = meta.get("co_concepts", ())
    if co_concepts:
        parts.append("co concepts: " + ", ".join(str(item) for item in co_concepts if str(item).strip()))
    buckets = meta.get("tag_buckets", {})
    if isinstance(buckets, Mapping):
        for bucket, values in buckets.items():
            if isinstance(values, (list, tuple)) and values:
                parts.append(f"{bucket}: " + ", ".join(str(item) for item in values if str(item).strip()))
    if not parts:
        parts.append(", ".join(str(item) for item in meta.get("tags", ()) if str(item).strip()) or sample.stem)
    return "\n".join(parts)


class TextEmbeddingProvider:
    """Small provider interface for semantic Concept Geometry prep sources."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class TextTranslator:
    """Prep-time translator interface for optional NL-to-English normalization."""

    def translate_texts(self, texts: Sequence[str]) -> List[str]:
        raise NotImplementedError


class LocalPyTorchTranslator(TextTranslator):
    """Transformers seq2seq translator for local/offline NL normalization."""

    def __init__(self, model_name_or_path: str, *, device: str = "cpu", batch_size: int = 4) -> None:
        self.model_name_or_path = str(model_name_or_path or "")
        if not self.model_name_or_path:
            raise ValueError("local translation provider requires --translation-model-path")
        self.device = str(device or "cpu")
        self.batch_size = max(int(batch_size or 4), 1)
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional runtime packages
            raise RuntimeError(f"local translation requires torch + transformers ({type(exc).__name__}: {exc})") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path, local_files_only=True)
        self._model.eval().to(self.device)

    def translate_texts(self, texts: Sequence[str]) -> List[str]:
        results: List[str] = []
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(item or "") for item in texts[start : start + self.batch_size]]
                inputs = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output_ids = self._model.generate(**inputs, max_new_tokens=160)
                results.extend(self._tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        return [item.strip() for item in results]


class OpenAICompatibleTranslator(TextTranslator):
    """OpenAI-compatible /v1/chat/completions translator for optional API use."""

    def __init__(self, *, api_base: str, api_key: str = "", model: str = "", batch_size: int = 8) -> None:
        self.api_base = str(api_base or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or "gpt-4o-mini")
        self.batch_size = max(int(batch_size or 8), 1)
        if not self.api_base:
            raise ValueError("api translation provider requires --translation-api-base")

    def translate_texts(self, texts: Sequence[str]) -> List[str]:
        endpoint = f"{self.api_base}/chat/completions" if self.api_base.endswith("/v1") else f"{self.api_base}/v1/chat/completions"
        results: List[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(item or "") for item in texts[start : start + self.batch_size]]
            prompt = (
                "Translate each caption to concise English for image concept clustering. "
                "Preserve names, clothing, pose, setting, and style. Return JSON array only.\n"
                + json.dumps(batch, ensure_ascii=False)
            )
            payload = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You translate training captions into concise English tags."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
            except Exception as exc:
                raise RuntimeError(f"api translation request failed: {exc}") from exc
            if not isinstance(parsed, list) or len(parsed) != len(batch):
                raise RuntimeError(f"api translation returned invalid batch of size {len(parsed) if isinstance(parsed, list) else 'unknown'}")
            results.extend(str(item).strip() for item in parsed)
        return results


def _translate_metadata_captions(
    metadata: Sequence[Dict[str, Any]],
    *,
    enabled: bool,
    provider_name: str,
    model_path: str,
    api_base: str,
    api_key: str,
    api_model: str,
    batch_size: int,
    device: str,
) -> List[str]:
    if not enabled:
        return []
    candidates = [str(meta.get("caption", "") or "") for meta in metadata]
    indexes = [idx for idx, text in enumerate(candidates) if text.strip() and _CJK_RE.search(text)]
    if not indexes:
        return []
    provider_key = str(provider_name or "local_path").strip().lower().replace("-", "_")
    if provider_key == "api":
        translator: TextTranslator = OpenAICompatibleTranslator(
            api_base=api_base,
            api_key=api_key,
            model=api_model,
            batch_size=batch_size,
        )
    else:
        translator = LocalPyTorchTranslator(model_path, device=device, batch_size=batch_size)
    translated = translator.translate_texts([candidates[idx] for idx in indexes])
    for idx, text in zip(indexes, translated):
        metadata[idx]["translated_caption"] = text
    return [f"translated {len(translated)} CJK captions via {provider_key}"]


class LocalPyTorchTextEmbeddingProvider(TextEmbeddingProvider):
    """Transformers/PyTorch embedding provider used by default for local semantic enhancement."""

    def __init__(self, model_name_or_path: str, *, device: str = "cpu", batch_size: int = 8) -> None:
        self.model_name_or_path = str(model_name_or_path or _DEFAULT_BGE_M3_REPO)
        self.device = str(device or "cpu")
        self.batch_size = max(int(batch_size or 8), 1)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages
            raise RuntimeError(f"pytorch text embedding requires torch + transformers ({type(exc).__name__}: {exc})") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_name_or_path, local_files_only=True)
        self._model.eval().to(self.device)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(item or "") for item in texts[start : start + self.batch_size]]
                inputs = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self._model(**inputs)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    pooled = getattr(outputs, "pooler_output", None)
                    if pooled is None:
                        raise RuntimeError("pytorch text embedding model produced no hidden or pooled output")
                    emb = pooled
                else:
                    mask = inputs.get("attention_mask")
                    if mask is None:
                        emb = hidden.mean(dim=1)
                    else:
                        mask_f = mask.unsqueeze(-1).to(dtype=hidden.dtype)
                        emb = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
                vectors.append(emb.detach().cpu().numpy().astype(np.float32))
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


class OpenAICompatibleTextEmbeddingProvider(TextEmbeddingProvider):
    """OpenAI-compatible /v1/embeddings provider for optional online or local API use."""

    def __init__(self, *, api_base: str, api_key: str = "", model: str = "", batch_size: int = 32) -> None:
        self.api_base = str(api_base or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or "text-embedding-3-small")
        self.batch_size = max(int(batch_size or 32), 1)
        if not self.api_base:
            raise ValueError("api text embedding provider requires --embedding-api-base")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        endpoint = f"{self.api_base}/embeddings" if self.api_base.endswith("/v1") else f"{self.api_base}/v1/embeddings"
        for start in range(0, len(texts), self.batch_size):
            batch = [str(item or "") for item in texts[start : start + self.batch_size]]
            payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"api text embedding request failed: {exc}") from exc
            data = body.get("data", [])
            if len(data) != len(batch):
                raise RuntimeError(f"api text embedding returned {len(data)} vectors for {len(batch)} inputs")
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            vectors.extend(np.asarray(item.get("embedding", []), dtype=np.float32) for item in ordered)
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalize_matrix(np.stack(vectors, axis=0))


class OnnxTextEmbeddingProvider(TextEmbeddingProvider):
    """Reserved ONNX extension point for contributors.

    The default implementation intentionally uses PyTorch/Transformers because it
    is easier to support arbitrary Hugging Face text embedding folders.  This
    class is left as a narrow adapter point for future contributors who want to
    wire ONNX Runtime, DirectML, CUDA EP selection, and tokenizer handling.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("ONNX text embedding provider is reserved for future developer integration.")


def _resolve_auto_download_model(model_name: str, cache_dir: str, allow_download: bool) -> str:
    model_name = str(model_name or _DEFAULT_BGE_M3_REPO)
    if not allow_download:
        raise RuntimeError(
            "auto_download provider requires explicit download approval. "
            f"Recommended model: {_DEFAULT_BGE_M3_REPO}; repo={_DEFAULT_BGE_M3_REPO_URL}; "
            f"pytorch_weight={_DEFAULT_BGE_M3_MODEL_URL}"
        )
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"auto_download requires huggingface_hub ({type(exc).__name__}: {exc})") from exc
    return snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_dir or ""),
        local_files_only=False,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.bin",
            "*.safetensors",
            "sentencepiece.*",
            "tokenizer.*",
        ],
    )


def _build_text_embedding_source(
    samples: Sequence[SampleRef],
    metadata: Sequence[Dict[str, Any]],
    *,
    provider_name: str,
    model_name: str,
    model_path: str,
    api_base: str,
    api_key: str,
    api_model: str,
    backend: str,
    cache_dir: str,
    allow_download: bool,
    feature_dim: int,
    batch_size: int,
    device: str,
) -> SourceFeature:
    provider_key = str(provider_name or "local_path").strip().lower().replace("-", "_")
    backend_key = str(backend or "pytorch").strip().lower().replace("-", "_")
    summaries: Dict[str, Dict[str, Any]] = {}
    fallbacks: List[str] = []
    texts = [_text_for_embedding(sample, meta) for sample, meta in zip(samples, metadata)]

    path = Path(str(model_path or "")).expanduser()
    if path.is_file() and path.suffix.lower() == ".npz":
        matrix = _load_npz_feature_map(path, samples)
        if matrix is None:
            raise RuntimeError(f"text embedding feature map is missing sample vectors: {path}")
        features = matrix
        for sample, text in zip(samples, texts):
            summaries[sample.sample_id] = {"provider": "npz", "text_chars": len(text), "path": str(path)}
        return SourceFeature(_TEXT_EMBEDDING_SOURCE, _normalize_matrix(features), summaries, fallbacks)

    if backend_key == "onnx":
        # Reserved for future ONNX Runtime contributors; keep this explicit so
        # users get a clear fallback instead of a silent partial implementation.
        provider: TextEmbeddingProvider = OnnxTextEmbeddingProvider()
    elif provider_key == "api":
        provider = OpenAICompatibleTextEmbeddingProvider(
            api_base=api_base,
            api_key=api_key,
            model=api_model or model_name,
            batch_size=batch_size,
        )
    else:
        resolved_path = str(path) if path.exists() else ""
        if provider_key == "auto_download":
            resolved_path = _resolve_auto_download_model(model_name or _DEFAULT_BGE_M3_REPO, cache_dir, allow_download)
        if not resolved_path:
            resolved_path = str(model_name or _DEFAULT_BGE_M3_REPO)
        provider = LocalPyTorchTextEmbeddingProvider(resolved_path, device=device, batch_size=batch_size)

    raw_features = provider.embed_texts(texts)
    features = np.stack([_resize_feature(row, feature_dim) for row in raw_features], axis=0)
    for sample, text in zip(samples, texts):
        summaries[sample.sample_id] = {
            "provider": provider_key,
            "backend": backend_key,
            "model": str(model_path or model_name or api_model or ""),
            "text_chars": len(text),
        }
    return SourceFeature(_TEXT_EMBEDDING_SOURCE, _normalize_matrix(features), summaries, fallbacks)


def _load_npz_feature_map(path: Path, samples: Sequence[SampleRef]) -> Optional[np.ndarray]:
    if path.is_dir():
        path = path / "features.npz"
    if not path.is_file() or path.suffix.lower() != ".npz":
        return None
    data = np.load(path, allow_pickle=False)
    vectors: List[np.ndarray] = []
    for sample in samples:
        key = sample.sample_id if sample.sample_id in data.files else sample.stem if sample.stem in data.files else f"{sample.stem}_feature"
        if key not in data.files:
            return None
        vectors.append(_normalize_vector(np.asarray(data[key], dtype=np.float32)))
    return np.stack(vectors, axis=0)


def _build_latent_source(samples: Sequence[SampleRef], feature_dim: int) -> SourceFeature:
    vectors: List[np.ndarray] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    fallbacks: List[str] = []
    for sample in samples:
        if sample.latent_path is None or not sample.latent_path.is_file():
            vectors.append(_build_hashed_feature([sample.sample_id], feature_dim))
            fallbacks.append(f"{sample.sample_id}: missing *_anima latent cache; used stem hash")
            summaries[sample.sample_id] = {"available": False}
            continue
        try:
            data = np.load(sample.latent_path, allow_pickle=False)
            arrays = [np.asarray(data[key], dtype=np.float32) for key in data.files if np.asarray(data[key]).ndim >= 2]
            if not arrays:
                raise ValueError("no tensor arrays in latent cache")
            stats: List[float] = []
            for array in arrays[:4]:
                flat = array.reshape(-1)
                stats.extend([float(flat.mean()), float(flat.std()), float(flat.min()), float(flat.max())])
                if array.ndim >= 3:
                    pooled = array.mean(axis=tuple(range(1, array.ndim))).reshape(-1)
                    stats.extend(float(x) for x in pooled[:64])
            vectors.append(_resize_feature(np.asarray(stats, dtype=np.float32), feature_dim))
            summaries[sample.sample_id] = {"available": True, "array_count": len(arrays), "path": sample.latent_path.name}
        except Exception as exc:
            vectors.append(_build_hashed_feature([sample.sample_id], feature_dim))
            fallbacks.append(f"{sample.sample_id}: failed to read latent cache ({type(exc).__name__}); used stem hash")
            summaries[sample.sample_id] = {"available": False, "error": type(exc).__name__}
    return SourceFeature("latent", np.stack(vectors, axis=0), summaries, fallbacks)


def _build_clip_source(
    samples: Sequence[SampleRef],
    metadata: Sequence[Dict[str, Any]],
    *,
    clip_model_path: str,
    feature_dim: int,
    device: str,
) -> SourceFeature:
    model_path = Path(str(clip_model_path or "").strip()).expanduser() if str(clip_model_path or "").strip() else None
    if model_path is not None:
        mapped = _load_npz_feature_map(model_path, samples)
        if mapped is not None:
            return SourceFeature("clip", _normalize_matrix(mapped), {sample.sample_id: {"mock_npz": True} for sample in samples}, [])

    fallbacks: List[str] = []
    if model_path is None:
        fallbacks.append("clip: no local model path; used caption text hash only")
    elif not model_path.exists():
        fallbacks.append(f"clip: local model path does not exist ({model_path}); used caption text hash only")
    else:
        try:
            from PIL import Image
            import torch
            from transformers import AutoProcessor, CLIPModel

            processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
            model = CLIPModel.from_pretrained(str(model_path), local_files_only=True)
            model.eval().to(device)
            vectors: List[np.ndarray] = []
            summaries: Dict[str, Dict[str, Any]] = {}
            with torch.no_grad():
                for sample, meta in zip(samples, metadata):
                    parts: List[torch.Tensor] = []
                    if sample.image_path is not None and sample.image_path.is_file() and hasattr(model, "get_image_features"):
                        image = Image.open(sample.image_path).convert("RGB")
                        inputs = {k: v.to(device) for k, v in processor(images=[image], return_tensors="pt").items()}
                        parts.append(model.get_image_features(**inputs)[0])
                    if meta["caption"] and hasattr(model, "get_text_features"):
                        inputs = {k: v.to(device) for k, v in processor(text=[meta["caption"]], return_tensors="pt", padding=True, truncation=True).items()}
                        parts.append(model.get_text_features(**inputs)[0])
                    if parts:
                        fused = torch.stack(parts, dim=0).mean(dim=0)
                        vectors.append(_normalize_vector(fused.detach().cpu().float().numpy()))
                        summaries[sample.sample_id] = {"image": sample.image_path is not None, "text": bool(meta["caption"])}
                    else:
                        vectors.append(_build_hashed_feature(meta["tags"] or [sample.sample_id], feature_dim))
                        summaries[sample.sample_id] = {"text_hash_fallback": True}
            return SourceFeature("clip", np.stack(vectors, axis=0), summaries, fallbacks)
        except Exception as exc:
            text_tower_error = exc
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
                text_model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
                text_model.eval().to(device)
                vectors = []
                summaries = {}
                with torch.no_grad():
                    for sample, meta in zip(samples, metadata):
                        text = str(meta["caption"] or ", ".join(meta["tags"]) or sample.stem)
                        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
                        inputs = {key: value.to(device) for key, value in inputs.items()}
                        outputs = text_model(**inputs)
                        hidden = getattr(outputs, "last_hidden_state", None)
                        if hidden is None:
                            pooled = getattr(outputs, "pooler_output", None)
                        else:
                            mask = inputs.get("attention_mask")
                            if mask is not None:
                                weights = mask.unsqueeze(-1).float()
                                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                            else:
                                pooled = hidden.mean(dim=1)
                        if pooled is None:
                            raise RuntimeError("text tower produced no pooled embedding")
                        vectors.append(_normalize_vector(pooled[0].detach().cpu().float().numpy()))
                        summaries[sample.sample_id] = {"text_tower_only": True}
                fallbacks.append("clip: image tower unavailable; used local text tower only")
                return SourceFeature("clip", np.stack(vectors, axis=0), summaries, fallbacks)
            except Exception:
                fallbacks.append(f"clip: failed to load local CLIP ({type(text_tower_error).__name__}: {text_tower_error}); used caption text hash only")

    tags = _build_tags_source(samples, metadata, feature_dim)
    return SourceFeature("clip", tags.features.copy(), {sample.sample_id: {"text_hash_fallback": True} for sample in samples}, fallbacks)


def _build_dino_source(samples: Sequence[SampleRef], *, dino_model_path: str, feature_dim: int, device: str) -> SourceFeature:
    model_path = Path(str(dino_model_path or "").strip()).expanduser() if str(dino_model_path or "").strip() else None
    if model_path is None:
        raise RuntimeError("dino backend requires --dino-model-path pointing to a local DINO/DINOv2 checkpoint")
    mapped = _load_npz_feature_map(model_path, samples)
    if mapped is not None:
        return SourceFeature("dino", _normalize_matrix(mapped), {sample.sample_id: {"mock_npz": True} for sample in samples}, [])
    if not model_path.exists():
        raise RuntimeError(f"dino backend local model path does not exist: {model_path}")
    try:
        from PIL import Image
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise RuntimeError(f"dino backend requires PIL/torch/transformers: {exc}") from exc

    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.eval().to(device)
    vectors: List[np.ndarray] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for sample in samples:
            if sample.image_path is None or not sample.image_path.is_file():
                raise RuntimeError(f"dino backend requires image files; missing image for sample {sample.sample_id}")
            image = Image.open(sample.image_path).convert("RGB")
            inputs = {k: v.to(device) for k, v in processor(images=[image], return_tensors="pt").items()}
            outputs = model(**inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            vectors.append(_normalize_vector(pooled[0].detach().cpu().float().numpy()))
            summaries[sample.sample_id] = {"image": True}
    return SourceFeature("dino", np.stack(vectors, axis=0), summaries, [])


def _compute_density(features: np.ndarray, neighbors: int) -> np.ndarray:
    sample_count = int(features.shape[0])
    if sample_count <= 1:
        return np.ones((sample_count,), dtype=np.float32)
    normalized = _normalize_matrix(features)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -1.0)
    k = max(1, min(int(neighbors or 1), sample_count - 1))
    topk = np.partition(similarity, -k, axis=1)[:, -k:]
    raw = np.clip(topk, 0.0, 1.0).mean(axis=1)
    span = float(raw.max() - raw.min())
    if span <= 1e-8:
        return np.full_like(raw, 0.5, dtype=np.float32)
    return ((raw - raw.min()) / span).astype(np.float32)


def _neighbors(features: np.ndarray, samples: Sequence[SampleRef], k: int) -> Dict[str, List[str]]:
    if len(samples) <= 1:
        return {sample.sample_id: [] for sample in samples}
    normalized = _normalize_matrix(features)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -1.0)
    count = max(1, min(int(k or 1), len(samples) - 1))
    result: Dict[str, List[str]] = {}
    for index, sample in enumerate(samples):
        order = np.argsort(-similarity[index])[:count]
        result[sample.sample_id] = [samples[int(item)].sample_id for item in order if similarity[index, int(item)] > -0.5]
    return result


def _siblings(
    features: np.ndarray,
    samples: Sequence[SampleRef],
    concept_groups: Sequence[str],
    k: int,
) -> Dict[str, List[str]]:
    if len(samples) <= 1:
        return {sample.sample_id: [] for sample in samples}
    normalized = _normalize_matrix(features)
    similarity = normalized @ normalized.T
    count = max(1, min(max(int(k or 1), 1) * 3, len(samples) - 1))
    result: Dict[str, List[str]] = {}
    for index, sample in enumerate(samples):
        group = str(concept_groups[index] or "")
        if not group:
            result[sample.sample_id] = []
            continue
        candidates = [
            other_index
            for other_index, other_group in enumerate(concept_groups)
            if other_index != index and str(other_group or "") == group
        ]
        candidates.sort(key=lambda other_index: float(similarity[index, other_index]), reverse=True)
        result[sample.sample_id] = [samples[int(item)].sample_id for item in candidates[:count]]
    return result


def _source_densities(sources: Mapping[str, SourceFeature], neighbors: int) -> Dict[str, np.ndarray]:
    return {name: _compute_density(source.features, neighbors) for name, source in sources.items()}


def _fuse_sources(sources: Mapping[str, SourceFeature], feature_dim: int, weights: Mapping[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
    active = {name: max(float(weights.get(name, 0.0)), 0.0) for name in sources}
    total = sum(active.values())
    if total <= 0.0:
        active = {name: 1.0 for name in sources}
        total = float(len(active))
    normalized_weights = {name: value / total for name, value in active.items()}
    fused = np.zeros((next(iter(sources.values())).features.shape[0], max(int(feature_dim), 1)), dtype=np.float32)
    for name, source in sources.items():
        resized = np.stack([_resize_feature(row, feature_dim) for row in source.features], axis=0)
        fused += resized * float(normalized_weights[name])
    return _normalize_matrix(fused), normalized_weights


def build_concept_geometry(
    data_dir: str | Path,
    *,
    output_path: str | Path = "",
    caption_extension: str = ".txt",
    backend: str = "auto",
    clip_model_path: str = "",
    dino_model_path: str = "",
    concept_depth: int = 3,
    feature_dim: int = 384,
    neighbors: int = 8,
    core_quantile: float = 0.33,
    edge_quantile: float = 0.67,
    device: str = "cpu",
    save_feature_cache: bool = False,
    semantic_enhance: bool = False,
    embedding_provider: str = "local_path",
    embedding_backend: str = "pytorch",
    embedding_model: str = _DEFAULT_BGE_M3_REPO,
    embedding_model_path: str = "",
    embedding_cache_dir: str = "",
    embedding_allow_download: bool = False,
    embedding_api_base: str = "",
    embedding_api_key: str = "",
    embedding_api_model: str = "",
    embedding_batch_size: int = 8,
    translation_enabled: bool = False,
    translation_provider: str = "local_path",
    translation_model_path: str = "",
    translation_api_base: str = "",
    translation_api_key: str = "",
    translation_api_model: str = "",
    translation_batch_size: int = 8,
    alias_map: str | Mapping[str, Any] | None = None,
    alias_map_path: str | Path = "",
    concept_source_priority: str | Sequence[str] = "explicit,folder,nl,identity,tag,stem",
) -> Dict[str, Any]:
    data_root = Path(data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Concept Geometry data_dir does not exist: {data_root}")
    if not (0.0 < float(core_quantile) < float(edge_quantile) < 1.0):
        raise ValueError("core_quantile and edge_quantile must satisfy 0 < core < edge < 1")

    samples = _discover_samples(data_root, caption_extension)
    if not samples:
        raise ValueError(f"No samples discovered under {data_root}")

    requested = str(backend or "auto").strip().lower().replace("-", "_")
    if requested == "lexical":
        requested = "latent_tags"
    if requested == "auto":
        requested = "latent_tags"
    if requested not in _BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")

    aliases = _load_alias_map(alias_map, alias_map_path)
    source_priority = _parse_source_priority(concept_source_priority)
    metadata = _base_metadata(
        samples,
        max(int(concept_depth), 1),
        aliases=aliases,
        concept_source_priority=source_priority,
    )
    sources: Dict[str, SourceFeature] = {}
    fallback_reasons: List[str] = []
    translation_notes: List[str] = []
    if translation_enabled:
        try:
            translation_notes = _translate_metadata_captions(
                metadata,
                enabled=True,
                provider_name=translation_provider,
                model_path=translation_model_path,
                api_base=translation_api_base,
                api_key=translation_api_key,
                api_model=translation_api_model,
                batch_size=translation_batch_size,
                device=device,
            )
        except Exception as exc:
            fallback_reasons.append(f"translation: {type(exc).__name__}: {exc}")

    def add_source(source: SourceFeature) -> None:
        sources[source.name] = source
        fallback_reasons.extend(source.fallback_reasons)

    if requested in {"latent_tags", "hybrid"}:
        add_source(_build_latent_source(samples, feature_dim))
        add_source(_build_tags_source(samples, metadata, feature_dim))
    if requested in {"clip", "hybrid"}:
        add_source(_build_clip_source(samples, metadata, clip_model_path=clip_model_path, feature_dim=feature_dim, device=device))
    if requested in {"dino", "hybrid"}:
        try:
            add_source(_build_dino_source(samples, dino_model_path=dino_model_path, feature_dim=feature_dim, device=device))
        except Exception as exc:
            if requested == "dino":
                raise
            fallback_reasons.append(f"dino: {exc}")
    if semantic_enhance:
        try:
            add_source(
                _build_text_embedding_source(
                    samples,
                    metadata,
                    provider_name=embedding_provider,
                    model_name=embedding_model,
                    model_path=embedding_model_path,
                    api_base=embedding_api_base,
                    api_key=embedding_api_key,
                    api_model=embedding_api_model,
                    backend=embedding_backend,
                    cache_dir=embedding_cache_dir,
                    allow_download=embedding_allow_download,
                    feature_dim=feature_dim,
                    batch_size=embedding_batch_size,
                    device=device,
                )
            )
        except Exception as exc:
            fallback_reasons.append(f"text_embedding: {type(exc).__name__}: {exc}")
    if not sources:
        add_source(_build_tags_source(samples, metadata, feature_dim))
        fallback_reasons.append("no requested feature source was available; used tags only")

    fused, resolved_weights = _fuse_sources(sources, feature_dim, _DEFAULT_SOURCE_WEIGHTS)
    density = _compute_density(fused, neighbors=neighbors)
    source_density_arrays = _source_densities(sources, neighbors=neighbors)
    neighbor_map = _neighbors(fused, samples, neighbors)
    concept_groups = [str(meta["concept_group"]) for meta in metadata]
    sibling_map = _siblings(fused, samples, concept_groups, neighbors)

    path_depth = np.asarray([float(meta["path_depth"]) for meta in metadata], dtype=np.float32)
    depth_norm = path_depth / max(float(path_depth.max(initial=0.0)), 1.0)
    radius = np.clip((1.0 - density) * 0.7 + depth_norm * 0.3, 0.0, 1.0)
    curriculum_score = radius.copy()
    core_threshold = float(np.quantile(curriculum_score, core_quantile))
    edge_threshold = float(np.quantile(curriculum_score, edge_quantile))

    samples_payload: Dict[str, Dict[str, Any]] = {}
    stage_counts = {"core": 0, "mid": 0, "edge": 0}
    for idx, (sample, meta) in enumerate(zip(samples, metadata)):
        sample_score = float(curriculum_score[idx])
        stage = "core" if sample_score <= core_threshold else ("edge" if sample_score >= edge_threshold else "mid")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        sibling_ids = sibling_map.get(sample.sample_id, [])
        conflict_ids = [item for item in neighbor_map.get(sample.sample_id, []) if item not in sibling_ids]
        conflict_score = len(conflict_ids) / max(len(neighbor_map.get(sample.sample_id, [])), 1)
        source_density = {name: round(float(values[idx]), 6) for name, values in source_density_arrays.items()}
        samples_payload[sample.sample_id] = {
            "stage": stage,
            "density": round(float(density[idx]), 6),
            "radius": round(float(radius[idx]), 6),
            "curriculum_score": round(sample_score, 6),
            "loss_weight": round(float(np.clip(0.85 + 0.15 * float(density[idx]), 0.5, 1.25)), 6),
            "concept_group": str(meta["concept_group"]),
            "concept_group_source": str(meta.get("concept_group_source", "")),
            "concept_path": list(meta["concept_path"]),
            "path_depth": int(meta["path_depth"]),
            "geometry_version": 2,
            "backend_requested": requested,
            "backend_resolved": "+".join(sorted(sources.keys())),
            "feature_sources": {name: source.summaries.get(sample.sample_id, {}) for name, source in sources.items()},
            "fallback_reasons": list(fallback_reasons),
            "tag_buckets": meta["tag_buckets"],
            "co_concepts": list(meta.get("co_concepts", [])),
            "parse_confidence": round(float(meta.get("parse_confidence", 0.0)), 6),
            "parse_warnings": list(meta.get("parse_warnings", [])),
            "source_density": source_density,
            "neighbor_ids": neighbor_map.get(sample.sample_id, []),
            "sibling_ids": sibling_ids,
            "conflict_score": round(float(conflict_score), 6),
        }

    destination = Path(output_path).expanduser() if str(output_path or "").strip() else (data_root / "concept_geometry.json")
    feature_cache_path = ""
    if save_feature_cache:
        feature_cache_path = str(destination.with_suffix(".features.npz"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_cache_path,
            fused=fused.astype(np.float32),
            stems=np.asarray([sample.sample_id for sample in samples]),
            **{f"source_{name}": source.features.astype(np.float32) for name, source in sources.items()},
        )

    payload = {
        "meta": {
            "geometry_version": 2,
            "backend_requested": requested,
            "backend_resolved": "+".join(sorted(sources.keys())),
            "feature_sources": sorted(sources.keys()),
            "source_weights": {key: round(float(value), 6) for key, value in resolved_weights.items()},
            "semantic_enhanced": bool(_TEXT_EMBEDDING_SOURCE in sources),
            "embedding_provider": str(embedding_provider or ""),
            "embedding_backend": str(embedding_backend or ""),
            "embedding_model": str(embedding_model_path or embedding_api_model or embedding_model or ""),
            "embedding_auto_download_repo": _DEFAULT_BGE_M3_REPO,
            "embedding_auto_download_repo_url": _DEFAULT_BGE_M3_REPO_URL,
            "embedding_auto_download_pytorch_weight_url": _DEFAULT_BGE_M3_MODEL_URL,
            "translation_enabled": bool(translation_enabled),
            "translation_provider": str(translation_provider or ""),
            "translation_model": str(translation_model_path or translation_api_model or ""),
            "translation_notes": list(translation_notes),
            "fallback_reasons": list(fallback_reasons),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(data_root),
            "caption_extension": str(caption_extension),
            "neighbors": int(neighbors),
            "concept_depth": int(concept_depth),
            "feature_dim": int(feature_dim),
            "core_threshold": round(core_threshold, 6),
            "edge_threshold": round(edge_threshold, 6),
            "sample_count": len(samples),
            "stage_counts": stage_counts,
            "alias_count": len(aliases),
            "concept_source_priority": source_priority,
            "concept_source_counts": {
                source: sum(1 for meta in metadata if str(meta.get("concept_group_source", "")) == source)
                for source in sorted({str(meta.get("concept_group_source", "")) for meta in metadata})
                if source
            },
            "low_confidence_count": sum(1 for meta in metadata if float(meta.get("parse_confidence", 0.0)) < 0.55),
            "parse_warning_count": sum(len(meta.get("parse_warnings", [])) for meta in metadata),
            "co_concept_count": sum(len(meta.get("co_concepts", [])) for meta in metadata),
            "review_recommended_count": sum(
                1
                for meta in metadata
                if float(meta.get("parse_confidence", 0.0)) < 0.55 or bool(meta.get("parse_warnings"))
            ),
            "feature_cache": feature_cache_path,
        },
        "samples": samples_payload,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[concept-geometry] backend_requested={requested} backend_resolved={payload['meta']['backend_resolved']} "
        f"samples={len(samples)} sources={','.join(sorted(sources.keys()))}"
    )
    for reason in fallback_reasons:
        print(f"[concept-geometry-fallback] {reason}")
    return payload


build_h_lora_geometry = build_concept_geometry


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local Concept Geometry Sampling metadata")
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument("--output", default="", help="Output JSON path (default: <data_dir>/concept_geometry.json)")
    parser.add_argument("--caption-extension", default=".txt", help="Caption sidecar extension")
    parser.add_argument("--backend", default="latent_tags", choices=tuple(sorted(_BACKENDS)))
    parser.add_argument("--clip-model-path", default="", help="Local CLIP checkpoint path, or an .npz feature map for tests")
    parser.add_argument("--dino-model-path", default="", help="Local DINO/DINOv2 checkpoint path, or an .npz feature map for tests")
    parser.add_argument("--concept-depth", type=int, default=3, help="Maximum concept path depth")
    parser.add_argument("--feature-dim", type=int, default=384, help="Feature width for fused geometry")
    parser.add_argument("--neighbors", type=int, default=8, help="Neighbor count for density and adjacency")
    parser.add_argument("--core-quantile", type=float, default=0.33, help="Quantile threshold for core samples")
    parser.add_argument("--edge-quantile", type=float, default=0.67, help="Quantile threshold for edge samples")
    parser.add_argument("--device", default="cpu", help="Device for optional encoder backends")
    parser.add_argument("--save-feature-cache", action="store_true", help="Write a compact .features.npz next to the JSON")
    parser.add_argument("--semantic-enhance", action="store_true", help="Enable optional text embedding source for Concept Geometry")
    parser.add_argument("--embedding-provider", default="local_path", choices=("local_path", "auto_download", "api"), help="Text embedding provider")
    parser.add_argument("--embedding-backend", default="pytorch", choices=("pytorch", "onnx"), help="Local text embedding backend; ONNX is a reserved developer extension point")
    parser.add_argument("--embedding-model", default=_DEFAULT_BGE_M3_REPO, help="Embedding model id/name. Recommended default: BAAI/bge-m3")
    parser.add_argument("--embedding-model-path", default="", help="Local Hugging Face text embedding model directory, or .npz feature map for tests")
    parser.add_argument("--embedding-cache-dir", default="", help="Optional Hugging Face cache directory for auto-download")
    parser.add_argument("--embedding-allow-download", action="store_true", help="Allow auto-download provider to fetch the recommended model")
    parser.add_argument("--embedding-api-base", default="", help="OpenAI-compatible embedding API base URL")
    parser.add_argument("--embedding-api-key", default="", help="OpenAI-compatible embedding API key")
    parser.add_argument("--embedding-api-model", default="", help="OpenAI-compatible embedding model name")
    parser.add_argument("--embedding-batch-size", type=int, default=8, help="Text embedding batch size")
    parser.add_argument("--translation-enabled", action="store_true", help="Translate CJK captions to concise English before embedding")
    parser.add_argument("--translation-provider", default="local_path", choices=("local_path", "api"), help="Translation provider")
    parser.add_argument("--translation-model-path", default="", help="Local Transformers seq2seq translation model directory")
    parser.add_argument("--translation-api-base", default="", help="OpenAI-compatible chat completions API base URL")
    parser.add_argument("--translation-api-key", default="", help="OpenAI-compatible translation API key")
    parser.add_argument("--translation-api-model", default="", help="OpenAI-compatible translation model name")
    parser.add_argument("--translation-batch-size", type=int, default=8, help="Translation batch size")
    parser.add_argument("--alias-map", default="", help="Optional JSON object for canonical concept/tag aliases")
    parser.add_argument("--alias-map-path", default="", help="Optional JSON file for canonical concept/tag aliases")
    parser.add_argument(
        "--concept-source-priority",
        default="explicit,folder,nl,identity,tag,stem",
        help="Concept group source priority, comma-separated. Known values: explicit, folder, nl, identity, tag, stem",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    payload = build_concept_geometry(
        args.data_dir,
        output_path=args.output,
        caption_extension=args.caption_extension,
        backend=args.backend,
        clip_model_path=args.clip_model_path,
        dino_model_path=args.dino_model_path,
        concept_depth=max(int(args.concept_depth), 1),
        feature_dim=max(int(args.feature_dim), 32),
        neighbors=max(int(args.neighbors), 1),
        core_quantile=float(args.core_quantile),
        edge_quantile=float(args.edge_quantile),
        device=str(args.device),
        save_feature_cache=bool(args.save_feature_cache),
        semantic_enhance=bool(args.semantic_enhance),
        embedding_provider=str(args.embedding_provider),
        embedding_backend=str(args.embedding_backend),
        embedding_model=str(args.embedding_model),
        embedding_model_path=str(args.embedding_model_path),
        embedding_cache_dir=str(args.embedding_cache_dir),
        embedding_allow_download=bool(args.embedding_allow_download),
        embedding_api_base=str(args.embedding_api_base),
        embedding_api_key=str(args.embedding_api_key),
        embedding_api_model=str(args.embedding_api_model),
        embedding_batch_size=max(int(args.embedding_batch_size), 1),
        translation_enabled=bool(args.translation_enabled),
        translation_provider=str(args.translation_provider),
        translation_model_path=str(args.translation_model_path),
        translation_api_base=str(args.translation_api_base),
        translation_api_key=str(args.translation_api_key),
        translation_api_model=str(args.translation_api_model),
        translation_batch_size=max(int(args.translation_batch_size), 1),
        alias_map=str(args.alias_map),
        alias_map_path=str(args.alias_map_path),
        concept_source_priority=str(args.concept_source_priority),
    )
    meta = payload.get("meta", {})
    stage_counts = meta.get("stage_counts", {})
    output_path = str(args.output or (Path(args.data_dir) / "concept_geometry.json"))
    print(
        "Concept Geometry prepared: "
        f"backend={meta.get('backend_resolved', 'latent+tags')} "
        f"samples={meta.get('sample_count', 0)} "
        f"core={stage_counts.get('core', 0)} "
        f"mid={stage_counts.get('mid', 0)} "
        f"edge={stage_counts.get('edge', 0)} "
        f"output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


