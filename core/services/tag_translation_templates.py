"""Pre-translation templates for the Danbooru tag encyclopedia.

The templates are small, read-only prompt contracts used by the tag
translation background task.  They mirror the existing tag category system so
the translator can treat general tags, names, copyright terms, characters, and
meta tags differently without adding another runtime entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .tag_service import (
    CATEGORY_ARTIST,
    CATEGORY_CHARACTER,
    CATEGORY_COPYRIGHT,
    CATEGORY_GENERAL,
    CATEGORY_META,
)


DEFAULT_TAG_TRANSLATION_TEMPLATE_ID = "danbooru-cn-compact"
DEFAULT_TARGET_LANGUAGE = "Chinese"

CATEGORY_LABELS_BY_ID = {
    CATEGORY_GENERAL: "general",
    CATEGORY_ARTIST: "artist",
    CATEGORY_COPYRIGHT: "copyright",
    CATEGORY_CHARACTER: "character",
    CATEGORY_META: "meta",
}

CATEGORY_LABELS_ZH = {
    "general": "通用",
    "artist": "作者",
    "copyright": "作品",
    "character": "角色",
    "meta": "元信息",
}


@dataclass(frozen=True)
class TagTranslationTemplate:
    id: str
    label: str
    description: str
    tone: str
    category_policy: dict[str, str]
    rules: tuple[str, ...]
    target_lang: str = DEFAULT_TARGET_LANGUAGE
    output_format: str = "one_line_per_tag"

    def to_public_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "target_lang": self.target_lang,
            "output_format": self.output_format,
            "rules": list(self.rules),
            "category_policy": dict(self.category_policy),
        }


_BASE_POLICIES = {
    "general": "把可见视觉概念翻译成简短、可复用的中文词条。",
    "artist": "作者名优先保留为专名，除非已有稳定常用中文名。",
    "copyright": "作品、系列、企划名保持可识别，只翻译其中的通用描述词。",
    "character": "角色名保持可识别，只在必要时翻译称号或描述性部分。",
    "meta": "质量、媒介、评级、请求和技术类元信息要保守、字面、稳定。",
}


_TEMPLATES: dict[str, TagTranslationTemplate] = {
    "danbooru-cn-compact": TagTranslationTemplate(
        id="danbooru-cn-compact",
        label="Danbooru 中文精简",
        description="短中文词条，适合标签百科和自动补全。",
        tone="concise tag dictionary translation",
        category_policy=_BASE_POLICIES,
        rules=(
            "使用简短中文标签词，不写完整句子。",
            "已有 Stable Diffusion/Danbooru 常用译法时优先沿用。",
            "输出要适合在标签建议列表里快速浏览。",
        ),
    ),
    "danbooru-cn-training": TagTranslationTemplate(
        id="danbooru-cn-training",
        label="训练向中文标签",
        description="偏训练数据集用语，减少解释性翻译。",
        tone="training dataset vocabulary normalization",
        category_policy={
            **_BASE_POLICIES,
            "general": "使用可复用于 caption 和标签清洗规则的短视觉词。",
            "meta": "质量、构图、风格、媒介标签使用训练侧更稳定的短词。",
        },
        rules=(
            "翻译成紧凑、训练友好的中文词汇。",
            "能用稳定短词时不要写成长名词短语。",
            "不要加入源标签之外的额外解释。",
        ),
    ),
    "danbooru-cn-name-safe": TagTranslationTemplate(
        id="danbooru-cn-name-safe",
        label="名称保守翻译",
        description="对作者、作品、角色名更保守，降低误译。",
        tone="proper noun safe translation",
        category_policy={
            **_BASE_POLICIES,
            "artist": "作者名尽量原样保留，除非源标签本身是通用身份词。",
            "copyright": "系列名和作品名尽量保留，不要臆造官方中文名。",
            "character": "角色名尽量保留，不要猜测不存在的官方中文名。",
        },
        rules=(
            "疑似名称的标签保留为可识别专名。",
            "只翻译制服、换装、系列类型等通用描述部分。",
            "未知名称不要创造官方译名。",
        ),
    ),
    "danbooru-cn-meta-quality": TagTranslationTemplate(
        id="danbooru-cn-meta-quality",
        label="质量/元信息优先",
        description="强化质量、媒介、构图、请求类标签的一致性。",
        tone="meta and quality tag normalization",
        category_policy={
            **_BASE_POLICIES,
            "general": "可见视觉概念直接、简短翻译。",
            "meta": "质量、媒介、文件状态、请求、评级等元标签统一成稳定中文词。",
        },
        rules=(
            "质量、媒介、构图、风格、请求类标签要格外一致。",
            "安全、评级、状态类词保持中性和字面。",
            "源标签已经是技术 token 时保留其技术含义。",
        ),
    ),
    "curated-tag-cn-readable": TagTranslationTemplate(
        id="curated-tag-cn-readable",
        label="精选词库可读中文",
        description="偏人工浏览，适合精选标签分类展示。",
        tone="readable curated tag library translation",
        category_policy=_BASE_POLICIES,
        rules=(
            "使用适合精选标签库浏览的自然中文词。",
            "译文要短到适合标签 chip 和分类列表展示。",
            "不要包含解释、例子或源英文词。",
        ),
    ),
}


def list_tag_translation_templates() -> list[dict[str, object]]:
    return [template.to_public_dict() for template in _TEMPLATES.values()]


def get_tag_translation_template(template_id: str | None = None) -> TagTranslationTemplate:
    normalized = (template_id or DEFAULT_TAG_TRANSLATION_TEMPLATE_ID).strip() or DEFAULT_TAG_TRANSLATION_TEMPLATE_ID
    template = _TEMPLATES.get(normalized)
    if template is None:
        raise ValueError(f"Unknown tag translation template: {normalized}")
    return template


def build_tag_translation_system_prompt(
    template_id: str | None = None,
    *,
    target_lang: str = DEFAULT_TARGET_LANGUAGE,
) -> str:
    template = get_tag_translation_template(template_id)
    language = (target_lang or template.target_lang or DEFAULT_TARGET_LANGUAGE).strip()
    category_lines = [
        f"- {name} ({CATEGORY_LABELS_ZH.get(name, name)}): {policy}"
        for name, policy in template.category_policy.items()
    ]
    rule_lines = [f"- {rule}" for rule in template.rules]
    return "\n".join(
        [
            "You are a Danbooru/Stable Diffusion tag translator.",
            f"Translate English tags into {language} using this style: {template.tone}.",
            "Input lines may be plain tags or '<category>\\t<tag>'.",
            "Output EXACTLY one translated tag per input line, in the same order as input.",
            "Output only the translated tag text. Do not add numbering, bullets, markdown, explanations, or source tags.",
            "If unsure, return a concise literal translation instead of an explanation.",
            "Category policy:",
            *category_lines,
            "Template rules (Chinese wording below describes the desired translation style):",
            *rule_lines,
        ]
    )


def build_tag_translation_user_prompt(
    tags: list[str],
    *,
    category_lookup: Callable[[str], int | str | None] | None = None,
) -> str:
    lines: list[str] = []
    for tag in tags:
        clean_tag = str(tag or "").strip()
        if not clean_tag:
            continue
        category = _normalize_category(category_lookup(clean_tag) if category_lookup else None)
        if category:
            lines.append(f"{category}\t{clean_tag}")
        else:
            lines.append(clean_tag)
    return "\n".join(lines)


def _normalize_category(value: int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return CATEGORY_LABELS_BY_ID.get(value, "")
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.isdigit():
        return CATEGORY_LABELS_BY_ID.get(int(text), "")
    return text if text in CATEGORY_LABELS_ZH else ""
