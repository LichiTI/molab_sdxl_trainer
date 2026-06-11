"""LLM/API image tagging helpers used by interrogation adapters.

This module is an original implementation for Lulynx Trainer. It intentionally
keeps provider calls small and explicit so the runtime/request boundary stays
inside the trainer backend.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image


PostJson = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


DEFAULT_LLM_TAG_TEMPLATES: dict[str, dict[str, str]] = {
    "anime-tags": {
        "label": "动漫标签 / Anime Tags",
        "mode": "tags",
        "system": (
            "You create concise comma-separated booru-style tags for image model training. "
            "Return tags only, no markdown, no explanation, no numbering."
        ),
        "user": (
            "Tag the image with visible subjects, clothing, pose, expression, camera view, "
            "background, lighting, and important visual details. Use short lowercase tags."
        ),
    },
    "natural-caption": {
        "label": "自然语言描述 / Natural Caption",
        "mode": "caption",
        "system": (
            "You write compact natural-language captions for image model training. "
            "Return one caption only, no markdown and no analysis."
        ),
        "user": "Describe the image accurately in one concise training caption.",
    },
    "character-lora": {
        "label": "角色 LoRA 标签 / Character LoRA Tags",
        "mode": "tags",
        "system": (
            "You create comma-separated tags for character LoRA datasets. Focus on stable "
            "identity, hair, eyes, outfit, pose, expression, and scene. Return tags only."
        ),
        "user": (
            "Tag the character and visible training-relevant details. Avoid guesses about "
            "names, artist, copyright, or hidden traits."
        ),
    },
    "caption-rewrite": {
        "label": "补全/改写已有 Caption / Caption Rewrite",
        "mode": "caption",
        "system": (
            "You improve image captions for training. Preserve correct existing details, "
            "add missing visible details, and return only the final caption."
        ),
        "user": (
            "Existing caption: {existing_caption}\n"
            "Rewrite or complete it using the image. Keep it concise and training-friendly."
        ),
    },
}


@dataclass(frozen=True)
class LlmTaggerConfig:
    provider: str
    api_key: str
    api_base: str
    model: str
    template_id: str
    system_prompt: str
    user_prompt: str
    output_mode: str
    temperature: float = 0.2
    max_tokens: int = 300
    timeout: float = 120.0
    image_max_size: int = 1280


def llm_template_presets() -> list[dict[str, str]]:
    return [
        {"id": key, "label": value["label"], "mode": value["mode"]}
        for key, value in DEFAULT_LLM_TAG_TEMPLATES.items()
    ]


def normalize_llm_tagger_config(params: dict[str, Any]) -> LlmTaggerConfig:
    provider = _normalize_provider(params.get("llm_provider") or params.get("interrogator_model") or params.get("provider"))
    template_id = str(params.get("llm_template_preset") or params.get("template_id") or "anime-tags").strip() or "anime-tags"
    template = DEFAULT_LLM_TAG_TEMPLATES.get(template_id) or DEFAULT_LLM_TAG_TEMPLATES["anime-tags"]
    api_key = str(params.get("llm_api_key") or params.get("api_key") or _env_api_key(provider) or "").strip()
    api_base = _normalize_base_url(str(params.get("llm_api_base") or params.get("api_base") or params.get("base_url") or ""), provider)
    model = str(params.get("llm_model") or params.get("model") or _default_model(provider)).strip()
    system_prompt = str(params.get("llm_system_prompt") or params.get("system_prompt") or template["system"]).strip()
    user_prompt = str(params.get("llm_user_prompt") or params.get("prompt") or template["user"]).strip()
    output_mode = str(params.get("llm_output_mode") or template.get("mode") or "tags").strip().lower()
    if output_mode not in {"tags", "caption"}:
        output_mode = "tags"
    return LlmTaggerConfig(
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        model=model,
        template_id=template_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_mode=output_mode,
        temperature=_float_param(params.get("llm_temperature"), 0.2),
        max_tokens=_int_param(params.get("llm_max_tokens"), 300),
        timeout=_float_param(params.get("llm_timeout"), 120.0),
        image_max_size=_int_param(params.get("llm_image_max_size"), 1280),
    )


def build_llm_health_report(params: dict[str, Any]) -> dict[str, Any]:
    config = normalize_llm_tagger_config(params)
    errors: list[str] = []
    warnings: list[str] = []
    if not config.api_key:
        errors.append("LLM/API key is required.")
    if not config.model:
        errors.append("LLM model is required.")
    if config.provider == "openai_compatible" and not config.api_base:
        errors.append("OpenAI-compatible API base URL is required.")
    if config.template_id not in DEFAULT_LLM_TAG_TEMPLATES:
        warnings.append(f"Unknown template preset '{config.template_id}', anime-tags will be used.")
    return {
        "ok": not errors,
        "provider": config.provider,
        "api_base": config.api_base,
        "model": config.model,
        "template_id": config.template_id,
        "output_mode": config.output_mode,
        "errors": errors,
        "warnings": warnings,
    }


class LlmImageTagger:
    def __init__(self, config: LlmTaggerConfig, *, post_json: PostJson | None = None) -> None:
        self.config = config
        self._post_json = post_json or _requests_post_json

    def interrogate(self, image_path: str, *, existing_caption: str = "", image_name: str = "") -> str:
        encoded = encode_image_for_llm(Path(image_path), max_size=self.config.image_max_size)
        system_prompt = render_prompt_template(self.config.system_prompt, existing_caption=existing_caption, image_name=image_name)
        user_prompt = render_prompt_template(self.config.user_prompt, existing_caption=existing_caption, image_name=image_name)
        if self.config.provider == "gemini":
            raw_text = self._call_gemini(encoded, system_prompt, user_prompt)
        elif self.config.provider == "anthropic":
            raw_text = self._call_anthropic(encoded, system_prompt, user_prompt)
        else:
            raw_text = self._call_openai_compatible(encoded, system_prompt, user_prompt)
        return clean_llm_output(raw_text, mode=self.config.output_mode)

    def _call_openai_compatible(self, encoded: dict[str, str], system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": encoded["data_url"]}},
                    ],
                },
            ],
        }
        url = f"{self.config.api_base.rstrip('/')}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        response = self._post_json(url, headers, payload, self.config.timeout)
        return _extract_openai_text(response)

    def _call_gemini(self, encoded: dict[str, str], system_prompt: str, user_prompt: str) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        {"inlineData": {"mimeType": encoded["mime_type"], "data": encoded["base64"]}},
                    ],
                }
            ],
            "generationConfig": {"temperature": self.config.temperature, "maxOutputTokens": self.config.max_tokens},
        }
        url = f"{self.config.api_base.rstrip('/')}/v1beta/models/{self.config.model}:generateContent"
        headers = {"x-goog-api-key": self.config.api_key, "Content-Type": "application/json"}
        response = self._post_json(url, headers, payload, self.config.timeout)
        return _extract_gemini_text(response)

    def _call_anthropic(self, encoded: dict[str, str], system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "system": system_prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": encoded["mime_type"], "data": encoded["base64"]},
                        },
                    ],
                }
            ],
        }
        url = f"{self.config.api_base.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        response = self._post_json(url, headers, payload, self.config.timeout)
        return _extract_anthropic_text(response)


def build_llm_image_tagger(params: dict[str, Any], *, post_json: PostJson | None = None) -> LlmImageTagger:
    config = normalize_llm_tagger_config(params)
    health = build_llm_health_report(params)
    if not health["ok"]:
        raise RuntimeError("; ".join(health["errors"]))
    return LlmImageTagger(config, post_json=post_json)


def encode_image_for_llm(path: Path, *, max_size: int) -> dict[str, str]:
    with Image.open(path) as image:
        image.load()
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if image.mode == "RGBA":
            output = image
            mime_type = "image/png"
            fmt = "PNG"
        else:
            output = image.convert("RGB")
            mime_type = "image/jpeg"
            fmt = "JPEG"
        buffer = io.BytesIO()
        output.save(buffer, format=fmt, quality=92)
    raw = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"base64": raw, "mime_type": mime_type, "data_url": f"data:{mime_type};base64,{raw}"}


def render_prompt_template(template: str, *, existing_caption: str, image_name: str) -> str:
    return str(template or "").format(
        existing_caption=existing_caption or "",
        image_name=image_name or "",
    )


def clean_llm_output(raw_text: str, *, mode: str) -> str:
    text = _strip_markdown(str(raw_text or "").strip())
    if mode == "caption":
        return _clean_caption_output(text)
    return ", ".join(_extract_tag_items(text))


def _extract_tag_items(text: str) -> list[str]:
    parsed = _try_json_payload(text)
    if parsed is not None:
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = parsed.get("tags") or parsed.get("caption") or parsed.get("result") or []
        else:
            candidates = []
        if isinstance(candidates, str):
            text = candidates
        else:
            return _dedupe_tags([str(item) for item in candidates])
    text = re.sub(r"^(tags?|caption|result|output)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(r"[,\n;，；]+", text)
    cleaned: list[str] = []
    for part in parts:
        item = re.sub(r"^[-*\d.\s]+", "", part).strip().strip('"\'`')
        item = item.replace(" ", "_").lower()
        item = re.sub(r"[^a-z0-9_()\\-]", "", item)
        if 1 < len(item) <= 80 and item not in {"tag", "tags", "none", "unknown", "image"}:
            cleaned.append(item)
    return _dedupe_tags(cleaned)


def _clean_caption_output(text: str) -> str:
    parsed = _try_json_payload(text)
    if isinstance(parsed, dict):
        text = str(parsed.get("caption") or parsed.get("result") or parsed.get("text") or text)
    elif isinstance(parsed, list):
        text = ", ".join(str(item).strip() for item in parsed if str(item).strip())
    text = re.sub(r"^(caption|result|output)\s*[:：]\s*", "", text.strip(), flags=re.IGNORECASE)
    return " ".join(text.split())


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```(?:json|text)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").strip().strip("` ")


def _try_json_payload(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _dedupe_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        item = str(tag or "").strip().strip('"\'`').replace(" ", "_").lower()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_openai_text(payload: dict[str, Any]) -> str:
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in payload.get("candidates", []) or []:
        for part in candidate.get("content", {}).get("parts", []) or []:
            if isinstance(part, dict) and "text" in part:
                chunks.append(str(part.get("text") or ""))
    return "".join(chunks)


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for part in payload.get("content", []) or []:
        if isinstance(part, dict) and part.get("type") == "text":
            chunks.append(str(part.get("text") or ""))
    return "".join(chunks)


def _requests_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    import requests

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _normalize_provider(value: Any) -> str:
    text = str(value or "llm-openai").strip().lower().replace("_", "-")
    if text in {"gemini", "llm-gemini", "google", "google-gemini"}:
        return "gemini"
    if text in {"claude", "llm-claude", "anthropic"}:
        return "anthropic"
    return "openai_compatible"


def _normalize_base_url(value: str, provider: str) -> str:
    base = str(value or "").strip().rstrip("/")
    if not base:
        if provider == "gemini":
            return "https://generativelanguage.googleapis.com"
        if provider == "anthropic":
            return "https://api.anthropic.com"
        return "https://api.openai.com"
    if provider == "openai_compatible" and base.lower().endswith("/v1"):
        return base[:-3]
    return base


def _default_model(provider: str) -> str:
    if provider == "gemini":
        return "gemini-2.0-flash"
    if provider == "anthropic":
        return "claude-3-5-sonnet-latest"
    return "gpt-4o-mini"


def _env_api_key(provider: str) -> str:
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY", "")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    return os.environ.get("OPENAI_API_KEY", "")


def _float_param(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int_param(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
