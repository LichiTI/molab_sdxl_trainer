"""Fallback and quality orchestration for LLM image tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.core.services.llm_tagger_channels import build_llm_execution_plan, step_to_tagger_params


TaggerFactory = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class LlmTagResult:
    text: str
    channel_id: str
    channel_name: str
    provider: str
    model: str
    attempt_count: int
    fallback_count: int
    quality_errors: list[str]


class LlmTaggerOrchestrator:
    def __init__(self, base_config: dict[str, Any], *, tagger_factory: TaggerFactory | None = None) -> None:
        self.base_config = dict(base_config)
        self.plan = build_llm_execution_plan(base_config)
        self._tagger_factory = tagger_factory or self._default_tagger_factory

    def describe_plan(self) -> dict[str, Any]:
        return {
            "steps": [_public_step(step) for step in self.plan.get("steps", [])],
            "usable_step_count": len(self.plan.get("valid_steps", [])),
        }

    def interrogate(self, image_path: str, *, existing_caption: str = "", image_name: str = "") -> LlmTagResult:
        steps = list(self.plan.get("valid_steps") or [])
        if not steps:
            raise RuntimeError("No usable LLM/API channel configured.")

        attempts = 0
        failures: list[str] = []
        fallback_count = 0
        for step_index, step in enumerate(steps):
            if step_index > 0:
                fallback_count += 1
            keys = list(step.get("api_keys") or [])
            retries = max(1, int(step.get("retries") or 0) + 1)
            for key_index, api_key in enumerate(keys):
                for retry_index in range(retries):
                    attempts += 1
                    try:
                        tagger = self._tagger_factory(step_to_tagger_params(self.base_config, step, api_key))
                        text = str(tagger.interrogate(image_path, existing_caption=existing_caption, image_name=image_name) or "").strip()
                        errors = evaluate_llm_output_quality(text, self.base_config)
                        if not errors:
                            return LlmTagResult(
                                text=text,
                                channel_id=str(step.get("channel_id") or step.get("id") or ""),
                                channel_name=str(step.get("name") or ""),
                                provider=str(step.get("provider") or ""),
                                model=str(step.get("model") or ""),
                                attempt_count=attempts,
                                fallback_count=fallback_count,
                                quality_errors=[],
                            )
                        failures.append(_format_failure(step, key_index, retry_index, "; ".join(errors)))
                    except Exception as exc:
                        failures.append(_format_failure(step, key_index, retry_index, str(exc)))
        raise RuntimeError("LLM/API tagging failed after fallback: " + " | ".join(failures[-6:]))

    @staticmethod
    def _default_tagger_factory(config: dict[str, Any]) -> Any:
        from backend.core.services.llm_image_tagger import build_llm_image_tagger

        return build_llm_image_tagger(config)


def evaluate_llm_output_quality(text: str, config: dict[str, Any]) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return ["empty output"]
    mode = str(config.get("llm_output_mode") or "tags").strip().lower()
    if _looks_like_refusal(value):
        return ["model returned refusal/explanation text"]
    if mode == "caption":
        return _caption_quality_errors(value, config)
    return _tag_quality_errors(value, config)


def build_llm_orchestrator(config: dict[str, Any], *, tagger_factory: TaggerFactory | None = None) -> LlmTaggerOrchestrator:
    return LlmTaggerOrchestrator(config, tagger_factory=tagger_factory)


def _tag_quality_errors(text: str, config: dict[str, Any]) -> list[str]:
    tags = [item.strip() for item in text.split(",") if item.strip()]
    min_tags = _int_param(config.get("llm_min_tags"), 1)
    max_tags = _int_param(config.get("llm_max_tags"), 120)
    errors: list[str] = []
    if len(tags) < min_tags:
        errors.append(f"too few tags: {len(tags)} < {min_tags}")
    if max_tags > 0 and len(tags) > max_tags:
        errors.append(f"too many tags: {len(tags)} > {max_tags}")
    if any(len(tag) > 96 for tag in tags):
        errors.append("tag item too long")
    if "```" in text:
        errors.append("markdown fence detected")
    return errors


def _caption_quality_errors(text: str, config: dict[str, Any]) -> list[str]:
    min_chars = _int_param(config.get("llm_min_caption_chars"), 8)
    max_chars = _int_param(config.get("llm_max_caption_chars"), 1000)
    errors: list[str] = []
    if len(text) < min_chars:
        errors.append(f"caption too short: {len(text)} < {min_chars}")
    if max_chars > 0 and len(text) > max_chars:
        errors.append(f"caption too long: {len(text)} > {max_chars}")
    if "```" in text:
        errors.append("markdown fence detected")
    return errors


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "i can't",
        "i cannot",
        "sorry",
        "as an ai",
        "i'm unable",
    ]
    return any(marker in lowered for marker in markers)


def _format_failure(step: dict[str, Any], key_index: int, retry_index: int, reason: str) -> str:
    name = step.get("name") or step.get("channel_id") or step.get("id") or "channel"
    return f"{name} key#{key_index + 1} try#{retry_index + 1}: {reason}"


def _public_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel_id": step.get("channel_id") or step.get("id") or "",
        "name": step.get("name") or "",
        "provider": step.get("provider") or "",
        "model": step.get("model") or "",
        "key_count": len(step.get("api_keys") or []),
        "retries": int(step.get("retries") or 0),
    }


def _int_param(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
