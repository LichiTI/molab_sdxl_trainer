"""
HTTP provider bridge for text-only LLM calls.

This module intentionally stays lightweight: no torch/transformers imports and no
long-lived workers. Local GGUF inference remains owned by LLMEngine.
"""

from __future__ import annotations

import json
import logging
import asyncio
import threading
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from fastapi.concurrency import run_in_threadpool

try:
    import requests
except ImportError:  # pragma: no cover - launcher env should include requests
    requests = None

from .llm_client import LLM_PRESETS

logger = logging.getLogger("LLMProviderBridge")


@dataclass
class ProviderConfig:
    provider: str = "openai"
    api_format: str = "openai_chat"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    proxy: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


def _preset_for(provider: str) -> Dict[str, Any]:
    return LLM_PRESETS.get(provider, LLM_PRESETS.get("custom", {}))


def _provider_config(config: Dict[str, Any]) -> ProviderConfig:
    provider = str(config.get("provider") or "openai")
    preset = _preset_for(provider)
    api_format = str(config.get("api_format") or preset.get("api_format") or "openai_chat")
    model = str(config.get("model") or config.get("model_name") or preset.get("default_model") or "")
    base_url = str(config.get("base_url") or preset.get("base_url") or "")
    return ProviderConfig(
        provider=provider,
        api_format=api_format,
        api_key=str(config.get("api_key") or ""),
        base_url=base_url,
        model=model,
        proxy=str(config.get("proxy") or ""),
        max_tokens=int(config.get("max_tokens") or 2048),
        temperature=float(config.get("temperature") if config.get("temperature") is not None else 0.7),
        top_p=float(config.get("top_p") if config.get("top_p") is not None else 0.9),
    )


def _proxies(config: ProviderConfig) -> Optional[Dict[str, str]]:
    if not config.proxy:
        return None
    return {"http": config.proxy, "https": config.proxy}


def _join_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    suffix = suffix if suffix.startswith("/") else f"/{suffix}"
    return f"{base}{suffix}"


def _openai_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _headers(config: ProviderConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_format == "anthropic_messages":
        headers["anthropic-version"] = "2023-06-01"
        if config.api_key:
            headers["x-api-key"] = config.api_key
    elif config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError("requests is not installed")


def _require_common(config: ProviderConfig) -> None:
    if not config.base_url:
        raise ValueError("base_url is required for online LLM provider")
    if not config.model:
        raise ValueError("model is required for online LLM provider")
    if not config.api_key and config.provider not in {"ollama", "custom"}:
        raise ValueError("api_key is required for this provider")


def _messages_to_anthropic(messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
    system_parts: List[str] = []
    user_messages: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            user_messages.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    return "\n\n".join(part for part in system_parts if part), user_messages


def _extract_openai_chat_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _extract_responses_text(payload: Dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    texts: List[str] = []
    for item in payload.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") in {"output_text", "text"}:
                texts.append(str(part.get("text") or ""))
    return "".join(texts)


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    texts: List[str] = []
    for part in payload.get("content", []) or []:
        if part.get("type") == "text":
            texts.append(str(part.get("text") or ""))
    return "".join(texts)


def _post_json(config: ProviderConfig, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_requests()
    response = requests.post(
        url,
        headers=_headers(config),
        json=payload,
        proxies=_proxies(config),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _chat_sync(messages: List[Dict[str, str]], config: ProviderConfig) -> str:
    _require_common(config)
    if config.api_format in {"openai_chat", "openai_compatible", "chat_completions"}:
        url = _join_url(_openai_base(config.base_url), "/chat/completions")
        payload = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        return _extract_openai_chat_text(_post_json(config, url, payload)).strip()

    if config.api_format == "openai_responses":
        url = _join_url(_openai_base(config.base_url), "/responses")
        payload = {
            "model": config.model,
            "input": messages,
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        return _extract_responses_text(_post_json(config, url, payload)).strip()

    if config.api_format == "anthropic_messages":
        url = _join_url(_openai_base(config.base_url), "/messages")
        system_prompt, anthropic_messages = _messages_to_anthropic(messages)
        payload: Dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if system_prompt:
            payload["system"] = system_prompt
        return _extract_anthropic_text(_post_json(config, url, payload)).strip()

    raise ValueError(f"Unsupported api_format: {config.api_format}")


def _iter_sse_lines(response: Any) -> Iterable[str]:
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            yield line[5:].strip()


def _stream_sync(messages: List[Dict[str, str]], config: ProviderConfig) -> Iterable[str]:
    _require_requests()
    _require_common(config)

    if config.api_format in {"openai_chat", "openai_compatible", "chat_completions"}:
        url = _join_url(_openai_base(config.base_url), "/chat/completions")
        payload = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        response = requests.post(url, headers=_headers(config), json=payload, proxies=_proxies(config), timeout=120, stream=True)
        response.raise_for_status()
        for raw in _iter_sse_lines(response):
            if raw == "[DONE]":
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed SSE line: %s", raw[:100])
                continue
            choices = data.get("choices") or []
            if choices:
                token = choices[0].get("delta", {}).get("content")
                if token:
                    yield str(token)
        return

    if config.api_format == "openai_responses":
        url = _join_url(_openai_base(config.base_url), "/responses")
        payload = {
            "model": config.model,
            "input": messages,
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        response = requests.post(url, headers=_headers(config), json=payload, proxies=_proxies(config), timeout=120, stream=True)
        response.raise_for_status()
        for raw in _iter_sse_lines(response):
            if raw == "[DONE]":
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed SSE line: %s", raw[:100])
                continue
            if data.get("type") == "response.output_text.delta" and data.get("delta"):
                yield str(data["delta"])
        return

    if config.api_format == "anthropic_messages":
        url = _join_url(_openai_base(config.base_url), "/messages")
        system_prompt, anthropic_messages = _messages_to_anthropic(messages)
        payload: Dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        response = requests.post(url, headers=_headers(config), json=payload, proxies=_proxies(config), timeout=120, stream=True)
        response.raise_for_status()
        for raw in _iter_sse_lines(response):
            if raw == "[DONE]":
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed SSE line: %s", raw[:100])
                continue
            if data.get("type") == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("text"):
                    yield str(delta["text"])
        return

    raise ValueError(f"Unsupported api_format: {config.api_format}")


async def chat_online(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    provider_config = _provider_config(config)
    return await run_in_threadpool(_chat_sync, messages, provider_config)


async def stream_online(messages: List[Dict[str, str]], config: Dict[str, Any]) -> AsyncGenerator[str, None]:
    provider_config = _provider_config(config)
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            for token in _stream_sync(messages, provider_config):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            logger.error("[LLMProviderBridge] stream failed: %s", exc)
            loop.call_soon_threadsafe(queue.put_nowait, f"Error: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        token = await queue.get()
        if token is None:
            break
        yield token


def list_preset_providers() -> List[Dict[str, Any]]:
    providers = []
    for provider_id, preset in LLM_PRESETS.items():
        providers.append({
            "id": provider_id,
            "name": preset.get("name", provider_id),
            "api_format": preset.get("api_format", "openai_chat"),
            "base_url": preset.get("base_url", ""),
            "models": preset.get("models", []),
            "default_model": preset.get("default_model", ""),
            "note": preset.get("note", ""),
        })
    return providers


def fetch_remote_models(config: Dict[str, Any]) -> List[str]:
    provider_config = _provider_config(config)
    if not provider_config.base_url:
        return []
    _require_requests()

    url = _join_url(_openai_base(provider_config.base_url), "/models")
    response = requests.get(
        url,
        headers=_headers(provider_config),
        proxies=_proxies(provider_config),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        return [str(item.get("id") or item.get("name") or item) for item in payload["models"]]
    return []
