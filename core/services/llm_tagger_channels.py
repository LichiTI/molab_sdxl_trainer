"""Persistent LLM/API channel configuration for image tagging."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


PROVIDERS = {"openai_compatible", "gemini", "anthropic"}


def default_llm_channel_store_path() -> Path:
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / "data" / "llm_tagger_channels.json"


def list_llm_channels(*, store_path: Path | str | None = None, include_secrets: bool = False) -> dict[str, Any]:
    channels = _load_channels(_store_path(store_path))
    merged = _merge_builtin_channels(channels)
    return {"channels": [_public_channel(channel, include_secrets=include_secrets) for channel in merged]}


def save_llm_channel(params: dict[str, Any], *, store_path: Path | str | None = None) -> dict[str, Any]:
    path = _store_path(store_path)
    channels = _load_channels(path)
    existing = _find_channel(channels, params)
    incoming = _normalize_channel(params, require_id=False)
    incoming["api_keys"] = _resolve_saved_api_keys(params, existing, incoming["api_keys"])
    next_channels = [channel for channel in channels if channel.get("id") != incoming["id"]]
    next_channels.append(incoming)
    _save_channels(path, next_channels)
    return list_llm_channels(store_path=path)


def clear_llm_channel_keys(channel_id: str, *, store_path: Path | str | None = None) -> dict[str, Any]:
    path = _store_path(store_path)
    normalized_id = _safe_id(channel_id)
    if not normalized_id:
        raise ValueError("Missing channel id")
    channels = _load_channels(path)
    changed = False
    next_channels: list[dict[str, Any]] = []
    for channel in channels:
        if channel.get("id") == normalized_id:
            channel = dict(channel)
            channel["api_keys"] = []
            changed = True
        next_channels.append(channel)
    if not changed:
        raise ValueError(f"LLM channel not found: {normalized_id}")
    _save_channels(path, next_channels)
    return list_llm_channels(store_path=path)


def delete_llm_channel(channel_id: str, *, store_path: Path | str | None = None) -> dict[str, Any]:
    path = _store_path(store_path)
    normalized_id = _safe_id(channel_id)
    if not normalized_id:
        raise ValueError("Missing channel id")
    channels = [channel for channel in _load_channels(path) if channel.get("id") != normalized_id]
    _save_channels(path, channels)
    return list_llm_channels(store_path=path)


def build_llm_execution_plan(params: dict[str, Any], *, store_path: Path | str | None = None) -> dict[str, Any]:
    channels = list_llm_channels(store_path=store_path, include_secrets=True)["channels"]
    by_id = {str(channel.get("id") or ""): channel for channel in channels}
    steps: list[dict[str, Any]] = []
    requested_ids = _requested_channel_ids(params)
    if requested_ids:
        for channel_id in requested_ids:
            channel = by_id.get(channel_id)
            if not channel:
                steps.append({"channel_id": channel_id, "enabled": False, "errors": [f"LLM channel not found: {channel_id}"]})
                continue
            steps.append(_channel_to_step(channel, params))
    else:
        inline = _inline_channel_from_params(params)
        if inline:
            steps.append(_channel_to_step(inline, params))
    valid_steps = [step for step in steps if step.get("enabled") and step.get("api_keys") and not step.get("errors")]
    return {"steps": steps, "valid_steps": valid_steps}


def build_llm_channel_health_report(params: dict[str, Any], *, store_path: Path | str | None = None) -> dict[str, Any]:
    plan = build_llm_execution_plan(params, store_path=store_path)
    errors: list[str] = []
    warnings: list[str] = []
    public_steps: list[dict[str, Any]] = []
    for step in plan["steps"]:
        step_errors = list(step.get("errors") or [])
        if not step.get("enabled"):
            step_errors.append(f"LLM channel disabled: {step.get('name') or step.get('channel_id')}")
        if not step.get("api_keys"):
            step_errors.append(f"LLM channel has no API key: {step.get('name') or step.get('channel_id')}")
        errors.extend(step_errors)
        public_steps.append(_public_step(step))
    if plan["steps"] and plan["valid_steps"]:
        errors = []
    if not plan["steps"]:
        errors.append("No LLM/API channel configured.")
    if len(plan["valid_steps"]) > 1:
        warnings.append(f"LLM fallback enabled with {len(plan['valid_steps'])} usable channels.")
    return {
        "ok": bool(plan["valid_steps"]),
        "steps": public_steps,
        "usable_step_count": len(plan["valid_steps"]),
        "errors": _dedupe(errors),
        "warnings": warnings,
    }


def step_to_tagger_params(base: dict[str, Any], step: dict[str, Any], api_key: str) -> dict[str, Any]:
    merged = dict(base)
    merged.update(
        {
            "llm_provider": step.get("provider") or base.get("llm_provider"),
            "interrogator_model": step.get("provider") or base.get("interrogator_model"),
            "llm_api_key": api_key,
            "llm_api_base": step.get("api_base") or base.get("llm_api_base") or "",
            "llm_model": step.get("model") or base.get("llm_model") or "",
            "llm_timeout": step.get("timeout") or base.get("llm_timeout"),
        }
    )
    return merged


def _store_path(path: Path | str | None) -> Path:
    return Path(path) if path else default_llm_channel_store_path()


def _load_channels(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_channels = payload.get("channels") if isinstance(payload, dict) else payload
    if not isinstance(raw_channels, list):
        return []
    channels: list[dict[str, Any]] = []
    for item in raw_channels:
        try:
            channels.append(_normalize_channel(item, require_id=True))
        except Exception:
            continue
    return channels


def _save_channels(path: Path, channels: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "channels": channels}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_channel(channels: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any] | None:
    requested_id = _safe_id(params.get("id") or params.get("channel_id") or params.get("name") or params.get("label"))
    if not requested_id:
        return None
    return next((channel for channel in channels if channel.get("id") == requested_id), None)


def _resolve_saved_api_keys(params: dict[str, Any], existing: dict[str, Any] | None, parsed_keys: list[str]) -> list[str]:
    key_mode = str(params.get("key_mode") or params.get("api_key_mode") or "").strip().lower()
    clear_keys = bool(params.get("clear_api_keys") or key_mode == "clear")
    replace_keys = bool(params.get("replace_api_keys") or key_mode in {"replace", "overwrite"})
    if clear_keys:
        return []
    if parsed_keys:
        return parsed_keys
    if existing and not replace_keys:
        return list(existing.get("api_keys") or [])
    return []


def _normalize_channel(params: dict[str, Any], *, require_id: bool) -> dict[str, Any]:
    name = str(params.get("name") or params.get("label") or "LLM Channel").strip() or "LLM Channel"
    channel_id = _safe_id(params.get("id") or params.get("channel_id") or name)
    if require_id and not channel_id:
        raise ValueError("Missing channel id")
    provider = _normalize_provider(params.get("provider") or params.get("llm_provider"))
    model = str(params.get("model") or params.get("llm_model") or _default_model(provider)).strip()
    api_base = _normalize_base_url(str(params.get("api_base") or params.get("llm_api_base") or ""), provider)
    api_keys = _parse_api_keys(params.get("api_keys") or params.get("api_key") or params.get("llm_api_key"))
    return {
        "id": channel_id or _safe_id(name),
        "name": name,
        "provider": provider,
        "api_base": api_base,
        "model": model,
        "api_keys": api_keys,
        "enabled": bool(params.get("enabled", True)),
        "retries": _int_param(params.get("retries"), 1),
        "timeout": _float_param(params.get("timeout"), 120.0),
    }


def _merge_builtin_channels(channels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {channel.get("id"): channel for channel in channels}
    merged = list(channels)
    for channel in _builtin_env_channels():
        if channel["id"] not in by_id:
            merged.append(channel)
    return merged


def _builtin_env_channels() -> list[dict[str, Any]]:
    return [
        _env_channel("env-openai", "OpenAI Environment", "openai_compatible", "OPENAI_API_KEY"),
        _env_channel("env-gemini", "Gemini Environment", "gemini", "GEMINI_API_KEY"),
        _env_channel("env-anthropic", "Anthropic Environment", "anthropic", "ANTHROPIC_API_KEY"),
    ]


def _env_channel(channel_id: str, name: str, provider: str, env_key: str) -> dict[str, Any]:
    api_key = os.environ.get(env_key, "").strip()
    return {
        "id": channel_id,
        "name": name,
        "provider": provider,
        "api_base": _normalize_base_url("", provider),
        "model": _default_model(provider),
        "api_keys": [api_key] if api_key else [],
        "enabled": bool(api_key),
        "retries": 1,
        "timeout": 120.0,
        "source": "env",
    }


def _requested_channel_ids(params: dict[str, Any]) -> list[str]:
    selected = str(params.get("llm_channel_id") or "").strip()
    fallback_enabled = bool(params.get("llm_fallback_enabled", True))
    fallback_ids = _parse_id_list(params.get("llm_fallback_channel_ids") or params.get("llm_channel_ids"))
    ids: list[str] = []
    if selected:
        ids.append(selected)
    if fallback_enabled:
        ids.extend(fallback_ids)
    return _dedupe([_safe_id(item) for item in ids if _safe_id(item)])


def _inline_channel_from_params(params: dict[str, Any]) -> dict[str, Any] | None:
    api_keys = _parse_api_keys(params.get("llm_api_keys") or params.get("llm_api_key") or params.get("api_key") or "")
    provider_value = params.get("llm_provider") or params.get("interrogator_model") or params.get("provider")
    model = str(params.get("llm_model") or params.get("model") or "").strip()
    if not (api_keys or model or provider_value or params.get("llm_api_base") or params.get("api_base")):
        return None
    provider = _normalize_provider(provider_value)
    return _normalize_channel(
        {
            "id": "inline",
            "name": "Inline API",
            "provider": provider,
            "api_base": params.get("llm_api_base") or params.get("api_base") or "",
            "model": model or _default_model(provider),
            "api_keys": api_keys,
            "enabled": True,
            "retries": params.get("llm_retries", 1),
            "timeout": params.get("llm_timeout", 120),
        },
        require_id=True,
    )


def _channel_to_step(channel: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    step = dict(channel)
    step["channel_id"] = step.get("id", "")
    if params.get("llm_model"):
        step["model"] = str(params.get("llm_model") or "").strip()
    if params.get("llm_timeout"):
        step["timeout"] = _float_param(params.get("llm_timeout"), _float_param(step.get("timeout"), 120.0))
    if params.get("llm_retries") is not None:
        step["retries"] = _int_param(params.get("llm_retries"), _int_param(step.get("retries"), 1))
    step["errors"] = []
    if step.get("provider") not in PROVIDERS:
        step["errors"].append(f"Unsupported provider: {step.get('provider')}")
    if not step.get("model"):
        step["errors"].append("LLM model is required.")
    if step.get("provider") == "openai_compatible" and not step.get("api_base"):
        step["errors"].append("OpenAI-compatible API base URL is required.")
    return step


def _public_channel(channel: dict[str, Any], *, include_secrets: bool) -> dict[str, Any]:
    public = {key: value for key, value in channel.items() if key != "api_keys"}
    keys = list(channel.get("api_keys") or [])
    public["key_count"] = len(keys)
    public["has_key"] = bool(keys)
    public["api_keys"] = keys if include_secrets else []
    return public


def _public_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel_id": step.get("channel_id") or step.get("id") or "",
        "name": step.get("name") or "",
        "provider": step.get("provider") or "",
        "model": step.get("model") or "",
        "api_base": step.get("api_base") or "",
        "enabled": bool(step.get("enabled")),
        "key_count": len(step.get("api_keys") or []),
        "retries": _int_param(step.get("retries"), 1),
        "timeout": _float_param(step.get("timeout"), 120.0),
        "errors": list(step.get("errors") or []),
    }


def _parse_api_keys(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[\n,;]+", str(value or ""))
    return _dedupe([str(item).strip() for item in raw_items if str(item).strip()])


def _parse_id_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[\n,;]+", str(value or ""))
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _normalize_provider(value: Any) -> str:
    text = str(value or "openai_compatible").strip().lower().replace("_", "-")
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


def _safe_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    return text.strip("-")[:80]


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _int_param(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return default


def _float_param(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
