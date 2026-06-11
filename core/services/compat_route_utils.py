"""Small serialization helpers shared by compatibility routes."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from pydantic import BaseModel


def format_compat_message(value: Any) -> str:
    """Normalize arbitrary exception/detail payloads into UI-safe text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(filter(None, (format_compat_message(item) for item in value)))
    if isinstance(value, dict):
        for key in ("message", "detail", "error", "reason"):
            text = format_compat_message(value.get(key))
            if text:
                return text
        for key in ("errors", "issues"):
            text = format_compat_message(value.get(key))
            if text:
                return text
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def dump_compat_model(model: BaseModel) -> Dict[str, Any]:
    """Dump a Pydantic model while preserving extra legacy payload fields."""

    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        data = dict(model_dump())
    else:
        data = dict(model.dict())
    extra = getattr(model, "model_extra", None)
    if isinstance(extra, dict):
        data.update(extra)
    legacy_extra = getattr(model, "__pydantic_extra__", None)
    if isinstance(legacy_extra, dict):
        data.update(legacy_extra)
    if data.get("attentionBackend") and str(data.get("attention_backend") or "auto").lower() == "auto":
        data["attention_backend"] = data.get("attentionBackend")
    return data


def serialize_resolved_result(value: Any) -> Dict[str, Any]:
    """Best-effort serialization for execution resolver outputs."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return dict(data)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
        if isinstance(data, dict):
            return dict(data)
    try:
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    except Exception:
        return {"value": str(value)}
