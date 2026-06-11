"""Read-only official UI configuration payload helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .plugin_settings_adapter import (
    PYTORCH_OPTIMIZER_PLUGIN_ID,
    build_plugin_settings_payload,
    resolve_pytorch_optimizer_visible_names,
)


FALLBACK_OPTIMIZERS = [
    "AdamW",
    "AdamW8bit",
    "PagedAdamW8bit",
    "Lion",
    "Lion8bit",
    "DAdaptation",
    "DAdaptAdam",
    "DAdaptLion",
    "AdaFactor",
    "Prodigy",
    "prodigyplus.ProdigyPlusScheduleFree",
]

FALLBACK_SCHEDULERS = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "adafactor",
    "inverse_sqrt",
    "piecewise_constant",
]

DEFAULT_LLM_TEMPLATE_PRESETS = [
    {"id": "anime-tags", "label": "动漫标签 / Anime Tags"},
    {"id": "natural-caption", "label": "自然语言描述 / Natural Caption"},
    {"id": "character-lora", "label": "角色 LoRA 标签 / Character LoRA Tags"},
    {"id": "caption-rewrite", "label": "补全/改写已有 Caption / Caption Rewrite"},
]


def build_presets_payload() -> dict[str, Any]:
    """Return pure training config presets consumed by the official UI."""

    try:
        from backend.core.preset_manager import preset_manager

        result = []
        for _preset_id, preset_data in preset_manager.presets.items():
            modules = preset_data.get("modules", {})
            start_training = modules.get("start_training", {})
            trainer_config = start_training.get("trainer_config", {})
            if trainer_config is not None:
                result.append(dict(trainer_config))
        return {"presets": result}
    except Exception:
        return {"presets": []}


def load_saved_params_payload(*, project_root: Path, output_root: Path) -> dict[str, Any]:
    """Load the official UI's last parameter snapshot with legacy fallback."""

    saved_params_file = project_root / "assets" / "ui_state" / "saved_params.json"
    for candidate in (saved_params_file, output_root / ".last_training_config.json"):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def build_config_options_payload() -> dict[str, list[str]]:
    try:
        from backend.core.configs import OptimizerType, SchedulerType

        optimizers = [item.value for item in OptimizerType]
        schedulers = [item.value for item in SchedulerType]
    except Exception:
        optimizers = FALLBACK_OPTIMIZERS
        schedulers = FALLBACK_SCHEDULERS

    optimizer_options = _dedupe([*optimizers, *_pytorch_optimizer_options()])
    scheduler_options = _dedupe(schedulers)
    return {
        "optimizers": optimizer_options,
        "optimizer_type": optimizer_options,
        "schedulers": scheduler_options,
        "lr_scheduler": scheduler_options,
    }


def _pytorch_optimizer_options(project_root: Path | None = None) -> list[str]:
    root = project_root or Path(__file__).resolve().parents[3]
    plugin_root = root / "plugin"
    manifest_path = plugin_root / "pytorch_optimizer-main" / "plugin_manifest.json"
    settings_path = root / "data" / "plugins" / "settings.json"
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    try:
        all_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        all_settings = {"plugins": {}}
    payload = build_plugin_settings_payload(
        plugin_id=PYTORCH_OPTIMIZER_PLUGIN_ID,
        manifest_payload=manifest_payload if isinstance(manifest_payload, dict) else {},
        all_settings=all_settings if isinstance(all_settings, dict) else {"plugins": {}},
        plugin_root=plugin_root,
        plugin_dir=plugin_root / "pytorch_optimizer-main",
    )
    names = resolve_pytorch_optimizer_visible_names(plugin_root, payload.get("values", {}))
    return [f"pytorch_optimizer.{name}" for name in names]


def build_interrogators_payload(*, env: dict[str, str] | None = None) -> dict[str, Any]:
    interrogators: list[dict[str, str]] = []
    try:
        from backend.core.wd14_tagger import get_available_models

        interrogators.extend({"name": name, "kind": "wd"} for name in get_available_models())
    except Exception:
        pass

    environment = env if env is not None else os.environ
    try:
        from backend.core.gemini_tagger import GeminiTagger

        tagger = GeminiTagger()
        api_key = getattr(tagger, "api_key", None) or environment.get("GEMINI_API_KEY", "")
        if api_key:
            interrogators.append({"name": "gemini-2.0-flash", "kind": "llm"})
    except Exception:
        pass

    try:
        from backend.core.services.llm_image_tagger import llm_template_presets

        template_presets = llm_template_presets()
    except Exception:
        template_presets = list(DEFAULT_LLM_TEMPLATE_PRESETS)
    try:
        from backend.core.services.llm_tagger_channels import list_llm_channels

        llm_channels = list_llm_channels().get("channels", [])
    except Exception:
        llm_channels = []
    return {"interrogators": interrogators, "llm_template_presets": template_presets, "llm_channels": llm_channels}


def build_config_summary_payload() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    try:
        from backend.core.locator import Locator

        cfg = Locator.get_config()
        if cfg and hasattr(cfg, "config"):
            summary["config"] = dict(cfg.config) if isinstance(cfg.config, dict) else {}
    except Exception:
        pass
    try:
        from backend.core.execution_resolver import get_execution_resolver

        profiles = get_execution_resolver().list_profiles()
        summary["profiles"] = [{"id": profile.id, "installed": profile.installed} for profile in profiles]
    except Exception:
        pass
    try:
        from backend.core.preset_manager import preset_manager

        summary["current_preset"] = preset_manager.current_preset
    except Exception:
        pass
    return {"summary": summary}


def build_scripts_payload() -> dict[str, list[Any]]:
    return {"scripts": []}


def build_turbocore_status_payload() -> dict[str, Any]:
    return {
        "available": False,
        "status": "not_configured",
        "backend": None,
        "message": "TurboCore backend is not enabled in this build.",
    }


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
