"""Legacy training request normalization shared by compatibility routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from backend.core.contracts import RequestSource, TrainingRequest
from backend.core.contracts.training import LEGACY_TRAINING_ALIASES


SCHEMA_ALIASES = {
    "sdxl_lora": "sdxl-lora",
    "sd_lora": "sd-lora",
    "anima_lora": "anima-lora",
    "newbie_lora": "newbie-lora",
    "flux_lora": "flux-lora",
    "flux_finetune": "flux-finetune",
    "flux_controlnet": "flux-controlnet",
    "lumina_lora": "lumina-lora",
    "lumina2_lora": "lumina2-lora",
    "lumina_finetune": "lumina-finetune",
    "qwen_image_lora": "qwen-image-lora",
    "hunyuan_dit_lora": "hunyuan-dit-lora",
    "hunyuan_image_lora": "hunyuan-image-lora",
    "sdxl_finetune": "sdxl-finetune",
    "sdxl_dreambooth": "sdxl-dreambooth",
    "anima_finetune": "anima-finetune",
    "sd_dreambooth": "sd-dreambooth",
    "sdxl_controlnet_lllite": "sdxl-controlnet-lllite",
    "sdxl_controlnet": "sdxl-controlnet",
    "sd_controlnet": "sd-controlnet",
    "sdxl_textual_inversion": "sdxl-textual-inversion",
    "sd_textual_inversion": "sd-textual-inversion",
    "sdxl_ip_adapter": "sdxl-ip-adapter",
    "sd_ip_adapter": "sd-ip-adapter",
}

WEBUI_EXPERIMENTAL_SCHEMA_IDS = {
    "lab-distiller",
    "sdxl-turbo-lora",
    "anima-few-step-lora",
    "newbie-few-step-lora",
}

PLACEHOLDER_SCHEMA_IDS = {
    "lumina-lora",
    "lumina2-lora",
    "lumina-finetune",
    "qwen-image-lora",
    "hunyuan-dit-lora",
    "hunyuan-image-lora",
}

PLACEHOLDER_SCHEMA_LABELS = {
    "lumina-lora": "Lumina LoRA",
    "lumina2-lora": "Lumina2 LoRA",
    "lumina-finetune": "Lumina Finetune",
    "qwen-image-lora": "Qwen Image LoRA",
    "hunyuan-dit-lora": "HunyuanDiT LoRA",
    "hunyuan-image-lora": "HunyuanDiT LoRA",
}


def placeholder_schema_error(schema_id: str) -> str | None:
    normalized = normalize_schema_id(schema_id)
    if normalized not in PLACEHOLDER_SCHEMA_IDS:
        return None
    label = PLACEHOLDER_SCHEMA_LABELS.get(normalized, normalized)
    return f"{label} 当前只是轻量选择入口，训练核心尚未接入。可以保存配置，暂不能直接启动训练。"

def str_field(data: Dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def bool_field(data: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value or "").strip().lower()
    if value_str in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if value_str in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def normalize_attention_backend_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "flash": "flash2",
        "flashattn": "flash2",
        "flashattention": "flash2",
        "flashattention2": "flash2",
        "fa2": "flash2",
        "sage": "sageattn",
        "sageattention": "sageattn",
        "flex": "flexattn",
        "flexattention": "flexattn",
    }
    return aliases.get(normalized, normalized)


def normalize_schema_id(value: str) -> str:
    candidate = str(value or "").strip().lower()
    if not candidate:
        return ""
    return SCHEMA_ALIASES.get(candidate, candidate.replace("_", "-"))


def derive_schema_id(data: Dict[str, Any]) -> str:
    for key in ("schema_id", "model_train_type", "training_schema", "training_type_id"):
        candidate = normalize_schema_id(str_field(data, key))
        if candidate:
            return candidate

    model_type = str_field(data, "model_type", "sdxl").lower()
    training_type = str_field(data, "training_type", "lora").lower().replace("_", "-")

    if training_type in {"lora", "network"}:
        return {
            "sdxl": "sdxl-lora",
            "sd15": "sd-lora",
            "sd": "sd-lora",
            "anima": "anima-lora",
            "newbie": "newbie-lora",
            "flux": "flux-lora",
            "lumina": "lumina-lora",
            "lumina2": "lumina2-lora",
            "qwen-image": "qwen-image-lora",
            "qwen_image": "qwen-image-lora",
            "hunyuan-dit": "hunyuan-dit-lora",
            "hunyuan_dit": "hunyuan-dit-lora",
            "hunyuan-image": "hunyuan-image-lora",
            "hunyuan_image": "hunyuan-image-lora",
        }.get(model_type, "")
    if training_type in {"full-finetune", "full_finetune", "finetune"}:
        return {"sdxl": "sdxl-finetune", "anima": "anima-finetune", "flux": "flux-finetune"}.get(model_type, "")
    if training_type == "dreambooth":
        return {"sdxl": "sdxl-dreambooth", "sd15": "sd-dreambooth", "sd": "sd-dreambooth"}.get(model_type, "")
    if training_type == "controlnet":
        return {"sdxl": "sdxl-controlnet", "sd15": "sd-controlnet", "sd": "sd-controlnet", "flux": "flux-controlnet"}.get(model_type, "")
    if training_type in {"textual-inversion", "textual_inversion"}:
        return {
            "sdxl": "sdxl-textual-inversion",
            "sd15": "sd-textual-inversion",
            "sd": "sd-textual-inversion",
        }.get(model_type, "")
    if training_type in {"ip-adapter", "ip_adapter"}:
        return {"sdxl": "sdxl-ip-adapter", "sd15": "sd-ip-adapter", "sd": "sd-ip-adapter"}.get(model_type, "")
    return ""


def derive_attention_backend(data: Dict[str, Any], current: str) -> str:
    requested = normalize_attention_backend_id(str_field(data, "attention_backend", current or "auto"))
    if requested and requested != "auto":
        return requested
    if bool_field(data, "flashattn", False):
        return "flash2"
    attn_mode = str_field(data, "attn_mode")
    if attn_mode:
        return normalize_attention_backend_id(attn_mode)
    if "xformers" in data and bool_field(data, "xformers", False):
        return "xformers"
    if bool_field(data, "sdpa", False) or bool_field(data, "use_sdpa", False):
        return "sdpa"
    return requested or "auto"


def normalize_compat_training_request(data: Dict[str, Any]) -> TrainingRequest:
    try:
        return TrainingRequest.from_legacy_payload(data, source=RequestSource.FASTAPI)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def compat_request_fields_set(request: Any) -> set[str] | None:
    """Return Pydantic-submitted field names when available."""

    fields = getattr(request, "model_fields_set", None)
    if fields is None:
        fields = getattr(request, "__fields_set__", None)
    if fields is None:
        return None
    return {str(item) for item in fields}


def _apply_legacy_training_aliases(data: Dict[str, Any]) -> Dict[str, Any]:
    for old, new in LEGACY_TRAINING_ALIASES.items():
        if old not in data:
            continue
        current = data.get(new)
        if new not in data or current in {None, ""} or (new == "attention_backend" and str(current).lower() == "auto"):
            data[new] = data[old]
    return data


def normalize_compat_training_route_data(data: Dict[str, Any], request: Any | None = None) -> Dict[str, Any]:
    """Merge route-model payloads without letting defaults overwrite saved config.

    FastAPI/Pydantic request models dump declared defaults such as
    ``network_dim=32`` and ``output_name='lora'`` even when the user only sent a
    nested saved ``config`` object.  For that shape, use the model's submitted
    field set so the nested saved config remains authoritative, while explicit
    top-level edits still override it.
    """

    raw = dict(data or {})
    nested = raw.get("config") if isinstance(raw.get("config"), dict) else None
    fields_set = compat_request_fields_set(request)
    if not nested or fields_set is None:
        return _apply_legacy_training_aliases(raw)

    merged: Dict[str, Any] = dict(nested)
    merged["config"] = dict(nested)
    for key, value in raw.items():
        if key == "config":
            continue
        if key in fields_set:
            merged[key] = value
    return _apply_legacy_training_aliases(merged)


def effective_execution_profile_id(profile_id: Any) -> str:
    """Return the compatibility default execution profile id."""

    value = str(profile_id or "").strip()
    if value:
        return value
    return "standard"


def build_training_config_from_schema(
    schema_id: str,
    data: Dict[str, Any],
    *,
    backend_root: Path,
    extra_config_layers: List[Dict[str, Any]] | None = None,
) -> tuple[dict, str, str, dict]:
    placeholder_error = placeholder_schema_error(schema_id)
    if placeholder_error is not None:
        raise ValueError(placeholder_error)

    from backend.lulynx_launcher.services.training_config_resolver import TrainingConfigResolver
    from backend.lulynx_launcher.services.training_profile_registry import TrainingProfileRegistry
    from backend.lulynx_launcher.services.training_registry import LulynxTrainingRegistry
    from backend.lulynx_launcher.services.training_route_service import TrainingRouteService

    schema = LulynxTrainingRegistry.default().get_by_id(schema_id)
    if schema is None:
        raise ValueError(f"Unknown training schema: {schema_id}")

    route_service = TrainingRouteService(project_root=backend_root.parent, backend_root=backend_root)
    route = route_service.resolve(schema_id)
    if not route.is_known:
        raise ValueError(f"Schema is not wired to a native trainer route: {schema_id}")

    config_values = dict(data)
    request_extra_layers = extra_config_layers if extra_config_layers is not None else config_values.pop("extra_config_layers", [])
    attention_backend = derive_attention_backend(data, str_field(data, "attention_backend", "auto"))
    if attention_backend:
        config_values["attention_backend"] = attention_backend
        if schema_id == "anima-lora" and attention_backend != "auto":
            config_values["attn_mode"] = attention_backend
    for key in ("execution_profile_id", "allow_attention_fallback", "schema_id"):
        config_values.pop(key, None)

    resolved = TrainingConfigResolver(TrainingProfileRegistry.default()).resolve(
        schema,
        config_values,
        extra_layers=request_extra_layers,
    )
    config_json = route_service.build_config_json(
        schema=schema,
        route=route,
        config_values=resolved.values,
        output_dir=str_field(resolved.values, "output_dir"),
        train_data_dir=str_field(resolved.values, "train_data_dir"),
    )
    return config_json, route.model_type, route.training_type, resolved.summary_dict()


def build_training_config_from_schema_route_payload(
    schema_id: str,
    data: Dict[str, Any],
    extra_config_layers: List[Dict[str, Any]] | None = None,
    *,
    backend_root: Path,
) -> tuple[dict, str, str, dict]:
    return build_training_config_from_schema(
        schema_id,
        data,
        backend_root=backend_root,
        extra_config_layers=extra_config_layers,
    )
