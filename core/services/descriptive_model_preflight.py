"""Structured preflight payloads for known future model routes."""

from __future__ import annotations

from typing import Any, Dict

from backend.core.warehouse.training_features.model_optimization_capabilities import capability_dict_for_schema
from backend.core.lulynx_trainer.model_acceleration_application import model_acceleration_preflight_payload
from backend.core.warehouse.training_features.training_config_checks import (
    TrainingPreflightConfig,
    run_training_config_checks,
)


DESCRIPTIVE_MODEL_SCHEMA_IDS = {
    "flux-lora",
    "flux-finetune",
    "flux-controlnet",
    "lumina-lora",
    "lumina2-lora",
    "lumina-finetune",
    "qwen-image-lora",
    "hunyuan-dit-lora",
    "hunyuan-image-lora",
}


def descriptive_model_preflight_payload(
    *,
    raw_data: Dict[str, Any],
    schema_id: str,
    model_type: str,
    training_type: str,
    error_message: str,
) -> Dict[str, Any] | None:
    """Return a structured report for schema ids known only as future routes."""

    if schema_id not in DESCRIPTIVE_MODEL_SCHEMA_IDS:
        return None
    capability = capability_dict_for_schema(schema_id, model_type) or {}
    config = dict(raw_data or {})
    config.setdefault("schema_id", schema_id)
    config.setdefault("model_type", model_type or capability.get("family") or "")
    config.setdefault("training_type", training_type or "")
    checks = run_training_config_checks(
        TrainingPreflightConfig(
            config=config,
            training_type=training_type,
            schema_id=schema_id,
        )
    )
    model_acceleration = model_acceleration_preflight_payload(
        config,
        schema_id=schema_id,
        training_type=training_type,
    )
    errors = [
        {
            "severity": "error",
            "code": "native_route_not_wired",
            "message": error_message,
        }
    ]
    for message in checks.errors:
        item = {"severity": "error", "code": "config_error", "message": message}
        if item not in errors:
            errors.append(item)
    warnings = [{"severity": "warning", "code": "config_warning", "message": message} for message in checks.warnings]
    notes = [{"severity": "note", "code": "config_note", "message": message} for message in checks.notes]
    return {
        "can_start": False,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "config_resolution": {
            "schema_id": schema_id,
            "model_type": model_type or capability.get("family") or "",
            "training_type": training_type,
            "native_route_status": capability.get("native_route_status", "not_wired"),
            "message": error_message,
        },
        "model_optimization_capability": capability,
        "model_acceleration": model_acceleration,
        "recommended_config_patch": dict(checks.recommended_config_patch),
        "training_advisor": {"available": False, "error": error_message},
    }


__all__ = ["DESCRIPTIVE_MODEL_SCHEMA_IDS", "descriptive_model_preflight_payload"]
