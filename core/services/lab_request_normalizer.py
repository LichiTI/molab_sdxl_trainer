"""Request normalization helpers for Lulynx LAB WebUI compatibility routes."""

from __future__ import annotations

from typing import Any

from backend.core.contracts.tools import (
    ArtifactReportRequest,
    ArtifactValidationRequest,
    DitFewStepLoraRequest,
    LabDistillerRequest,
    TurboLoraRequest,
)


def extract_lab_config(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return the legacy config payload and runtime id from a LAB route body."""

    payload = body.get("config") if isinstance(body.get("config"), dict) else body
    config = dict(payload or {})
    runtime_id = str(
        body.get("runtime_id")
        or body.get("runtimeId")
        or config.get("runtime_id")
        or config.get("runtimeId")
        or config.get("execution_profile_id")
        or ""
    ).strip()
    return config, runtime_id


def normalize_turbo_lora_request(config: dict[str, Any]) -> TurboLoraRequest:
    """Validate Turbo/LCM LoRA config through the request-native contract."""

    return TurboLoraRequest.from_legacy_payload(dict(config or {}))


def normalize_lab_distiller_request(config: dict[str, Any]) -> LabDistillerRequest:
    """Validate LAB Distiller config through the request-native contract."""

    return LabDistillerRequest.from_legacy_payload(dict(config or {}))


def normalize_dit_few_step_lora_request(config: dict[str, Any]) -> DitFewStepLoraRequest:
    """Validate Anima/Newbie few-step config through the request-native contract."""

    data = dict(config or {})
    schema_id = str(data.get("model_train_type") or data.get("schema_id") or "anima-few-step-lora").strip()
    data["schema_id"] = schema_id
    data.setdefault("model_train_type", schema_id)
    data.setdefault("model_family", "newbie" if "newbie" in schema_id.lower() else "anima")
    return DitFewStepLoraRequest.from_legacy_payload(data)


def normalize_artifact_validation_request(config: dict[str, Any]) -> ArtifactValidationRequest:
    """Validate artifact validation config while accepting legacy path aliases."""

    data = dict(config or {})
    data.setdefault("artifact_path", data.get("output_path") or data.get("path") or "")
    return ArtifactValidationRequest.from_legacy_payload(data)


def normalize_artifact_report_request(config: dict[str, Any]) -> ArtifactReportRequest:
    """Validate artifact report config while accepting legacy path aliases."""

    data = dict(config or {})
    data.setdefault("artifact_path", data.get("output_path") or data.get("path") or "")
    return ArtifactReportRequest.from_legacy_payload(data)


__all__ = [
    "extract_lab_config",
    "normalize_artifact_report_request",
    "normalize_artifact_validation_request",
    "normalize_dit_few_step_lora_request",
    "normalize_lab_distiller_request",
    "normalize_turbo_lora_request",
]
