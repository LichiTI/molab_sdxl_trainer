"""Manifest payload adapters for request-native plugin SDK declarations."""

from __future__ import annotations

from typing import Any

from backend.core.contracts import (
    PluginArtifactHandlerRegistration,
    PluginPermissionRequest,
    PluginRequestSchemaRegistration,
    PluginResourceDetectorRegistration,
    PluginRunnerRegistration,
    PluginUiSlotRegistration,
)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


def collect_sdk_registrations_from_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect request-native plugin declarations without executing code."""

    permissions: list[PluginPermissionRequest] = []
    request_schemas: list[PluginRequestSchemaRegistration] = []
    runners: list[PluginRunnerRegistration] = []
    resource_detectors: list[PluginResourceDetectorRegistration] = []
    artifact_handlers: list[PluginArtifactHandlerRegistration] = []
    ui_slots: list[PluginUiSlotRegistration] = []
    errors: list[dict[str, str]] = []

    for payload in payloads:
        plugin_id = str(payload.get("id") or payload.get("plugin_id") or "").strip()
        if not plugin_id:
            continue

        raw_permissions = payload.get("permission_requests") or payload.get("permissions") or []
        try:
            items = _as_list(raw_permissions)
            if items and all(isinstance(item, str) for item in items):
                permissions.append(PluginPermissionRequest(plugin_id=plugin_id, permissions=[str(item) for item in items]))
            else:
                for item in items:
                    if isinstance(item, dict):
                        permissions.append(PluginPermissionRequest(plugin_id=plugin_id, **item))
        except Exception as exc:
            errors.append({"plugin_id": plugin_id, "kind": "permission", "error": str(exc)})

        for item in _as_list(payload.get("request_schema_registrations") or payload.get("request_schemas") or []):
            if not isinstance(item, dict):
                continue
            try:
                schema_payload = dict(item)
                if "id" in schema_payload and "request_schema_id" not in schema_payload:
                    schema_payload["request_schema_id"] = schema_payload.pop("id")
                request_schemas.append(PluginRequestSchemaRegistration(plugin_id=plugin_id, **schema_payload))
            except Exception as exc:
                errors.append({"plugin_id": plugin_id, "kind": "request_schema", "error": str(exc)})

        for item in _as_list(payload.get("runner_registrations") or payload.get("runners") or []):
            if not isinstance(item, dict):
                continue
            try:
                runner_payload = dict(item)
                if "id" in runner_payload and "runner_id" not in runner_payload:
                    runner_payload["runner_id"] = runner_payload.pop("id")
                if "request_schema" in runner_payload and "request_schema_id" not in runner_payload:
                    runner_payload["request_schema_id"] = runner_payload.pop("request_schema")
                if "entry" in runner_payload and "entrypoint" not in runner_payload:
                    runner_payload["entrypoint"] = runner_payload.pop("entry")
                if "artifacts" in runner_payload and "artifact_types" not in runner_payload:
                    runner_payload["artifact_types"] = runner_payload.pop("artifacts")
                runners.append(PluginRunnerRegistration(plugin_id=plugin_id, **runner_payload))
            except Exception as exc:
                errors.append({"plugin_id": plugin_id, "kind": "runner", "error": str(exc)})

        for item in _as_list(payload.get("resource_detector_registrations") or payload.get("resource_detectors") or []):
            if not isinstance(item, dict):
                continue
            try:
                detector_payload = dict(item)
                if "id" in detector_payload and "detector_id" not in detector_payload:
                    detector_payload["detector_id"] = detector_payload.pop("id")
                if "entry" in detector_payload and "entrypoint" not in detector_payload:
                    detector_payload["entrypoint"] = detector_payload.pop("entry")
                resource_detectors.append(PluginResourceDetectorRegistration(plugin_id=plugin_id, **detector_payload))
            except Exception as exc:
                errors.append({"plugin_id": plugin_id, "kind": "resource_detector", "error": str(exc)})

        for item in _as_list(payload.get("artifact_handler_registrations") or payload.get("artifact_handlers") or []):
            if not isinstance(item, dict):
                continue
            try:
                handler_payload = dict(item)
                if "id" in handler_payload and "handler_id" not in handler_payload:
                    handler_payload["handler_id"] = handler_payload.pop("id")
                if "entry" in handler_payload and "entrypoint" not in handler_payload:
                    handler_payload["entrypoint"] = handler_payload.pop("entry")
                artifact_handlers.append(PluginArtifactHandlerRegistration(plugin_id=plugin_id, **handler_payload))
            except Exception as exc:
                errors.append({"plugin_id": plugin_id, "kind": "artifact_handler", "error": str(exc)})

        for item in _as_list(payload.get("ui_slot_registrations") or payload.get("ui_slots") or []):
            if not isinstance(item, dict):
                continue
            try:
                ui_slots.append(PluginUiSlotRegistration(plugin_id=plugin_id, **item))
            except Exception as exc:
                errors.append({"plugin_id": plugin_id, "kind": "ui_slot", "error": str(exc)})

    return {
        "permissions": [item.model_dump(mode="json") for item in permissions],
        "request_schemas": [item.model_dump(mode="json") for item in request_schemas],
        "runners": [item.model_dump(mode="json") for item in runners],
        "resource_detectors": [item.model_dump(mode="json") for item in resource_detectors],
        "artifact_handlers": [item.model_dump(mode="json") for item in artifact_handlers],
        "ui_slots": [item.model_dump(mode="json") for item in ui_slots],
        "errors": errors,
    }
