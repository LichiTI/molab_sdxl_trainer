# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Read-only advisory plugin extension surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


ADVISORY_EVENTS = (
    "preflight_advisory",
    "dataset_health_advisory",
    "caption_tag_advisory",
    "validation_prompt_advisory",
    "metrics_advisory",
    "report_advisory",
)

ADVISORY_CAPABILITIES = (
    "read_training_context",
    "read_dataset_summary",
    "suggest_preflight_repair",
    "suggest_caption_tags",
    "select_validation_prompts",
    "write_aux_logs",
)


@dataclass(frozen=True)
class AdvisoryPluginResult:
    status: str
    event: str
    sections: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class AdvisoryPluginService:
    def __init__(self, runtime: Any | None = None):
        self.runtime = runtime

    def emit(self, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if event not in ADVISORY_EVENTS:
            return AdvisoryPluginResult(status="unsupported", event=event, diagnostics={"allowed_events": list(ADVISORY_EVENTS)}).__dict__
        runtime = self.runtime or self._runtime()
        if runtime is None or not self._has_handlers(runtime, event):
            return AdvisoryPluginResult(status="no_plugins", event=event).__dict__
        safe_payload = self._readonly_payload(event, payload)
        report = runtime.emit_event(event, safe_payload)
        return AdvisoryPluginResult(
            status="ready",
            event=event,
            sections=self._sections(report),
            diagnostics={"dispatch": report},
        ).__dict__

    def diagnostics(self) -> Dict[str, Any]:
        runtime = self.runtime or self._runtime()
        if runtime is None:
            return {"active": False, "events": list(ADVISORY_EVENTS), "capabilities": list(ADVISORY_CAPABILITIES)}
        manifests = runtime.get_manifests() if hasattr(runtime, "get_manifests") else {}
        return {
            "active": bool(manifests),
            "events": list(ADVISORY_EVENTS),
            "capabilities": list(ADVISORY_CAPABILITIES),
            "plugins": [
                {
                    "plugin_id": plugin_id,
                    "capabilities": [cap for cap in manifest.get("capabilities", []) if cap in ADVISORY_CAPABILITIES],
                    "hooks": [hook for hook in manifest.get("hooks", []) if hook.get("event") in ADVISORY_EVENTS],
                }
                for plugin_id, manifest in manifests.items()
            ],
        }

    def _runtime(self) -> Any | None:
        try:
            from core.services.plugin_runtime import get_plugin_runtime

            return get_plugin_runtime()
        except Exception:
            return None

    def _has_handlers(self, runtime: Any, event: str) -> bool:
        try:
            bus = runtime.get_bus()
            return bool(bus.has_handlers(event))
        except Exception:
            return False

    def _readonly_payload(self, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "protocol_version": "advisory.training.v1",
            "event": event,
            "readonly": True,
            "payload": dict(payload or {}),
        }

    def _sections(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = report.get("result_payload", {}) if isinstance(report, dict) else {}
        sections = payload.get("sections", []) if isinstance(payload, dict) else []
        return [section for section in sections if isinstance(section, dict)]
