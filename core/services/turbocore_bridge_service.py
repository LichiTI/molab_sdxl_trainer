"""Service facade for request-native TurboCore bridge probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.contracts import RunContext, TurboCoreBridgeRequest, TurboCoreBridgeResult
from backend.core.runners import create_turbocore_bridge_registry
from core.turbocore_capabilities import probe_native_training_bridge
from core.turbocore_workspace_pipeline import build_turbocore_native_training_capability_stub


class TurboCoreBridgeService:
    """Small runtime service wrapping the TurboCore bridge runner registry."""

    def __init__(self, *, project_root: Path | None = None, backend_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.backend_root = backend_root or self.project_root / "backend"
        self._registry = create_turbocore_bridge_registry()

    def run(self, payload: TurboCoreBridgeRequest | dict[str, Any]) -> TurboCoreBridgeResult:
        request = payload if isinstance(payload, TurboCoreBridgeRequest) else TurboCoreBridgeRequest.model_validate(payload)
        context = RunContext(
            project_root=self.project_root,
            backend_root=self.backend_root,
            safe_roots=(self.project_root,),
            runtime_id=request.runtime_id,
        )
        result = self._registry.run(request, context)
        return TurboCoreBridgeResult.model_validate(result)

    def capabilities(self) -> list[dict[str, Any]]:
        return self._registry.capabilities()

    def native_capability_stub(self) -> dict[str, Any]:
        """Return native capability report when available, else canonical stub."""

        bridge = probe_native_training_bridge()
        features = bridge.get("features") if isinstance(bridge.get("features"), dict) else {}
        report = {
            "schema_version": int(bridge.get("schema_version", 1) or 1),
            "training_path_enabled": bool(bridge.get("training_path_enabled", False)),
            "training_bridge": {
                "available": bool(bridge.get("available", False)),
                "status": str(bridge.get("status", "unknown") or "unknown"),
                "reason": str(bridge.get("reason", "") or ""),
            },
            "features": dict(features),
        }
        diagnostic = bridge.get("diagnostic")
        if isinstance(diagnostic, dict):
            report["native_probe"] = diagnostic
        if bool(report["training_bridge"]["available"]) and report["features"]:
            return report
        fallback = build_turbocore_native_training_capability_stub()
        fallback["native_probe"] = diagnostic if isinstance(diagnostic, dict) else {
            "module": "lulynx_native",
            "provider": "python_stub",
            "status": "unavailable",
            "reason": "native_probe_unavailable",
        }
        return fallback
