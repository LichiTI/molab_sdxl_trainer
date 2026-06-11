"""TurboCore request-native runners."""

from __future__ import annotations

from typing import Any

from backend.core.contracts import (
    PlatformIssue,
    RunContext,
    RunStatus,
    RunnerRegistry,
    TurboCoreBridgeAction,
    TurboCoreBridgeRequest,
    TurboCoreBridgeResult,
)


class TurboCoreBridgeRunner:
    """Developer-only TurboCore bridge runner.

    The runner exposes probe/capability/lifecycle evidence through the same
    request-native path as generation dry-runs. It does not activate training.
    """

    runner_id = "turbocore.bridge"
    schema_ids = ("turbocore.bridge", "turbocore.workspace_pipeline")
    capability_metadata = {
        "supports_dry_run": True,
        "permissions": ["read_runtime_capabilities"],
        "resources": ["turbocore_capability_report"],
        "heavy_dependencies": [],
        "estimated_cost": "light",
        "metadata": {
            "training_path_enabled": False,
            "native_runtime": False,
            "developer_only": True,
        },
    }

    def run(self, request: Any, context: RunContext) -> TurboCoreBridgeResult:
        if not isinstance(request, TurboCoreBridgeRequest):
            request = TurboCoreBridgeRequest.model_validate(request)

        if not request.dry_run:
            return TurboCoreBridgeResult.failure(
                "TurboCore bridge runner is research-only; use dry_run=true.",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="turbocore.bridge.training_activation_blocked",
                        message="TurboCore bridge probes cannot activate training paths.",
                        severity="error",
                        field="dry_run",
                    )
                ],
                data={"runner_id": self.runner_id, "schema_id": request.schema_id},
            )

        action = str(request.action)
        data = self._run_action(request, context)
        return TurboCoreBridgeResult(
            request_id=request.request_id,
            status=RunStatus.SUCCEEDED,
            message=f"TurboCore bridge {action} completed.",
            data={
                "runner_id": self.runner_id,
                "schema_id": request.schema_id,
                "action": action,
                "training_path_enabled": False,
                **data,
            },
            metrics=self._metrics(data),
        )

    def _run_action(self, request: TurboCoreBridgeRequest, _context: RunContext) -> dict[str, Any]:
        action = str(request.action)
        if action == TurboCoreBridgeAction.CAPABILITY.value:
            return {"capability": self._capability_report(request)}
        if action == TurboCoreBridgeAction.WORKSPACE_PIPELINE_LIFECYCLE.value:
            return {"lifecycle": self._lifecycle_report(request)}
        if action == TurboCoreBridgeAction.VALIDATE_NATIVE_ABI.value:
            return {"native_abi_validation": self._native_abi_validation(request)}
        return {
            "capability": self._capability_report(request),
            "lifecycle": self._lifecycle_report(request),
            "native_capability_stub": self._native_capability_stub(),
            "native_abi_validation": self._native_abi_validation(request),
        }

    def _capability_report(self, request: TurboCoreBridgeRequest) -> dict[str, Any]:
        from core.turbocore_capabilities import build_turbocore_capability_report

        return build_turbocore_capability_report(self._config(request))

    def _lifecycle_report(self, request: TurboCoreBridgeRequest) -> dict[str, Any]:
        from core.turbocore_workspace_pipeline import run_workspace_pipeline_lifecycle_probe

        dtype = self._torch_dtype(request.dtype)
        return run_workspace_pipeline_lifecycle_probe(
            batches=max(int(request.lifecycle_batches), 0),
            prefetch_depth=max(int(request.prefetch_depth), 1),
            workspace_mb=max(int(request.workspace_mb), 0),
            dtype=dtype,
            device=str(request.device or "cpu"),
            chunk_size=self._native_chunk_size(request),
        )

    def _native_abi_validation(self, request: TurboCoreBridgeRequest) -> dict[str, Any]:
        from core.turbocore_native_abi import validate_workspace_pipeline_native_capabilities

        return validate_workspace_pipeline_native_capabilities(self._native_report_or_stub(request))

    def _native_capability_stub(self) -> dict[str, Any]:
        from core.turbocore_capabilities import probe_native_training_bridge
        from core.turbocore_workspace_pipeline import build_turbocore_native_training_capability_stub

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

    def _native_report_or_stub(self, request: TurboCoreBridgeRequest) -> dict[str, Any]:
        report = request.native_report if isinstance(request.native_report, dict) else {}
        return dict(report) if report else self._native_capability_stub()

    def _config(self, request: TurboCoreBridgeRequest) -> dict[str, Any]:
        return {
            "model_type": request.model_type,
            "training_type": request.training_type,
            "execution_core": "turbo",
            "turbocore_features": request.normalized_features(),
            "turbocore_workspace_mb": max(int(request.workspace_mb), 0),
            "turbocore_prefetch_depth": max(int(request.prefetch_depth), 1),
            "turbocore_allow_fallback": True,
        }

    @staticmethod
    def _torch_dtype(value: str) -> Any:
        import torch

        text = str(value or "float32").strip().lower()
        if text in {"fp16", "float16", "half"}:
            return torch.float16
        if text in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _native_chunk_size(request: TurboCoreBridgeRequest) -> int:
        extra = request.model_extra if isinstance(request.model_extra, dict) else {}
        try:
            return max(int(extra.get("native_chunk_size", 256) or 256), 1)
        except Exception:
            return 256

    @staticmethod
    def _metrics(data: dict[str, Any]) -> dict[str, Any]:
        lifecycle = data.get("lifecycle") if isinstance(data.get("lifecycle"), dict) else {}
        stats = lifecycle.get("stats") if isinstance(lifecycle.get("stats"), dict) else {}
        pool = lifecycle.get("workspace_pool") if isinstance(lifecycle.get("workspace_pool"), dict) else {}
        return {
            "lifecycle_ok": bool(lifecycle.get("ok", False)) if lifecycle else False,
            "submitted": int(lifecycle.get("submitted_batches", 0) or 0) if lifecycle else 0,
            "consumed": int(lifecycle.get("consumed_batches", 0) or 0) if lifecycle else 0,
            "queue_full_stalls": int(stats.get("queue_full_stalls", 0) or 0),
            "workspace_hits": int(pool.get("hits", 0) or 0),
            "workspace_misses": int(pool.get("misses", 0) or 0),
            "native_runtime": bool(lifecycle.get("native_runtime", False)) if lifecycle else False,
            "lifecycle_provider": str(lifecycle.get("provider", "")) if lifecycle else "",
            "chunk_size": int(lifecycle.get("chunk_size", 0) or 0) if lifecycle else 0,
        }


def create_turbocore_bridge_registry() -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.register(TurboCoreBridgeRunner())
    return registry
