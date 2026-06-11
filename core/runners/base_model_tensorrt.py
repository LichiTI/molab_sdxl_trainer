"""Base-model TensorRT LAB runtime runner."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from backend.core.contracts import (
    BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID,
    BaseModelTensorRtRuntimeMode,
    BaseModelTensorRtRuntimeRequest,
    BaseModelTensorRtRuntimeResult,
    PlatformIssue,
    RunContext,
    RunnerRegistry,
)


RuntimeService = Callable[[BaseModelTensorRtRuntimeRequest | Mapping[str, Any]], BaseModelTensorRtRuntimeResult]


class BaseModelTensorRtRuntimeRunner:
    """Request-native wrapper for LAB-only static TensorRT runtime checks.

    This runner deliberately exposes only the static runtime gate/smoke path.
    It does not activate product image generation or training dispatch.
    """

    runner_id = BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID
    schema_ids = (BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID,)
    capability_metadata = {
        "supports_dry_run": True,
        "permissions": ["read_models", "run_lab_runtime"],
        "resources": ["tensorrt_engine", "cuda_runtime"],
        "heavy_dependencies": ["torch", "tensorrt"],
        "estimated_cost": "heavy",
        "metadata": {
            "lab_only": True,
            "family": "newbie",
            "component": "transformer",
            "static_shape_only": True,
            "generation_path_enabled": False,
            "training_path_enabled": False,
            "note": "Runtime gate/smoke only; product generation remains disabled.",
        },
    }

    def __init__(self, runtime_service: RuntimeService | None = None) -> None:
        self._runtime_service = runtime_service

    def run(self, request: Any, context: RunContext) -> BaseModelTensorRtRuntimeResult:
        if not isinstance(request, BaseModelTensorRtRuntimeRequest):
            request = BaseModelTensorRtRuntimeRequest.model_validate(request)

        if not request.dry_run:
            return self._failure(
                "Base-model TensorRT runner is LAB-only; use dry_run=true.",
                request=request,
                issue=PlatformIssue(
                    code="base_model_tensorrt.product_activation_blocked",
                    message="Base-model TensorRT runtime checks cannot activate generation or training paths.",
                    severity="error",
                    field="dry_run",
                ),
            )

        unsafe_issue = self._unsafe_engine_path_issue(request, context)
        if unsafe_issue is not None:
            return self._failure(
                "Base-model TensorRT request failed path validation.",
                request=request,
                issue=unsafe_issue,
            )

        if self._mode(request) == BaseModelTensorRtRuntimeMode.SMOKE.value:
            permissions = set(context.metadata.get("permissions") or [])
            if "run_lab_runtime" not in permissions:
                return self._failure(
                    "Base-model TensorRT smoke requires run_lab_runtime permission.",
                    request=request,
                    issue=PlatformIssue(
                        code="base_model_tensorrt.permission_required",
                        message="run_lab_runtime permission is required for TensorRT smoke inference.",
                        severity="error",
                        field="permissions",
                    ),
                    data={"required_permission": "run_lab_runtime"},
                )

        try:
            result = self._service()(request)
        except Exception as exc:
            return self._failure(
                f"Base-model TensorRT runtime failed: {exc}",
                request=request,
                issue=PlatformIssue(
                    code="base_model_tensorrt.runtime_failed",
                    message=str(exc),
                    severity="error",
                ),
            )

        result.data.setdefault("runner_id", self.runner_id)
        result.data.setdefault("schema_id", request.schema_id)
        result.data.setdefault("mode", self._mode(request))
        result.data["generation_path_enabled"] = False
        result.data["training_path_enabled"] = False
        result.generation_path_enabled = False
        result.training_path_enabled = False
        return result

    def _service(self) -> RuntimeService:
        if self._runtime_service is not None:
            return self._runtime_service
        from backend.core.services.base_model_tensorrt_runtime_service import run_base_model_tensorrt_runtime_request

        return run_base_model_tensorrt_runtime_request

    def _unsafe_engine_path_issue(
        self,
        request: BaseModelTensorRtRuntimeRequest,
        context: RunContext,
    ) -> PlatformIssue | None:
        if context.is_safe_path(request.engine_path):
            return None
        return PlatformIssue(
            code="base_model_tensorrt.path_outside_safe_roots",
            message=f"engine_path is outside allowed safe roots: {request.engine_path}",
            severity="error",
            field="engine_path",
            hint="Use a TensorRT engine under the project roots or pass an explicit LAB temp safe root.",
        )

    def _failure(
        self,
        message: str,
        *,
        request: BaseModelTensorRtRuntimeRequest,
        issue: PlatformIssue,
        data: Mapping[str, Any] | None = None,
    ) -> BaseModelTensorRtRuntimeResult:
        payload = {
            "runner_id": self.runner_id,
            "schema_id": request.schema_id,
            "mode": self._mode(request),
            "generation_path_enabled": False,
            "training_path_enabled": False,
        }
        payload.update(dict(data or {}))
        return BaseModelTensorRtRuntimeResult.failure(
            message,
            request_id=request.request_id,
            issues=[issue],
            data=payload,
            runtime_loadable=False,
            lab_runtime_allowed=False,
            generation_path_enabled=False,
            training_path_enabled=False,
        )

    @staticmethod
    def _mode(request: BaseModelTensorRtRuntimeRequest) -> str:
        return str(request.mode or BaseModelTensorRtRuntimeMode.GATE.value)


def create_base_model_tensorrt_runtime_registry(
    *,
    runtime_service: RuntimeService | None = None,
) -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.register(BaseModelTensorRtRuntimeRunner(runtime_service=runtime_service))
    return registry


__all__ = ["BaseModelTensorRtRuntimeRunner", "create_base_model_tensorrt_runtime_registry"]
