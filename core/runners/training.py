"""Training runners for request-native orchestration contracts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from backend.core.contracts import (
    PlatformIssue,
    RunContext,
    RunResult,
    RunStatus,
    RunnerRegistry,
    TrainingRequest,
)


def _request_to_config_payload(request: TrainingRequest) -> dict[str, Any]:
    payload = request.to_legacy_config()
    payload.update(dict(request.config or {}))
    payload.pop("dry_run", None)
    payload.pop("config", None)
    for key in ("schema_id", "model_type", "training_type"):
        if getattr(request, key, ""):
            payload[key] = getattr(request, key)
    return payload


def _value_from_result(result: Any, key: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _result_data(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if is_dataclass(result):
        return asdict(result)
    data: dict[str, Any] = {}
    for key in (
        "status",
        "run_id",
        "config_name",
        "execution_profile_id",
        "requested_attention_backend",
        "resolved_attention_backend",
        "applied_attention_backend",
        "queue_position",
        "queue_depth",
        "message",
    ):
        value = getattr(result, key, None)
        if value is not None:
            data[key] = value
    return data


class _NativeTrainingConfig:
    """Small config wrapper matching the queue service contract."""

    def __init__(self, **kwargs: Any) -> None:
        defaults = {
            "model_type": "sdxl",
            "training_type": "lora",
            "trainer_engine": "lulynx",
            "train_data_dir": "",
            "output_dir": "",
            "output_name": "lora",
            "resume_path": None,
        }
        self._payload = {**defaults, **dict(kwargs)}
        for key, value in self._payload.items():
            setattr(self, key, value)

    def model_dump(self) -> dict[str, Any]:
        return dict(self._payload)


def _default_native_training_launcher(request: TrainingRequest, context: RunContext) -> Any:
    """Start training through the native core queue without importing UI code."""

    backend_root = context.backend_root or context.project_root / "backend"
    raw_data = _request_to_config_payload(request)

    from backend.core.execution_profile import ResolutionError
    from backend.core.execution_resolver import TrainingExecutionResolver
    from backend.core.security import validate_path
    from backend.core.services.training_queue_service import get_training_queue_service
    from backend.core.services.training_request_adapter import (
        derive_attention_backend,
        derive_schema_id,
        effective_execution_profile_id,
    )
    schema_id = request.schema_id or derive_schema_id(raw_data)
    attention_backend = derive_attention_backend(raw_data, request.attention_backend)
    execution_profile_id = effective_execution_profile_id(request.execution_profile_id)
    config_dict = dict(raw_data)
    for key in ("execution_profile_id", "attention_backend", "allow_attention_fallback", "schema_id", "extra_config_layers"):
        config_dict.pop(key, None)
    model_type = request.model_type or str(config_dict.get("model_type") or "sdxl")
    training_type = request.training_type or str(config_dict.get("training_type") or "lora")

    config = _NativeTrainingConfig(**config_dict)
    validate_path(config.train_data_dir, must_exist=True, allow_dirs=True)
    output_path = validate_path(config.output_dir, allow_dirs=True)
    output_path.mkdir(parents=True, exist_ok=True)
    if config.resume_path:
        validate_path(config.resume_path, allow_files=True, must_exist=True)
    if config.trainer_engine != "lulynx":
        raise ValueError(f"Unsupported trainer_engine={config.trainer_engine!r}. Only 'lulynx' is supported.")

    resolver = TrainingExecutionResolver(Path(backend_root))
    try:
        resolved = resolver.resolve(
            execution_profile_id=execution_profile_id,
            requested_attention=attention_backend,
            schema_id=schema_id,
            allow_attention_fallback=request.allow_attention_fallback,
            model_type=str(model_type or config.model_type).strip().lower(),
            training_type=str(training_type or config.training_type).strip().lower(),
        )
    except ResolutionError:
        raise

    return get_training_queue_service().enqueue_or_start(config=config, resolved=resolved)


class TrainingRequestRunner:
    """Request/runtime boundary for normalized training launch requests.

    The runner owns dry-run planning, path checks, permission gating, native
    queue dispatch, and result shaping. It deliberately delegates the real
    training process to backend runtime services instead of owning the trainer
    loop.
    """

    runner_id = "training.request-runner"
    schema_ids = (
        "sdxl-lora",
        "sd-lora",
        "anima-lora",
        "newbie-lora",
        "sdxl-finetune",
        "anima-finetune",
        "sdxl-dreambooth",
        "sd-dreambooth",
        "sdxl-controlnet",
        "sdxl-controlnet-lllite",
        "sd-controlnet",
        "sdxl-textual-inversion",
        "sd-textual-inversion",
        "sdxl-ip-adapter",
        "sd-ip-adapter",
        "sdxl-turbo-lora",
        "lab-distiller",
        "anima-few-step-lora",
        "newbie-few-step-lora",
    )
    capability_metadata = {
        "supports_dry_run": True,
        "permissions": ["read_models", "read_dataset", "write_output", "start_training"],
        "resources": ["pretrained_model", "train_data_dir", "output_dir", "runtime"],
        "heavy_dependencies": [],
        "estimated_cost": "heavy",
        "metadata": {
            "real_training": True,
            "note": "Request boundary, dry-run planner, and native training queue dispatcher.",
        },
    }

    def __init__(self, launcher: Any | None = None) -> None:
        self._launcher = launcher or _default_native_training_launcher

    def run(self, request: Any, context: RunContext) -> RunResult:
        if not isinstance(request, TrainingRequest):
            request = TrainingRequest.model_validate(request)

        issues = self._validate_paths(request, context)
        if issues:
            return RunResult.failure(
                "Training request failed path validation.",
                request_id=request.request_id,
                issues=issues,
                data={"runner_id": self.runner_id, "schema_id": request.schema_id},
            )

        if not request.dry_run:
            return self._run_native_training(request, context)

        return RunResult(
            request_id=request.request_id,
            status=RunStatus.SUCCEEDED,
            message="Training dry-run plan created.",
            data={
                "runner_id": self.runner_id,
                "schema_id": request.schema_id,
                "model_type": request.model_type,
                "training_type": request.training_type,
                "execution_profile_id": request.execution_profile_id,
                "attention_backend": request.attention_backend,
                "dry_run": True,
                "output_name": request.output_name,
            },
            metrics={
                "network_dim": request.network_dim,
                "network_alpha": request.network_alpha,
            },
        )

    def _run_native_training(self, request: TrainingRequest, context: RunContext) -> RunResult:
        if "start_training" not in set(context.metadata.get("permissions") or []):
            return RunResult.failure(
                "Training execution requires start_training permission.",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="training.permission_required",
                        message="start_training permission is required for native training dispatch.",
                        severity="error",
                        field="permissions",
                    )
                ],
                data={"runner_id": self.runner_id, "schema_id": request.schema_id, "required_permission": "start_training"},
            )

        try:
            launch_result = self._launcher(request, context)
        except Exception as exc:
            return RunResult.failure(
                f"Training launch failed: {exc}",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="training.launch_failed",
                        message=str(exc),
                        severity="error",
                    )
                ],
                data={"runner_id": self.runner_id, "schema_id": request.schema_id},
            )

        data = _result_data(launch_result)
        run_id = str(_value_from_result(launch_result, "run_id", "") or "")
        status = str(_value_from_result(launch_result, "status", "training_started") or "training_started")
        queued = status == "queued"
        return RunResult(
            request_id=request.request_id,
            run_id=run_id or "",
            status=RunStatus.QUEUED if queued else RunStatus.RUNNING,
            message=str(_value_from_result(launch_result, "message", "Training queued." if queued else "Training started.")),
            data={
                "runner_id": self.runner_id,
                "schema_id": request.schema_id,
                "run_id": run_id,
                "task_id": run_id,
                "native_status": status,
                "training": data,
            },
        )

    def _validate_paths(self, request: TrainingRequest, context: RunContext) -> list[PlatformIssue]:
        issues: list[PlatformIssue] = []
        for field in ("pretrained_model_name_or_path", "train_data_dir", "output_dir"):
            value = str(getattr(request, field, "") or "").strip()
            if value and not context.is_safe_path(value):
                issues.append(
                    PlatformIssue(
                        code="training.path_outside_safe_roots",
                        message=f"{field} is outside allowed safe roots: {value}",
                        severity="error",
                        field=field,
                        hint="Use project-local paths or configure an explicit safe root in the run context.",
                    )
                )
        return issues


def create_training_registry() -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.register(TrainingRequestRunner())
    return registry


TrainingRunner = TrainingRequestRunner
