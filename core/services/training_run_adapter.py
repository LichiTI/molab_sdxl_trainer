"""Compatibility helper for the legacy `/api/run` training route."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from backend.core.contracts import RequestSource, RunContext, RunResult, RunStatus, TrainingRequest
from backend.core.runners import TrainingRunner
from backend.core.services.training_request_adapter import (
    build_training_config_from_schema_route_payload,
    derive_attention_backend,
    derive_schema_id,
    effective_execution_profile_id,
    normalize_compat_training_request,
    normalize_compat_training_route_data,
)


def _safe_roots_for_training(project_root: Path) -> tuple[Path, ...]:
    try:
        from backend.core import security

        roots = list(getattr(security, "READ_ROOTS", []) or [])
        roots.extend(list(getattr(security, "WRITE_ROOTS", []) or []))
        extra_roots = getattr(security, "_get_extra_roots", None)
        if callable(extra_roots):
            roots.extend(list(extra_roots() or []))
    except Exception:
        roots = [project_root]

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots or [project_root]:
        try:
            resolved = Path(root).resolve()
        except Exception:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return tuple(unique or [project_root])


def _training_result_payload(result: RunResult) -> dict[str, Any]:
    if str(result.status) == RunStatus.FAILED.value:
        raise ValueError(result.message or "Training launch failed")

    run_id = str(result.data.get("task_id") or result.data.get("run_id") or result.run_id or "")
    payload = {"task_id": run_id, "id": run_id}
    training = result.data.get("training") if isinstance(result.data.get("training"), dict) else {}
    for key in (
        "status",
        "run_id",
        "native_status",
        "execution_profile_id",
        "requested_attention_backend",
        "resolved_attention_backend",
        "applied_attention_backend",
        "queue_position",
        "queue_depth",
        "message",
    ):
        if key in training:
            payload[key] = training[key]
    return payload


async def run_training_compat_payload(
    *,
    raw_data: Dict[str, Any],
    request: Any,
    normalize_training_request: Callable[[Dict[str, Any]], Any],
    derive_schema_id: Callable[[Dict[str, Any]], str],
    derive_attention_backend: Callable[[Dict[str, Any], str], str],
    effective_execution_profile_id: Callable[[Any], str],
    build_training_config_from_schema: Callable[[str, Dict[str, Any], Any], tuple[dict, str, str, dict]],
    training_runner: Any | None = None,
    context: RunContext | None = None,
) -> dict[str, Any]:
    raw_data = normalize_compat_training_route_data(raw_data, request)
    training_request = normalize_training_request(raw_data)
    schema_id = getattr(training_request, "schema_id", "") or derive_schema_id(raw_data)
    attention_backend = derive_attention_backend(
        raw_data,
        getattr(training_request, "attention_backend", "") or getattr(request, "attention_backend", "auto"),
    )
    execution_profile_id = effective_execution_profile_id(
        getattr(training_request, "execution_profile_id", "") or getattr(request, "execution_profile_id", "")
    )
    raw_data["execution_profile_id"] = execution_profile_id
    extra_config_layers = getattr(request, "extra_config_layers", [])

    if schema_id:
        config_dict, _, _, _ = build_training_config_from_schema(schema_id, raw_data, extra_config_layers)
    else:
        config_dict = dict(raw_data)
        for key in ("execution_profile_id", "attention_backend", "allow_attention_fallback", "schema_id", "extra_config_layers"):
            config_dict.pop(key, None)

    runner_request = TrainingRequest.from_legacy_payload(
        {
            **config_dict,
            "schema_id": schema_id,
            "execution_profile_id": execution_profile_id,
            "attention_backend": attention_backend,
            "allow_attention_fallback": getattr(request, "allow_attention_fallback", True),
            "extra_config_layers": extra_config_layers,
            "config": config_dict,
            "dry_run": False,
        },
        source=RequestSource.FASTAPI,
        compat_mode=True,
    )
    run_context = context or RunContext(
        project_root=Path.cwd(),
        metadata={"permissions": ["start_training"], "source": "compat.api.run"},
    )
    return _training_result_payload((training_runner or TrainingRunner()).run(runner_request, run_context))


async def run_training_compat_route_payload(
    *,
    raw_data: Dict[str, Any],
    request: Any,
    backend_root: Path | None = None,
    build_training_config_from_schema: Callable[[str, Dict[str, Any], Any], tuple[dict, str, str, dict]] | None = None,
    training_runner: Any | None = None,
) -> dict[str, Any]:
    """Route-facing `/api/run` adapter using the request-native TrainingRunner."""

    if backend_root is None:
        raise ValueError("backend_root is required")
    if build_training_config_from_schema is None:

        def build_training_config_from_schema(schema_id: str, data: Dict[str, Any], extra_config_layers: Any = None):
            return build_training_config_from_schema_route_payload(
                schema_id,
                data,
                extra_config_layers,
                backend_root=backend_root,
            )

    project_root = backend_root.parent
    return await run_training_compat_payload(
        raw_data=raw_data,
        request=request,
        normalize_training_request=normalize_compat_training_request,
        derive_schema_id=derive_schema_id,
        derive_attention_backend=derive_attention_backend,
        effective_execution_profile_id=effective_execution_profile_id,
        build_training_config_from_schema=build_training_config_from_schema,
        training_runner=training_runner,
        context=RunContext(
            project_root=project_root,
            backend_root=backend_root,
            safe_roots=_safe_roots_for_training(project_root),
            metadata={"permissions": ["start_training"], "source": "compat.api.run"},
        ),
    )
