"""Official training UI compatibility helpers.

These helpers keep FastAPI routes as transport/envelope adapters while the
launcher/WebUI-facing training metadata services remain injectable and tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from backend.core.services.training_request_adapter import derive_schema_id


def training_profiles_payload(
    *,
    profile_registry: Any | None = None,
    schema_id: str = "",
    training_registry: Any | None = None,
) -> dict[str, Any]:
    if profile_registry is None:
        from backend.lulynx_launcher.services.training_profile_registry import TrainingProfileRegistry

        profile_registry = TrainingProfileRegistry.default()
    normalized_schema_id = str(schema_id or "").strip()
    if normalized_schema_id:
        if training_registry is None:
            from backend.lulynx_launcher.services.training_registry import LulynxTrainingRegistry

            training_registry = LulynxTrainingRegistry.default()
        schema = training_registry.get_by_id(normalized_schema_id)
        if schema is None:
            raise LookupError(f"Unknown training schema: {normalized_schema_id}")
        return {"schema_id": normalized_schema_id, "profiles": profile_registry.describe_all(normalized_schema_id)}
    return {"profiles": profile_registry.describe_all()}


def preview_training_wizard_payload(
    params: Mapping[str, Any],
    *,
    project_root: Path,
    backend_root: Path,
    wizard_service_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if wizard_service_factory is None:
        from backend.lulynx_launcher.services.training_wizard_service import TrainingWizardService

        wizard_service_factory = TrainingWizardService
    service = wizard_service_factory(project_root=project_root, backend_root=backend_root)
    return service.preview(
        goal=str(params.get("goal", "") or ""),
        schema_id=str(params.get("schema_id", "sdxl-lora") or "sdxl-lora"),
        dataset_path=str(params.get("dataset_path", "") or ""),
        base_config=dict(params.get("config") or {}),
        recipe=params.get("recipe"),
        recipe_path=str(params.get("recipe_path", "") or ""),
        source_url=str(params.get("source_url", "") or ""),
        hardware_summary=dict(params.get("hardware_summary") or {}),
        backend_capabilities=dict(params.get("backend_capabilities") or {}),
        benchmark_results=dict(params.get("benchmark_results") or {}),
    )


def preview_training_wizard_route_payload(params: Mapping[str, Any], *, backend_root: Path) -> dict[str, Any]:
    return preview_training_wizard_payload(
        params,
        project_root=backend_root.parent,
        backend_root=backend_root,
    )


def preview_training_recipe_payload(
    params: Mapping[str, Any],
    *,
    project_root: Path,
    backend_root: Path,
    recipe_service_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if recipe_service_factory is None:
        from backend.lulynx_launcher.services.training_recipe_service import TrainingRecipeService

        recipe_service_factory = TrainingRecipeService
    service = recipe_service_factory()
    return service.preview(
        recipe=params.get("recipe"),
        recipe_path=str(params.get("recipe_path", "") or ""),
        source_url=str(params.get("source_url", "") or ""),
        user_config=dict(params.get("config") or {}),
        project_root=project_root,
        backend_root=backend_root,
    )


def preview_training_recipe_route_payload(params: Mapping[str, Any], *, backend_root: Path) -> dict[str, Any]:
    return preview_training_recipe_payload(
        params,
        project_root=backend_root.parent,
        backend_root=backend_root,
    )


def resolve_training_config_payload(
    params: Mapping[str, Any],
    *,
    project_root: Path,
    backend_root: Path,
    derive_schema_id: Callable[[dict[str, Any]], str],
    training_registry: Any | None = None,
    profile_registry: Any | None = None,
    config_resolver_cls: Callable[..., Any] | None = None,
    route_service_cls: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if training_registry is None:
        from backend.lulynx_launcher.services.training_registry import LulynxTrainingRegistry

        training_registry = LulynxTrainingRegistry.default()
    if profile_registry is None:
        from backend.lulynx_launcher.services.training_profile_registry import TrainingProfileRegistry

        profile_registry = TrainingProfileRegistry.default()
    if config_resolver_cls is None:
        from backend.lulynx_launcher.services.training_config_resolver import TrainingConfigResolver

        config_resolver_cls = TrainingConfigResolver
    if route_service_cls is None:
        from backend.lulynx_launcher.services.training_route_service import TrainingRouteService

        route_service_cls = TrainingRouteService

    raw_data = dict(params.get("config") or {})
    schema_id = str(params.get("schema_id", "") or derive_schema_id(raw_data) or "")
    if not schema_id:
        raise ValueError("schema_id is required")
    schema = training_registry.get_by_id(schema_id)
    if schema is None:
        raise LookupError(f"Unknown training schema: {schema_id}")

    if params.get("config_schema_version") is not None:
        raw_data["config_schema_version"] = params.get("config_schema_version")
    if params.get("target_schema_version") is not None:
        raw_data["target_schema_version"] = params.get("target_schema_version")
    profile_id = str(params.get("profile_id", "") or "")
    if profile_id:
        raw_data["profile_id"] = profile_id

    resolved = config_resolver_cls(profile_registry).resolve(
        schema,
        raw_data,
        profile_id=profile_id or None,
        extra_layers=list(params.get("extra_config_layers") or []),
    )
    payload = resolved.summary_dict()
    payload["trainer_config_preview"] = {}
    if bool(params.get("include_trainer_config_preview", True)):
        route_service = route_service_cls(project_root, backend_root)
        route = route_service.resolve(schema_id)
        if getattr(route, "is_known", False):
            values = dict(getattr(resolved, "values", {}) or {})
            payload["trainer_config_preview"] = route_service.build_config_json(
                schema,
                route,
                values,
                _str_field(values, "output_dir"),
                _str_field(values, "train_data_dir"),
            )
    return payload


def resolve_training_config_route_payload(params: Mapping[str, Any], *, backend_root: Path) -> dict[str, Any]:
    return resolve_training_config_payload(
        params,
        project_root=backend_root.parent,
        backend_root=backend_root,
        derive_schema_id=derive_schema_id,
    )


def _str_field(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    return str(value)
