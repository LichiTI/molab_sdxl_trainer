"""Route-facing helpers for launcher compatibility endpoints exposed by WebUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


OfflineRuntimePackServiceFactory = Callable[[], Any]
LauncherContextBuilder = Callable[..., Any]


def _default_offline_runtime_pack_service_factory() -> Any:
    from backend.lulynx_launcher.services.offline_runtime_pack_service import OfflineRuntimePackService

    return OfflineRuntimePackService()


def _default_launcher_context_builder() -> LauncherContextBuilder:
    from backend.lulynx_launcher.app.composition import build_context

    return build_context


def launcher_offline_runtime_packs_payload(
    *,
    project_root: Path,
    backend_root: Path,
    service_factory: OfflineRuntimePackServiceFactory | None = None,
) -> dict[str, Any]:
    service = (service_factory or _default_offline_runtime_pack_service_factory)()
    return {"packs": service.describe_packs(project_root=project_root, backend_root=backend_root)}


def launcher_local_cache_status_payload(
    *,
    project_root: Path,
    backend_root: Path,
    context_builder: LauncherContextBuilder | None = None,
) -> dict[str, Any]:
    build_context = context_builder or _default_launcher_context_builder()
    ctx = build_context(project_root=project_root, backend_root=backend_root, skip_probes=True)
    return ctx.api.get_local_cache_status()


def launcher_offline_runtime_packs_route_payload(*, backend_root: Path) -> dict[str, Any]:
    return launcher_offline_runtime_packs_payload(
        project_root=backend_root.parent,
        backend_root=backend_root,
    )


def launcher_local_cache_status_route_payload(*, backend_root: Path) -> dict[str, Any]:
    return launcher_local_cache_status_payload(
        project_root=backend_root.parent,
        backend_root=backend_root,
    )
