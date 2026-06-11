"""Route-facing helpers for dataset preview and local media compatibility endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.services.dataset_preview import (
    audit_masked_loss_images,
    build_resized_image_file,
    list_dataset_image_preview,
    list_sample_images,
    resolve_sample_file,
)
from backend.core.services.dataset_tags_adapter import list_dataset_tags, save_dataset_tag_caption
from backend.core.services.local_desktop_adapter import open_folder_payload
from backend.core.services.preprocess_preview import (
    collect_image_resize_preview,
    image_resize_params_from_request,
    image_resize_status_payload,
    normalize_image_resize_request,
    submit_image_resize_task,
)
from backend.core.services.tageditor_service_locator import tag_editor_service


def _default_image_resize_dependencies() -> tuple[Any, Any]:
    from backend.core.dataset.image_preprocess import parse_resize_options, run_image_resize

    return parse_resize_options, run_image_resize


def list_dataset_tags_route_payload(directory: str, *, tag_editor: Any) -> dict[str, Any]:
    return list_dataset_tags(directory, tag_editor=tag_editor)


def list_dataset_tags_default_route_payload(directory: str) -> dict[str, Any]:
    return list_dataset_tags_route_payload(directory, tag_editor=tag_editor_service())


def save_dataset_tag_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return save_dataset_tag_caption(params, tag_editor=tag_editor)


def save_dataset_tag_default_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return save_dataset_tag_route_payload(params, tag_editor=tag_editor_service())


def submit_image_resize_route_payload(
    params: dict[str, Any],
    *,
    state: dict[str, Any],
    parse_resize_options: Any | None = None,
    run_image_resize: Any | None = None,
) -> dict[str, Any] | None:
    if parse_resize_options is None or run_image_resize is None:
        parse_resize_options, run_image_resize = _default_image_resize_dependencies()
    resize_request = normalize_image_resize_request(params)
    resize_params = image_resize_params_from_request(resize_request, params)
    submit_image_resize_task(
        state=state,
        request=resize_request,
        resize_params=resize_params,
        parse_resize_options=parse_resize_options,
        run_image_resize=run_image_resize,
    )
    return None


def image_resize_preview_route_payload(
    *,
    input_dir: str,
    recursive: bool,
    limit: int,
) -> dict[str, Any]:
    resize_request = normalize_image_resize_request(
        {"input_dir": input_dir, "recursive": recursive, "dry_run": True}
    )
    return collect_image_resize_preview(resize_request, limit=limit)


def image_resize_file_route_payload(
    path: str,
    *,
    output_root: Path,
    project_root: Path,
    max_size: int = 512,
) -> dict[str, Any]:
    allowed_dirs = [
        output_root.resolve(),
        (project_root / "train").resolve(),
        (project_root / "data").resolve(),
        (project_root / "models").resolve(),
        (project_root / "sd-models").resolve(),
    ]
    return build_resized_image_file(path, allowed_dirs=allowed_dirs, max_size=max_size)


def list_dataset_images_route_payload(folder: str, *, limit: int = 6) -> dict[str, Any]:
    return list_dataset_image_preview(folder, limit=limit)


def masked_loss_audit_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    dir_path = params.get("path", "") or params.get("dir", "")
    return audit_masked_loss_images(str(dir_path or ""))


def sample_images_route_payload(*, output_root: Path) -> dict[str, Any]:
    return list_sample_images(str(output_root / "sample"))


def sample_file_route_path(name: str, *, output_root: Path) -> Path:
    return resolve_sample_file(str(output_root / "sample"), name)


def open_folder_route_payload(body: dict[str, Any] | None, *, project_root: Path) -> dict[str, str]:
    return open_folder_payload(body, project_root=project_root)


def image_resize_status_route_payload(state: dict[str, Any]) -> dict[str, Any]:
    return image_resize_status_payload(state)
