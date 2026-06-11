"""Preview helpers for request-native preprocess routes."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import PreprocessRequest, RequestSource

from .preprocess_artifacts import attach_preprocess_artifacts


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_image_resize_request(params: dict[str, Any]) -> PreprocessRequest:
    """Build the request-native view for the legacy image resize endpoint."""

    data = dict(params or {})
    input_path = data.get("input_path") or data.get("input_dir") or data.get("dir") or data.get("path") or ""
    output_path = data.get("output_path") or data.get("output_dir") or ""
    options = dict(data.get("options") or {}) if isinstance(data.get("options"), dict) else {}
    for key, value in data.items():
        if key not in {
            "action",
            "input_path",
            "input_dir",
            "dir",
            "path",
            "output_path",
            "output_dir",
            "dataset_path",
            "caption_extension",
            "recursive",
            "dry_run",
            "options",
            "metadata",
            "schema_id",
            "schema_version",
            "compat_mode",
        }:
            options.setdefault(key, value)
    request_payload = {
        **data,
        "schema_id": data.get("schema_id") or "preprocess.image-resize",
        "action": data.get("action") or "resize-image",
        "input_path": str(input_path or ""),
        "dataset_path": str(data.get("dataset_path") or input_path or ""),
        "output_path": str(output_path or ""),
        "recursive": data.get("recursive", False),
        "dry_run": data.get("dry_run", False),
        "options": options,
    }
    return PreprocessRequest.from_legacy_payload(request_payload, source=RequestSource.WEBUI)


def image_resize_params_from_request(request: PreprocessRequest, original: dict[str, Any]) -> dict[str, Any]:
    params = dict(original or {})
    primary_input = request.primary_input()
    if primary_input:
        params.setdefault("input_dir", primary_input)
    if request.output_path:
        params.setdefault("output_dir", request.output_path)
    params["recursive"] = request.recursive
    params.update(request.options)
    return params


def image_resize_status_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": state.get("status", "idle"),
        "process_status": state.get("status", "idle"),
        "lines": state.get("lines", []),
        "result": state.get("result"),
    }


def submit_image_resize_task(
    *,
    state: dict[str, Any],
    request: PreprocessRequest,
    resize_params: dict[str, Any],
    parse_resize_options: Callable[[dict[str, Any]], Any],
    run_image_resize: Callable[[Any, Callable[[str], None]], Any],
) -> None:
    """Start the legacy image-resize background task and mutate shared state."""

    if state.get("status") == "running":
        raise RuntimeError("A resize task is already running")
    if not request.primary_input():
        raise ValueError("Missing input_dir")

    options = parse_resize_options(resize_params)
    state.clear()
    state.update({"status": "running", "lines": [], "pid": None, "result": None})

    def _run_resize() -> None:
        try:
            if resize_params.get("delete_original"):
                state["lines"].append("delete_original ignored by clean-room safe resize service")

            def _progress(line: str) -> None:
                lines = state.setdefault("lines", [])
                lines.append(line)
                if len(lines) > 500:
                    state["lines"] = lines[-500:]

            result = run_image_resize(options, _progress)
            result_payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
            state["result"] = attach_preprocess_artifacts(request, result_payload)
            errors = getattr(result, "errors", None)
            if errors is None and isinstance(result_payload, dict):
                errors = result_payload.get("errors")
            state["status"] = "done" if not errors else "error"
        except Exception as exc:
            state["status"] = "error"
            state.setdefault("lines", []).append(f"Error: {exc}")

    thread = threading.Thread(target=_run_resize, daemon=True)
    thread.start()


def collect_image_resize_preview(request: PreprocessRequest, *, limit: int = 8) -> dict[str, Any]:
    """Return legacy-compatible image resize preview data for a preprocess request."""

    input_path = request.primary_input()
    if not input_path:
        return attach_preprocess_artifacts(request, {"images": []})

    dataset_dir = Path(input_path)
    if not dataset_dir.is_dir():
        return attach_preprocess_artifacts(request, {"images": []})

    images: list[dict[str, Any]] = []
    pattern = "**/*" if request.recursive else "*"
    max_items = max(0, int(limit or 0))

    for entry in sorted(dataset_dir.glob(pattern)):
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        info: dict[str, Any] = {"path": str(entry), "name": entry.name}
        try:
            from PIL import Image

            with Image.open(entry) as im:
                width, height = im.size
                info["width"] = width
                info["height"] = height
        except Exception:
            pass
        images.append(info)
        if max_items and len(images) >= max_items:
            break

    return attach_preprocess_artifacts(request, {"images": images})

