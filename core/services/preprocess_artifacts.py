"""Artifact helpers for request-native preprocess routes."""

from __future__ import annotations

from typing import Any

from backend.core.contracts import ArtifactFile, ArtifactManifest, PreprocessRequest


def guess_preprocess_artifact_kind(request: PreprocessRequest) -> str:
    if request.action == "resize-image":
        return "dataset-resize"
    if request.action in {"clean-captions", "analyze-tags"}:
        return "dataset-report"
    if request.action == "build-cache":
        return "cache"
    return "dataset-report"


def build_preprocess_artifacts(
    request: PreprocessRequest,
    *,
    result: dict[str, Any] | None = None,
    output_path: str = "",
) -> list[dict[str, Any]]:
    """Build common artifact manifests for legacy preprocess route responses."""

    source_path = request.primary_input()
    effective_output = output_path or request.output_path or source_path
    files: list[ArtifactFile] = []
    if source_path:
        files.append(ArtifactFile(path=source_path, role="input"))
    if effective_output:
        files.append(ArtifactFile(path=effective_output, role="output" if effective_output != source_path else "dataset"))
    artifact = ArtifactManifest(
        artifact_kind=guess_preprocess_artifact_kind(request),
        schema_id=request.schema_id,
        producer="webui-preprocess",
        request_id=request.request_id,
        files=files,
        metadata={
            "action": request.action,
            "caption_extension": request.caption_extension,
            "recursive": request.recursive,
            "dry_run": request.dry_run,
            "summary": (result or {}).get("summary", {}) if isinstance(result, dict) else {},
        },
    )
    return [artifact.model_dump(mode="json")]


def attach_preprocess_artifacts(request: PreprocessRequest, payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload or {})
    data.setdefault("request_id", request.request_id)
    data.setdefault("schema_id", request.schema_id)
    data.setdefault("artifacts", build_preprocess_artifacts(request, result=data))
    return data
