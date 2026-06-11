"""Preprocess and dataset pipeline request contracts."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator

from .base import BaseRequest


class PreprocessRequest(BaseRequest):
    """Canonical request for dataset/image preprocessing actions."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "preprocess.generic"
    action: str = "analyze"
    input_path: str = ""
    output_path: str = ""
    dataset_path: str = ""
    caption_extension: str = ".txt"
    recursive: bool = False
    dry_run: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)
    resources: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("action")
    @classmethod
    def _normalize_action(cls, value: str) -> str:
        value = str(value or "analyze").strip().lower().replace("_", "-")
        aliases = {
            "image-resize": "resize-image",
            "resize": "resize-image",
            "tag-analysis": "analyze-tags",
            "caption-cleanup": "clean-captions",
            "cache-build": "build-cache",
        }
        return aliases.get(value, value)

    @field_validator("caption_extension")
    @classmethod
    def _normalize_caption_extension(cls, value: str) -> str:
        value = str(value or ".txt").strip()
        if not value.startswith("."):
            value = f".{value}"
        return value

    def primary_input(self) -> str:
        return self.input_path or self.dataset_path


class DatasetArtifactManifest(BaseRequest):
    """Small request-linked manifest for dataset preprocessing outputs."""

    schema_id: str = "artifact.dataset"
    artifact_kind: str = "dataset-report"
    source_path: str = ""
    output_path: str = ""
    action: str = ""
    file_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("file_count")
    @classmethod
    def _non_negative_file_count(cls, value: int) -> int:
        value = int(value)
        if value < 0:
            raise ValueError("file_count must be >= 0")
        return value
