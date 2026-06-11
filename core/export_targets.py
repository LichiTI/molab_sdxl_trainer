# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Export target registry and validation-only adapter packaging contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

_SAFETENSORS_HEADER_MAX_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class ExportTarget:
    target_id: str
    name: str
    supported_adapter_types: List[str]
    supported_model_families: List[str]
    output_format: str
    metadata_requirements: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExportValidationResult:
    schema_version: int
    target: Dict[str, Any]
    ok: bool
    adapter_type: str
    model_family: str
    warnings: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    manifest_preview: Dict[str, Any]
    validation_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_EXPORT_TARGETS: Dict[str, ExportTarget] = {
    "kohya-safetensors": ExportTarget(
        target_id="kohya-safetensors",
        name="Kohya / WebUI safetensors adapter",
        supported_adapter_types=["lora", "dora", "lokr", "loha"],
        supported_model_families=["sd15", "sdxl", "sd3", "flux", "anima", "newbie"],
        output_format="safetensors",
        metadata_requirements=["ss_network_dim", "ss_network_alpha"],
        validation_checks=["safetensors_header", "adapter_type", "metadata", "digest"],
    ),
    "diffusers-lora": ExportTarget(
        target_id="diffusers-lora",
        name="Diffusers LoRA adapter",
        supported_adapter_types=["lora", "dora"],
        supported_model_families=["sd15", "sdxl", "sd3", "flux"],
        output_format="safetensors",
        metadata_requirements=[],
        validation_checks=["safetensors_header", "adapter_type", "digest"],
    ),
}


class ExportTargetRegistry:
    def __init__(self, targets: Iterable[ExportTarget] | None = None) -> None:
        if targets is None:
            self._targets = dict(_EXPORT_TARGETS)
        else:
            self._targets = {target.target_id: target for target in targets}

    def list_targets(self) -> List[Dict[str, Any]]:
        return [target.to_dict() for target in self._targets.values()]

    def get(self, target_id: str) -> ExportTarget | None:
        return self._targets.get(target_id or "kohya-safetensors")

    def validate_adapter_package(
        self,
        *,
        file_path: str | Path,
        target_id: str = "kohya-safetensors",
        model_family: str = "",
        base_model_metadata: Dict[str, Any] | None = None,
        source_run_id: str = "",
    ) -> Dict[str, Any]:
        target = self.get(target_id)
        if target is None:
            return ExportValidationResult(
                schema_version=1,
                target={"target_id": target_id},
                ok=False,
                adapter_type="unknown",
                model_family=model_family,
                warnings=[],
                errors=[{"code": "export.unknown_target", "message": f"Unknown export target: {target_id}"}],
                manifest_preview={},
            ).to_dict()

        path = Path(file_path)
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        tensor_names: List[str] = []
        metadata: Dict[str, Any] = {}
        file_digest = ""
        file_size = 0

        if not path.is_file():
            errors.append({"code": "export.file_missing", "message": f"Adapter file not found: {path}"})
        elif path.suffix.lower() != ".safetensors":
            errors.append({"code": "export.unsupported_format", "message": "Validation-only export checks currently require a .safetensors adapter."})
        else:
            try:
                header = _read_safetensors_header(path)
                tensor_names = sorted(key for key in header if key != "__metadata__")
                metadata = dict(header.get("__metadata__") or {})
                file_digest = _sha256_file(path)
                file_size = path.stat().st_size
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append({"code": "export.invalid_safetensors", "message": str(exc)})

        adapter_type = infer_adapter_type(tensor_names, metadata)
        normalized_family = str(model_family or metadata.get("ss_base_model_family") or "").lower()
        if adapter_type != "unknown" and adapter_type not in target.supported_adapter_types:
            errors.append({"code": "export.adapter_unsupported", "message": f"{adapter_type} is not supported by {target.target_id}"})
        if normalized_family and normalized_family not in target.supported_model_families:
            warnings.append({"code": "export.family_unverified", "message": f"{normalized_family} is not declared compatible with {target.target_id}"})
        missing_metadata = [key for key in target.metadata_requirements if key not in metadata]
        if missing_metadata:
            warnings.append({"code": "export.metadata_missing", "message": "Target metadata is incomplete.", "missing": missing_metadata})

        manifest = build_export_manifest(
            source_run_id=source_run_id,
            adapter_type=adapter_type,
            base_model_metadata=base_model_metadata or {},
            target=target,
            files=[{
                "path": str(path),
                "name": path.name,
                "size_bytes": file_size,
                "sha256": file_digest,
            }] if file_digest else [],
            compatibility_warnings=warnings,
            metadata=metadata,
        )
        return ExportValidationResult(
            schema_version=1,
            target=target.to_dict(),
            ok=not errors,
            adapter_type=adapter_type,
            model_family=normalized_family,
            warnings=warnings,
            errors=errors,
            manifest_preview=manifest,
        ).to_dict()


def infer_adapter_type(tensor_names: Iterable[str], metadata: Dict[str, Any] | None = None) -> str:
    names = "\n".join(name.lower() for name in tensor_names)
    network_module = str((metadata or {}).get("ss_network_module") or "").lower()
    if "lokr" in names or "lokr" in network_module:
        return "lokr"
    if "hada" in names or "loha" in network_module or "lycoris" in network_module:
        return "loha"
    if ".m" in names or "dora" in network_module:
        return "dora"
    if "lora_down" in names or "lora_up" in names or "lora_a" in names or "lora_b" in names:
        return "lora"
    return "unknown"


def build_export_manifest(
    *,
    source_run_id: str = "",
    adapter_type: str,
    base_model_metadata: Dict[str, Any],
    target: ExportTarget,
    files: List[Dict[str, Any]],
    compatibility_warnings: List[Dict[str, Any]] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "source_run_id": source_run_id,
        "adapter_type": adapter_type,
        "base_model_metadata": dict(base_model_metadata),
        "target_platform": target.to_dict(),
        "files": [dict(file_info) for file_info in files],
        "compatibility_warnings": list(compatibility_warnings or []),
        "metadata": dict(metadata or {}),
    }


def _read_safetensors_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        if header_size <= 0 or header_size > _SAFETENSORS_HEADER_MAX_BYTES:
            raise ValueError("Invalid safetensors header size")
        return json.loads(handle.read(header_size).decode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
