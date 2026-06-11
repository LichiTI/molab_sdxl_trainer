"""Python reference shape loader for Lulynx image GGUF containers.

This loader is intentionally shape-only: it reads GGUF metadata and tensor
descriptors, but it does not instantiate image models or run inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from core.tools.image_gguf_runtime_contract import build_image_gguf_runtime_contract
    from core.tools.image_gguf_shape_contracts import inspect_image_gguf_shape_contract
except ImportError:
    from backend.core.tools.image_gguf_runtime_contract import build_image_gguf_runtime_contract
    from backend.core.tools.image_gguf_shape_contracts import inspect_image_gguf_shape_contract


GGUF_ARCH = "lulynx_image"


def load_image_gguf_shape_contract(path: str | Path, *, sidecar_path: str | Path | None = None) -> dict[str, Any]:
    gguf_path = Path(path)
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    if gguf_path.suffix.lower() != ".gguf":
        raise ValueError("path must end with .gguf")

    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "Image GGUF shape loader requires the optional gguf module in Launcher support dependency. "
            "Install or repair Launcher 支持依赖 with gguf included."
        ) from exc

    reader = gguf.GGUFReader(str(gguf_path))
    fields = _read_fields(reader)
    tensors = _read_tensor_infos(reader, gguf)
    tensor_names = [str(item["name"]) for item in tensors]
    sidecar = _read_sidecar(gguf_path, sidecar_path)
    sidecar_payload = sidecar["payload"]
    probe_manifests, probe_source = _read_probe_manifests(fields, sidecar_payload)
    component = str(fields.get("lulynx.image_gguf.component") or sidecar_payload.get("component") or "")
    family = str(fields.get("lulynx.image_gguf.family") or sidecar_payload.get("family") or "")
    compatibility = str(fields.get("lulynx.image_gguf.compatibility") or sidecar_payload.get("compatibility") or "")
    arch = str(fields.get("general.architecture") or "")
    container_contract = _inspect_container_contract(
        component=component,
        family=family,
        tensor_names=tensor_names,
        probe_manifests=probe_manifests,
        probe_source=probe_source,
    )
    shape_contract = inspect_image_gguf_shape_contract(component=component, family=family, tensors=tensors)

    issues = _top_level_issues(
        arch=arch,
        component=component,
        family=family,
        compatibility=compatibility,
        sidecar=sidecar,
        sidecar_payload=sidecar_payload,
        tensor_count=len(tensor_names),
        container_contract=container_contract,
        shape_contract=shape_contract,
    )
    runtime_contract = build_image_gguf_runtime_contract(
        component=component,
        issues=issues,
        container_contract=container_contract,
        shape_contract=shape_contract,
    )
    return {
        "ok": not issues,
        "path": str(gguf_path),
        "sidecar_path": str(sidecar["path"] or ""),
        "gguf_arch": arch,
        "component": component,
        "family": family,
        "compatibility": compatibility,
        "tensor_count": len(tensor_names),
        "fields": fields,
        "tensor_names_sample": tensor_names[:20],
        "tensor_descriptors_sample": tensors[:20],
        "probe_manifest_source": probe_source,
        "probe_manifest_count": len(probe_manifests),
        "probe_manifest_summary": _probe_manifest_summary(probe_manifests),
        "container_contract": container_contract,
        "shape_contract": shape_contract,
        "runtime_loadable": runtime_contract["runtime_loadable"],
        "runtime_contract": runtime_contract,
        "runtime_blockers": runtime_contract["blockers"],
        "sidecar_present": bool(sidecar["path"]),
        "issues": issues,
    }


def _read_fields(reader: Any) -> dict[str, Any]:
    return {str(key): _field_value(field) for key, field in reader.fields.items()}


def _field_value(field: Any) -> Any:
    parts = getattr(field, "parts", None)
    if not parts:
        return getattr(field, "value", None)
    data = parts[-1]
    try:
        value = data.tolist() if hasattr(data, "tolist") else data
        if isinstance(value, list):
            if len(value) == 1:
                value = value[0]
            elif value and all(isinstance(item, int) for item in value):
                try:
                    return bytes(value).decode("utf-8")
                except Exception:
                    return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
    except Exception:
        return None


def _read_tensor_infos(reader: Any, gguf: Any) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for tensor in reader.tensors:
        storage_shape = _shape_list(getattr(tensor, "shape", []))
        logical_shape = list(reversed(storage_shape))
        tensor_type_id = _safe_int(getattr(tensor, "tensor_type", None), -1)
        infos.append(
            {
                "name": str(getattr(tensor, "name", "")),
                "storage_shape": storage_shape,
                "logical_shape": logical_shape,
                "rank": len(logical_shape),
                "numel": _safe_int(getattr(tensor, "n_elements", None), _numel(logical_shape)),
                "n_bytes": _safe_int(getattr(tensor, "n_bytes", None), 0),
                "tensor_type": _tensor_type_name(gguf, tensor_type_id),
                "tensor_type_id": tensor_type_id,
            }
        )
    return sorted(infos, key=lambda item: str(item["name"]))


def _read_sidecar(gguf_path: Path, sidecar_path: str | Path | None) -> dict[str, Any]:
    path = Path(sidecar_path) if sidecar_path else gguf_path.with_suffix(gguf_path.suffix + ".manifest.json")
    if not path.is_file():
        return {"path": None, "payload": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read image GGUF sidecar: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Image GGUF sidecar must be a JSON object: {path}")
    return {"path": path, "payload": payload}


def _read_probe_manifests(fields: dict[str, Any], sidecar_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    sidecar_manifests = sidecar_payload.get("probe_manifests")
    if isinstance(sidecar_manifests, list):
        return [item for item in sidecar_manifests if isinstance(item, dict)], "sidecar"
    raw = fields.get("lulynx.image_gguf.probe_manifest")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            return [], "gguf_metadata_parse_failed"
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)], "gguf_metadata"
    return [], ""


def _inspect_container_contract(
    *,
    component: str,
    family: str,
    tensor_names: list[str],
    probe_manifests: list[dict[str, Any]],
    probe_source: str,
) -> dict[str, Any]:
    tensor_set = set(tensor_names)
    required_tensors = sorted({str(key) for manifest in probe_manifests for key in manifest.get("required_tensors", []) or [] if key})
    required_prefixes = sorted({str(prefix) for manifest in probe_manifests for prefix in manifest.get("required_prefixes", []) or [] if prefix})
    missing_tensors = [key for key in required_tensors if key not in tensor_set]
    missing_prefixes = [prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in tensor_names)]
    manifest_components = sorted({str(manifest.get("component") or "") for manifest in probe_manifests if manifest.get("component")})
    manifest_families = sorted({str(manifest.get("family") or "") for manifest in probe_manifests if manifest.get("family")})
    issues: list[str] = []
    if not probe_manifests:
        issues.append("missing embedded probe manifest; tensor namespace contract cannot be verified")
    if component and manifest_components and component not in manifest_components:
        issues.append("probe manifest component does not match GGUF metadata")
    if family and manifest_families and family not in manifest_families:
        issues.append("probe manifest family does not match GGUF metadata")
    if missing_tensors:
        issues.append(f"missing required tensors from container: {missing_tensors[:8]}")
    if missing_prefixes:
        issues.append(f"missing required tensor prefixes from container: {missing_prefixes[:8]}")
    return {
        "ok": not issues,
        "schema_version": 1,
        "component": component,
        "family": family,
        "probe_manifest_source": probe_source,
        "probe_manifest_count": len(probe_manifests),
        "tensor_count": len(tensor_names),
        "required_tensor_count": len(required_tensors),
        "required_prefix_count": len(required_prefixes),
        "missing_required_tensors": missing_tensors,
        "missing_required_prefixes": missing_prefixes,
        "manifest_components": manifest_components,
        "manifest_families": manifest_families,
        "issues": issues,
    }


def _top_level_issues(
    *,
    arch: str,
    component: str,
    family: str,
    compatibility: str,
    sidecar: dict[str, Any],
    sidecar_payload: dict[str, Any],
    tensor_count: int,
    container_contract: dict[str, Any],
    shape_contract: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if arch != GGUF_ARCH:
        issues.append(f"unexpected architecture: {arch or '<missing>'}")
    if not component:
        issues.append("missing lulynx image component metadata")
    if not family:
        issues.append("missing lulynx image family metadata")
    if compatibility != "container_compatible":
        issues.append(f"unexpected compatibility: {compatibility or '<missing>'}")
    if sidecar["path"] and sidecar_payload:
        expected_count = int(sidecar_payload.get("tensor_count") or -1)
        if expected_count != tensor_count:
            issues.append(f"sidecar tensor_count mismatch: {expected_count} != {tensor_count}")
        if str(sidecar_payload.get("component") or "") != component:
            issues.append("sidecar component does not match GGUF metadata")
        if str(sidecar_payload.get("family") or "") != family:
            issues.append("sidecar family does not match GGUF metadata")
    if not container_contract["ok"]:
        issues.extend(str(item) for item in container_contract["issues"])
    if not shape_contract["ok"]:
        issues.extend(str(item) for item in shape_contract["issues"])
    return issues


def _probe_manifest_summary(probe_manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "component": str(manifest.get("component") or ""),
            "family": str(manifest.get("family") or ""),
            "ok": bool(manifest.get("ok")),
            "tensor_count": int(manifest.get("tensor_count") or 0),
            "matched_tensors": int(manifest.get("matched_tensors") or 0),
            "missing_required_tensor_count": len(manifest.get("missing_required_tensors") or []),
            "missing_required_prefix_count": len(manifest.get("missing_required_prefixes") or []),
        }
        for manifest in probe_manifests
    ]


def _shape_list(value: Any) -> list[int]:
    try:
        raw = value.tolist() if hasattr(value, "tolist") else list(value)
    except Exception:
        raw = []
    return [int(item) for item in raw]


def _tensor_type_name(gguf: Any, tensor_type_id: int) -> str:
    enum_cls = getattr(gguf, "GGMLQuantizationType", None)
    if enum_cls is None:
        return str(tensor_type_id)
    try:
        return str(enum_cls(tensor_type_id).name).lower()
    except Exception:
        return str(tensor_type_id)


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


__all__ = ["GGUF_ARCH", "load_image_gguf_shape_contract"]
