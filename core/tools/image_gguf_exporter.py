"""Image-model GGUF container exporter.

This exporter writes Lulynx-owned GGUF containers plus a JSON sidecar manifest.
It does not claim runtime loadability; UNet and DiT components are container-only
until the target runtime loader contract is implemented.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch

try:
    from core.image_gguf_contracts import ImageGGUFCompatibility, ImageGGUFExportPlan, ImageGGUFExportResult, TensorInfo
    from core.image_gguf_probe import ImageGGUFAdapterRegistry, probe_image_gguf_manifest, read_safetensors_tensor_info
except ImportError:
    from backend.core.image_gguf_contracts import ImageGGUFCompatibility, ImageGGUFExportPlan, ImageGGUFExportResult, TensorInfo
    from backend.core.image_gguf_probe import ImageGGUFAdapterRegistry, probe_image_gguf_manifest, read_safetensors_tensor_info


CONTAINER_ONLY_COMPONENTS = {"anima_dit", "newbie_dit", "sd15_unet", "sdxl_unet"}
EXPORTABLE_COMPONENTS = {"vae", "clip", "t5"} | CONTAINER_ONLY_COMPONENTS
GGUF_ARCH = "lulynx_image"


def plan_image_gguf_export(
    input_paths: str | Path | Iterable[str | Path],
    *,
    family_hint: str = "",
    file_type: str = "f16",
) -> ImageGGUFExportPlan:
    sources = _normalize_sources(input_paths)
    normalized_file_type = _normalize_file_type(file_type)
    manifests = [probe_image_gguf_manifest(path, family_hint=family_hint).to_dict() for path in sources]
    errors = _export_manifest_errors(manifests)
    component = str(manifests[0]["component"])
    family = str(manifests[0]["family"])
    infos, duplicate_count = _collect_tensor_info(sources)
    tensor_bytes, converted, skipped = _estimate_tensor_bytes(infos, normalized_file_type)
    overhead = _estimate_container_overhead_bytes(infos, manifests)
    return ImageGGUFExportPlan(
        schema_version=1,
        ok=not errors,
        component=component,
        family=family,
        compatibility=(
            ImageGGUFCompatibility.CONTAINER_CANDIDATE.value
            if not errors
            else ImageGGUFCompatibility.PROBE_ONLY.value
        ),
        source_paths=[str(path) for path in sources],
        tensor_count=sum(int(manifest.get("tensor_count") or 0) for manifest in manifests),
        unique_tensor_count=len(infos),
        duplicate_tensor_count=duplicate_count,
        converted_tensors=converted,
        skipped_tensors=skipped,
        estimated_tensor_bytes=tensor_bytes,
        estimated_container_overhead_bytes=overhead,
        estimated_output_size_bytes=tensor_bytes + overhead,
        gguf_arch=GGUF_ARCH,
        gguf_file_type=normalized_file_type,
        dtype_counts=_info_dtype_counts(infos),
        rank_counts=_info_rank_counts(infos),
        warnings=_collect_warnings(manifests),
        errors=errors,
    )


def export_image_gguf_component(
    input_paths: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    family_hint: str = "",
    name: str = "",
    file_type: str = "f16",
    overwrite: bool = False,
) -> ImageGGUFExportResult:
    sources = _normalize_sources(input_paths)
    dst = Path(output_path)
    _validate_output_path(dst, overwrite=overwrite)
    normalized_file_type = _normalize_file_type(file_type)

    manifests = [probe_image_gguf_manifest(path, family_hint=family_hint).to_dict() for path in sources]
    _validate_export_manifests(manifests)
    component = str(manifests[0]["component"])
    family = str(manifests[0]["family"])
    tensors, skipped = _load_component_tensors(sources, normalized_file_type)
    if not tensors:
        raise ValueError("image GGUF export requires at least one tensor")

    dst.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = dst.with_suffix(dst.suffix + ".manifest.json")
    if sidecar_path.exists() and not overwrite:
        raise FileExistsError(f"Sidecar already exists: {sidecar_path}")

    stats = _write_gguf_container(
        dst,
        tensors,
        component=component,
        family=family,
        name=name or sources[0].stem,
        file_type=normalized_file_type,
        manifests=manifests,
    )

    result = ImageGGUFExportResult(
        schema_version=1,
        ok=True,
        output_path=str(dst),
        sidecar_path=str(sidecar_path),
        component=component,
        family=family,
        compatibility=ImageGGUFCompatibility.CONTAINER_COMPATIBLE.value,
        source_paths=[str(path) for path in sources],
        tensor_count=len(tensors),
        converted_tensors=int(stats["converted_tensors"]),
        skipped_tensors=skipped,
        output_size_bytes=os.path.getsize(dst),
        gguf_arch=GGUF_ARCH,
        gguf_file_type=normalized_file_type,
        dtype_counts=_dtype_counts(tensors),
        rank_counts=_rank_counts(tensors),
        warnings=_collect_warnings(manifests),
    )
    _write_sidecar(sidecar_path, result, manifests)
    return result


def _normalize_sources(input_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(input_paths, (str, Path)):
        raw_paths = [input_paths]
    else:
        raw_paths = list(input_paths)
    paths = [Path(path) for path in raw_paths]
    if not paths:
        raise ValueError("input_paths must not be empty")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError("image GGUF export currently accepts .safetensors inputs only")
    return paths


def _validate_output_path(dst: Path, *, overwrite: bool) -> None:
    if dst.suffix.lower() != ".gguf":
        raise ValueError("output_path must end with .gguf")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst}")


def _normalize_file_type(value: Any) -> str:
    text = str(value or "f16").strip().lower().replace("-", "_")
    aliases = {"fp16": "f16", "float16": "f16", "half": "f16", "fp32": "f32", "float32": "f32"}
    text = aliases.get(text, text)
    if text not in {"f16", "f32"}:
        raise ValueError("image GGUF export file_type must be f16 or f32")
    return text


def _validate_export_manifests(manifests: list[dict[str, Any]]) -> None:
    errors = _export_manifest_errors(manifests)
    if errors:
        raise ValueError(errors[0])


def _export_manifest_errors(manifests: list[dict[str, Any]]) -> list[str]:
    components = {str(manifest["component"]) for manifest in manifests}
    families = {str(manifest["family"]) for manifest in manifests}
    if len(components) != 1:
        return [f"image GGUF export cannot mix components: {sorted(components)}"]
    component = next(iter(components))
    if component not in EXPORTABLE_COMPONENTS:
        return [f"image GGUF export does not support component yet: {component}"]
    if len(families) != 1:
        return [f"image GGUF export cannot mix families: {sorted(families)}"]
    if component != "t5":
        bad = [manifest for manifest in manifests if not manifest.get("ok")]
        if bad:
            return [f"image GGUF export probe failed for {bad[0].get('source_path')}"]
        return []

    if not any(not manifest.get("missing_required_tensors") for manifest in manifests):
        return ["T5 image GGUF export requires at least one shard containing the required base tensors"]
    bad_prefixes = [manifest for manifest in manifests if manifest.get("missing_required_prefixes")]
    if bad_prefixes:
        return [f"T5 image GGUF export has missing required prefixes in {bad_prefixes[0].get('source_path')}"]
    return []


def _collect_tensor_info(paths: list[Path]) -> tuple[dict[str, TensorInfo], int]:
    infos: dict[str, TensorInfo] = {}
    duplicates = 0
    for path in paths:
        for key, info in read_safetensors_tensor_info(path).items():
            if key in infos:
                duplicates += 1
                continue
            infos[key] = info
    return infos, duplicates


def _estimate_tensor_bytes(infos: dict[str, TensorInfo], file_type: str) -> tuple[int, int, int]:
    total = 0
    converted = 0
    skipped = 0
    target_float_bytes = 2 if file_type == "f16" else 4
    for info in infos.values():
        source_bytes = _dtype_size_bytes(info.dtype)
        if _is_float_dtype(info.dtype):
            total += info.numel * target_float_bytes
            converted += 1
        else:
            total += info.numel * source_bytes
            skipped += 1
    return total, converted, skipped


def _estimate_container_overhead_bytes(infos: dict[str, TensorInfo], manifests: list[dict[str, Any]]) -> int:
    key_bytes = sum(len(key.encode("utf-8")) + 96 for key in infos)
    manifest_bytes = len(json.dumps(manifests, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return key_bytes + manifest_bytes + 4096


def _dtype_size_bytes(dtype: str) -> int:
    text = str(dtype).upper()
    if text in {"F64", "I64", "U64"}:
        return 8
    if text in {"F32", "I32", "U32"}:
        return 4
    if text in {"F16", "BF16", "I16", "U16"}:
        return 2
    return 1


def _is_float_dtype(dtype: str) -> bool:
    return str(dtype).upper() in {"F64", "F32", "F16", "BF16"}


def _load_component_tensors(paths: list[Path], file_type: str) -> tuple[dict[str, torch.Tensor], int]:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    skipped = 0
    target_dtype = torch.float16 if file_type == "f16" else torch.float32
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in tensors:
                    skipped += 1
                    continue
                tensor = handle.get_tensor(key).detach().cpu().contiguous()
                if tensor.is_floating_point():
                    tensor = tensor.to(target_dtype)
                tensors[str(key)] = tensor
    return tensors, skipped


def _write_gguf_container(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    component: str,
    family: str,
    name: str,
    file_type: str,
    manifests: list[dict[str, Any]],
) -> dict[str, int]:
    try:
        import numpy as np
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "Image GGUF export requires the optional gguf module in Launcher support dependency. "
            "Install or repair Launcher 支持依赖 with gguf included."
        ) from exc

    writer = gguf.GGUFWriter(str(path), GGUF_ARCH)
    converted = 0
    try:
        _add_metadata(writer, gguf, component=component, family=family, name=name, file_type=file_type, manifests=manifests)
        for key, tensor in tensors.items():
            value = tensor.detach().cpu().contiguous()
            array = value.numpy()
            if value.is_floating_point():
                array = array.astype(np.float16 if file_type == "f16" else np.float32, copy=False)
                converted += 1
            writer.add_tensor(str(key), array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        close = getattr(writer, "close", None)
        if callable(close):
            close()
    if not path.is_file():
        raise RuntimeError("GGUF writer completed without creating the output file")
    return {"converted_tensors": converted}


def _add_metadata(
    writer: Any,
    gguf: Any,
    *,
    component: str,
    family: str,
    name: str,
    file_type: str,
    manifests: list[dict[str, Any]],
) -> None:
    quant_type = gguf.GGMLQuantizationType.F16 if file_type == "f16" else gguf.GGMLQuantizationType.F32
    writer.add_name(str(name))
    writer.add_file_type(quant_type)
    writer.add_string("lulynx.image_gguf.schema", "1")
    writer.add_string("lulynx.image_gguf.component", component)
    writer.add_string("lulynx.image_gguf.family", family)
    writer.add_string("lulynx.image_gguf.compatibility", ImageGGUFCompatibility.CONTAINER_COMPATIBLE.value)
    writer.add_string("lulynx.image_gguf.source_count", str(len(manifests)))
    writer.add_string("lulynx.image_gguf.probe_manifest", json.dumps(manifests, ensure_ascii=False, separators=(",", ":")))


def _write_sidecar(path: Path, result: ImageGGUFExportResult, manifests: list[dict[str, Any]]) -> None:
    payload = result.to_dict()
    payload["probe_manifests"] = manifests
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dtype_counts(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in tensors.values():
        key = str(tensor.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _rank_counts(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in tensors.values():
        key = str(tensor.dim())
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _info_dtype_counts(infos: dict[str, TensorInfo]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for info in infos.values():
        counts[info.dtype] = counts.get(info.dtype, 0) + 1
    return dict(sorted(counts.items()))


def _info_rank_counts(infos: dict[str, TensorInfo]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for info in infos.values():
        key = str(info.rank)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _collect_warnings(manifests: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    components = {str(manifest.get("component") or "") for manifest in manifests}
    for component in sorted(components & CONTAINER_ONLY_COMPONENTS):
        warnings.append(f"{component} image GGUF export is container-compatible only; runtime loader contract is not implemented yet")
    for manifest in manifests:
        warnings.extend(str(item) for item in manifest.get("warnings", []) or [])
        if manifest.get("missing_required_tensors"):
            source = manifest.get("source_path")
            missing = manifest.get("missing_required_tensors")
            warnings.append(f"source shard has missing required tensors: {source}: {missing}")
    return warnings


__all__ = ["CONTAINER_ONLY_COMPONENTS", "EXPORTABLE_COMPONENTS", "GGUF_ARCH", "export_image_gguf_component", "plan_image_gguf_export"]
