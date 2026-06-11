from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .static_engine import compare_tensor_outputs


def save_tensor_output_artifact(
    path: str | Path,
    tensor: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    output_name: str = "sample_out",
) -> dict[str, Any]:
    started = time.perf_counter()
    import torch

    dst = Path(path)
    if not str(dst):
        raise ValueError("Tensor output artifact path is required")
    if dst.suffix.lower() != ".pt":
        dst = dst.with_suffix(".pt")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cpu_tensor = tensor.detach().cpu().contiguous()
    summary = summarize_tensor_artifact_value(cpu_tensor)
    payload = {
        "schema_version": 1,
        "kind": "base_model_tensorrt_tensor_output_artifact",
        "output_name": str(output_name or "sample_out"),
        "metadata": _json_safe(dict(metadata or {})),
        "summary": summary,
        "tensor": cpu_tensor,
    }
    torch.save(payload, str(dst))
    return {
        "schema_version": 1,
        "kind": "base_model_tensorrt_tensor_output_artifact_write",
        "success": True,
        "path": str(dst),
        "bytes": dst.stat().st_size,
        "sha256": _file_sha256(dst),
        "output_name": payload["output_name"],
        "metadata": payload["metadata"],
        "summary": summary,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def load_tensor_output_artifact(path: str | Path) -> dict[str, Any]:
    import torch

    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Tensor output artifact not found: {src}")
    try:
        payload = torch.load(str(src), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(src), map_location="cpu")
    if not isinstance(payload, dict) or "tensor" not in payload:
        raise ValueError(f"Invalid tensor output artifact: {src}")
    tensor = payload["tensor"]
    if not hasattr(tensor, "detach"):
        raise ValueError(f"Tensor output artifact has no tensor payload: {src}")
    return {
        "path": str(src),
        "bytes": src.stat().st_size,
        "sha256": _file_sha256(src),
        "schema_version": int(payload.get("schema_version") or 0),
        "kind": str(payload.get("kind") or ""),
        "output_name": str(payload.get("output_name") or "sample_out"),
        "metadata": dict(payload.get("metadata") or {}),
        "summary": dict(payload.get("summary") or summarize_tensor_artifact_value(tensor)),
        "tensor": tensor.detach().cpu().contiguous(),
    }


def compare_tensor_output_artifacts(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    started = time.perf_counter()
    reference = load_tensor_output_artifact(reference_path)
    candidate = load_tensor_output_artifact(candidate_path)
    comparison = compare_tensor_outputs(reference["tensor"], candidate["tensor"])
    compatibility = _metadata_compatibility(reference, candidate)
    return {
        "schema_version": 1,
        "kind": "base_model_tensorrt_tensor_output_artifact_compare",
        "success": bool(comparison.get("same_shape")) and bool(comparison.get("all_finite", True)),
        "parity_acceptable": bool(comparison.get("parity_acceptable", False)),
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "reference_artifact": _artifact_report(reference),
        "candidate_artifact": _artifact_report(candidate),
        "metadata_compatibility": compatibility,
        "comparison": comparison,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def summarize_tensor_artifact_value(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().float().cpu()
    finite = torch.isfinite(value)
    return {
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "finite_ratio": float(finite.float().mean().item()) if value.numel() else 1.0,
        "mean": float(value.mean().item()) if value.numel() else 0.0,
        "std": float(value.std(unbiased=False).item()) if value.numel() else 0.0,
        "min": float(value.min().item()) if value.numel() else 0.0,
        "max": float(value.max().item()) if value.numel() else 0.0,
    }


def _artifact_report(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": artifact.get("path", ""),
        "bytes": int(artifact.get("bytes") or 0),
        "sha256": artifact.get("sha256", ""),
        "output_name": artifact.get("output_name", "sample_out"),
        "metadata": artifact.get("metadata", {}),
        "summary": artifact.get("summary", {}),
    }


def _metadata_compatibility(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    ref_meta = dict(reference.get("metadata") or {})
    got_meta = dict(candidate.get("metadata") or {})
    keys = ("model_family", "layer_indices", "shape", "seed", "dtype", "device", "input_signature", "output_name")
    mismatches = [key for key in keys if _metadata_value(ref_meta, key) != _metadata_value(got_meta, key)]
    if reference.get("output_name") != candidate.get("output_name"):
        mismatches.append("artifact_output_name")
    return {
        "compatible": not mismatches,
        "checked_keys": list(keys) + ["artifact_output_name"],
        "mismatches": sorted(set(mismatches)),
    }


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if key == "device":
        return _normalize_device_name(value)
    return value


def _normalize_device_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "cuda":
        return "cuda:0"
    return text


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
