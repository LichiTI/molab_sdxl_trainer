"""Python reference state_dict loader for image GGUF containers.

This loader reads GGUF tensor payloads and returns a PyTorch state_dict-shaped
mapping. It is a reference/tooling bridge only: it does not instantiate model
modules, run inference, enable training dispatch, or mark GGUF runtime-loadable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from core.tools.image_gguf_shape_loader import load_image_gguf_shape_contract
except ImportError:
    from backend.core.tools.image_gguf_shape_loader import load_image_gguf_shape_contract


DEFAULT_MAX_TENSORS = 0


def load_image_gguf_state_dict(
    path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    clone_tensors: bool = True,
) -> dict[str, torch.Tensor]:
    """Load GGUF tensor payloads into a PyTorch state_dict mapping.

    ``max_tensors`` limits how many tensors are materialized. ``0`` means all
    tensors. Returned tensors are CPU tensors and detached from the GGUF memmap.
    """

    return load_image_gguf_state_dict_with_report(
        path,
        sidecar_path=sidecar_path,
        max_tensors=max_tensors,
        clone_tensors=clone_tensors,
    )["state_dict"]


def load_image_gguf_state_dict_with_report(
    path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    max_tensors: int = DEFAULT_MAX_TENSORS,
    clone_tensors: bool = True,
) -> dict[str, Any]:
    gguf_path = Path(path)
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("image GGUF state_dict loader requires gguf") from exc

    shape_report = load_image_gguf_shape_contract(gguf_path, sidecar_path=sidecar_path)
    reader = gguf.GGUFReader(str(gguf_path))
    limit = int(max_tensors or 0)
    state_dict: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    dtype_counts: Counter[str] = Counter()
    total_numel = 0
    total_bytes = 0
    for index, tensor in enumerate(sorted(reader.tensors, key=lambda item: str(item.name))):
        if limit > 0 and index >= limit:
            break
        name = str(tensor.name)
        array = np.asarray(tensor.data)
        torch_tensor = _torch_tensor_from_array(array, clone=clone_tensors)
        state_dict[name] = torch_tensor
        dtype_name = str(torch_tensor.dtype).replace("torch.", "")
        dtype_counts[dtype_name] += 1
        numel = int(torch_tensor.numel())
        bytes_count = int(numel * torch_tensor.element_size())
        total_numel += numel
        total_bytes += bytes_count
        records.append(
            {
                "name": name,
                "shape": [int(dim) for dim in torch_tensor.shape],
                "dtype": dtype_name,
                "numel": numel,
                "n_bytes": bytes_count,
                "gguf_tensor_type": _tensor_type_name(gguf, int(getattr(tensor, "tensor_type", -1))),
            }
        )
    truncated = limit > 0 and len(reader.tensors) > limit
    return {
        "schema_version": 1,
        "loader": "image_gguf_state_dict_reference_loader_v1",
        "ok": bool(state_dict) and shape_report.get("container_contract", {}).get("ok") is True,
        "report_only": True,
        "reads_tensor_payloads": True,
        "builds_model_modules": False,
        "runs_forward_pass": False,
        "runtime_loadable_enabled": False,
        "training_path_enabled": False,
        "path": str(gguf_path),
        "component": shape_report.get("component", ""),
        "family": shape_report.get("family", ""),
        "compatibility": shape_report.get("compatibility", ""),
        "state_dict": state_dict,
        "state_dict_tensor_count": len(state_dict),
        "gguf_tensor_count": len(reader.tensors),
        "truncated": truncated,
        "max_tensors": limit,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "total_numel": total_numel,
        "memory_estimate_bytes": total_bytes,
        "records": records,
        "shape_contract_ok": bool(shape_report.get("shape_contract", {}).get("ok")),
        "runtime_loadable": bool(shape_report.get("runtime_loadable")),
        "runtime_blockers": list(shape_report.get("runtime_blockers") or []),
    }


def summarize_image_gguf_state_dict_load(
    path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    max_tensors: int = 8,
) -> dict[str, Any]:
    """Load a bounded state_dict sample and return a JSON-friendly summary."""

    report = load_image_gguf_state_dict_with_report(path, sidecar_path=sidecar_path, max_tensors=max_tensors)
    return {
        key: value
        for key, value in report.items()
        if key != "state_dict"
    }


def _torch_tensor_from_array(array: np.ndarray, *, clone: bool) -> torch.Tensor:
    source = np.asarray(array)
    if clone or not source.flags.writeable:
        source = np.array(source, copy=True)
    tensor = torch.from_numpy(source)
    return tensor.detach().cpu().contiguous()


def _tensor_type_name(gguf: Any, tensor_type_id: int) -> str:
    enum_cls = getattr(gguf, "GGMLQuantizationType", None)
    if enum_cls is None:
        return str(tensor_type_id)
    try:
        return str(enum_cls(tensor_type_id).name).lower()
    except Exception:
        return str(tensor_type_id)


__all__ = [
    "load_image_gguf_state_dict",
    "load_image_gguf_state_dict_with_report",
    "summarize_image_gguf_state_dict_load",
]
