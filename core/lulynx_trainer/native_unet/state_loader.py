"""Streaming state loading helpers for native backend keymaps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

try:
    from ..safetensors_loader import open_safetensors
except ImportError:
    from safetensors import safe_open as _safe_open

    def open_safetensors(
        path: str | Path,
        *,
        framework: str = "pt",
        device: str = "cpu",
        disable_mmap: bool = False,
    ):
        del disable_mmap
        return _safe_open(str(path), framework=framework, device=device)
from .keymap_inspector import StateTensorPlan, build_state_mapping_plan


@dataclass(frozen=True)
class MappedTensor:
    source_key: str
    target_key: str
    tensor: Any
    dtype: str
    shape: list[int]
    rule_name: str


def iter_mapped_safetensors(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    *,
    limit: int | None = None,
    target_prefixes: tuple[str, ...] | list[str] | None = None,
    target_keys: set[str] | frozenset[str] | None = None,
) -> Iterator[MappedTensor]:
    """Yield mapped tensors one at a time.

    This intentionally avoids building a full native state dict by default.
    Callers that need all tensors can consume the iterator explicitly, while
    low-memory paths can load module-by-module.
    """

    requires_full_plan = (
        limit is None
        or int(limit or 0) > 20
        or target_prefixes is not None
        or target_keys is not None
    )
    if requires_full_plan:
        tensor_plans, resolved_model_path = _build_full_tensor_plan(manifest_path, model_path)
    else:
        plan = build_state_mapping_plan(manifest_path, model_path)
        if not plan.ok:
            raise ValueError(f"state mapping plan is not valid for {manifest_path}")
        tensor_plans = [
            StateTensorPlan(
                source_key=item["source_key"],
                target_key=item["target_key"],
                shape=list(item["shape"]),
                dtype=str(item["dtype"]),
                rule_name=str(item["rule_name"]),
            )
            for item in plan.tensor_plan_sample
        ]
        resolved_model_path = Path(plan.model_path)

    if target_prefixes is not None:
        prefixes = tuple(str(prefix) for prefix in target_prefixes)
        tensor_plans = [item for item in tensor_plans if item.target_key.startswith(prefixes)]
    if target_keys is not None:
        allowed_keys = {str(key) for key in target_keys}
        tensor_plans = [item for item in tensor_plans if item.target_key in allowed_keys]

    selected_plans = tensor_plans if limit is None else tensor_plans[:limit]
    with open_safetensors(str(resolved_model_path), framework="pt", device="cpu") as handle:
        for tensor_plan in selected_plans:
            tensor = handle.get_tensor(tensor_plan.source_key)
            yield MappedTensor(
                source_key=tensor_plan.source_key,
                target_key=tensor_plan.target_key,
                tensor=tensor,
                dtype=tensor_plan.dtype,
                shape=tensor_plan.shape,
                rule_name=tensor_plan.rule_name,
            )


def load_mapped_state_dict(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    *,
    limit: int | None = None,
    target_prefixes: tuple[str, ...] | list[str] | None = None,
    target_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Load a mapped state dict.

    This can materialize a large model in memory.  Prefer
    `iter_mapped_safetensors` for production low-VRAM loading paths.
    """

    return {
        item.target_key: item.tensor
        for item in iter_mapped_safetensors(
            manifest_path,
            model_path,
            limit=limit,
            target_prefixes=target_prefixes,
            target_keys=target_keys,
        )
    }


def load_mapped_state_dict_into_module(
    module: torch.nn.Module,
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    *,
    key_transform: Callable[[str], str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Stream mapped tensors directly into an existing module.

    This avoids materializing a full intermediate state dict and then copying it
    again through ``load_state_dict``.  The target module should already be on
    the desired device/dtype.
    """

    target_state = module.state_dict()
    seen: set[str] = set()
    unexpected: list[str] = []
    mismatched: list[dict[str, Any]] = []
    copied = 0
    with torch.no_grad():
        for item in iter_mapped_safetensors(manifest_path, model_path):
            target_key = key_transform(item.target_key) if key_transform is not None else item.target_key
            target = target_state.get(target_key)
            if target is None:
                unexpected.append(target_key)
                continue
            if list(target.shape) != list(item.shape):
                mismatched.append(
                    {
                        "target_key": target_key,
                        "source_shape": list(item.shape),
                        "target_shape": [int(dim) for dim in target.shape],
                    }
                )
                continue
            tensor = item.tensor
            if tensor.device != target.device or (tensor.is_floating_point() and tensor.dtype != target.dtype):
                tensor = tensor.to(
                    device=target.device,
                    dtype=target.dtype if tensor.is_floating_point() else None,
                )
            target.copy_(tensor)
            seen.add(target_key)
            copied += 1
    missing = sorted(set(target_state) - seen)
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            "mapped state streaming mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, mismatched={mismatched[:5]}"
        )
    return {
        "copied": copied,
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def _build_full_tensor_plan(
    manifest_path: str | Path,
    model_path: str | Path | None,
) -> tuple[list[StateTensorPlan], Path]:
    from .keymap_inspector import build_resolved_keymap_entries

    manifest, entries, unmatched, _metadata = build_resolved_keymap_entries(manifest_path, model_path)
    resolved_model_path = Path(model_path or manifest.get("expected_local_model") or "")
    if not resolved_model_path:
        raise ValueError(f"{manifest_path}: model_path is required")
    if unmatched:
        raise ValueError(f"state mapping plan has unmatched source keys for {manifest_path}: {unmatched[:5]}")
    target_counts = Counter(entry.target_key for entry in entries)
    duplicates = sorted(target for target, count in target_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"state mapping plan has duplicate target keys for {manifest_path}: {duplicates[:5]}")
    tensor_plans = [
        StateTensorPlan(
            source_key=entry.source_key,
            target_key=entry.target_key,
            shape=entry.shape,
            dtype=entry.dtype,
            rule_name=entry.rule_name,
        )
        for entry in entries
    ]
    return tensor_plans, resolved_model_path
