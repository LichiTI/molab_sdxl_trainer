"""Lulynx runtime contract helpers for native multi-batch training.

This module describes and validates our own batch semantics. It intentionally
keeps the implementation small and independent from reference projects: the
goal is to prove whether a training batch is a real physical batch, whether its
fields agree on the leading dimension, and which execution strategy is safest
for the current feature mix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # torch is optional for pure metadata tests.
    import torch
except Exception:  # pragma: no cover - torch is available in normal runtime
    torch = None  # type: ignore[assignment]


TENSOR_BATCH_FIELDS = {
    "images",
    "latents",
    "encoder_hidden_states",
    "pooled_prompt_embeds",
    "attention_mask",
    "qwen3_hidden_states",
    "qwen3_attention_mask",
    "guidance_images",
    "control_images",
    "loss_masks",
    "caption_weights",
    "padding_mask",
    "ip_adapter_images",
    "ip_adapter_image_features",
}

SEQUENCE_BATCH_FIELDS = {
    "captions",
    "original_sizes",
    "target_sizes",
    "crop_coords",
    "filenames",
}

OPTIONAL_SINGLETON_FIELDS = {
    "network_weights",
}

LULYNX_MULTI_BATCH_DATALOADER_CONTRACT_ATTR = "_lulynx_multi_batch_dataloader_contract"


@dataclass(frozen=True)
class MultiBatchRequest:
    """Normalized batch-size request from runtime/config state."""

    physical_batch_size: int
    gradient_accumulation_steps: int = 1
    data_parallel_world_size: int = 1

    @property
    def effective_batch_size(self) -> int:
        return self.physical_batch_size * self.gradient_accumulation_steps * self.data_parallel_world_size


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return max(int(value if value is not None else default), 1)
    except (TypeError, ValueError, OverflowError):
        return max(int(default), 1)


def normalize_multi_batch_request(
    *,
    train_batch_size: Any = 1,
    gradient_accumulation_steps: Any = 1,
    data_parallel_world_size: Any = 1,
) -> MultiBatchRequest:
    """Return explicit physical/effective batch semantics."""

    return MultiBatchRequest(
        physical_batch_size=_safe_int(train_batch_size),
        gradient_accumulation_steps=_safe_int(gradient_accumulation_steps),
        data_parallel_world_size=_safe_int(data_parallel_world_size),
    )


def _is_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _sequence_len(value: Any) -> int | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return len(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _find_wrapped_attr(dataloader: Any, attr_name: str, *, _depth: int = 0) -> Any:
    if dataloader is None or _depth > 3:
        return None
    if hasattr(dataloader, attr_name):
        return getattr(dataloader, attr_name, None)
    for child_name in ("_dl", "_dataloader", "dataloader"):
        child = getattr(dataloader, child_name, None)
        if child is not None and child is not dataloader:
            found = _find_wrapped_attr(child, attr_name, _depth=_depth + 1)
            if found is not None:
                return found
    return None


def _rebuild_descriptor(dataloader: Any) -> Mapping[str, Any]:
    public_descriptor = _mapping(_find_wrapped_attr(dataloader, "dataloader_rebuild_descriptor"))
    if public_descriptor:
        return public_descriptor
    try:
        from .dataloader_rebuild_runtime import dataloader_rebuild_descriptor
    except Exception:
        return {}
    return _mapping(dataloader_rebuild_descriptor(dataloader))


def _field_batch_dim(name: str, value: Any) -> tuple[int | None, str]:
    if _is_tensor(value):
        if value.dim() == 0:
            return None, "scalar_tensor"
        return int(value.shape[0]), "tensor"
    if name in SEQUENCE_BATCH_FIELDS:
        length = _sequence_len(value)
        return length, "sequence" if length is not None else "not_sequence"
    if name in OPTIONAL_SINGLETON_FIELDS:
        length = _sequence_len(value)
        return length, "optional_sequence" if length is not None else "optional_singleton"
    return None, "untracked"


def inspect_batch_contract(
    batch: Mapping[str, Any],
    *,
    expected_physical_batch_size: Any | None = None,
) -> dict[str, Any]:
    """Inspect leading-dimension agreement for a single training batch.

    The result is diagnostic evidence only. It never mutates tensors and never
    starts GPU work.
    """

    expected = _safe_int(expected_physical_batch_size, 0) if expected_physical_batch_size else 0
    field_reports: list[dict[str, Any]] = []
    dims: dict[int, list[str]] = {}
    for name, value in batch.items():
        dim, kind = _field_batch_dim(str(name), value)
        report = {"name": str(name), "kind": kind, "batch_dim": dim}
        if _is_tensor(value):
            report["shape"] = [int(part) for part in value.shape]
            report["dtype"] = str(value.dtype).replace("torch.", "")
        field_reports.append(report)
        if dim is not None:
            dims.setdefault(dim, []).append(str(name))

    inferred = max(dims, key=lambda dim: len(dims[dim])) if dims else 0
    mismatched = {
        str(dim): names
        for dim, names in sorted(dims.items())
        if inferred and dim != inferred
    }
    missing_core = [
        name
        for name in ("latents", "images", "encoder_hidden_states", "captions")
        if name not in batch
    ]
    has_train_payload = ("latents" in batch or "images" in batch) and (
        "encoder_hidden_states" in batch or "captions" in batch
    )
    warnings: list[str] = []
    if expected and inferred and expected != inferred:
        warnings.append("inferred_batch_size_differs_from_requested_physical_batch_size")
    if mismatched:
        warnings.append("batch_field_leading_dimensions_disagree")
    if not has_train_payload:
        warnings.append("missing_minimum_training_payload")

    return {
        "schema_version": 1,
        "contract": "lulynx_multi_batch_training_batch_contract_v0",
        "ok": bool(inferred > 0 and not mismatched and has_train_payload and (not expected or expected == inferred)),
        "expected_physical_batch_size": expected,
        "inferred_physical_batch_size": inferred,
        "real_multi_batch": inferred > 1,
        "has_cached_native_payload": "latents" in batch and "encoder_hidden_states" in batch,
        "has_live_image_payload": "images" in batch and "captions" in batch,
        "missing_core_fields": missing_core,
        "mismatched_batch_dims": mismatched,
        "warnings": warnings,
        "fields": sorted(field_reports, key=lambda item: item["name"]),
    }


def dataloader_batching_contract(
    dataloader: Any,
    *,
    requested_physical_batch_size: Any = 1,
    gradient_accumulation_steps: Any = 1,
    data_parallel_world_size: Any = 1,
) -> dict[str, Any]:
    """Describe whether a dataloader is wired for native physical batch N."""

    request = normalize_multi_batch_request(
        train_batch_size=requested_physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        data_parallel_world_size=data_parallel_world_size,
    )
    batch_sampler = _find_wrapped_attr(dataloader, "batch_sampler")
    descriptor = _rebuild_descriptor(dataloader)
    uses_batch_sampler = bool(
        getattr(batch_sampler, "__class__", type("", (), {})).__name__ == "BucketBatchSampler"
        or descriptor.get("uses_batch_sampler")
    )
    dataset = _find_wrapped_attr(dataloader, "dataset")
    bucket_manager = getattr(dataset, "bucket_manager", None)
    drop_last = bool(getattr(batch_sampler, "drop_last", descriptor.get("drop_last", False)))
    warnings: list[str] = []
    if request.physical_batch_size > 1 and bucket_manager is not None and not uses_batch_sampler:
        warnings.append("bucketed_dataset_batch_gt1_without_bucket_batch_sampler")
    if request.physical_batch_size > 1 and not drop_last:
        warnings.append("tail_batch_may_be_smaller_than_physical_batch_size")

    return {
        "schema_version": 1,
        "contract": "lulynx_multi_batch_dataloader_contract_v0",
        "ok": not warnings,
        "physical_batch_size": request.physical_batch_size,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "data_parallel_world_size": request.data_parallel_world_size,
        "effective_batch_size": request.effective_batch_size,
        "uses_bucket_batch_sampler": uses_batch_sampler,
        "has_bucket_manager": bucket_manager is not None,
        "drop_last": drop_last,
        "dataloader_route": str(descriptor.get("route") or ""),
        "descriptor": str(descriptor.get("descriptor") or ""),
        "warnings": warnings,
        "recommended_next_checks": [
            "inspect_first_real_batch_contract",
            "run_batch2_4_8_long_window_stability_matrix",
            "separate_physical_batch_from_gradient_accumulation_in_reports",
        ],
    }


def attach_dataloader_batching_contract(
    dataloader: Any,
    *,
    requested_physical_batch_size: Any = 1,
    gradient_accumulation_steps: Any = 1,
    data_parallel_world_size: Any = 1,
) -> Any:
    """Attach Lulynx multi-batch metadata to a DataLoader-like object."""

    report = dataloader_batching_contract(
        dataloader,
        requested_physical_batch_size=requested_physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        data_parallel_world_size=data_parallel_world_size,
    )
    try:
        setattr(dataloader, LULYNX_MULTI_BATCH_DATALOADER_CONTRACT_ATTR, report)
    except Exception:
        pass
    return dataloader


def dataloader_attached_batching_contract(dataloader: Any, *, _depth: int = 0) -> dict[str, Any]:
    """Return the attached Lulynx multi-batch contract, following common wrappers."""

    if dataloader is None or _depth > 3:
        return {}
    attached = _mapping(getattr(dataloader, LULYNX_MULTI_BATCH_DATALOADER_CONTRACT_ATTR, None))
    if attached:
        return dict(attached)
    for child_name in ("_dl", "_dataloader", "dataloader"):
        child = getattr(dataloader, child_name, None)
        if child is not None and child is not dataloader:
            found = dataloader_attached_batching_contract(child, _depth=_depth + 1)
            if found:
                return found
    return {}


def recommend_execution_strategy(
    *,
    batch_contract: Mapping[str, Any],
    allow_single_item_debug: bool = True,
    allow_microbatch_fallback: bool = True,
) -> dict[str, Any]:
    """Recommend the next execution strategy from contract evidence."""

    warnings = [str(item) for item in batch_contract.get("warnings", [])]
    inferred = _safe_int(batch_contract.get("inferred_physical_batch_size"), 1)
    if batch_contract.get("ok"):
        strategy = "native_batch_forward"
        reasons = ["batch_contract_passed", "real_physical_batch_detected" if inferred > 1 else "batch1_reference"]
    elif allow_microbatch_fallback and inferred > 1:
        strategy = "microbatch_forward_diagnostic"
        reasons = ["batch_contract_failed", "preserve_effective_batch_for_diagnostics", *warnings]
    elif allow_single_item_debug:
        strategy = "single_item_forward_debug"
        reasons = ["batch_contract_failed", "isolate_per_sample_failure", *warnings]
    else:
        strategy = "manual_review_required"
        reasons = ["batch_contract_failed", *warnings]
    return {
        "schema_version": 1,
        "contract": "lulynx_multi_batch_execution_strategy_v0",
        "strategy": strategy,
        "release_claim_allowed": False,
        "diagnostic_only": strategy != "native_batch_forward",
        "reasons": reasons,
    }


__all__ = [
    "LULYNX_MULTI_BATCH_DATALOADER_CONTRACT_ATTR",
    "MultiBatchRequest",
    "attach_dataloader_batching_contract",
    "dataloader_attached_batching_contract",
    "dataloader_batching_contract",
    "inspect_batch_contract",
    "normalize_multi_batch_request",
    "recommend_execution_strategy",
]
