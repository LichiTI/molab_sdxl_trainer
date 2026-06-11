"""Report-only planning for the data side of the Lulynx training pipeline.

These builders describe dataset_scan, bucket_plan, and batch_collate stage
metadata. They do not scan paths, start DataLoader iteration, move tensors, or
sample randomness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


LULYNX_DATASET_SCAN_STAGE_PLAN = "lulynx_training_data_dataset_scan_stage_plan_v0"
LULYNX_BUCKET_PLAN_STAGE_PLAN = "lulynx_training_data_bucket_plan_stage_plan_v0"
LULYNX_BATCH_COLLATE_STAGE_PLAN = "lulynx_training_data_batch_collate_stage_plan_v0"
LULYNX_DATA_PIPELINE_REPORT = "lulynx_training_data_pipeline_report_v0"
LULYNX_DATA_PIPELINE_REPORT_ATTR = "_lulynx_training_data_pipeline_report"

_REPORT_ONLY_GUARDS = {
    "report_only": True,
    "does_not_scan_disk": True,
    "starts_dataloader": False,
    "moves_tensors": False,
    "changes_rng_state": False,
}


@dataclass(frozen=True)
class LulynxDatasetScanStagePlan:
    dataset_count: int
    source_count: int
    sample_count: int
    source_kinds: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.dataset_count > 0 and self.source_count > 0 and not self.warnings

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_DATASET_SCAN_STAGE_PLAN,
            "stage_id": "dataset_scan",
            "ok": self.ok,
            "dataset_count": self.dataset_count,
            "source_count": self.source_count,
            "sample_count": self.sample_count,
            "source_kinds": list(self.source_kinds),
            "warnings": list(self.warnings),
            **_REPORT_ONLY_GUARDS,
        }


@dataclass(frozen=True)
class LulynxBucketPlanStagePlan:
    uses_bucket_sampler: bool
    drop_last: bool
    requested_physical_batch_size: int
    per_bucket_batch_size: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = ()

    @property
    def tail_batch_risk(self) -> bool:
        return self.requested_physical_batch_size > 1 and not self.drop_last

    @property
    def ok(self) -> bool:
        return not self.warnings

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_BUCKET_PLAN_STAGE_PLAN,
            "stage_id": "bucket_plan",
            "ok": self.ok,
            "uses_bucket_sampler": self.uses_bucket_sampler,
            "drop_last": self.drop_last,
            "requested_physical_batch_size": self.requested_physical_batch_size,
            "per_bucket_batch_size": list(self.per_bucket_batch_size),
            "tail_batch_risk": self.tail_batch_risk,
            "warnings": list(self.warnings),
            **_REPORT_ONLY_GUARDS,
        }


@dataclass(frozen=True)
class LulynxBatchCollateStagePlan:
    expected_physical_batch_size: int
    inferred_leading_dim: int
    required_fields: tuple[str, ...]
    missing_required_fields: tuple[str, ...]
    field_leading_dims: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = ()

    @property
    def tail_batch_risk(self) -> bool:
        return (
            self.expected_physical_batch_size > 1
            and self.inferred_leading_dim > 0
            and self.inferred_leading_dim < self.expected_physical_batch_size
        )

    @property
    def ok(self) -> bool:
        return not self.missing_required_fields and not self.warnings

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_BATCH_COLLATE_STAGE_PLAN,
            "stage_id": "batch_collate",
            "ok": self.ok,
            "expected_physical_batch_size": self.expected_physical_batch_size,
            "inferred_leading_dim": self.inferred_leading_dim,
            "required_fields": list(self.required_fields),
            "missing_required_fields": list(self.missing_required_fields),
            "field_leading_dims": list(self.field_leading_dims),
            "tail_batch_risk": self.tail_batch_risk,
            "warnings": list(self.warnings),
            **_REPORT_ONLY_GUARDS,
        }


def build_lulynx_dataset_scan_stage_plan(
    *,
    dataset_descriptor: Mapping[str, Any] | None = None,
    source_descriptors: Sequence[Mapping[str, Any]] | None = None,
) -> LulynxDatasetScanStagePlan:
    """Describe dataset/source counts from supplied metadata only."""

    descriptor = _mapping(dataset_descriptor)
    sources = tuple(item for item in (source_descriptors or ()) if isinstance(item, Mapping))
    dataset_count = _first_positive_int(
        descriptor.get("dataset_count"),
        _sequence_count(descriptor.get("datasets")),
        1 if descriptor else 0,
    )
    source_count = _first_positive_int(
        descriptor.get("source_count"),
        _sequence_count(descriptor.get("sources")),
        _sequence_count(descriptor.get("source_descriptors")),
        len(sources),
    )
    sample_count = _first_positive_int(
        descriptor.get("sample_count"),
        descriptor.get("num_samples"),
        descriptor.get("total_samples"),
        0,
    )
    source_kinds = _source_kinds(descriptor=descriptor, source_descriptors=sources)
    warnings: list[str] = []
    if dataset_count <= 0:
        warnings.append("dataset_count_not_reported")
    if source_count <= 0:
        warnings.append("source_count_not_reported")
    if sample_count <= 0:
        warnings.append("sample_count_not_reported")
    return LulynxDatasetScanStagePlan(
        dataset_count=dataset_count,
        source_count=source_count,
        sample_count=sample_count,
        source_kinds=source_kinds,
        warnings=tuple(warnings),
    )


def build_lulynx_bucket_plan_stage_plan(
    *,
    bucket_descriptor: Mapping[str, Any] | None = None,
    requested_physical_batch_size: Any = 1,
) -> LulynxBucketPlanStagePlan:
    """Describe bucket sampler, drop_last, and per-bucket batch metadata."""

    descriptor = _mapping(bucket_descriptor)
    requested = _safe_int(requested_physical_batch_size, default=1)
    uses_bucket_sampler = bool(
        descriptor.get("uses_bucket_sampler")
        or descriptor.get("uses_bucket_batch_sampler")
        or descriptor.get("bucket_sampler")
    )
    drop_last = bool(descriptor.get("drop_last", False))
    per_bucket = _normalize_per_bucket_batch_sizes(
        descriptor.get("per_bucket_batch_size")
        or descriptor.get("per_bucket_batch_sizes")
        or descriptor.get("buckets")
    )
    warnings: list[str] = []
    if requested > 1 and not uses_bucket_sampler:
        warnings.append("physical_batch_gt1_without_bucket_sampler")
    if requested > 1 and not drop_last:
        warnings.append("tail_batch_may_be_smaller_than_physical_batch_size")
    if uses_bucket_sampler and not per_bucket:
        warnings.append("per_bucket_batch_size_not_reported")
    return LulynxBucketPlanStagePlan(
        uses_bucket_sampler=uses_bucket_sampler,
        drop_last=drop_last,
        requested_physical_batch_size=requested,
        per_bucket_batch_size=per_bucket,
        warnings=tuple(warnings),
    )


def build_lulynx_batch_collate_stage_plan(
    *,
    batch: Mapping[str, Any] | None,
    expected_physical_batch_size: Any = 1,
    required_fields: Sequence[str] | None = None,
) -> LulynxBatchCollateStagePlan:
    """Describe collated batch leading dim and required fields."""

    batch_mapping = _mapping(batch)
    expected = _safe_int(expected_physical_batch_size, default=1)
    required = tuple(str(item) for item in (required_fields or ("latents", "encoder_hidden_states", "captions")))
    observed_batch = isinstance(batch, Mapping)
    missing = tuple(name for name in required if name not in batch_mapping) if observed_batch else ()
    field_dims = tuple(
        {"name": name, "kind": kind, "leading_dim": dim}
        for name, value in sorted(batch_mapping.items(), key=lambda item: str(item[0]))
        for dim, kind in [_leading_dim(value)]
        if dim is not None
    )
    inferred = _majority_dim(field_dims)
    warnings: list[str] = []
    if not observed_batch:
        warnings.append("batch_collate_not_observed_without_dataloader_iteration")
    if missing:
        warnings.append("missing_required_collate_fields")
    if _has_mismatched_dims(field_dims, inferred):
        warnings.append("collate_field_leading_dimensions_disagree")
    if expected > 1 and inferred > 0 and inferred < expected:
        warnings.append("tail_batch_may_break_static_batch_shape")
    if inferred <= 0:
        warnings.append("collate_leading_dim_not_observable")
    return LulynxBatchCollateStagePlan(
        expected_physical_batch_size=expected,
        inferred_leading_dim=inferred,
        required_fields=required,
        missing_required_fields=missing,
        field_leading_dims=field_dims,
        warnings=tuple(warnings),
    )


def build_lulynx_dataloader_data_pipeline_report(
    dataloader: Any,
    *,
    requested_physical_batch_size: Any = None,
    route: str = "",
    required_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a report-only data pipeline summary from attached DataLoader metadata."""

    descriptor = _dataloader_rebuild_descriptor(dataloader)
    dataset = _find_wrapped_attr(dataloader, "dataset")
    batch_sampler = _find_wrapped_attr(dataloader, "batch_sampler")
    resolved_batch_size = _safe_int(
        requested_physical_batch_size
        if requested_physical_batch_size is not None
        else descriptor.get("batch_size") or getattr(batch_sampler, "batch_size", 1),
        default=1,
    )
    resolved_route = str(route or descriptor.get("route") or "unknown")
    dataset_plan = build_lulynx_dataset_scan_stage_plan(
        dataset_descriptor=_dataset_descriptor(dataset, route=resolved_route),
        source_descriptors=[{"kind": resolved_route}] if resolved_route else None,
    )
    bucket_plan = build_lulynx_bucket_plan_stage_plan(
        bucket_descriptor=_bucket_descriptor(
            dataloader=dataloader,
            descriptor=descriptor,
            batch_sampler=batch_sampler,
            requested_physical_batch_size=resolved_batch_size,
        ),
        requested_physical_batch_size=resolved_batch_size,
    )
    collate_plan = build_lulynx_batch_collate_stage_plan(
        batch=None,
        expected_physical_batch_size=resolved_batch_size,
        required_fields=required_fields,
    )
    missing_runtime_evidence = ["batch_collate_not_observed_without_dataloader_iteration"]
    return {
        "schema_version": 1,
        "report": LULYNX_DATA_PIPELINE_REPORT,
        "route": resolved_route,
        "ok": bool(dataset_plan.ok and bucket_plan.ok),
        "stage_ids": ["dataset_scan", "bucket_plan", "batch_collate"],
        "dataset_scan_stage_plan": dataset_plan.as_dict(),
        "bucket_plan_stage_plan": bucket_plan.as_dict(),
        "batch_collate_stage_plan": collate_plan.as_dict(),
        "missing_runtime_evidence": missing_runtime_evidence,
        **_REPORT_ONLY_GUARDS,
    }


def attach_lulynx_dataloader_data_pipeline_report(
    dataloader: Any,
    *,
    requested_physical_batch_size: Any = None,
    route: str = "",
    required_fields: Sequence[str] | None = None,
) -> Any:
    """Attach report-only data pipeline metadata to a DataLoader-like object."""

    report = build_lulynx_dataloader_data_pipeline_report(
        dataloader,
        requested_physical_batch_size=requested_physical_batch_size,
        route=route,
        required_fields=required_fields,
    )
    try:
        setattr(dataloader, LULYNX_DATA_PIPELINE_REPORT_ATTR, report)
    except Exception:
        pass
    return dataloader


def observe_lulynx_data_pipeline_batch_collate(
    report: Mapping[str, Any] | None,
    *,
    batch: Mapping[str, Any] | None,
    expected_physical_batch_size: Any = 1,
    required_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a data pipeline report updated with an already-realized batch."""

    base = dict(_mapping(report))
    collate_plan = build_lulynx_batch_collate_stage_plan(
        batch=batch,
        expected_physical_batch_size=expected_physical_batch_size,
        required_fields=required_fields,
    ).as_dict()
    missing_evidence = [
        item
        for item in _string_list(base.get("missing_runtime_evidence"))
        if item != "batch_collate_not_observed_without_dataloader_iteration"
    ]
    observed_batch = isinstance(batch, Mapping)
    if not observed_batch:
        missing_evidence.append("batch_collate_not_observed_without_dataloader_iteration")
    updated = {
        "schema_version": 1,
        "report": LULYNX_DATA_PIPELINE_REPORT,
        "route": str(base.get("route") or "runtime_batch"),
        "stage_ids": ["dataset_scan", "bucket_plan", "batch_collate"],
        **base,
        "batch_collate_stage_plan": collate_plan,
        "batch_collate_runtime_observed": observed_batch,
        "missing_runtime_evidence": _dedupe(missing_evidence),
        **_REPORT_ONLY_GUARDS,
    }
    dataset_ok = bool(_mapping(updated.get("dataset_scan_stage_plan")).get("ok", bool(base)))
    bucket_ok = bool(_mapping(updated.get("bucket_plan_stage_plan")).get("ok", bool(base)))
    updated["ok"] = bool(dataset_ok and bucket_ok and collate_plan.get("ok"))
    return updated


def merge_lulynx_data_pipeline_reports(
    dataloader_report: Mapping[str, Any] | None,
    runtime_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge DataLoader stage plans with runtime batch-collate observation."""

    base = dict(_mapping(dataloader_report))
    runtime = dict(_mapping(runtime_report))
    if not base:
        return runtime
    if not runtime:
        return base

    merged = {
        "schema_version": 1,
        "report": LULYNX_DATA_PIPELINE_REPORT,
        "route": str(runtime.get("route") or base.get("route") or ""),
        "stage_ids": ["dataset_scan", "bucket_plan", "batch_collate"],
        **base,
        **runtime,
        "dataset_scan_stage_plan": dict(
            _mapping(runtime.get("dataset_scan_stage_plan"))
            or _mapping(base.get("dataset_scan_stage_plan"))
        ),
        "bucket_plan_stage_plan": dict(
            _mapping(runtime.get("bucket_plan_stage_plan"))
            or _mapping(base.get("bucket_plan_stage_plan"))
        ),
        "batch_collate_stage_plan": dict(
            _mapping(runtime.get("batch_collate_stage_plan"))
            or _mapping(base.get("batch_collate_stage_plan"))
        ),
        "batch_collate_runtime_observed": bool(
            runtime.get("batch_collate_runtime_observed")
            or base.get("batch_collate_runtime_observed")
        ),
        **_REPORT_ONLY_GUARDS,
    }
    missing = _dedupe(
        [
            *_string_list(base.get("missing_runtime_evidence")),
            *_string_list(runtime.get("missing_runtime_evidence")),
        ]
    )
    if merged["batch_collate_runtime_observed"]:
        missing = [
            item
            for item in missing
            if item != "batch_collate_not_observed_without_dataloader_iteration"
        ]
    merged["missing_runtime_evidence"] = missing
    dataset_ok = bool(_mapping(merged.get("dataset_scan_stage_plan")).get("ok", False))
    bucket_ok = bool(_mapping(merged.get("bucket_plan_stage_plan")).get("ok", False))
    collate_ok = bool(_mapping(merged.get("batch_collate_stage_plan")).get("ok", False))
    merged["ok"] = bool(dataset_ok and bucket_ok and collate_ok)
    return merged


def dataloader_attached_data_pipeline_report(dataloader: Any, *, _depth: int = 0) -> dict[str, Any]:
    """Return attached Lulynx data pipeline metadata, following common wrappers."""

    if dataloader is None or _depth > 3:
        return {}
    attached = _mapping(getattr(dataloader, LULYNX_DATA_PIPELINE_REPORT_ATTR, None))
    if attached:
        return dict(attached)
    for child_name in ("_dl", "_dataloader", "dataloader"):
        child = getattr(dataloader, child_name, None)
        if child is not None and child is not dataloader:
            found = dataloader_attached_data_pipeline_report(child, _depth=_depth + 1)
            if found:
                return found
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _dataloader_rebuild_descriptor(dataloader: Any) -> Mapping[str, Any]:
    try:
        from .dataloader_rebuild_runtime import dataloader_rebuild_descriptor

        return _mapping(dataloader_rebuild_descriptor(dataloader))
    except Exception:
        return {}


def _find_wrapped_attr(source: Any, name: str, *, _depth: int = 0) -> Any:
    if source is None or _depth > 3:
        return None
    if hasattr(source, name):
        return getattr(source, name)
    for child_name in ("_dl", "_dataloader", "dataloader"):
        child = getattr(source, child_name, None)
        if child is not None and child is not source:
            found = _find_wrapped_attr(child, name, _depth=_depth + 1)
            if found is not None:
                return found
    return None


def _dataset_descriptor(dataset: Any, *, route: str) -> dict[str, Any]:
    if dataset is None:
        return {"source_kinds": [route] if route else []}
    samples = getattr(dataset, "samples", None)
    sample_count = _sequence_count(samples)
    if sample_count <= 0:
        sample_count = _safe_int(
            getattr(dataset, "sample_count", None)
            or getattr(dataset, "num_samples", None)
            or getattr(dataset, "total_samples", None),
            default=0,
        )
    return {
        "dataset_count": 1,
        "source_count": 1,
        "sample_count": sample_count,
        "source_kinds": [route] if route else [],
    }


def _bucket_descriptor(
    *,
    dataloader: Any,
    descriptor: Mapping[str, Any],
    batch_sampler: Any,
    requested_physical_batch_size: int,
) -> dict[str, Any]:
    uses_bucket_sampler = bool(
        descriptor.get("uses_batch_sampler")
        or getattr(batch_sampler, "__class__", type("", (), {})).__name__ == "BucketBatchSampler"
    )
    batch_size = _safe_int(getattr(batch_sampler, "batch_size", None), default=requested_physical_batch_size)
    per_bucket = []
    if uses_bucket_sampler and batch_size > 0:
        per_bucket.append({"bucket": "uniform_bucket_policy", "batch_size": batch_size})
    return {
        "uses_bucket_sampler": uses_bucket_sampler,
        "drop_last": bool(getattr(batch_sampler, "drop_last", descriptor.get("drop_last", False))),
        "per_bucket_batch_size": per_bucket,
    }


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(int(value if value is not None else default), 0)
    except (TypeError, ValueError, OverflowError):
        return max(int(default), 0)


def _first_positive_int(*values: Any) -> int:
    for value in values:
        number = _safe_int(value)
        if number > 0:
            return number
    return 0


def _sequence_count(value: Any) -> int:
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return 0
    return len(value)


def _source_kinds(
    *,
    descriptor: Mapping[str, Any],
    source_descriptors: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    kinds: list[str] = []
    for item in source_descriptors:
        kind = str(item.get("kind") or item.get("type") or "").strip()
        if kind:
            kinds.append(kind)
    for key in ("source_kinds", "source_types"):
        raw = descriptor.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            kinds.extend(str(item).strip() for item in raw if str(item).strip())
    return tuple(sorted(set(kinds)))


def _normalize_per_bucket_batch_sizes(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Mapping):
        return tuple(
            {"bucket": str(bucket), "batch_size": _safe_int(batch_size)}
            for bucket, batch_size in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        reports: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                bucket = item.get("bucket") or item.get("bucket_id") or index
                batch_size = item.get("batch_size") or item.get("per_bucket_batch_size")
                reports.append({"bucket": str(bucket), "batch_size": _safe_int(batch_size)})
        return tuple(reports)
    return ()


def _leading_dim(value: Any) -> tuple[int | None, str]:
    shape = getattr(value, "shape", None)
    if shape:
        try:
            return max(int(shape[0]), 0), "tensor_like"
        except (TypeError, ValueError, IndexError):
            return None, "tensor_like_unknown"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value), "sequence"
    return None, "untracked"


def _majority_dim(field_dims: Sequence[Mapping[str, Any]]) -> int:
    counts: dict[int, int] = {}
    for report in field_dims:
        dim = report.get("leading_dim")
        if isinstance(dim, int) and dim > 0:
            counts[dim] = counts.get(dim, 0) + 1
    if not counts:
        return 0
    return max(counts, key=lambda dim: (counts[dim], dim))


def _has_mismatched_dims(field_dims: Sequence[Mapping[str, Any]], inferred: int) -> bool:
    return inferred > 0 and any(report.get("leading_dim") not in (None, inferred) for report in field_dims)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_BATCH_COLLATE_STAGE_PLAN",
    "LULYNX_BUCKET_PLAN_STAGE_PLAN",
    "LULYNX_DATA_PIPELINE_REPORT",
    "LULYNX_DATA_PIPELINE_REPORT_ATTR",
    "LULYNX_DATASET_SCAN_STAGE_PLAN",
    "LulynxBatchCollateStagePlan",
    "LulynxBucketPlanStagePlan",
    "LulynxDatasetScanStagePlan",
    "attach_lulynx_dataloader_data_pipeline_report",
    "build_lulynx_batch_collate_stage_plan",
    "build_lulynx_bucket_plan_stage_plan",
    "build_lulynx_dataloader_data_pipeline_report",
    "build_lulynx_dataset_scan_stage_plan",
    "dataloader_attached_data_pipeline_report",
    "merge_lulynx_data_pipeline_reports",
    "observe_lulynx_data_pipeline_batch_collate",
]
