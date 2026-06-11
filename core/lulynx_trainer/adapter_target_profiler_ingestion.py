"""Profiler-ingestion contract for adapter target layer selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .adapter_target_policy import (
    AdapterLayerMetric,
    AdapterTargetPolicyConfig,
    AdapterTargetPolicyPlan,
    build_adapter_target_policy_plan,
)


@dataclass(frozen=True)
class AdapterTargetProfilerIngestion:
    metrics: tuple[AdapterLayerMetric, ...]
    source_kind: str
    skipped_count: int = 0

    @property
    def metric_count(self) -> int:
        return len(self.metrics)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ingestion": "adapter_target_profiler_ingestion_v0",
            "source_kind": self.source_kind,
            "metric_count": int(self.metric_count),
            "skipped_count": int(self.skipped_count),
            "metrics": [
                {
                    "name": metric.name,
                    "parameter_count": int(metric.parameter_count),
                    "gradient_norm": float(metric.gradient_norm),
                    "cka_dissimilarity": float(metric.cka_dissimilarity),
                    "sensitivity": float(metric.sensitivity),
                }
                for metric in self.metrics
            ],
        }


def ingest_adapter_target_profiler_metrics(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> AdapterTargetProfilerIngestion:
    rows, source_kind = _extract_rows(payload)
    merged: dict[str, AdapterLayerMetric] = {}
    skipped = 0
    for row in rows:
        metric = AdapterLayerMetric.from_mapping(row)
        if not metric.name:
            skipped += 1
            continue
        previous = merged.get(metric.name)
        merged[metric.name] = metric if previous is None else _merge_metric(previous, metric)
    return AdapterTargetProfilerIngestion(
        metrics=tuple(merged[name] for name in sorted(merged)),
        source_kind=source_kind,
        skipped_count=skipped,
    )


def build_adapter_target_policy_plan_from_profile(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    config: AdapterTargetPolicyConfig | Mapping[str, Any] | None = None,
) -> tuple[AdapterTargetPolicyPlan, dict[str, Any]]:
    ingestion = ingest_adapter_target_profiler_metrics(payload)
    plan = build_adapter_target_policy_plan(ingestion.metrics, config)
    profile = ingestion.as_dict()
    return plan, {
        "schema_version": 1,
        "contract": "adapter_target_policy_profile_plan_v0",
        "profile_ingestion": profile,
        "plan": plan.as_dict(),
        "profile_ready": ingestion.metric_count > 0,
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def build_adapter_target_profiler_ingestion_scorecard(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    ingestion = dict(payload.get("profile_ingestion") or {})
    plan = dict(payload.get("plan") or {})
    blockers = []
    if int(ingestion.get("metric_count") or 0) <= 0:
        blockers.append("profiler_metrics_missing")
    if int(plan.get("selected_count") or 0) <= 0:
        blockers.append("no_adapter_targets_selected")
    blockers.append("trainer_config_wiring_missing")
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_profiler_ingestion_v0",
        "ok": bool(payload.get("profile_ready")) and int(plan.get("selected_count") or 0) > 0,
        "profile_ready": bool(payload.get("profile_ready")),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "report": payload,
        "blocked_reasons": blockers,
        "recommended_next_step": "wire selected layer/rank plan into default-off trainer adapter config",
    }


def _extract_rows(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[tuple[Mapping[str, Any], ...], str]:
    if isinstance(payload, Mapping):
        for key in ("layers", "metrics", "modules", "adapter_layers"):
            rows = payload.get(key)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                return tuple(row for row in rows if isinstance(row, Mapping)), str(payload.get("source_kind") or key)
        nested = payload.get("adapter_target_profile") or payload.get("profile") or payload.get("profiler")
        if isinstance(nested, Mapping):
            return _extract_rows(nested)
        return (payload,), str(payload.get("source_kind") or "single_metric")
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return tuple(row for row in payload if isinstance(row, Mapping)), "sequence"
    raise TypeError("profiler payload must be a mapping or sequence of mappings")


def _merge_metric(left: AdapterLayerMetric, right: AdapterLayerMetric) -> AdapterLayerMetric:
    return AdapterLayerMetric(
        name=left.name,
        parameter_count=max(left.parameter_count, right.parameter_count),
        gradient_norm=max(left.gradient_norm, right.gradient_norm),
        cka_dissimilarity=max(left.cka_dissimilarity, right.cka_dissimilarity),
        sensitivity=max(left.sensitivity, right.sensitivity),
    )


__all__ = [
    "AdapterTargetProfilerIngestion",
    "build_adapter_target_policy_plan_from_profile",
    "build_adapter_target_profiler_ingestion_scorecard",
    "ingest_adapter_target_profiler_metrics",
]
