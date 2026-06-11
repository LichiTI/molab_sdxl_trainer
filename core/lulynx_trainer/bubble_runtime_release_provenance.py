"""Evidence provenance buckets for release-safe bubble runtime claims."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping


PROVENANCE_REPORT = "bubble_runtime_release_evidence_provenance_v0"


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _evidence_ref(item: Mapping[str, Any]) -> dict[str, Any]:
    ref = {
        "case_id": str(item.get("case_id") or ""),
        "family": str(item.get("family") or ""),
        "kind": str(item.get("kind") or ""),
        "status": str(item.get("status") or ""),
    }
    speedup = _safe_float(item.get("speedup_pct"))
    if speedup:
        ref["speedup_pct"] = speedup
    reasons = _string_list(item.get("release_probe_only_reasons"))
    if reasons:
        ref["release_probe_only_reasons"] = reasons
    return ref


def release_provenance_tags(item: Mapping[str, Any], *, min_throughput_gain_pct: float) -> list[str]:
    kind = str(item.get("kind") or "")
    tags: list[str] = []
    perf_eligible = _safe_bool(item.get("release_perf_eligible"), True)
    probe_only = _safe_bool(item.get("release_probe_only"), False) or (
        not perf_eligible and bool(item.get("release_probe_only_reasons"))
    )
    if probe_only:
        tags.append("probe_only")
    elif kind in {"gpu_bubble_experiment", "bubble_ab_evidence"}:
        tags.append("benchmark_only")
    if kind == "bubble_natural_data_wait_ab_evidence":
        tags.append("natural_ab")
    if kind == "bubble_closed_loop_evidence":
        tags.append("closed_loop_safety")
    throughput_kinds = {"gpu_bubble_experiment", "bubble_ab_evidence", "bubble_natural_data_wait_ab_evidence"}
    natural_ab_ok = kind != "bubble_natural_data_wait_ab_evidence" or _safe_bool(item.get("natural_ab_release_eligible"), False)
    if (
        perf_eligible
        and natural_ab_ok
        and kind in throughput_kinds
        and _safe_float(item.get("speedup_pct")) >= max(float(min_throughput_gain_pct or 0.0), 0.0)
    ):
        tags.append("throughput_gain")
    if not tags:
        tags.append("supporting_context")
    return tags


def release_claim_role(tags: Sequence[str]) -> str:
    for role in ("probe_only", "throughput_gain", "closed_loop_safety", "natural_ab", "benchmark_only"):
        if role in tags:
            return role
    return "supporting_context"


def build_release_evidence_provenance(
    items: Sequence[Mapping[str, Any]],
    *,
    min_throughput_gain_pct: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {
        "probe_only": [],
        "benchmark_only": [],
        "natural_ab": [],
        "closed_loop_safety": [],
        "throughput_gain": [],
        "supporting_context": [],
    }
    for source in items:
        item = dict(source)
        tags = release_provenance_tags(item, min_throughput_gain_pct=min_throughput_gain_pct)
        item["release_provenance_tags"] = tags
        item["release_claim_role"] = release_claim_role(tags)
        annotated.append(item)
        ref = _evidence_ref(item)
        for tag in tags:
            buckets.setdefault(tag, []).append(ref)
    summary = {f"{key}_count": len(value) for key, value in buckets.items()}
    return annotated, {
        "schema_version": 1,
        "provenance": PROVENANCE_REPORT,
        "summary": summary,
        "buckets": buckets,
    }


__all__ = [
    "PROVENANCE_REPORT",
    "build_release_evidence_provenance",
    "release_claim_role",
    "release_provenance_tags",
]
