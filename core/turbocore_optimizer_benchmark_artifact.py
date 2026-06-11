"""Adapter for optimizer benchmark artifacts used by TurboCore reports.

The adapter is intentionally report-only. It normalizes an already-generated
optimizer benchmark JSON into a small evidence object that can be consumed by
native update performance gates without running benchmarks from the matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_optimizer_performance_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return normalize_optimizer_performance_artifact(payload, source_path=str(artifact_path))


def normalize_optimizer_performance_artifact(
    payload: Mapping[str, Any],
    *,
    source_path: str = "",
) -> dict[str, Any]:
    root = _as_dict(payload)
    gate = _find_optimizer_gate(root)
    summary = _as_dict(root.get("summary"))
    artifact = {
        "schema_version": 1,
        "artifact": "turbocore_optimizer_benchmark_artifact_v0",
        "source_path": str(source_path or ""),
        "benchmark": str(root.get("benchmark", "") or gate.get("gate", "") or ""),
        "ok": bool(root.get("ok", False) or gate.get("ok", False)),
        "iters": _int_or_none(root.get("iters") or gate.get("iters")),
        "warmup": _int_or_none(root.get("warmup") or gate.get("warmup")),
        "summary": summary,
        "optimizer_performance_gate": gate,
        "gate_present": bool(gate),
        "gate_ok": bool(gate.get("ok", False)) if gate else False,
        "promotion_gate_ok": bool(gate.get("promotion_gate_ok", False)) if gate else False,
        "evidence_quality": str(gate.get("evidence_quality", "") or "") if gate else "",
        "best_speedup_vs_baseline": _best_speedup(gate),
    }
    if not gate:
        artifact["blocked_reasons"] = ["optimizer_performance_gate_missing"]
    return artifact


def optimizer_artifact_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    gate = _as_dict(artifact.get("optimizer_performance_gate"))
    return {
        "artifact": str(artifact.get("artifact", "") or ""),
        "source_path": str(artifact.get("source_path", "") or ""),
        "benchmark": str(artifact.get("benchmark", "") or ""),
        "gate_present": bool(artifact.get("gate_present", False)),
        "gate_ok": bool(artifact.get("gate_ok", False)),
        "promotion_gate_ok": bool(artifact.get("promotion_gate_ok", False)),
        "evidence_quality": str(artifact.get("evidence_quality", "") or ""),
        "best_speedup_vs_baseline": _best_speedup(gate),
        "status": str(gate.get("status", "") or ""),
    }


def _find_optimizer_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("native_update_optimizer_performance_gate", "optimizer_performance_gate", "performance_gate"):
        value = _as_dict(report.get(key))
        if value:
            return value
    if report.get("gate") == "turbocore_optimizer_performance_gate":
        return dict(report)
    return {}


def _best_speedup(gate: Mapping[str, Any]) -> float | None:
    best = _as_dict(gate.get("best_candidate") or gate.get("best_measured_candidate"))
    return _float_or_none(best.get("speedup_vs_baseline"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "load_optimizer_performance_artifact",
    "normalize_optimizer_performance_artifact",
    "optimizer_artifact_summary",
]
