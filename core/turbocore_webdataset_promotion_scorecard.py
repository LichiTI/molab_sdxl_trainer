"""Promotion scorecard for native WebDataset materialization."""

from __future__ import annotations

from typing import Any, Mapping


def build_webdataset_materializer_promotion_scorecard(
    *,
    native_default_enabled: bool,
    validation_mode: str,
    benchmark_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a promotion scorecard for the native WebDataset path."""

    mode = _normalize_validation_mode(validation_mode)
    benchmark = _as_dict(benchmark_report)
    header_gain = _header_gain(benchmark)
    pil_equivalent = mode in {"pil", "pillow"} and bool(benchmark.get("pil_verify_equivalent", False))
    blockers: list[str] = []
    if not native_default_enabled:
        blockers.append("native_webdataset_default_disabled")
    if mode not in {"pil", "pillow"}:
        blockers.append("native_webdataset_header_validation_weaker_than_pil_verify")
    if not pil_equivalent:
        blockers.append("pil_verify_equivalent_native_default_not_promoted")
    promotion_ready = bool(native_default_enabled and pil_equivalent and not blockers)
    return {
        "schema_version": 1,
        "scorecard": "native_webdataset_materializer_promotion_scorecard_v0",
        "ok": True,
        "promotion_ready": promotion_ready,
        "native_default_enabled": bool(native_default_enabled),
        "validation_mode": mode,
        "default_training_path_ready": promotion_ready,
        "training_path_enabled": promotion_ready,
        "header_validation_has_benchmark_gain": header_gain,
        "pil_verify_equivalent_mode_default": pil_equivalent,
        "benchmark": {
            "present": bool(benchmark),
            "header_speedup_vs_python": benchmark.get("header_speedup_vs_python"),
            "pil_verify_speedup_vs_python": benchmark.get("pil_verify_speedup_vs_python"),
            "native_tar_passes": int(benchmark.get("native_tar_passes", 1) or 1),
        },
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
    }


def _normalize_validation_mode(value: str) -> str:
    mode = str(value or "pil").strip().lower()
    if mode in {"native", "native_header"}:
        return "header"
    return mode or "pil"


def _header_gain(report: Mapping[str, Any]) -> bool:
    if not report:
        return True
    speedup = report.get("header_speedup_vs_python")
    try:
        return float(speedup) > 1.0
    except (TypeError, ValueError):
        return bool(report.get("header_validation_has_benchmark_gain", True))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_webdataset_materializer_promotion_scorecard"]
