"""Conservative policy for the experimental LXCS replacement loader.

The replacement loader is still a P3 probe.  This policy only decides whether
diagnostic benchmarks should try the LXCS path or keep the raw DataLoader path;
it is not a production trainer switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LosslessCacheReplacementPolicyConfig:
    mode: str = "adaptive"  # off | always | adaptive
    min_raw_bytes: int = 64 * 1024
    light_compute_min_raw_bytes: int = 512 * 1024
    light_compute_repeat: int = 4
    min_saved_bytes: int = 64 * 1024
    max_compression_ratio: float = 0.85
    cuda_min_raw_bytes: int = 1 * 1024 * 1024
    cuda_light_compute_min_raw_bytes: int = 2 * 1024 * 1024
    cuda_min_saved_bytes: int = 512 * 1024


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reports_from_prepare_report(prepare_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    reports = prepare_report.get("reports", [])
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes, bytearray)):
        return []
    return [item for item in reports if isinstance(item, Mapping)]


def _summarize_prepare_report(prepare_report: Mapping[str, Any]) -> dict[str, Any]:
    reports = _reports_from_prepare_report(prepare_report)
    ok_reports = [item for item in reports if bool(item.get("ok"))]
    raw_bytes = sum(_as_int(item.get("raw_bytes")) for item in ok_reports)
    sidecar_bytes = sum(_as_int(item.get("sidecar_bytes")) for item in ok_reports)
    saved_bytes = raw_bytes - sidecar_bytes
    compression_ratio = sidecar_bytes / max(float(raw_bytes), 1.0) if raw_bytes > 0 else 1.0
    return {
        "prepare_ok": bool(prepare_report.get("ok")) if prepare_report else False,
        "prepare_skipped": bool(prepare_report.get("skipped")),
        "case_count": len(reports),
        "ok_count": len(ok_reports),
        "raw_bytes": raw_bytes,
        "sidecar_bytes": sidecar_bytes,
        "saved_bytes": saved_bytes,
        "compression_ratio": round(float(compression_ratio), 6),
    }


def evaluate_lossless_cache_replacement_policy(
    prepare_report: Mapping[str, Any] | None,
    *,
    workload: Mapping[str, Any] | None = None,
    config: LosslessCacheReplacementPolicyConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LosslessCacheReplacementPolicyConfig()
    mode = str(cfg.mode or "adaptive").lower()
    workload_info = dict(workload or {})
    summary = _summarize_prepare_report(dict(prepare_report or {}))
    raw_bytes = _as_int(summary.get("raw_bytes"))
    saved_bytes = _as_int(summary.get("saved_bytes"))
    compression_ratio = _as_float(summary.get("compression_ratio"), 1.0)
    compute_repeat = max(_as_int(workload_info.get("compute_repeat"), 1), 1)
    device = str(workload_info.get("device") or "").lower()
    cuda_workload = device.startswith("cuda")

    reasons: list[str] = []
    if mode == "off":
        reasons.append("policy_mode_off")
    elif mode == "always":
        reasons.append("policy_mode_always")
    elif not summary["prepare_ok"]:
        reasons.append("sidecar_prepare_not_ok")
    elif summary["ok_count"] <= 0:
        reasons.append("no_ok_sidecar_reports")
    elif saved_bytes <= 0:
        reasons.append("no_real_byte_savings")
    elif raw_bytes < max(int(cfg.min_raw_bytes), 0):
        reasons.append("raw_bytes_below_floor")
    elif saved_bytes < max(int(cfg.min_saved_bytes), 0):
        reasons.append("saved_bytes_below_floor")
    elif compression_ratio > float(cfg.max_compression_ratio):
        reasons.append("compression_ratio_too_high")
    elif cuda_workload and raw_bytes < max(int(cfg.cuda_min_raw_bytes), 0):
        reasons.append("cuda_raw_bytes_below_floor")
    elif cuda_workload and saved_bytes < max(int(cfg.cuda_min_saved_bytes), 0):
        reasons.append("cuda_saved_bytes_below_floor")
    elif (
        cuda_workload
        and compute_repeat <= max(int(cfg.light_compute_repeat), 1)
        and raw_bytes < max(int(cfg.cuda_light_compute_min_raw_bytes), 0)
    ):
        reasons.append("cuda_light_compute_raw_bytes_below_floor")
    elif (
        compute_repeat <= max(int(cfg.light_compute_repeat), 1)
        and raw_bytes < max(int(cfg.light_compute_min_raw_bytes), 0)
    ):
        reasons.append("light_compute_raw_bytes_below_floor")
    else:
        reasons.append("adaptive_policy_passed")

    enabled = mode == "always" or (mode == "adaptive" and reasons == ["adaptive_policy_passed"])
    selected_path = "lxcs_replacement" if enabled else "raw_dataloader"
    return {
        "provider": "lxcs_replacement_policy_v1",
        "mode": mode,
        "enabled": bool(enabled),
        "recommended": bool(enabled),
        "selected_path": selected_path,
        "reason": reasons[0] if reasons else "",
        "reasons": reasons,
        "workload": workload_info,
        "prepare_summary": summary,
        "thresholds": {
            "min_raw_bytes": max(int(cfg.min_raw_bytes), 0),
            "light_compute_min_raw_bytes": max(int(cfg.light_compute_min_raw_bytes), 0),
            "light_compute_repeat": max(int(cfg.light_compute_repeat), 1),
            "min_saved_bytes": max(int(cfg.min_saved_bytes), 0),
            "max_compression_ratio": float(cfg.max_compression_ratio),
            "cuda_min_raw_bytes": max(int(cfg.cuda_min_raw_bytes), 0),
            "cuda_light_compute_min_raw_bytes": max(int(cfg.cuda_light_compute_min_raw_bytes), 0),
            "cuda_min_saved_bytes": max(int(cfg.cuda_min_saved_bytes), 0),
        },
        "training_path_enabled": False,
    }


__all__ = [
    "LosslessCacheReplacementPolicyConfig",
    "evaluate_lossless_cache_replacement_policy",
]
