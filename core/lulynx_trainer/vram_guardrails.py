from __future__ import annotations

import sys
from typing import Any, Dict, List, Mapping


_AUTO_RESIDENCY_VALUES = {"", "auto", "resident", "off", "none", "gpu"}


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def classify_vram_hardware_profile(vram: Mapping[str, Any], *, platform: str | None = None) -> Dict[str, Any]:
    available_gb = _as_float(vram.get("available_gb"), 0.0)
    estimated_gb = _as_float(vram.get("estimated_gb"), 0.0)
    usage_ratio = _as_float(vram.get("usage_ratio"), 0.0)
    runtime_platform = str(platform or sys.platform or "").lower()

    if available_gb <= 0.0:
        tier = "unknown"
    elif available_gb <= 6.5:
        tier = "very_low"
    elif available_gb <= 8.5:
        tier = "low"
    elif available_gb <= 12.5:
        tier = "constrained"
    elif available_gb <= 16.5:
        tier = "standard"
    else:
        tier = "ample"

    low_vram_target = tier in {"very_low", "low"}
    constrained_vram_target = tier in {"very_low", "low", "constrained"}
    if usage_ratio >= 1.0:
        pressure = "oom_risk"
    elif usage_ratio >= 0.9:
        pressure = "critical"
    elif usage_ratio >= 0.78:
        pressure = "watch"
    elif available_gb > 0.0:
        pressure = "comfortable"
    else:
        pressure = "unknown"

    if runtime_platform.startswith("win"):
        shared_vram_detection = "runtime_smart_sensing_only"
        shared_vram_note = (
            "Windows/WDDM shared or pageable GPU memory is not exposed here as a reliable preflight counter; "
            "confirm shared-memory pressure from runtime smart sensing."
        )
    else:
        shared_vram_detection = "runtime_smart_sensing_only"
        shared_vram_note = (
            "Preflight estimates use device-visible VRAM only; shared-memory pressure still needs runtime observation."
        )

    return {
        "available_gb": round(available_gb, 2) if available_gb > 0.0 else None,
        "estimated_gb": round(estimated_gb, 2) if estimated_gb > 0.0 else None,
        "usage_ratio": round(usage_ratio, 3) if usage_ratio > 0.0 else None,
        "device_tier": tier,
        "low_vram_target": low_vram_target,
        "constrained_vram_target": constrained_vram_target,
        "pressure_level": pressure,
        "shared_vram_detection": shared_vram_detection,
        "shared_vram_note": shared_vram_note,
        "platform": runtime_platform,
    }


def apply_low_vram_guardrails(
    config: Any,
    *,
    vram_report: Mapping[str, Any],
    smart_enabled: bool,
    auto_enabled: bool,
) -> Dict[str, Any]:
    hardware = classify_vram_hardware_profile(vram_report)
    family = str(vram_report.get("family") or _get(config, "model_arch", "") or "").strip().lower()
    dit_runtime = vram_report.get("dit_runtime", {})
    action_plan = vram_report.get("action_plan", {})
    safety = str(vram_report.get("safety") or "unknown")
    recommendations = list(vram_report.get("recommendations") or [])
    result: Dict[str, Any] = {
        "enabled": bool(smart_enabled and auto_enabled),
        "family": family,
        "hardware": hardware,
        "safety": safety,
        "estimated_gb": vram_report.get("estimated_gb"),
        "available_gb": vram_report.get("available_gb"),
        "usage_ratio": vram_report.get("usage_ratio"),
        "triggered": False,
        "changes": {},
        "skipped": [],
        "notes": [],
        "recommendations": recommendations,
        "action_plan_steps": list(action_plan.get("steps") or []),
    }
    if not smart_enabled or not auto_enabled:
        result["notes"].append("VRAM smart sensing or auto enhancement is disabled.")
        return result
    if family not in {"anima", "newbie"} or not isinstance(dit_runtime, Mapping) or not dit_runtime.get("available"):
        result["notes"].append("Native DiT low-VRAM guardrails are only active for Anima/Newbie.")
        return result

    low_vram_target = bool(hardware.get("low_vram_target"))
    constrained_target = bool(hardware.get("constrained_vram_target"))
    trigger = (
        safety in {"danger", "tight"}
        or bool(dit_runtime.get("risk"))
        or bool(dit_runtime.get("checkpoint_missing"))
        or low_vram_target
    )
    result["triggered"] = bool(trigger)
    if not trigger:
        result["notes"].append("Preflight did not detect a low-VRAM guardrail trigger.")
        return result

    residency_key = str(dit_runtime.get("residency_key") or "")
    checkpoint_key = str(dit_runtime.get("checkpoint_key") or "")
    prefetch_key = str(dit_runtime.get("prefetch_key") or "")
    prefetch_depth_key = str(dit_runtime.get("prefetch_depth_key") or "")
    current_mode = str(dit_runtime.get("mode") or "resident")
    recommendation = str(dit_runtime.get("recommendation") or current_mode)
    streaming_allowed = bool(_get(config, "vram_smart_sensing_streaming_enabled", True))
    sparse_allowed = bool(_get(config, "vram_smart_sensing_sparse_swap_enabled", True))
    delta_cache_allowed = bool(_get(config, "vram_smart_sensing_delta_cache_enabled", False))
    enhanced_protection_mode = bool(_get(config, "enhanced_protection_mode", False))

    raw_residency = str(_get(config, residency_key, current_mode) or current_mode).strip().lower().replace("-", "_")
    residency_autotunable = raw_residency in _AUTO_RESIDENCY_VALUES
    residency_changed = False

    if recommendation != current_mode:
        if recommendation in {"streaming_offload", "block_cpu_pinned"} and not streaming_allowed:
            result["skipped"].append(
                {
                    "key": residency_key,
                    "reason": "streaming_guard_disabled",
                    "requested": recommendation,
                }
            )
        elif residency_autotunable:
            setattr(config, residency_key, recommendation)
            result["changes"][residency_key] = recommendation
            current_mode = recommendation
            residency_changed = True
        else:
            result["skipped"].append(
                {
                    "key": residency_key,
                    "reason": "explicit_residency_preserved",
                    "current": raw_residency,
                    "requested": recommendation,
                }
            )

    if checkpoint_key and bool(dit_runtime.get("checkpoint_missing")):
        if residency_changed and not _as_bool(_get(config, checkpoint_key, False)):
            setattr(config, checkpoint_key, True)
            result["changes"][checkpoint_key] = True
        elif not _as_bool(_get(config, checkpoint_key, False)):
            result["skipped"].append(
                {
                    "key": checkpoint_key,
                    "reason": "checkpoint_left_for_user_choice",
                    "requested": True,
                }
            )

    if residency_changed and current_mode == "streaming_offload":
        # Low-end cards prefer sparse on-demand buckets over aggressive warm prefetch.
        if sparse_allowed and not _as_bool(_get(config, "sparse_swap_enabled", False)):
            setattr(config, "sparse_swap_enabled", True)
            result["changes"]["sparse_swap_enabled"] = True
        elif not sparse_allowed:
            result["skipped"].append({"key": "sparse_swap_enabled", "reason": "sparse_swap_guard_disabled"})
        if sparse_allowed and _as_float(_get(config, "sparse_swap_warm_fraction", 0.0), 0.0) <= 0.0:
            warm_fraction = 0.25 if low_vram_target else 0.35
            setattr(config, "sparse_swap_warm_fraction", warm_fraction)
            result["changes"]["sparse_swap_warm_fraction"] = warm_fraction

        if prefetch_key:
            if low_vram_target:
                result["skipped"].append(
                    {
                        "key": prefetch_key,
                        "reason": "prefetch_left_off_for_low_vram",
                        "requested": False,
                    }
                )
            elif not _as_bool(_get(config, prefetch_key, False)):
                setattr(config, prefetch_key, True)
                result["changes"][prefetch_key] = True
                if prefetch_depth_key and _as_int(_get(config, prefetch_depth_key, 0), 0) <= 0:
                    setattr(config, prefetch_depth_key, 1)
                    result["changes"][prefetch_depth_key] = 1

        if delta_cache_allowed and not _as_bool(_get(config, "pcie_delta_cache_enabled", False)):
            result["skipped"].append(
                {
                    "key": "pcie_delta_cache_enabled",
                    "reason": "observe_only_candidate",
                    "requested": True,
                }
            )

        pcie_before = str(_get(config, "pcie_transfer_format", "off") or "off").strip().lower()
        if enhanced_protection_mode and pcie_before in {"", "off", "none", "disabled"}:
            setattr(config, "pcie_transfer_format", "fp8_e4m3")
            result["changes"]["pcie_transfer_format"] = "fp8_e4m3"

    compile_runtime = str(_get(config, "compile_runtime", "off") or "off").strip().lower().replace("-", "_")
    if constrained_target and compile_runtime in {"auto", "compile", "compile_cache", "compile_cudagraph"}:
        result["notes"].append(
            "Low-VRAM target detected: keep compile runtime conservative unless long-run cache reuse clearly pays back the first-step peak."
        )

    if low_vram_target:
        result["notes"].append("Low-end VRAM tier detected; sparse on-demand residency is preferred over warm prefetch.")
    elif constrained_target:
        result["notes"].append("Constrained VRAM tier detected; enable nonresident DiT paths before reducing batch.")
    if result["skipped"]:
        result["notes"].append("Some guardrail suggestions were left as report-only to preserve user-selected runtime knobs.")
    return result
