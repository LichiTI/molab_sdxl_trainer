"""Report-only canary router for V2 native training paths."""

from __future__ import annotations

from typing import Any, Mapping


def build_native_training_router_canary(
    *,
    lora_report: Mapping[str, Any] | None = None,
    optimizer_report: Mapping[str, Any] | None = None,
    data_report: Mapping[str, Any] | None = None,
    mode: str = "auto",
) -> dict[str, Any]:
    lora = dict(lora_report or {})
    optimizer = dict(optimizer_report or {})
    data = dict(data_report or {})
    normalized_mode = _normalize_mode(mode)
    route_decisions = [
        _route_decision("lora_fused", lora, normalized_mode),
        _route_decision("native_optimizer", optimizer, normalized_mode),
        _route_decision("data_pipeline", data, normalized_mode),
    ]
    native_route_hit_count = sum(1 for item in route_decisions if item["decision"] == "native_canary")
    would_native_count = sum(1 for item in route_decisions if item["decision"] in {"native_canary", "would_native"})
    fallback_count = sum(1 for item in route_decisions if item["decision"] == "fallback")
    observe_only = normalized_mode == "observe"
    gate_ready = all(_is_ready(report) for report in (lora, optimizer, data))
    promotion_ready = bool(
        gate_ready
        and not observe_only
        and native_route_hit_count >= 1
        and normalized_mode in {"auto", "canary"}
    )
    blockers: list[str] = []
    if observe_only:
        blockers.append("runtime_native_router_observe_only_mode")
    if not gate_ready:
        blockers.append("runtime_native_router_upstream_gates_not_ready")
    if native_route_hit_count <= 0 and not observe_only:
        blockers.append("runtime_native_router_no_native_hits")
    return {
        "schema_version": 1,
        "gate": "runtime_native_router_canary",
        "scorecard": "turbocore_native_router_canary_v0",
        "ok": True,
        "mode": normalized_mode,
        "promotion_ready": promotion_ready,
        "training_path_enabled": bool(promotion_ready),
        "gate_ready": gate_ready,
        "route_decisions": route_decisions,
        "native_route_hit_count": native_route_hit_count,
        "would_native_count": would_native_count,
        "fallback_count": fallback_count,
        "manifest_summary": {
            "lora_ready": _is_ready(lora),
            "optimizer_ready": _is_ready(optimizer),
            "data_ready": _is_ready(data),
            "native_route_hit_count": native_route_hit_count,
            "would_native_count": would_native_count,
            "fallback_count": fallback_count,
        },
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe(blockers),
    }


def _route_decision(name: str, report: Mapping[str, Any], mode: str) -> dict[str, Any]:
    ready = _is_ready(report)
    if mode == "observe":
        decision = "would_native" if ready else "fallback"
    elif mode in {"canary", "auto"} and ready:
        decision = "native_canary"
    else:
        decision = "fallback"
    return {
        "feature": name,
        "promotion_ready": ready,
        "decision": decision,
        "reason": "gate_ready" if ready else "gate_not_ready",
    }


def _is_ready(report: Mapping[str, Any]) -> bool:
    if not report:
        return False
    return bool(report.get("promotion_ready", False) or report.get("ok", False))


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "auto"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_native_training_router_canary"]
