"""Report-only runtime canary shadow for PagedAdamW8bit live-buffer launch."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_paged_adamw8bit_native_live_buffer_launch_scorecard import (
    build_paged_adamw8bit_native_live_buffer_launch_scorecard,
)


OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"
LAUNCH_PLAN = "paged_adamw8bit_cloned_live_buffer_launch_plan_v0"


def build_paged_adamw8bit_live_canary_shadow_scorecard(
    *,
    live_launch_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Build a request-shaped canary shadow without enabling dispatch."""

    launch = dict(
        live_launch_report
        or build_paged_adamw8bit_native_live_buffer_launch_scorecard(run_live_probe=False)
    )
    mode = _normalize_mode(native_training_mode)
    route = _route_decision(launch, mode)
    shadow_ready = (
        bool(launch.get("native_live_launch_probe_ready", False))
        and route["decision"] in {"would_shadow_cloned_live_launch", "blocked_before_training_dispatch"}
        and not bool(route.get("training_path_enabled", True))
    )
    validations = _validations(launch, route, shadow_ready)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_live_canary_shadow_scorecard_v0",
        "gate": "paged_adamw8bit_live_canary_shadow_manifest",
        "ok": not failed,
        "promotion_ready": False,
        "runtime_canary_shadow_ready": shadow_ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "native_training_mode": mode,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "route_decision": route,
        "live_launch_summary": dict(launch.get("summary") or {}),
        "validations": validations,
        "manifest_summary": {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_kind": OPTIMIZER_KIND,
            "native_training_mode": mode,
            "native_live_launch_probe_ready": bool(launch.get("native_live_launch_probe_ready", False)),
            "native_live_launch_parity_ready": bool(launch.get("native_live_launch_parity_ready", False)),
            "runtime_canary_shadow_ready": shadow_ready,
            "runtime_canary_ready": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_training_tensor_binding_missing",
                "paged_adamw8bit_checkpoint_adapter_runtime_missing",
                "runtime_canary_e2e_no_regression_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add checkpoint adapter runtime proof before PagedAdamW8bit training tensor binding"
            if shadow_ready
            else "complete PagedAdamW8bit native live-buffer launch probe before runtime canary shadow"
        ),
        "notes": [
            "This manifest is request-shaped and canary-aware, but never dispatches a training optimizer update.",
            "P8I evidence is accepted only as cloned live-buffer launch evidence, not as training tensor binding.",
            "Real dispatch remains blocked until checkpoint adapter runtime, tensor binding, and e2e no-regression exist.",
        ],
    }


def _route_decision(launch: Mapping[str, Any], mode: str) -> dict[str, Any]:
    launch_ready = bool(launch.get("native_live_launch_probe_ready", False))
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif mode == "observe" and launch_ready:
        decision = "would_shadow_cloned_live_launch"
        reason = "training_tensor_binding_and_checkpoint_runtime_missing"
    elif mode in {"canary", "auto"} and launch_ready:
        decision = "blocked_before_training_dispatch"
        reason = "training_tensor_binding_and_checkpoint_runtime_missing"
    else:
        decision = "fallback"
        reason = "native_live_buffer_launch_gate_not_ready"
    return {
        "schema_version": 1,
        "feature": "paged_adamw8bit_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "request_fields": {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_kind": OPTIMIZER_KIND,
            "native_training_mode": mode,
            "launch_plan": LAUNCH_PLAN,
        },
        "evidence": {
            "native_live_launch_probe_ready": bool(launch.get("native_live_launch_probe_ready", False)),
            "native_live_launch_parity_ready": bool(launch.get("native_live_launch_parity_ready", False)),
            "cloned_live_buffers_only": True,
        },
        "missing_before_dispatch": [
            "training_tensor_binding",
            "checkpoint_adapter_runtime",
            "runtime_canary_e2e_no_regression",
        ],
    }


def _validations(
    launch: Mapping[str, Any],
    route: Mapping[str, Any],
    shadow_ready: bool,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8i_live_launch_probe_ready",
            bool(launch.get("native_live_launch_probe_ready", False)),
            "paged_adamw8bit_native_live_launch_probe_missing",
        ),
        _validation(
            "runtime_canary_shadow_manifest",
            shadow_ready,
            "paged_adamw8bit_runtime_canary_shadow_missing",
        ),
        _validation(
            "route_blocks_training_dispatch",
            route.get("decision") in {"would_shadow_cloned_live_launch", "blocked_before_training_dispatch"}
            and not bool(route.get("native_dispatch_allowed", True)),
            "paged_adamw8bit_runtime_canary_shadow_enabled_dispatch",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _normalize_mode(value: str) -> str:
    normalized = str(value or "canary").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "canary"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_paged_adamw8bit_live_canary_shadow_scorecard"]
