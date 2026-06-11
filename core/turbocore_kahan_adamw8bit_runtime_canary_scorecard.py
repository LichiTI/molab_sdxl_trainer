"""Report-only runtime canary manifest for KahanAdamW8bit native work."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_kahan_adamw8bit_native_scratch_kernel_scorecard import (
    build_kahan_adamw8bit_native_scratch_kernel_scorecard,
)
from core.turbocore_kahan_adamw8bit_scratch_update_scorecard import (
    build_kahan_adamw8bit_scratch_update_scorecard,
)


OPTIMIZER_KIND = "kahan_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_kahan"
MANIFEST_KIND = "kahan_adamw8bit_runtime_canary_manifest_v0"
LAUNCH_PLAN = "kahan_adamw8bit_flat_quantized_kahan_launch_plan_v0"


def build_kahan_adamw8bit_runtime_canary_scorecard(
    *,
    scratch_update_report: Mapping[str, Any] | None = None,
    native_kernel_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Build a request-shaped canary manifest without dispatching training updates."""

    scratch = dict(scratch_update_report or build_kahan_adamw8bit_scratch_update_scorecard())
    kernel = dict(
        native_kernel_report
        or build_kahan_adamw8bit_native_scratch_kernel_scorecard(
            scratch_update_report=scratch,
        )
    )
    mode = _normalize_mode(native_training_mode)
    route = _route_decision(scratch, kernel, mode)
    validations = _validations(scratch, kernel, route, mode)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    manifest_ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_runtime_canary_scorecard_v0",
        "gate": "kahan_adamw8bit_runtime_canary_manifest",
        "ok": bool(scratch.get("ok", False)) and bool(kernel.get("ok", False)) and mode in _MODES,
        "promotion_ready": False,
        "runtime_canary_manifest_ready": manifest_ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "native_training_mode": mode,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "manifest_kind": MANIFEST_KIND,
        "launch_plan": LAUNCH_PLAN,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "canary_shadow_route_only": True,
        "route_decision": route,
        "validations": validations,
        "manifest_summary": {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_kind": OPTIMIZER_KIND,
            "native_training_mode": mode,
            "scratch_update_parity_ready": bool(scratch.get("scratch_update_parity_ready", False)),
            "native_scratch_kernel_parity_ready": bool(kernel.get("native_scratch_kernel_parity_ready", False)),
            "runtime_canary_manifest_ready": manifest_ready,
            "runtime_canary_ready": False,
            "training_path_enabled": False,
        },
        "scratch_update_summary": dict(scratch.get("summary") or {}),
        "native_kernel_summary": dict(kernel.get("summary") or {}),
        "promotion_blockers": _dedupe(
            blockers
            + [
                "kahan_adamw8bit_training_tensor_binding_missing",
                "kahan_adamw8bit_runtime_canary_e2e_missing",
                "kahan_adamw8bit_bf16_native_dtype_matrix_missing",
                "kahan_adamw8bit_checkpoint_resume_adapter_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add KahanAdamW8bit training tensor binding canary with native dispatch still disabled"
            if manifest_ready
            else "complete KahanAdamW8bit scratch update and native scratch kernel gates before runtime canary"
        ),
        "notes": [
            "This manifest is request-shaped but never dispatches a native optimizer update.",
            "P8S proves the Python scratch formula; P8T proves the native synthetic scratch kernel.",
            "Native dispatch remains blocked until tensor binding, dtype matrix, resume, and e2e canary evidence exist.",
        ],
    }


_MODES = {"off", "observe", "canary", "auto"}


def _route_decision(
    scratch: Mapping[str, Any],
    kernel: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    scratch_ready = bool(scratch.get("scratch_update_parity_ready", False))
    kernel_ready = bool(kernel.get("native_scratch_kernel_parity_ready", False))
    ready = scratch_ready and kernel_ready
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif mode == "observe" and ready:
        decision = "would_native_shadow_but_blocked"
        reason = "training_tensor_binding_and_dtype_matrix_missing"
    elif mode in {"canary", "auto"} and ready:
        decision = "blocked_before_canary"
        reason = "training_tensor_binding_and_dtype_matrix_missing"
    else:
        decision = "fallback"
        reason = "p8s_or_p8t_gate_not_ready"
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": "kahan_adamw8bit_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "canary_shadow_route_only": True,
        "request_fields": {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_kind": OPTIMIZER_KIND,
            "native_training_mode": mode,
            "launch_plan": LAUNCH_PLAN,
            "manifest_kind": MANIFEST_KIND,
        },
        "evidence": {
            "scratch_update_parity_ready": scratch_ready,
            "native_scratch_kernel_parity_ready": kernel_ready,
        },
        "missing_before_dispatch": [
            "training_tensor_binding",
            "bf16_native_dtype_matrix",
            "checkpoint_resume_adapter",
            "runtime_canary_e2e_no_regression",
        ],
    }


def _validations(
    scratch: Mapping[str, Any],
    kernel: Mapping[str, Any],
    route: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8s_scratch_update_parity_ready",
            bool(scratch.get("scratch_update_parity_ready", False)),
            "kahan_adamw8bit_scratch_update_parity_missing",
        ),
        _validation(
            "p8t_native_scratch_kernel_ready",
            bool(kernel.get("native_scratch_kernel_parity_ready", False)),
            "kahan_adamw8bit_native_scratch_kernel_parity_missing",
        ),
        _validation(
            "runtime_canary_route_blocked",
            mode != "off"
            and route.get("decision") in {"would_native_shadow_but_blocked", "blocked_before_canary"},
            "kahan_adamw8bit_runtime_canary_route_missing",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True)),
            "kahan_adamw8bit_runtime_canary_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(scratch.get("training_path_enabled", True))
            and not bool(kernel.get("training_path_enabled", True))
            and not bool(route.get("training_path_enabled", True)),
            "kahan_adamw8bit_runtime_canary_changed_default_behavior",
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
    return normalized if normalized in _MODES else "canary"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "LAUNCH_PLAN",
    "MANIFEST_KIND",
    "OPTIMIZER_FAMILY",
    "OPTIMIZER_KIND",
    "build_kahan_adamw8bit_runtime_canary_scorecard",
]
