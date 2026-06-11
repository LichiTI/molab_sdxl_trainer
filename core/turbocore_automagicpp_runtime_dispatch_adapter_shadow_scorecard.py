"""Report-only runtime dispatch adapter shadow for Automagic++."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_automagicpp_training_tensor_binding_canary_scorecard import (
    build_automagicpp_training_tensor_binding_canary_scorecard,
)


OPTIMIZER_KIND = "automagicpp"
OPTIMIZER_FAMILY = "factored_custom"
ADAPTER_KIND = "automagicpp_runtime_dispatch_adapter_shadow_v0"
FALLBACK_BACKEND = "python_automagicpp"


def build_automagicpp_runtime_dispatch_adapter_shadow_scorecard(
    *,
    tensor_binding_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
    run_live_probe: bool = True,
) -> dict[str, Any]:
    tensor_binding = dict(
        tensor_binding_report
        or build_automagicpp_training_tensor_binding_canary_scorecard(run_live_probe=run_live_probe)
    )
    route = _adapter_route(tensor_binding, native_training_mode)
    envelope = _adapter_envelope(tensor_binding, route)
    validations = _validations(tensor_binding, route, envelope)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_automagicpp_runtime_dispatch_adapter_shadow_scorecard_v0",
        "gate": "automagicpp_runtime_dispatch_adapter_shadow",
        "ok": ready,
        "promotion_ready": False,
        "runtime_dispatch_adapter_shadow_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": str(native_training_mode),
        "adapter_route": route,
        "adapter_envelope": envelope,
        "tensor_binding_summary": dict(tensor_binding.get("summary") or {}),
        "validations": validations,
        "summary": {
            "runtime_dispatch_adapter_shadow_ready": ready,
            "adapter_decision": route.get("decision"),
            "fallback_backend_authoritative": True,
            "native_shadow_call_allowed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "automagicpp_training_loop_canary_missing",
                "automagicpp_end_to_end_shadow_matrix_missing",
                "automagicpp_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Automagic++ TrainingLoop explicit native canary"
            if ready
            else "fix Automagic++ runtime dispatch adapter shadow blockers"
        ),
        "notes": [
            "This adapter shadow is an internal runtime/request contract only.",
            "The Python Automagic++ training update remains authoritative.",
            "No native optimizer update is dispatched from this scorecard.",
        ],
    }


def _adapter_route(tensor_binding: Mapping[str, Any], native_training_mode: str) -> dict[str, Any]:
    binding_ready = bool(tensor_binding.get("training_tensor_binding_canary_ready", False))
    if binding_ready:
        decision = "shadow_adapter_prepared_fallback_authoritative"
        reason = "runtime_dispatch_disabled_pending_training_loop_canary"
    else:
        decision = "fallback"
        reason = "training_tensor_binding_canary_missing"
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": "automagicpp_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": str(native_training_mode),
        "decision": decision,
        "reason": reason,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_call_allowed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "missing_before_dispatch": [
            "training_loop_explicit_canary",
            "end_to_end_shadow_matrix",
            "canary_rollout_policy",
            "manual_promotion_review",
        ],
    }


def _adapter_envelope(tensor_binding: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "training_update_authority": FALLBACK_BACKEND,
        "native_update_authority": "none",
        "fallback_backend": route.get("fallback_backend"),
        "runtime_request_fields": [
            "optimizer_kind",
            "optimizer_family",
            "native_training_mode",
            "param_dtype",
            "grad_dtype",
            "device_type",
            "contiguous",
            "local_lr_dtype",
            "prev_sign_dtype",
            "full_var_dtype",
            "training_tensor_binding_canary",
        ],
        "required_evidence": [
            "native_scratch_kernel_parity",
            "training_tensor_binding_canary",
            "training_loop_explicit_canary",
        ],
        "tensor_binding_probe_kind": tensor_binding.get("probe_kind"),
        "canary_dispatch_armed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
    }


def _validations(
    tensor_binding: Mapping[str, Any],
    route: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p21_training_tensor_binding_canary_ready",
            bool(tensor_binding.get("training_tensor_binding_canary_ready", False)),
            "automagicpp_training_tensor_binding_canary_missing",
        ),
        _validation(
            "adapter_shadow_envelope_ready",
            envelope.get("training_update_authority") == FALLBACK_BACKEND
            and envelope.get("tensor_binding_probe_kind") == "automagicpp_training_tensor_binding_canary_v0",
            "automagicpp_runtime_dispatch_adapter_shadow_envelope_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(route.get("fallback_backend_authoritative", False))
            and bool(envelope.get("fallback_backend_authoritative", False)),
            "automagicpp_runtime_dispatch_adapter_shadow_non_authoritative_fallback",
        ),
        _validation(
            "native_shadow_call_disabled",
            not bool(route.get("native_shadow_call_allowed", True))
            and not bool(envelope.get("native_shadow_call_allowed", True)),
            "automagicpp_runtime_dispatch_adapter_shadow_enabled_native_call",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("training_path_enabled", True)),
            "automagicpp_runtime_dispatch_adapter_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(tensor_binding.get("training_path_enabled", True))
            and not bool(tensor_binding.get("default_behavior_changed", True)),
            "automagicpp_runtime_dispatch_adapter_shadow_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_automagicpp_runtime_dispatch_adapter_shadow_scorecard"]
