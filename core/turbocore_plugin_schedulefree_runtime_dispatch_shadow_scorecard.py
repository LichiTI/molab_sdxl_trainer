"""Report-only runtime dispatch shadow for selected schedule-free plugin optimizers."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_plugin_schedulefree_training_tensor_binding_canary_scorecard import (
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_schedulefree_training_tensor_binding_canary_scorecard,
)


ADAPTER_KIND = "plugin_schedulefree_runtime_dispatch_shadow_v0"
OPTIMIZER_FAMILY = "schedule_free_state_machine"
FALLBACK_BACKEND = "selected_pytorch_optimizer_plugin"


def build_plugin_schedulefree_runtime_dispatch_shadow_scorecard(
    *,
    training_tensor_binding_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Build a runtime/request dispatch shadow without native execution."""

    binding = dict(
        training_tensor_binding_report or build_plugin_schedulefree_training_tensor_binding_canary_scorecard()
    )
    mode = _normalize_mode(native_training_mode)
    route = _route_decision(binding, mode)
    envelope = _dispatch_envelope(binding, route, mode)
    validations = _validations(binding, route, envelope)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(str(reason) for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_runtime_dispatch_shadow_scorecard_v0",
        "gate": "plugin_schedulefree_runtime_dispatch_shadow",
        "ok": ready,
        "promotion_ready": False,
        "runtime_dispatch_shadow_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
        "native_training_mode": mode,
        "adapter_route": route,
        "dispatch_envelope": envelope,
        "training_tensor_binding_summary": dict(binding.get("summary") or {}),
        "validations": validations,
        "summary": {
            "runtime_dispatch_shadow_ready": ready,
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
                "selected_schedulefree_native_kernel_missing",
                "selected_schedulefree_runtime_checkpoint_adapter_missing",
                "selected_schedulefree_e2e_shadow_matrix_missing",
                "selected_schedulefree_canary_rollout_policy_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "build selected schedule-free e2e shadow matrix before any native canary review"
            if ready
            else "fix selected schedule-free runtime dispatch shadow blockers"
        ),
        "notes": [
            "This shadow is a runtime/request envelope only.",
            "The selected pytorch_optimizer plugin remains the update authority.",
            "Canary and auto modes remain blocked until a real native schedule-free kernel and review exist.",
        ],
    }


def _route_decision(binding: Mapping[str, Any], mode: str) -> dict[str, Any]:
    binding_ready = bool(binding.get("training_tensor_binding_canary_ready", False))
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif not binding_ready:
        decision = "fallback"
        reason = "training_tensor_binding_canary_missing"
    elif mode == "observe":
        decision = "would_select_schedulefree_native_but_dispatch_disabled"
        reason = "observe_mode_and_native_kernel_missing"
    elif mode in {"canary", "auto"}:
        decision = "blocked_before_native_schedulefree_kernel"
        reason = "native_kernel_and_promotion_review_missing"
    else:
        decision = "fallback"
        reason = "unknown_native_training_mode"
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": "plugin_schedulefree_native_optimizer",
        "optimizer_family": OPTIMIZER_FAMILY,
        "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_call_allowed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "missing_before_dispatch": [
            "native_schedulefree_kernel",
            "runtime_checkpoint_adapter",
            "e2e_shadow_matrix",
            "canary_rollout_policy",
            "manual_promotion_review",
        ],
    }


def _dispatch_envelope(
    binding: Mapping[str, Any],
    route: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
        "native_training_mode": mode,
        "training_update_authority": FALLBACK_BACKEND,
        "native_update_authority": "none_until_review",
        "fallback_backend": route.get("fallback_backend"),
        "runtime_request_fields": [
            "optimizer_type",
            "optimizer_args.name",
            "scheduler_policy",
            "train_eval_mode",
            "param_handles",
            "grad_handles",
            "state_handles",
            "checkpoint_adapter_proof",
            "training_tensor_binding_canary",
        ],
        "required_evidence": [
            "selected_optimizer_abi",
            "native_abi_sketch",
            "checkpoint_adapter_proof",
            "training_tensor_binding_canary",
            "native_kernel",
            "runtime_checkpoint_adapter",
            "e2e_shadow_matrix",
            "canary_rollout_policy",
        ],
        "binding_schema": binding.get("binding_schema"),
        "canary_dispatch_armed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
    }


def _validations(
    binding: Mapping[str, Any],
    route: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p15_training_tensor_binding_canary_ready",
            bool(binding.get("training_tensor_binding_canary_ready", False)),
            "selected_schedulefree_training_tensor_binding_canary_missing",
        ),
        _validation(
            "runtime_dispatch_shadow_envelope_ready",
            envelope.get("training_update_authority") == FALLBACK_BACKEND
            and bool(envelope.get("binding_schema")),
            "selected_schedulefree_runtime_dispatch_shadow_envelope_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(route.get("fallback_backend_authoritative", False))
            and bool(envelope.get("fallback_backend_authoritative", False)),
            "selected_schedulefree_runtime_dispatch_shadow_non_authoritative_fallback",
        ),
        _validation(
            "native_shadow_call_disabled",
            not bool(route.get("native_shadow_call_allowed", True))
            and not bool(envelope.get("native_shadow_call_allowed", True)),
            "selected_schedulefree_runtime_dispatch_shadow_enabled_native_call",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("training_path_enabled", True)),
            "selected_schedulefree_runtime_dispatch_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(binding.get("training_path_enabled", True))
            and not bool(binding.get("default_behavior_changed", True)),
            "selected_schedulefree_runtime_dispatch_shadow_changed_default_behavior",
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


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_plugin_schedulefree_runtime_dispatch_shadow_scorecard"]
