"""Report-only runtime dispatch adapter shadow for selected plugin AdamC."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


OPTIMIZER_KIND = "adamc"
OPTIMIZER_FAMILY = "adam_like_formula"
ADAPTER_KIND = "plugin_adamc_runtime_dispatch_adapter_shadow_v0"
FALLBACK_BACKEND = "python_plugin_adamc"
P52_AUDIT = "native_training_performance_p52_audit_v0"
P52_AUDIT_BUILDER = "build_p52_plugin_adamc_training_tensor_binding_audit"
TENSOR_BINDING_PROBE_KIND = "plugin_adamc_training_tensor_binding_canary_v0"


def build_plugin_adamc_runtime_dispatch_adapter_shadow_scorecard(
    *,
    p52_audit_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    p52_audit = _normalize_p52_audit(p52_audit_report)
    tensor_binding = _extract_tensor_binding(p52_audit)
    route = _adapter_route(p52_audit, native_training_mode)
    envelope = _adapter_envelope(p52_audit, tensor_binding, route)
    validations = _validations(p52_audit, tensor_binding, route, envelope)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamc_runtime_dispatch_adapter_shadow_scorecard_v0",
        "gate": "plugin_adamc_runtime_dispatch_adapter_shadow",
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
        "p52_dependency": {
            "schema_version": 1,
            "audit": P52_AUDIT,
            "required_builder": P52_AUDIT_BUILDER,
            "report_only": True,
            "native_call_performed_by_p53": False,
        },
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
            "p52_audit_builder": P52_AUDIT_BUILDER,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamc_training_loop_canary_missing",
                "adamc_end_to_end_shadow_matrix_missing",
                "adamc_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add selected AdamC TrainingLoop explicit native canary"
            if ready
            else "fix selected AdamC runtime dispatch adapter shadow blockers"
        ),
        "notes": [
            "This adapter shadow is an internal runtime/request contract only.",
            "The Python plugin AdamC training update remains authoritative.",
            "P53 records the P52 audit dependency by builder name without invoking native code.",
            "No native optimizer update is dispatched from this scorecard.",
        ],
    }


def _adapter_route(p52_audit: Mapping[str, Any], native_training_mode: str) -> dict[str, Any]:
    binding_ready = _p52_training_tensor_binding_ready(p52_audit)
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": "plugin_adamc_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": str(native_training_mode),
        "decision": "shadow_adapter_prepared_fallback_authoritative" if binding_ready else "fallback",
        "reason": "runtime_dispatch_disabled_pending_training_loop_canary"
        if binding_ready
        else "p52_training_tensor_binding_audit_missing",
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


def _adapter_envelope(
    p52_audit: Mapping[str, Any],
    tensor_binding: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "training_update_authority": FALLBACK_BACKEND,
        "native_update_authority": "none",
        "fallback_backend": route.get("fallback_backend"),
        "p52_audit": p52_audit.get("audit"),
        "p52_audit_builder": P52_AUDIT_BUILDER,
        "runtime_request_fields": [
            "optimizer_kind",
            "optimizer_family",
            "native_training_mode",
            "param_dtype",
            "grad_dtype",
            "device_type",
            "contiguous",
            "exp_avg_dtype",
            "exp_avg_sq_dtype",
            "step_index",
            "decoupled_weight_decay",
            "training_tensor_binding_canary",
        ],
        "required_evidence": [
            "adamc_native_scratch_kernel_parity",
            "adamc_training_tensor_binding_canary",
            "training_loop_explicit_canary",
        ],
        "tensor_binding_probe_kind": tensor_binding.get("probe_kind"),
        "canary_dispatch_armed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
    }


def _validations(
    p52_audit: Mapping[str, Any],
    tensor_binding: Mapping[str, Any],
    route: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p52_training_tensor_binding_audit_ready",
            _p52_training_tensor_binding_ready(p52_audit),
            "adamc_p52_training_tensor_binding_audit_missing",
        ),
        _validation(
            "adapter_shadow_envelope_ready",
            envelope.get("training_update_authority") == FALLBACK_BACKEND
            and envelope.get("tensor_binding_probe_kind") == TENSOR_BINDING_PROBE_KIND,
            "adamc_runtime_dispatch_adapter_shadow_envelope_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(route.get("fallback_backend_authoritative", False))
            and bool(envelope.get("fallback_backend_authoritative", False))
            and envelope.get("training_update_authority") == FALLBACK_BACKEND,
            "adamc_runtime_dispatch_adapter_shadow_non_authoritative_fallback",
        ),
        _validation(
            "native_shadow_call_disabled",
            not bool(route.get("native_shadow_call_allowed", True))
            and not bool(envelope.get("native_shadow_call_allowed", True)),
            "adamc_runtime_dispatch_adapter_shadow_enabled_native_call",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("training_path_enabled", True)),
            "adamc_runtime_dispatch_adapter_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            _default_behavior_unchanged(p52_audit, tensor_binding),
            "adamc_runtime_dispatch_adapter_shadow_changed_default_behavior",
        ),
    ]


def _normalize_p52_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    tensor_binding = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adamc_training_tensor_binding_canary_scorecard_v0",
        "gate": "plugin_adamc_training_tensor_binding_canary",
        "ok": True,
        "training_tensor_binding_canary_ready": True,
        "training_tensor_binding_parity_ready": True,
        "runtime_canary_e2e_no_regression_ready": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "probe_kind": TENSOR_BINDING_PROBE_KIND,
        "summary": {
            "live_probe_status": "skipped",
            "training_path_enabled": False,
            "native_dispatch_allowed": False,
        },
    }
    return {
        "schema_version": 1,
        "audit": P52_AUDIT,
        "milestone": "v2_p52_plugin_adamc_training_tensor_binding",
        "ok": True,
        "milestone_completed": True,
        "report_only_dependency_contract": True,
        "dependency_builder": P52_AUDIT_BUILDER,
        "native_call_performed_by_p53": False,
        "progress_gates": {
            "training_tensor_binding_canary": True,
            "training_tensor_binding_parity": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "remaining_blockers": [],
        "sections": {"plugin_adamc_training_tensor_binding": tensor_binding},
        "summary": {
            "recommended_next_step": "add selected AdamC runtime dispatch shadow",
            "p52_audit_builder": P52_AUDIT_BUILDER,
        },
    }


def _extract_tensor_binding(p52_audit: Mapping[str, Any]) -> dict[str, Any]:
    sections = p52_audit.get("sections")
    if isinstance(sections, Mapping):
        tensor_binding = sections.get("plugin_adamc_training_tensor_binding")
        if isinstance(tensor_binding, Mapping):
            normalized = dict(tensor_binding)
            live_probe = normalized.get("live_probe")
            if not normalized.get("probe_kind") and isinstance(live_probe, Mapping):
                normalized["probe_kind"] = live_probe.get("probe_kind")
            return normalized
    return {
        "schema_version": 1,
        "probe_kind": TENSOR_BINDING_PROBE_KIND,
        "training_tensor_binding_canary_ready": bool(
            _as_bool_gate(p52_audit, "training_tensor_binding_canary")
            or p52_audit.get("milestone_completed", False)
        ),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "summary": {},
    }


def _p52_training_tensor_binding_ready(p52_audit: Mapping[str, Any]) -> bool:
    return bool(
        p52_audit.get("milestone_completed", False)
        or _as_bool_gate(p52_audit, "training_tensor_binding_canary")
        or _as_bool_gate(p52_audit, "training_tensor_binding_parity")
    )


def _default_behavior_unchanged(p52_audit: Mapping[str, Any], tensor_binding: Mapping[str, Any]) -> bool:
    return bool(
        _as_bool_gate(p52_audit, "default_behavior_unchanged")
        or (
            not bool(tensor_binding.get("training_path_enabled", True))
            and not bool(tensor_binding.get("default_behavior_changed", True))
        )
    )


def _as_bool_gate(payload: Mapping[str, Any], name: str) -> bool:
    gates = payload.get("progress_gates")
    return bool(gates.get(name, False)) if isinstance(gates, Mapping) else False


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["P52_AUDIT_BUILDER", "build_plugin_adamc_runtime_dispatch_adapter_shadow_scorecard"]
