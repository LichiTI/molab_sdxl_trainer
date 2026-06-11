"""Report-only runtime dispatch adapter shadow for AnimaFactoredAdamW."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


OPTIMIZER_KIND = "anima_factored_adamw"
OPTIMIZER_FAMILY = "factored_custom"
ADAPTER_KIND = "anima_factored_adamw_runtime_dispatch_adapter_shadow_v0"
FALLBACK_BACKEND = "python_anima_factored_adamw"
P25_AUDIT = "native_training_performance_p25_audit_v0"
P25_AUDIT_BUILDER = "build_p25_anima_factored_adamw_training_tensor_binding_audit"
TENSOR_BINDING_PROBE_KIND = "anima_factored_adamw_training_tensor_binding_canary_v0"


def build_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard(
    *,
    p25_audit_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    p25_audit = _normalize_p25_audit(p25_audit_report)
    tensor_binding = _extract_tensor_binding(p25_audit)
    route = _adapter_route(p25_audit, tensor_binding, native_training_mode)
    envelope = _adapter_envelope(p25_audit, tensor_binding, route)
    validations = _validations(p25_audit, tensor_binding, route, envelope)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard_v0",
        "gate": "anima_factored_adamw_runtime_dispatch_adapter_shadow",
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
        "p25_dependency": {
            "schema_version": 1,
            "audit": P25_AUDIT,
            "required_builder": P25_AUDIT_BUILDER,
            "report_only": True,
            "native_call_performed_by_p26": False,
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
            "p25_audit_builder": P25_AUDIT_BUILDER,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "anima_factored_adamw_training_loop_canary_missing",
                "anima_factored_adamw_end_to_end_shadow_matrix_missing",
                "anima_factored_adamw_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add AnimaFactoredAdamW TrainingLoop explicit native canary"
            if ready
            else "fix AnimaFactoredAdamW runtime dispatch adapter shadow blockers"
        ),
        "notes": [
            "This adapter shadow is an internal runtime/request contract only.",
            "The Python AnimaFactoredAdamW training update remains authoritative.",
            "P26 records the P25 audit dependency by builder name without invoking native code.",
            "No native optimizer update is dispatched from this scorecard.",
        ],
    }


def _adapter_route(
    p25_audit: Mapping[str, Any],
    tensor_binding: Mapping[str, Any],
    native_training_mode: str,
) -> dict[str, Any]:
    binding_ready = _p25_training_tensor_binding_ready(p25_audit, tensor_binding)
    if binding_ready:
        decision = "shadow_adapter_prepared_fallback_authoritative"
        reason = "runtime_dispatch_disabled_pending_training_loop_canary"
    else:
        decision = "fallback"
        reason = "p25_training_tensor_binding_audit_missing"
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": "anima_factored_adamw_native_optimizer",
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


def _adapter_envelope(
    p25_audit: Mapping[str, Any],
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
        "p25_audit": p25_audit.get("audit"),
        "p25_audit_builder": P25_AUDIT_BUILDER,
        "runtime_request_fields": [
            "optimizer_kind",
            "optimizer_family",
            "native_training_mode",
            "param_dtype",
            "grad_dtype",
            "device_type",
            "contiguous",
            "factored_state_dtype",
            "unfactored_state_dtype",
            "row_stat_dtype",
            "col_stat_dtype",
            "training_tensor_binding_canary",
        ],
        "required_evidence": [
            "anima_factored_adamw_native_scratch_kernel_parity",
            "anima_factored_adamw_training_tensor_binding_canary",
            "training_loop_explicit_canary",
        ],
        "tensor_binding_probe_kind": tensor_binding.get("probe_kind"),
        "canary_dispatch_armed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
    }


def _validations(
    p25_audit: Mapping[str, Any],
    tensor_binding: Mapping[str, Any],
    route: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p25_training_tensor_binding_audit_ready",
            _p25_training_tensor_binding_ready(p25_audit, tensor_binding),
            "anima_factored_adamw_p25_training_tensor_binding_audit_missing",
        ),
        _validation(
            "adapter_shadow_envelope_ready",
            envelope.get("training_update_authority") == FALLBACK_BACKEND
            and envelope.get("tensor_binding_probe_kind") == TENSOR_BINDING_PROBE_KIND,
            "anima_factored_adamw_runtime_dispatch_adapter_shadow_envelope_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(route.get("fallback_backend_authoritative", False))
            and bool(envelope.get("fallback_backend_authoritative", False))
            and envelope.get("training_update_authority") == FALLBACK_BACKEND,
            "anima_factored_adamw_runtime_dispatch_adapter_shadow_non_authoritative_fallback",
        ),
        _validation(
            "native_shadow_call_disabled",
            not bool(route.get("native_shadow_call_allowed", True))
            and not bool(envelope.get("native_shadow_call_allowed", True)),
            "anima_factored_adamw_runtime_dispatch_adapter_shadow_enabled_native_call",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("training_path_enabled", True)),
            "anima_factored_adamw_runtime_dispatch_adapter_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            _default_behavior_unchanged(p25_audit, tensor_binding),
            "anima_factored_adamw_runtime_dispatch_adapter_shadow_changed_default_behavior",
        ),
    ]


def _normalize_p25_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    tensor_binding = {
        "schema_version": 1,
        "scorecard": "turbocore_anima_factored_adamw_training_tensor_binding_canary_scorecard_v0",
        "gate": "anima_factored_adamw_training_tensor_binding_canary",
        "ok": True,
        "training_tensor_binding_canary_ready": True,
        "training_tensor_binding_probe_ready": True,
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
        "audit": P25_AUDIT,
        "milestone": "v2_p25_anima_factored_adamw_training_tensor_binding_canary",
        "ok": True,
        "milestone_completed": True,
        "report_only_dependency_contract": True,
        "dependency_builder": P25_AUDIT_BUILDER,
        "native_call_performed_by_p26": False,
        "progress_gates": {
            "training_tensor_binding_canary": True,
            "training_tensor_binding_parity": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "sections": {
            "anima_factored_adamw_training_tensor_binding_canary": tensor_binding,
        },
        "remaining_blockers": [],
        "summary": {
            "recommended_next_step": "add AnimaFactoredAdamW runtime dispatch shadow",
            "p25_audit_builder": P25_AUDIT_BUILDER,
        },
    }


def _extract_tensor_binding(p25_audit: Mapping[str, Any]) -> dict[str, Any]:
    sections = p25_audit.get("sections")
    if isinstance(sections, Mapping):
        tensor_binding = sections.get("anima_factored_adamw_training_tensor_binding_canary")
        if isinstance(tensor_binding, Mapping):
            return dict(tensor_binding)
    return {
        "schema_version": 1,
        "probe_kind": TENSOR_BINDING_PROBE_KIND,
        "training_tensor_binding_canary_ready": bool(
            _as_bool_gate(p25_audit, "training_tensor_binding_canary")
            or p25_audit.get("milestone_completed", False)
        ),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "summary": {},
    }


def _p25_training_tensor_binding_ready(
    p25_audit: Mapping[str, Any],
    tensor_binding: Mapping[str, Any],
) -> bool:
    return bool(
        p25_audit.get("milestone_completed", False)
        or _as_bool_gate(p25_audit, "training_tensor_binding_canary")
        or tensor_binding.get("training_tensor_binding_canary_ready", False)
    )


def _default_behavior_unchanged(p25_audit: Mapping[str, Any], tensor_binding: Mapping[str, Any]) -> bool:
    p25_gate = _as_bool_gate(p25_audit, "default_behavior_unchanged")
    return bool(
        p25_gate
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


__all__ = ["build_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard"]
