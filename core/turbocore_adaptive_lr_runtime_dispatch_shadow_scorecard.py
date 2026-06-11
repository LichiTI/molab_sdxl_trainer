"""Report-only runtime dispatch shadow for built-in adaptive-LR optimizers."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES
from core.turbocore_adaptive_lr_training_tensor_binding_canary_scorecard import (
    PROBE_KIND,
    build_adaptive_lr_training_tensor_binding_canary_scorecard,
)


ADAPTER_KIND = "adaptive_lr_runtime_dispatch_shadow_v0"
OPTIMIZER_FAMILY = "built_in_adaptive_lr"
FALLBACK_BACKEND = "python_adaptive_lr_optimizer"


def build_adaptive_lr_runtime_dispatch_shadow_scorecard(
    *,
    training_tensor_binding_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Build an internal runtime/request shadow without native dispatch."""

    binding = _as_dict(training_tensor_binding_report or build_adaptive_lr_training_tensor_binding_canary_scorecard())
    mode = _normalize_mode(native_training_mode)
    rows = [_row(case.optimizer.value, binding) for case in TARGET_CASES]
    route = _route_decision(binding, mode)
    envelope = _dispatch_envelope(binding, route, rows, mode)
    validations = _validations(binding, rows, route, envelope)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_runtime_dispatch_shadow_scorecard_v0",
        "gate": "adaptive_lr_runtime_dispatch_shadow",
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
        "selected_optimizer_names": [case.optimizer.value for case in TARGET_CASES],
        "native_training_mode": mode,
        "adapter_route": route,
        "dispatch_envelope": envelope,
        "training_tensor_binding_summary": _as_dict(binding.get("summary")),
        "rows": rows,
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "runtime_dispatch_shadow_ready_count": sum(
                1 for row in rows if row["runtime_dispatch_shadow_ready"] is True
            ),
            "training_tensor_binding_canary_ready_count": sum(
                1 for row in rows if row["training_tensor_binding_canary_ready"] is True
            ),
            "training_tensor_binding_parity_ready_count": sum(
                1 for row in rows if row["training_tensor_binding_parity_ready"] is True
            ),
            "fallback_backend_authoritative": True,
            "native_shadow_call_allowed_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adaptive_lr_training_loop_canary_missing",
                "adaptive_lr_end_to_end_shadow_matrix_missing",
                "adaptive_lr_canary_rollout_policy_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add adaptive-LR TrainingLoop explicit canary with dispatch still default-off"
            if ready
            else "fix adaptive-LR runtime dispatch shadow blockers"
        ),
        "notes": [
            "This shadow is an internal runtime/request envelope only.",
            "The Python adaptive-LR optimizer remains the training update authority.",
            "No native optimizer update is dispatched from this scorecard.",
        ],
    }


def _route_decision(binding: Mapping[str, Any], mode: str) -> dict[str, Any]:
    binding_ready = binding.get("training_tensor_binding_canary_ready") is True
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif not binding_ready:
        decision = "fallback"
        reason = "training_tensor_binding_canary_missing"
    else:
        decision = "shadow_adapter_prepared_fallback_authoritative"
        reason = "runtime_dispatch_disabled_pending_training_loop_canary"
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": "adaptive_lr_native_optimizer",
        "optimizer_family": OPTIMIZER_FAMILY,
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
            "training_loop_explicit_canary",
            "end_to_end_shadow_matrix",
            "canary_rollout_policy",
            "manual_promotion_review",
        ],
    }


def _dispatch_envelope(
    binding: Mapping[str, Any],
    route: Mapping[str, Any],
    rows: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "selected_optimizer_names": [row["optimizer_type"] for row in rows],
        "native_training_mode": mode,
        "training_update_authority": FALLBACK_BACKEND,
        "native_update_authority": "none_until_review",
        "fallback_backend": route.get("fallback_backend"),
        "runtime_request_fields": [
            "optimizer_type",
            "optimizer_family",
            "native_training_mode",
            "param_handles",
            "grad_handles",
            "exp_avg_handles",
            "exp_avg_sq_handles",
            "adaptive_state_handles",
            "training_tensor_binding_canary",
        ],
        "required_evidence": [
            "adaptive_lr_state_machine_replay_executor",
            "adaptive_lr_cuda_kernel_implementation",
            "adaptive_lr_training_tensor_binding_canary",
            "training_loop_explicit_canary",
            "end_to_end_shadow_matrix",
            "canary_rollout_policy",
        ],
        "tensor_binding_probe_kind": binding.get("probe_kind") or PROBE_KIND,
        "canary_dispatch_armed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
    }


def _row(optimizer_type: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    binding_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in binding.get("rows", [])
        if isinstance(row, Mapping)
    }
    source = _as_dict(binding_rows.get(optimizer_type))
    binding_ready = source.get("training_tensor_binding_canary_ready") is True
    parity_ready = source.get("training_tensor_binding_parity_ready") is True
    ready = binding_ready and parity_ready
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": str(source.get("family") or OPTIMIZER_FAMILY),
        "runtime_dispatch_shadow_ready": ready,
        "training_tensor_binding_canary_ready": binding_ready,
        "training_tensor_binding_parity_ready": parity_ready,
        "kernel_executed": source.get("kernel_executed") is True,
        "native_shadow_call_allowed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "next_gate": "adaptive_lr_training_loop_canary",
        "blocked_reasons": [] if ready else [f"adaptive_lr_runtime_dispatch_shadow_binding_missing:{optimizer_type}"],
    }


def _validations(
    binding: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    route: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "training_tensor_binding_canary_ready",
            binding.get("training_tensor_binding_canary_ready") is True,
            "adaptive_lr_training_tensor_binding_canary_missing",
        ),
        _validation(
            "all_rows_shadow_ready",
            all(row.get("runtime_dispatch_shadow_ready") is True for row in rows),
            "adaptive_lr_runtime_dispatch_shadow_row_not_ready",
        ),
        _validation(
            "runtime_dispatch_shadow_envelope_ready",
            envelope.get("training_update_authority") == FALLBACK_BACKEND
            and envelope.get("tensor_binding_probe_kind") == PROBE_KIND,
            "adaptive_lr_runtime_dispatch_shadow_envelope_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            route.get("fallback_backend_authoritative") is True
            and envelope.get("fallback_backend_authoritative") is True,
            "adaptive_lr_runtime_dispatch_shadow_non_authoritative_fallback",
        ),
        _validation(
            "native_shadow_call_disabled",
            route.get("native_shadow_call_allowed") is False
            and envelope.get("native_shadow_call_allowed") is False,
            "adaptive_lr_runtime_dispatch_shadow_enabled_native_call",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            route.get("runtime_dispatch_ready") is False
            and route.get("native_dispatch_allowed") is False
            and route.get("training_path_enabled") is False
            and all(row.get("runtime_dispatch_ready") is False for row in rows)
            and all(row.get("native_dispatch_allowed") is False for row in rows)
            and all(row.get("training_path_enabled") is False for row in rows),
            "adaptive_lr_runtime_dispatch_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            binding.get("training_path_enabled") is False
            and binding.get("default_behavior_changed") is False
            and all(row.get("default_behavior_changed") is False for row in rows),
            "adaptive_lr_runtime_dispatch_shadow_changed_default_behavior",
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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_adaptive_lr_runtime_dispatch_shadow_scorecard"]
