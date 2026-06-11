"""Report-only runtime dispatch adapter shadow for KahanAdamW8bit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_kahan_adamw8bit_canary_dispatch_manifest_scorecard import (
    build_kahan_adamw8bit_canary_dispatch_manifest_scorecard,
)


OPTIMIZER_KIND = "kahan_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_kahan"
ADAPTER_KIND = "kahan_adamw8bit_runtime_dispatch_adapter_shadow_v0"
FALLBACK_BACKEND = "python_kahan_adamw8bit"


def build_kahan_adamw8bit_runtime_dispatch_adapter_shadow_scorecard(
    *,
    manifest_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
    run_live_probe: bool = True,
    require_live_matrix: bool = True,
) -> dict[str, Any]:
    """Build the runtime adapter envelope while keeping fallback authoritative."""

    manifest = dict(
        manifest_report
        or build_kahan_adamw8bit_canary_dispatch_manifest_scorecard(
            native_training_mode=native_training_mode,
            run_live_probe=run_live_probe,
            require_live_matrix=require_live_matrix,
        )
    )
    route = _adapter_route(manifest)
    envelope = _adapter_envelope(manifest, route)
    validations = _validations(manifest, route, envelope)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_runtime_dispatch_adapter_shadow_scorecard_v0",
        "gate": "kahan_adamw8bit_runtime_dispatch_adapter_shadow",
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
        "native_training_mode": str(manifest.get("native_training_mode") or native_training_mode),
        "adapter_route": route,
        "adapter_envelope": envelope,
        "manifest_summary": dict(manifest.get("summary") or {}),
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
                "kahan_adamw8bit_runtime_dispatch_disabled_pending_review",
                "kahan_adamw8bit_end_to_end_training_shadow_missing",
                "kahan_adamw8bit_canary_rollout_policy_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add KahanAdamW8bit end-to-end shadow training matrix before enabling canary"
            if ready
            else "fix KahanAdamW8bit runtime dispatch adapter shadow blockers"
        ),
        "notes": [
            "This adapter shadow is an internal runtime/request contract only.",
            "The Python KahanAdamW8bit training update remains authoritative.",
            "No native optimizer update is dispatched from this scorecard.",
        ],
    }


def _adapter_route(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_ready = bool(manifest.get("canary_dispatch_manifest_ready", False))
    manifest_route = manifest.get("route_decision")
    manifest_route = manifest_route if isinstance(manifest_route, Mapping) else {}
    if manifest_ready:
        decision = "shadow_adapter_prepared_fallback_authoritative"
        reason = "runtime_dispatch_disabled_pending_review"
    else:
        decision = "fallback"
        reason = "canary_dispatch_manifest_missing"
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "feature": "kahan_adamw8bit_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "decision": decision,
        "reason": reason,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "native_shadow_call_allowed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "launch_plan": manifest.get("launch_plan"),
        "manifest_decision": manifest_route.get("decision"),
        "missing_before_dispatch": [
            "manual_promotion_review",
            "end_to_end_training_shadow",
            "canary_rollout_policy",
        ],
    }


def _adapter_envelope(manifest: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_kind": ADAPTER_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "training_update_authority": "python_kahan_adamw8bit",
        "native_update_authority": "none",
        "fallback_backend": route.get("fallback_backend"),
        "runtime_request_fields": [
            "optimizer_kind",
            "optimizer_family",
            "native_training_mode",
            "param_dtype",
            "grad_dtype",
            "state_dtype",
            "device_type",
            "contiguous",
            "checkpoint_resume_adapter",
            "training_tensor_binding_canary",
            "kahan_comp_dtype",
        ],
        "required_evidence": [
            "bf16_native_dtype_matrix",
            "checkpoint_resume_adapter",
            "real_training_matrix",
            "canary_dispatch_manifest",
        ],
        "accepted_launch_plan": manifest.get("launch_plan"),
        "canary_dispatch_armed": False,
        "native_shadow_call_allowed": False,
        "fallback_backend_authoritative": True,
    }


def _validations(
    manifest: Mapping[str, Any],
    route: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8z_canary_dispatch_manifest_ready",
            bool(manifest.get("canary_dispatch_manifest_ready", False)),
            "kahan_adamw8bit_canary_dispatch_manifest_missing",
        ),
        _validation(
            "adapter_shadow_envelope_ready",
            bool(envelope.get("accepted_launch_plan"))
            and envelope.get("training_update_authority") == "python_kahan_adamw8bit",
            "kahan_adamw8bit_runtime_dispatch_adapter_shadow_envelope_missing",
        ),
        _validation(
            "fallback_backend_authoritative",
            bool(route.get("fallback_backend_authoritative", False))
            and bool(envelope.get("fallback_backend_authoritative", False)),
            "kahan_adamw8bit_runtime_dispatch_adapter_shadow_non_authoritative_fallback",
        ),
        _validation(
            "native_shadow_call_disabled",
            not bool(route.get("native_shadow_call_allowed", True))
            and not bool(envelope.get("native_shadow_call_allowed", True)),
            "kahan_adamw8bit_runtime_dispatch_adapter_shadow_enabled_native_call",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("training_path_enabled", True)),
            "kahan_adamw8bit_runtime_dispatch_adapter_shadow_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(manifest.get("training_path_enabled", True))
            and not bool(manifest.get("default_behavior_changed", True)),
            "kahan_adamw8bit_runtime_dispatch_adapter_shadow_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_kahan_adamw8bit_runtime_dispatch_adapter_shadow_scorecard"]
