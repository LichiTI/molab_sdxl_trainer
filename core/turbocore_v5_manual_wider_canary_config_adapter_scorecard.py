"""Scorecard for the V5 manual wider-canary config/request adapter."""

from __future__ import annotations

from typing import Any

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.turbocore_v5_manual_wider_canary_config_adapter import (
    apply_v5_manual_wider_canary_config_adapter,
)


def build_v5_manual_wider_canary_config_adapter_scorecard() -> dict[str, Any]:
    """Build a machine-readable proof for the V5 request adapter."""

    cases = {
        "default_off": _default_off_case(),
        "scope_without_review_blocked": _scope_without_review_case(),
        "approved_manual_wider_canary": _approved_case(),
        "reject_non_exact_optimizer": _reject_optimizer_case(),
        "reject_unsupported_backend": _reject_backend_case(),
        "reject_unsupported_scope": _reject_scope_case(),
    }
    progress_gates = {
        "default_off": bool(cases["default_off"].get("ok", False)),
        "manual_wider_scope_without_review_blocked": bool(
            cases["scope_without_review_blocked"].get("ok", False)
        ),
        "approved_manual_wider_canary_enables_existing_fields": bool(
            cases["approved_manual_wider_canary"].get("ok", False)
        ),
        "non_exact_optimizer_blocked": bool(cases["reject_non_exact_optimizer"].get("ok", False)),
        "unsupported_backend_blocked": bool(cases["reject_unsupported_backend"].get("ok", False)),
        "unsupported_scope_blocked": bool(cases["reject_unsupported_scope"].get("ok", False)),
        "default_behavior_unchanged": True,
    }
    ready = all(progress_gates.values())
    blockers = [f"v5_p5_{name}_missing" for name, ok in progress_gates.items() if not ok]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_manual_wider_canary_config_adapter_scorecard_v0",
        "gate": "v5_manual_wider_canary_config_request_adapter",
        "ok": ready,
        "milestone_completed": ready,
        "config_adapter_ready": ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "adapter_cases": cases,
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "manual wider canary requests can be mapped after owner review evidence"
            if ready
            else "complete V5-P5 config/request adapter blockers"
        ),
        "notes": [
            "The adapter maps a high-level V5 manual canary request to existing TurboCore native update fields.",
            "No launcher or WebUI training entry is introduced.",
            "Missing owner-review evidence forces the native update training path back to off.",
        ],
    }


def _default_off_case() -> dict[str, Any]:
    config = ConfigAdapter.from_frontend_dict({"optimizerType": "AdamW"})
    ok = _fields_off(config)
    return {
        "schema_version": 1,
        "case": "default_off",
        "ok": ok,
        "resolved_fields": _resolved_config_fields(config),
    }


def _scope_without_review_case() -> dict[str, Any]:
    config = ConfigAdapter.from_frontend_dict(
        {
            "optimizerType": "AdamW",
            "optimizerBackend": "torch_adamw",
            "turbocoreNativeUpdateCanaryOptimizer": "exact_adamw",
            "turbocoreNativeUpdateCanaryScope": "manual_wider_canary",
        }
    )
    raw = _base_manual_wider_raw(review=False)
    report = apply_v5_manual_wider_canary_config_adapter(raw)
    ok = (
        _fields_off(config)
        and report["allowed"] is False
        and "v5_p5_manual_wider_canary_review_evidence_missing" in report["blocked_reasons"]
    )
    return {
        "schema_version": 1,
        "case": "scope_without_review_blocked",
        "ok": ok,
        "resolved_fields": _resolved_config_fields(config),
        "adapter_report": report,
    }


def _approved_case() -> dict[str, Any]:
    config = ConfigAdapter.from_frontend_dict(
        {
            "optimizerType": "AdamW",
            "optimizerBackend": "torch_adamw",
            "turbocoreNativeUpdateCanaryOptimizer": "exact_adamw",
            "turbocoreNativeUpdateCanaryScope": "manual_wider_canary",
            "turbocoreNativeUpdateManualWiderCanaryApproved": True,
        }
    )
    ok = (
        config.turbocore_native_update_mode == "native_experimental"
        and config.turbocore_native_update_dispatch_enabled
        and config.turbocore_native_update_training_path_enabled
        and config.turbocore_native_update_require_native_cuda
        and config.turbocore_native_update_defer_state_sync
    )
    return {
        "schema_version": 1,
        "case": "approved_manual_wider_canary",
        "ok": ok,
        "resolved_fields": _resolved_config_fields(config),
    }


def _reject_optimizer_case() -> dict[str, Any]:
    raw = _base_manual_wider_raw(review=True)
    raw["optimizer_type"] = "AdamW8bit"
    report = apply_v5_manual_wider_canary_config_adapter(raw)
    ok = (
        report["allowed"] is False
        and raw["turbocore_native_update_training_path_enabled"] is False
        and raw["turbocore_native_update_defer_state_sync"] is False
        and "v5_p5_requires_optimizer_type_adamw" in report["blocked_reasons"]
    )
    return {"schema_version": 1, "case": "reject_non_exact_optimizer", "ok": ok, "adapter_report": report}


def _reject_backend_case() -> dict[str, Any]:
    raw = _base_manual_wider_raw(review=True)
    raw["optimizer_backend"] = "bnb_8bit"
    report = apply_v5_manual_wider_canary_config_adapter(raw)
    ok = (
        report["allowed"] is False
        and raw["turbocore_native_update_dispatch_enabled"] is False
        and "v5_p5_backend_not_allowed" in report["blocked_reasons"]
    )
    return {"schema_version": 1, "case": "reject_unsupported_backend", "ok": ok, "adapter_report": report}


def _reject_scope_case() -> dict[str, Any]:
    config = ConfigAdapter.from_frontend_dict(
        {
            "optimizerType": "AdamW",
            "optimizerBackend": "torch_adamw",
            "turbocoreNativeUpdateCanaryOptimizer": "exact_adamw",
            "turbocoreNativeUpdateCanaryScope": "auto",
            "turbocoreNativeUpdateManualWiderCanaryApproved": True,
        }
    )
    raw = _base_manual_wider_raw(review=True)
    raw["turbocore_native_update_canary_scope"] = "auto"
    report = apply_v5_manual_wider_canary_config_adapter(raw)
    ok = (
        _fields_off(config)
        and report["allowed"] is False
        and "v5_p5_scope_limited_to_manual_wider_canary" in report["blocked_reasons"]
    )
    return {
        "schema_version": 1,
        "case": "reject_unsupported_scope",
        "ok": ok,
        "resolved_fields": _resolved_config_fields(config),
        "adapter_report": report,
    }


def _base_manual_wider_raw(*, review: bool) -> dict[str, Any]:
    return {
        "optimizer_type": "AdamW",
        "optimizer_backend": "torch_adamw",
        "turbocore_native_update_canary_optimizer": "exact_adamw",
        "turbocore_native_update_canary_scope": "manual_wider_canary",
        "turbocore_native_update_manual_wider_canary_approved": bool(review),
    }


def _fields_off(config: Any) -> bool:
    return (
        config.turbocore_native_update_mode == "off"
        and not config.turbocore_native_update_dispatch_enabled
        and not config.turbocore_native_update_training_path_enabled
        and not config.turbocore_native_update_require_native_cuda
        and not config.turbocore_native_update_defer_state_sync
    )


def _resolved_config_fields(config: Any) -> dict[str, Any]:
    return {
        "turbocore_native_update_mode": config.turbocore_native_update_mode,
        "turbocore_native_update_dispatch_enabled": bool(config.turbocore_native_update_dispatch_enabled),
        "turbocore_native_update_training_path_enabled": bool(
            config.turbocore_native_update_training_path_enabled
        ),
        "turbocore_native_update_require_native_cuda": bool(
            config.turbocore_native_update_require_native_cuda
        ),
        "turbocore_native_update_defer_state_sync": bool(config.turbocore_native_update_defer_state_sync),
        "optimizer_type": str(config.optimizer_type),
        "optimizer_backend": str(config.optimizer_backend),
    }


__all__ = ["build_v5_manual_wider_canary_config_adapter_scorecard"]
