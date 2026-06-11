"""Scorecard for the V3 exact AdamW canary config/request adapter."""

from __future__ import annotations

from typing import Any, Mapping

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.turbocore_v3_exact_adamw_config_adapter import (
    apply_v3_exact_adamw_canary_config_adapter,
)
from core.turbocore_v3_exact_adamw_runtime_recovery_scorecard import (
    build_v3_exact_adamw_runtime_recovery_scorecard,
)


def build_v3_exact_adamw_config_adapter_scorecard(
    *,
    p3_recovery: Mapping[str, Any] | None = None,
    run_live_training: bool = True,
) -> dict[str, Any]:
    """Build a machine-readable proof for the V3 canary config adapter."""

    p3 = dict(
        p3_recovery
        or build_v3_exact_adamw_runtime_recovery_scorecard(run_live_training=run_live_training)
    )
    cases = {
        "default_off": _default_off_case(),
        "explicit_exact_adamw": _explicit_exact_adamw_case(),
        "reject_adamw8bit": _reject_optimizer_case("AdamW8bit"),
        "reject_paged_adamw8bit": _reject_optimizer_case("PagedAdamW8bit"),
        "reject_bnb_backend": _reject_backend_case(),
    }
    progress_gates = {
        "p3_runtime_recovery_complete": bool(p3.get("runtime_recovery_hardened", False)),
        "default_off": bool(cases["default_off"].get("ok", False)),
        "explicit_exact_adamw_enables_existing_fields": bool(cases["explicit_exact_adamw"].get("ok", False)),
        "non_exact_optimizer_blocked": bool(cases["reject_adamw8bit"].get("ok", False))
        and bool(cases["reject_paged_adamw8bit"].get("ok", False)),
        "unsupported_backend_blocked": bool(cases["reject_bnb_backend"].get("ok", False)),
        "default_behavior_unchanged": (
            not bool(p3.get("default_behavior_changed", True))
            and not bool(p3.get("default_training_path_enabled", True))
        ),
    }
    ready = all(progress_gates.values())
    blockers = [f"v3_p4_{name}_missing" for name, ok in progress_gates.items() if not ok]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v3_exact_adamw_config_adapter_scorecard_v0",
        "gate": "v3_exact_adamw_config_request_adapter",
        "ok": bool(p3.get("ok", False)) and all(bool(case.get("ok", False)) for case in cases.values()),
        "milestone_completed": ready,
        "config_adapter_ready": ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "p3_summary": {
            "runtime_recovery_hardened": bool(p3.get("runtime_recovery_hardened", False)),
            "milestone_completed": bool(p3.get("milestone_completed", False)),
        },
        "adapter_cases": cases,
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "build V3 exact AdamW promotion review gate"
            if ready
            else "complete V3-P4 config/request adapter blockers"
        ),
        "notes": [
            "The adapter maps a high-level explicit canary request to existing TurboCore native update fields.",
            "No launcher or WebUI training entry is introduced.",
            "Non-exact AdamW and unsupported backends are forced back to off for this V3 canary request.",
        ],
    }


def _default_off_case() -> dict[str, Any]:
    config = ConfigAdapter.from_frontend_dict({"optimizerType": "AdamW"})
    ok = (
        config.turbocore_native_update_mode == "off"
        and not config.turbocore_native_update_dispatch_enabled
        and not config.turbocore_native_update_training_path_enabled
        and not config.turbocore_native_update_require_native_cuda
    )
    return {
        "schema_version": 1,
        "case": "default_off",
        "ok": ok,
        "resolved_fields": _resolved_config_fields(config),
    }


def _explicit_exact_adamw_case() -> dict[str, Any]:
    config = ConfigAdapter.from_frontend_dict(
        {
            "optimizerType": "AdamW",
            "optimizerBackend": "torch_adamw",
            "turbocoreExactAdamwCanary": True,
        }
    )
    ok = (
        config.turbocore_native_update_mode == "native_experimental"
        and config.turbocore_native_update_dispatch_enabled
        and config.turbocore_native_update_training_path_enabled
        and config.turbocore_native_update_require_native_cuda
    )
    return {
        "schema_version": 1,
        "case": "explicit_exact_adamw",
        "ok": ok,
        "resolved_fields": _resolved_config_fields(config),
    }


def _reject_optimizer_case(optimizer_type: str) -> dict[str, Any]:
    raw = {
        "optimizer_type": optimizer_type,
        "optimizer_backend": "auto",
        "turbocore_exact_adamw_canary": True,
    }
    report = apply_v3_exact_adamw_canary_config_adapter(raw)
    ok = (
        report["allowed"] is False
        and raw["turbocore_native_update_mode"] == "off"
        and raw["turbocore_native_update_dispatch_enabled"] is False
        and "v3_exact_adamw_canary_requires_optimizer_type_adamw" in report["blocked_reasons"]
    )
    return {
        "schema_version": 1,
        "case": f"reject_{optimizer_type}",
        "ok": ok,
        "adapter_report": report,
    }


def _reject_backend_case() -> dict[str, Any]:
    raw = {
        "optimizer_type": "AdamW",
        "optimizer_backend": "bnb_8bit",
        "turbocore_exact_adamw_canary": True,
    }
    report = apply_v3_exact_adamw_canary_config_adapter(raw)
    ok = (
        report["allowed"] is False
        and raw["turbocore_native_update_training_path_enabled"] is False
        and "v3_exact_adamw_canary_backend_not_allowed" in report["blocked_reasons"]
    )
    return {
        "schema_version": 1,
        "case": "reject_bnb_backend",
        "ok": ok,
        "adapter_report": report,
    }


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
        "optimizer_type": str(config.optimizer_type),
        "optimizer_backend": str(config.optimizer_backend),
    }


__all__ = ["build_v3_exact_adamw_config_adapter_scorecard"]
