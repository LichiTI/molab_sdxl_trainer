"""Config/request adapter for V5 manual wider-canary native AdamW."""

from __future__ import annotations

from typing import Any, MutableMapping


EXACT_ADAMW_OPTIMIZERS = {"adamw"}
EXACT_CANARY_OPTIMIZERS = {"exact_adamw", "adamw_exact", "rust_cuda_adamw_v0"}
ALLOWED_BACKENDS = {"auto", "torch_adamw", "foreach_adamw", "torch_fused"}
MANUAL_WIDER_SCOPES = {"manual_wider_canary", "wider_manual_canary"}


def apply_v5_manual_wider_canary_config_adapter(config: MutableMapping[str, Any]) -> dict[str, Any]:
    """Resolve an explicit V5 manual wider canary request into existing fields.

    This adapter is deliberately narrower than the V3 single-run canary adapter:
    a manual wider canary must name the canary scope and optimizer, and must
    include internal owner-review evidence.  Missing or invalid V5 evidence
    forces the native update training path back to off, even if older canary
    fields were also present in the request.
    """

    scope = _scope_key(config.get("turbocore_native_update_canary_scope"))
    requested = _requested(config, scope)
    optimizer = _optimizer_key(config.get("optimizer_type"))
    backend = _backend_key(config.get("optimizer_backend", "auto"))
    canary_optimizer = _canary_optimizer_key(config)
    review_evidence = _review_evidence_present(config)

    blockers: list[str] = []
    if requested and scope not in MANUAL_WIDER_SCOPES:
        blockers.append("v5_p5_scope_limited_to_manual_wider_canary")
    if requested and canary_optimizer not in EXACT_CANARY_OPTIMIZERS:
        blockers.append("v5_p5_requires_canary_optimizer_exact_adamw")
    if requested and optimizer not in EXACT_ADAMW_OPTIMIZERS:
        blockers.append("v5_p5_requires_optimizer_type_adamw")
    if requested and backend not in ALLOWED_BACKENDS:
        blockers.append("v5_p5_backend_not_allowed")
    if requested and not review_evidence:
        blockers.append("v5_p5_manual_wider_canary_review_evidence_missing")

    allowed = bool(requested and not blockers)
    if requested:
        _write_resolved_fields(config, enabled=allowed)

    return {
        "schema_version": 1,
        "adapter": "v5_manual_wider_canary_config_adapter_v0",
        "requested": requested,
        "allowed": allowed,
        "default_off": not requested,
        "scope": scope,
        "scope_allowed": scope in MANUAL_WIDER_SCOPES,
        "optimizer_type": str(config.get("optimizer_type") or ""),
        "optimizer_key": optimizer,
        "optimizer_exact_adamw": optimizer in EXACT_ADAMW_OPTIMIZERS,
        "canary_optimizer": canary_optimizer,
        "canary_optimizer_allowed": canary_optimizer in EXACT_CANARY_OPTIMIZERS,
        "optimizer_backend": backend,
        "backend_allowed": backend in ALLOWED_BACKENDS,
        "manual_review_evidence_present": review_evidence,
        "resolved_fields": _resolved_fields(config),
        "blocked_reasons": _dedupe(blockers),
        "notes": [
            "V5 manual wider canary is explicit request only.",
            "Default and auto rollout are not enabled by this adapter.",
            "The adapter maps to existing turbocore_native_update_* fields instead of creating a new training entry.",
        ],
    }


def _requested(config: MutableMapping[str, Any], scope: str) -> bool:
    return bool(
        scope
        or _boolish(config.get("turbocore_native_update_manual_wider_canary"), False)
        or _boolish(config.get("turbocore_native_update_manual_wider_canary_requested"), False)
    )


def _review_evidence_present(config: MutableMapping[str, Any]) -> bool:
    return bool(
        _boolish(config.get("turbocore_native_update_manual_wider_canary_approved"), False)
        or _boolish(config.get("turbocore_native_update_manual_wider_canary_owner_approved"), False)
        or _boolish(config.get("turbocore_native_update_manual_wider_canary_review_ok"), False)
        or _boolish(config.get("turbocore_native_update_manual_wider_canary_review_ready"), False)
    )


def _write_resolved_fields(config: MutableMapping[str, Any], *, enabled: bool) -> None:
    config["turbocore_native_update_mode"] = "native_experimental" if enabled else "off"
    config["turbocore_native_update_dispatch_enabled"] = bool(enabled)
    config["turbocore_native_update_training_path_enabled"] = bool(enabled)
    config["turbocore_native_update_require_native_cuda"] = bool(enabled)
    config["turbocore_native_update_defer_state_sync"] = bool(enabled)


def _resolved_fields(config: MutableMapping[str, Any]) -> dict[str, Any]:
    return {
        "turbocore_native_update_mode": str(config.get("turbocore_native_update_mode", "off") or "off"),
        "turbocore_native_update_dispatch_enabled": bool(
            config.get("turbocore_native_update_dispatch_enabled", False)
        ),
        "turbocore_native_update_training_path_enabled": bool(
            config.get("turbocore_native_update_training_path_enabled", False)
        ),
        "turbocore_native_update_require_native_cuda": bool(
            config.get("turbocore_native_update_require_native_cuda", False)
        ),
        "turbocore_native_update_defer_state_sync": bool(
            config.get("turbocore_native_update_defer_state_sync", False)
        ),
    }


def _canary_optimizer_key(config: MutableMapping[str, Any]) -> str:
    raw = config.get("turbocore_native_update_canary_optimizer")
    if raw is None and _boolish(config.get("turbocore_exact_adamw_canary"), False):
        return "exact_adamw"
    return _optimizer_key(raw)


def _optimizer_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.replace(" ", "").replace("-", "_")


def _backend_key(value: Any) -> str:
    return str(value or "auto").strip().lower().replace("-", "_").replace(" ", "")


def _scope_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "")


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["apply_v5_manual_wider_canary_config_adapter"]
