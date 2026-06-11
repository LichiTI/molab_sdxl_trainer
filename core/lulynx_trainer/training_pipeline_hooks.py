"""Hook gate helpers for the Lulynx staged training pipeline.

This module is deliberately small and declarative. It validates plugin hook
declarations against the staged pipeline contract, groups accepted hooks by
stage order, and reports why unsafe requests are rejected. It does not execute
plugin code and does not alter the active training path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.core.lulynx_trainer.training_pipeline_contract import (
    LulynxHookAccess,
    LULYNX_TRAINING_STAGES,
    validate_lulynx_pipeline_hook_request,
)


LULYNX_TRAINING_PIPELINE_HOOK_GATE_REPORT = "lulynx_training_pipeline_hook_gate_v0"


@dataclass(frozen=True)
class LulynxPipelineHookDeclaration:
    plugin_id: str
    hook_id: str
    stage_id: str
    access: str
    requested_mutations: tuple[str, ...] = ()
    entry: str = ""
    enabled: bool = True


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item))


def _safe_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    return bool(value)


def normalize_lulynx_pipeline_hook_declaration(value: Any) -> LulynxPipelineHookDeclaration:
    raw = _mapping(value)
    plugin_id = str(raw.get("plugin_id") or raw.get("plugin") or "unknown_plugin").strip() or "unknown_plugin"
    hook_id = str(raw.get("hook_id") or raw.get("id") or "").strip()
    stage_id = str(raw.get("stage_id") or raw.get("stage") or "").strip()
    access = str(raw.get("access") or raw.get("hook_access") or LulynxHookAccess.READONLY.value).strip().lower()
    if not hook_id:
        hook_id = f"{plugin_id}:{stage_id or 'unknown_stage'}:{access or 'unknown_access'}"
    return LulynxPipelineHookDeclaration(
        plugin_id=plugin_id,
        hook_id=hook_id,
        stage_id=stage_id,
        access=access,
        requested_mutations=_string_tuple(raw.get("requested_mutations") or raw.get("mutations")),
        entry=str(raw.get("entry") or ""),
        enabled=_safe_bool(raw.get("enabled"), default=True),
    )


def _declaration_payload(declaration: LulynxPipelineHookDeclaration) -> dict[str, Any]:
    return {
        "plugin_id": declaration.plugin_id,
        "hook_id": declaration.hook_id,
        "stage_id": declaration.stage_id,
        "access": declaration.access,
        "requested_mutations": list(declaration.requested_mutations),
        "entry": declaration.entry,
        "enabled": declaration.enabled,
    }


def build_lulynx_pipeline_hook_gate(
    declarations: Sequence[Any],
    *,
    allow_experimental: bool = False,
) -> dict[str, Any]:
    """Validate staged training hook declarations without running hooks."""

    stage_order = {stage.id: stage.order for stage in LULYNX_TRAINING_STAGES}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw in declarations:
        declaration = normalize_lulynx_pipeline_hook_declaration(raw)
        payload = _declaration_payload(declaration)
        if not declaration.enabled:
            payload["status"] = "disabled"
            disabled.append(payload)
            continue

        validation = validate_lulynx_pipeline_hook_request(
            stage_id=declaration.stage_id,
            requested_access=declaration.access,
            requested_mutations=declaration.requested_mutations,
        )
        reasons = [str(item) for item in validation.get("reasons", [])]
        if declaration.hook_id in seen:
            reasons.append("duplicate_hook_id")
        if declaration.access == LulynxHookAccess.EXPERIMENTAL.value and not allow_experimental:
            reasons.append("experimental_hooks_disabled")

        item = {
            **payload,
            "stage_order": int(stage_order.get(declaration.stage_id, 9999)),
            "allowed_access": list(validation.get("allowed_access", [])),
            "allowed_mutations": list(validation.get("allowed_mutations", [])),
            "mutation_errors": list(validation.get("mutation_errors", [])),
            "reasons": sorted(set(reasons)),
        }
        if validation.get("ok") and not reasons:
            item["status"] = "accepted"
            accepted.append(item)
            seen.add(declaration.hook_id)
        else:
            item["status"] = "rejected"
            rejected.append(item)

    accepted.sort(key=lambda item: (int(item.get("stage_order") or 9999), str(item.get("hook_id") or "")))
    stage_groups: dict[str, list[str]] = {}
    for item in accepted:
        stage_groups.setdefault(str(item["stage_id"]), []).append(str(item["hook_id"]))

    return {
        "schema_version": 1,
        "report": LULYNX_TRAINING_PIPELINE_HOOK_GATE_REPORT,
        "status": "ready" if not rejected else "has_rejected_hooks",
        "does_not_execute_hooks": True,
        "release_claim_allowed": False,
        "allow_experimental": bool(allow_experimental),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "disabled_count": len(disabled),
        "accepted_hooks": accepted,
        "rejected_hooks": rejected,
        "disabled_hooks": disabled,
        "stage_groups": stage_groups,
    }


__all__ = [
    "LULYNX_TRAINING_PIPELINE_HOOK_GATE_REPORT",
    "LulynxPipelineHookDeclaration",
    "build_lulynx_pipeline_hook_gate",
    "normalize_lulynx_pipeline_hook_declaration",
]
