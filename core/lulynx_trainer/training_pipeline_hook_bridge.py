"""Report-only bridge from legacy plugin hook events to staged pipeline hooks.

The bridge treats plugin manifests as data. It maps known hook event names to
the internal staged training pipeline declaration shape, detects privilege
escalation in the source declaration, and delegates stage/access validation to
the existing hook gate. It never imports plugin code and never executes hooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.core.lulynx_trainer.training_pipeline_contract import LulynxHookAccess
from backend.core.lulynx_trainer.training_pipeline_hooks import build_lulynx_pipeline_hook_gate


LULYNX_TRAINING_PIPELINE_HOOK_BRIDGE_REPORT = "lulynx_training_pipeline_hook_bridge_v0"

_WRITE_ACCESS = {LulynxHookAccess.TRANSFORM.value, LulynxHookAccess.CONTROL.value}
_NON_EXECUTED_ACCESS = _WRITE_ACCESS | {LulynxHookAccess.EXPERIMENTAL.value}


@dataclass(frozen=True)
class LulynxEventPipelineMapping:
    event: str
    stage_id: str
    access: str
    default_mutations: tuple[str, ...] = ()
    note: str = ""


LULYNX_PLUGIN_EVENT_TO_PIPELINE_STAGE_ACCESS: tuple[LulynxEventPipelineMapping, ...] = (
    LulynxEventPipelineMapping("on_config_loaded", "dataset_scan", "readonly"),
    LulynxEventPipelineMapping("on_dataset_prepared", "dataset_scan", "readonly"),
    LulynxEventPipelineMapping("on_train_launch", "batch_contract", "readonly"),
    LulynxEventPipelineMapping("before_forward", "forward", "readonly"),
    LulynxEventPipelineMapping("after_loss", "loss", "readonly"),
    LulynxEventPipelineMapping("after_backward", "backward", "readonly"),
    LulynxEventPipelineMapping("before_optimizer_step", "optimizer_step", "readonly"),
    LulynxEventPipelineMapping("after_optimizer_step", "optimizer_step", "readonly"),
    LulynxEventPipelineMapping("on_train_complete", "telemetry", "readonly"),
    LulynxEventPipelineMapping("preflight_advisory", "dataset_scan", "advisory"),
    LulynxEventPipelineMapping("dataset_health_advisory", "dataset_scan", "advisory"),
    LulynxEventPipelineMapping("caption_tag_advisory", "bucket_plan", "advisory"),
    LulynxEventPipelineMapping("validation_prompt_advisory", "dataset_scan", "advisory"),
    LulynxEventPipelineMapping("metrics_advisory", "telemetry", "advisory"),
    LulynxEventPipelineMapping("report_advisory", "telemetry", "advisory"),
    LulynxEventPipelineMapping(
        "modify_loss",
        "loss",
        "transform",
        ("loss",),
        "report_only_transform_hook_not_executed",
    ),
    LulynxEventPipelineMapping(
        "modify_scheduler_step",
        "optimizer_step",
        "control",
        ("optimizer_policy",),
        "report_only_control_hook_not_executed",
    ),
    LulynxEventPipelineMapping(
        "modify_optimizer_step",
        "optimizer_step",
        "control",
        ("optimizer_policy",),
        "report_only_control_hook_not_executed",
    ),
)

_EVENT_MAPPING_BY_NAME = {item.event: item for item in LULYNX_PLUGIN_EVENT_TO_PIPELINE_STAGE_ACCESS}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(item) for item in _sequence(value) if item is not None and str(item))


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "disabled"}
    return bool(value)


def _event_from_capability(value: str) -> str:
    return value[5:] if value.startswith("hook_") else value


def _extract_manifest_hooks(manifest_or_hooks: Any) -> tuple[str, list[Any]]:
    raw = _mapping(manifest_or_hooks)
    if raw:
        plugin_id = str(raw.get("plugin_id") or raw.get("id") or raw.get("plugin") or "unknown_plugin")
        for key in ("hooks", "training_hooks", "hook_events"):
            hooks = raw.get(key)
            if isinstance(hooks, Mapping):
                return plugin_id, [{"event": event, **_mapping(item)} for event, item in hooks.items()]
            if isinstance(hooks, Sequence) and not isinstance(hooks, str):
                return plugin_id, list(hooks)
        if raw.get("event") or raw.get("hook_event") or raw.get("capability"):
            return plugin_id, [raw]
        return plugin_id, []
    if isinstance(manifest_or_hooks, Sequence) and not isinstance(manifest_or_hooks, str):
        return "unknown_plugin", list(manifest_or_hooks)
    return "unknown_plugin", []


def _normalize_source_hook(raw_hook: Any, *, plugin_id: str, index: int) -> dict[str, Any]:
    if isinstance(raw_hook, str):
        raw = {"event": raw_hook}
    else:
        raw = dict(_mapping(raw_hook))
    event = str(
        raw.get("event")
        or raw.get("hook_event")
        or raw.get("name")
        or _event_from_capability(str(raw.get("capability") or ""))
    ).strip()
    hook_plugin_id = str(raw.get("plugin_id") or raw.get("plugin") or plugin_id or "unknown_plugin").strip()
    hook_id = str(raw.get("hook_id") or raw.get("id") or "").strip()
    entry = str(raw.get("entry") or raw.get("handler") or "").strip()
    if not hook_id:
        hook_id = f"{hook_plugin_id}:{event or 'unknown_event'}:{entry or index}"
    return {
        "plugin_id": hook_plugin_id or "unknown_plugin",
        "hook_id": hook_id,
        "event": event,
        "entry": entry,
        "declared_access": str(raw.get("access") or raw.get("hook_access") or "").strip().lower(),
        "declared_mutable": _safe_bool(raw.get("mutable") or raw.get("mutates_payload"), default=False),
        "requested_mutations": _string_tuple(
            raw.get("requested_mutations") or raw.get("mutations") or raw.get("mutates")
        ),
        "enabled": _safe_bool(raw.get("enabled"), default=True),
        "source": raw,
    }


def _mapping_payload(mapping: LulynxEventPipelineMapping) -> dict[str, Any]:
    return {
        "event": mapping.event,
        "stage_id": mapping.stage_id,
        "access": mapping.access,
        "default_mutations": list(mapping.default_mutations),
        "note": mapping.note,
    }


def _bridge_declaration(source: Mapping[str, Any], mapping: LulynxEventPipelineMapping) -> dict[str, Any]:
    requested_mutations = tuple(source.get("requested_mutations") or ()) or mapping.default_mutations
    return {
        "plugin_id": source["plugin_id"],
        "hook_id": source["hook_id"],
        "stage_id": mapping.stage_id,
        "access": mapping.access,
        "requested_mutations": list(requested_mutations),
        "entry": source["entry"],
        "enabled": source["enabled"],
        "source_event": source["event"],
    }


def _source_overreach_reasons(source: Mapping[str, Any], mapping: LulynxEventPipelineMapping) -> list[str]:
    reasons: list[str] = []
    declared_access = str(source.get("declared_access") or "")
    mapped_access = mapping.access
    requested_mutations = tuple(source.get("requested_mutations") or ())
    if declared_access and declared_access != mapped_access:
        reasons.append("declared_access_does_not_match_event_mapping")
    if mapped_access not in _WRITE_ACCESS:
        if bool(source.get("declared_mutable")):
            reasons.append("mutable_hook_declared_for_non_mutating_event")
        if requested_mutations:
            reasons.append("mutations_declared_for_non_mutating_event")
    return reasons


def _hook_public_payload(source: Mapping[str, Any], mapping: LulynxEventPipelineMapping | None) -> dict[str, Any]:
    payload = {
        "plugin_id": source["plugin_id"],
        "hook_id": source["hook_id"],
        "event": source["event"],
        "entry": source["entry"],
        "enabled": source["enabled"],
        "declared_access": source["declared_access"],
        "declared_mutable": source["declared_mutable"],
        "requested_mutations": list(source.get("requested_mutations") or ()),
    }
    if mapping is not None:
        payload["mapped_stage_id"] = mapping.stage_id
        payload["mapped_access"] = mapping.access
    return payload


def build_lulynx_training_pipeline_hook_bridge_report(
    manifest_or_hooks: Any,
    *,
    plugin_id: str | None = None,
) -> dict[str, Any]:
    """Build staged hook declarations and diagnostics from plugin hook events."""

    manifest_plugin_id, raw_hooks = _extract_manifest_hooks(manifest_or_hooks)
    source_plugin_id = plugin_id or manifest_plugin_id
    normalized = [
        _normalize_source_hook(raw_hook, plugin_id=source_plugin_id, index=index)
        for index, raw_hook in enumerate(raw_hooks)
    ]

    declarations: list[dict[str, Any]] = []
    mapped_hooks: list[dict[str, Any]] = []
    bridge_blockers: list[dict[str, Any]] = []
    deferred_hooks: list[dict[str, Any]] = []

    for source in normalized:
        event = str(source.get("event") or "")
        mapping = _EVENT_MAPPING_BY_NAME.get(event)
        if mapping is None:
            bridge_blockers.append(
                {
                    **_hook_public_payload(source, None),
                    "reason": "unknown_event_blocker",
                    "known_events": sorted(_EVENT_MAPPING_BY_NAME),
                }
            )
            continue

        declaration = _bridge_declaration(source, mapping)
        declarations.append(declaration)
        overreach = _source_overreach_reasons(source, mapping)
        hook_payload = {
            **_hook_public_payload(source, mapping),
            "staged_declaration": declaration,
            "bridge_reasons": overreach,
        }
        mapped_hooks.append(hook_payload)
        if overreach:
            bridge_blockers.append({**hook_payload, "reason": "source_hook_privilege_overreach"})
        if mapping.access in _NON_EXECUTED_ACCESS:
            deferred_hooks.append(
                {
                    **hook_payload,
                    "reason": "transform_control_hooks_report_only_not_executed",
                }
            )

    gate_report = build_lulynx_pipeline_hook_gate(declarations)
    gate_rejected = list(gate_report.get("rejected_hooks", []))
    for rejected in gate_rejected:
        bridge_blockers.append(
            {
                "reason": "staged_hook_gate_rejected_declaration",
                "hook_id": rejected.get("hook_id"),
                "stage_id": rejected.get("stage_id"),
                "reasons": list(rejected.get("reasons", [])),
            }
        )

    if bridge_blockers:
        status = "blocked"
    elif deferred_hooks:
        status = "report_only_with_deferred_runtime_hooks"
    else:
        status = "ready"

    return {
        "schema_version": 1,
        "report": LULYNX_TRAINING_PIPELINE_HOOK_BRIDGE_REPORT,
        "status": status,
        "does_not_execute_hooks": True,
        "does_not_import_plugin_code": True,
        "release_claim_allowed": False,
        "source_plugin_id": source_plugin_id,
        "source_hook_count": len(normalized),
        "mapped_hook_count": len(declarations),
        "unknown_event_blocker_count": sum(
            1 for item in bridge_blockers if item.get("reason") == "unknown_event_blocker"
        ),
        "privilege_overreach_count": sum(
            1 for item in bridge_blockers if item.get("reason") == "source_hook_privilege_overreach"
        ),
        "transform_control_deferred_count": len(deferred_hooks),
        "event_stage_access_map": [_mapping_payload(item) for item in LULYNX_PLUGIN_EVENT_TO_PIPELINE_STAGE_ACCESS],
        "mapped_hooks": mapped_hooks,
        "staged_declarations": declarations,
        "deferred_hooks": deferred_hooks,
        "bridge_blockers": bridge_blockers,
        "staged_hook_gate_report": gate_report,
    }


__all__ = [
    "LULYNX_PLUGIN_EVENT_TO_PIPELINE_STAGE_ACCESS",
    "LULYNX_TRAINING_PIPELINE_HOOK_BRIDGE_REPORT",
    "LulynxEventPipelineMapping",
    "build_lulynx_training_pipeline_hook_bridge_report",
]
