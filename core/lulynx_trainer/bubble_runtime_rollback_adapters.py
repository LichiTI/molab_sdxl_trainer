"""Rollback adapters for bubble-aware current-run mutations."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .dataloader_rebuild_runtime import build_dataloader_rebuild_plan


RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID = "runtime_config_overlay_v0"
TRANSFER_RUNTIME_ROLLBACK_ADAPTER_ID = "transfer_runtime_overlay_v0"
TRANSFER_PREFETCH_NEXT_REQUEST_ADAPTER_ID = "transfer_prefetch_next_request_v0"
DATALOADER_REBUILD_RUNTIME_CONTRACT_ID = "dataloader_rebuild_runtime_contract_v0"

LOW_RISK_RUNTIME_ACTIONS = {
    "disable_sync_profiler_mode",
    "increase_logging_interval",
    "move_validation_after_training_window",
    "reduce_hot_path_sync",
}

LOW_RISK_RUNTIME_PATHS = {
    "adaptive_step_logging_enabled",
    "data_transfer_profile_mode",
    "eval_every_n_steps",
    "layer_monitor_interval",
    "step_phase_profile_enabled",
    "tensorboard_flush_interval_steps",
}

TRANSFER_PREFETCH_ACTIONS = {
    "enable_block_prefetch",
    "increase_block_prefetch_depth",
    "profile_transfer_path",
}

TRANSFER_RUNTIME_ACTIONS = {
    "enable_non_blocking_transfer",
}

TRANSFER_RUNTIME_PATHS = {
    "data_transfer_non_blocking",
}

DATALOADER_REBUILD_ACTIONS = {
    "enable_pin_memory",
    "set_dataloader_workers",
    "set_dataloader_prefetch_factor",
    "enable_persistent_workers",
    "recommend_cache_first",
}

DATALOADER_REBUILD_HANDLE_BLOCKERS = {
    "epoch_boundary_or_safe_step_pause": "missing_epoch_boundary_handle",
    "active_iterator_drain": "missing_iterator_drain_handle",
    "worker_shutdown_and_join": "missing_worker_shutdown_handle",
    "dataloader_rebuild_factory": "missing_dataloader_rebuild_handle",
    "ddp_sampler_rewrap_if_needed": "missing_ddp_sampler_rewrap_handle",
    "rollback_rebuild_factory": "missing_rollback_rebuild_handle",
}


def runtime_action_support(
    action_plan: Mapping[str, Any],
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    domain = str(action_plan.get("domain") or "")
    action_kind = str(action_plan.get("action_kind") or "")
    if domain == "transfer_offload" and action_kind in TRANSFER_RUNTIME_ACTIONS:
        return transfer_runtime_action_support(action_plan)
    if (
        (domain == "transfer_offload" and action_kind == "enable_pin_memory")
        or (domain == "data_supply" and action_kind in DATALOADER_REBUILD_ACTIONS)
    ):
        return dataloader_rebuild_runtime_boundary(action_plan, runtime_context=runtime_context)
    if domain == "transfer_offload" and action_kind in TRANSFER_PREFETCH_ACTIONS:
        return transfer_prefetch_next_request_boundary(action_plan)
    supported = domain == "host_scheduling" and action_kind in LOW_RISK_RUNTIME_ACTIONS
    return {
        "adapter_id": RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID,
        "domain": domain,
        "action_kind": action_kind,
        "supported": supported,
        "reason": "low_risk_host_scheduling" if supported else "action_not_low_risk",
        "blocked_reasons": [] if supported else ["action_not_low_risk"],
    }


def dataloader_rebuild_runtime_boundary(
    action_plan: Mapping[str, Any],
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    mutations = _mutation_list(action_plan)
    restore = dict(_mapping(_mapping(action_plan.get("rollback")).get("restore")))
    readiness = dict(_mapping(_mapping(runtime_context).get("dataloader_rebuild_readiness")))
    rebuild_plan = build_dataloader_rebuild_plan(action_plan, readiness)
    available_handles = _string_list(readiness.get("available_runtime_handles"))
    missing_handles = _string_list(readiness.get("missing_runtime_handles"))
    if not readiness:
        missing_handles = list(DATALOADER_REBUILD_HANDLE_BLOCKERS)
    blocked_reasons = [
        "dataloader_rebuild_runtime_contract_missing",
        "missing_current_run_rollback_adapter",
    ]
    if missing_handles:
        blocked_reasons.extend(
            reason
            for handle, reason in DATALOADER_REBUILD_HANDLE_BLOCKERS.items()
            if handle in missing_handles
        )
    if rebuild_plan.get("unsupported_mutation_paths"):
        blocked_reasons.append("unsupported_dataloader_rebuild_mutation")
    return {
        "adapter_id": DATALOADER_REBUILD_RUNTIME_CONTRACT_ID,
        "domain": str(action_plan.get("domain") or "data_supply"),
        "action_kind": str(action_plan.get("action_kind") or ""),
        "supported": False,
        "next_request_only": True,
        "current_run_reversible": False,
        "current_run_rebuild_ready": bool(readiness.get("current_run_rebuild_ready", False))
        and bool(rebuild_plan.get("current_run_reversible", False)),
        "reason": "dataloader_rebuild_requires_runtime_handle_contract",
        "blocked_reasons": _unique_strings(blocked_reasons),
        "runtime_rebuild_readiness": readiness,
        "runtime_rebuild_plan": rebuild_plan,
        "available_runtime_handles": available_handles,
        "missing_runtime_handles": missing_handles,
        "mutation_paths": [
            str(item.get("path") or "")
            for item in mutations
            if item.get("path") is not None
        ],
        "restore_paths": sorted(str(path) for path in restore),
        "required_runtime_handles": [
            "epoch_boundary_or_safe_step_pause",
            "active_iterator_drain",
            "worker_shutdown_and_join",
            "dataloader_rebuild_factory",
            "ddp_sampler_rewrap_if_needed",
            "rollback_rebuild_factory",
        ],
        "required_evidence": [
            "before_after_steady_samples_per_second",
            "data_wait_share_delta",
            "worker_startup_overhead_guard",
            "iterator_state_continuity_guard",
            "loss_regression_guard",
        ],
    }


def transfer_runtime_action_support(action_plan: Mapping[str, Any]) -> dict[str, Any]:
    domain = str(action_plan.get("domain") or "transfer_offload")
    action_kind = str(action_plan.get("action_kind") or "")
    mutations, skipped = filter_transfer_runtime_mutations(action_plan)
    supported = bool(mutations) and action_kind in TRANSFER_RUNTIME_ACTIONS
    blocked = [str(item.get("skip_reason") or "transfer_runtime_mutation_skipped") for item in skipped]
    if not mutations:
        blocked.append("no_low_risk_transfer_runtime_mutations")
    return {
        "adapter_id": TRANSFER_RUNTIME_ROLLBACK_ADAPTER_ID,
        "domain": domain,
        "action_kind": action_kind,
        "supported": supported,
        "reason": "low_risk_transfer_runtime_overlay" if supported else "transfer_runtime_action_not_reversible",
        "current_run_reversible": supported,
        "allowed_paths": sorted(TRANSFER_RUNTIME_PATHS),
        "blocked_reasons": [] if supported else blocked[:8],
        "required_evidence": [
            "before_after_steady_samples_per_second",
            "h2d_transfer_share_delta",
            "loss_regression_guard",
        ],
    }


def transfer_prefetch_next_request_boundary(action_plan: Mapping[str, Any]) -> dict[str, Any]:
    mutations = _mutation_list(action_plan)
    restore = dict(_mapping(_mapping(action_plan.get("rollback")).get("restore")))
    return {
        "adapter_id": TRANSFER_PREFETCH_NEXT_REQUEST_ADAPTER_ID,
        "domain": str(action_plan.get("domain") or "transfer_offload"),
        "action_kind": str(action_plan.get("action_kind") or ""),
        "supported": False,
        "next_request_only": True,
        "current_run_reversible": False,
        "reason": "transfer_prefetch_requires_next_request_ab_evidence",
        "blocked_reasons": [
            "transfer_prefetch_next_request_only",
            "missing_current_run_rollback_adapter",
        ],
        "mutation_paths": [
            str(item.get("path") or "")
            for item in mutations
            if item.get("path") is not None
        ],
        "restore_paths": sorted(str(path) for path in restore),
        "required_evidence": [
            "before_after_steady_samples_per_second",
            "h2d_transfer_share_delta",
            "peak_vram_ratio_guard",
            "loss_regression_guard",
        ],
    }


def build_runtime_apply_candidate(action_plan: Mapping[str, Any]) -> dict[str, Any]:
    support = runtime_action_support(action_plan)
    adapter_id = str(support.get("adapter_id") or RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID)
    mutations, skipped = filter_runtime_apply_mutations(action_plan)
    overlay = {str(item["path"]): item.get("recommended") for item in mutations}
    rollback_restore = {str(item["path"]): item.get("current") for item in mutations}
    return {
        "adapter_id": adapter_id,
        "action_id": stable_runtime_action_id(action_plan, mutations, adapter_id=adapter_id),
        "mutations": mutations,
        "skipped_mutations": skipped,
        "applied_overlay": overlay,
        "rollback_restore": rollback_restore,
        "rollback_adapter": {
            "adapter_id": adapter_id,
            "restore_paths": sorted(rollback_restore),
            "current_run_reversible": True,
        },
    }


def finalize_runtime_apply_candidate(
    candidate: Mapping[str, Any],
    *,
    step: int,
    cooldown_steps: int,
    before_metrics: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
) -> dict[str, Any]:
    cooldown_until = int(step) + max(int(cooldown_steps or 1), 1)
    profiler_handoff = _profiler_data_wait_handoff(
        before_metrics=before_metrics,
        diagnosis=diagnosis,
        step=step,
    )
    return {
        "adapter_id": str(candidate.get("adapter_id") or RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID),
        "action_id": str(candidate.get("action_id") or ""),
        "status": "pending_apply",
        "step": int(step),
        "cooldown_until_step": cooldown_until,
        "mutations": [dict(item) for item in _mapping_list(candidate.get("mutations"))],
        "skipped_mutations": [dict(item) for item in _mapping_list(candidate.get("skipped_mutations"))],
        "applied_overlay": dict(_mapping(candidate.get("applied_overlay"))),
        "rollback_restore": dict(_mapping(candidate.get("rollback_restore"))),
        "rollback_adapter": dict(_mapping(candidate.get("rollback_adapter"))),
        "before_metrics": dict(before_metrics),
        "diagnosis_kind": str(diagnosis.get("kind") or ""),
        "profiler_handoff": profiler_handoff,
    }


def build_runtime_rollback_plan(rollback_restore: Mapping[str, Any]) -> dict[str, Any]:
    restore = dict(_mapping(rollback_restore))
    adapter_id = _adapter_id_for_restore(restore)
    allowed_paths = _allowed_paths_for_adapter(adapter_id)
    mutations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path, value in restore.items():
        path_text = str(path)
        mutation = {
            "op": "set",
            "path": path_text,
            "current": None,
            "recommended": value,
            "reason": "restore pre-action value",
        }
        if path_text in allowed_paths:
            mutations.append(mutation)
        else:
            skipped.append({**mutation, "skip_reason": "not_low_risk_current_run_path"})
    return {
        "adapter_id": adapter_id,
        "restore": restore,
        "mutations": mutations,
        "skipped_restore_mutations": skipped,
        "current_run_reversible": bool(mutations) and not skipped,
    }


def filter_runtime_apply_mutations(action_plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support = runtime_action_support(action_plan)
    allowed_paths = _allowed_paths_for_adapter(str(support.get("adapter_id") or RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID))
    return _filter_apply_mutations(action_plan, allowed_paths=allowed_paths)


def filter_transfer_runtime_mutations(action_plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _filter_apply_mutations(action_plan, allowed_paths=TRANSFER_RUNTIME_PATHS)


def _filter_apply_mutations(
    action_plan: Mapping[str, Any],
    *,
    allowed_paths: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    mutations = action_plan.get("mutations", [])
    if not isinstance(mutations, list):
        return allowed, [{"skip_reason": "mutations_not_list"}]
    for item in mutations:
        mutation = dict(item) if isinstance(item, Mapping) else {}
        if mutation.get("op") != "set":
            skipped.append({**mutation, "skip_reason": "unsupported_mutation_op"})
            continue
        path = str(mutation.get("path") or "")
        if path not in allowed_paths:
            skipped.append({**mutation, "skip_reason": "not_low_risk_current_run_path"})
            continue
        allowed.append(mutation)
    return allowed, skipped


def stable_runtime_action_id(
    action_plan: Mapping[str, Any],
    mutations: list[Mapping[str, Any]],
    *,
    adapter_id: str = RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID,
) -> str:
    payload = {
        "adapter_id": adapter_id,
        "phase": action_plan.get("phase"),
        "domain": action_plan.get("domain"),
        "action_kind": action_plan.get("action_kind"),
        "mutations": [
            {"path": item.get("path"), "recommended": item.get("recommended")}
            for item in mutations
        ],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"bubble-runtime-{digest}"


def _adapter_id_for_restore(restore: Mapping[str, Any]) -> str:
    paths = {str(path) for path in restore}
    if paths and paths.issubset(TRANSFER_RUNTIME_PATHS):
        return TRANSFER_RUNTIME_ROLLBACK_ADAPTER_ID
    return RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID


def _allowed_paths_for_adapter(adapter_id: str) -> set[str]:
    if adapter_id == TRANSFER_RUNTIME_ROLLBACK_ADAPTER_ID:
        return set(TRANSFER_RUNTIME_PATHS)
    return set(LOW_RISK_RUNTIME_PATHS)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _profiler_data_wait_handoff(
    *,
    before_metrics: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    action = _mapping(diagnosis.get("recommended_action"))
    if str(action.get("kind") or "") != "disable_sync_profiler_mode":
        return {}
    data_wait = _safe_float(before_metrics.get("data_wait_share"))
    if data_wait < 0.08:
        return {}
    return {
        "schema_version": 1,
        "kind": "data_wait_after_sync_profiler_disable_v0",
        "source_action_kind": "disable_sync_profiler_mode",
        "recommended_action_kind": "set_dataloader_workers",
        "observed_step": _safe_int(step),
        "data_wait_share": _round(data_wait),
        "h2d_transfer_share": _round(before_metrics.get("h2d_transfer_share")),
        "optimizer_share": _round(before_metrics.get("optimizer_share")),
        "host_gap_share": _round(before_metrics.get("host_gap_share")),
        "logging_checkpoint_share": _round(before_metrics.get("logging_checkpoint_share")),
        "steady_samples_per_second": _round(before_metrics.get("steady_samples_per_second")),
    }


def _mutation_list(action_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in action_plan.get("mutations", [])
        if isinstance(item, Mapping)
    ] if isinstance(action_plan.get("mutations", []), list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = [
    "DATALOADER_REBUILD_ACTIONS",
    "DATALOADER_REBUILD_HANDLE_BLOCKERS",
    "DATALOADER_REBUILD_RUNTIME_CONTRACT_ID",
    "LOW_RISK_RUNTIME_ACTIONS",
    "LOW_RISK_RUNTIME_PATHS",
    "RUNTIME_CONFIG_ROLLBACK_ADAPTER_ID",
    "TRANSFER_PREFETCH_ACTIONS",
    "TRANSFER_PREFETCH_NEXT_REQUEST_ADAPTER_ID",
    "TRANSFER_RUNTIME_ACTIONS",
    "TRANSFER_RUNTIME_PATHS",
    "TRANSFER_RUNTIME_ROLLBACK_ADAPTER_ID",
    "build_runtime_apply_candidate",
    "build_runtime_rollback_plan",
    "dataloader_rebuild_runtime_boundary",
    "filter_runtime_apply_mutations",
    "filter_transfer_runtime_mutations",
    "finalize_runtime_apply_candidate",
    "runtime_action_support",
    "stable_runtime_action_id",
    "transfer_runtime_action_support",
    "transfer_prefetch_next_request_boundary",
]
