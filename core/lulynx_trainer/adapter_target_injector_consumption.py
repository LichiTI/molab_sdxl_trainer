"""Default-off injector-consumption bridge for profiled adapter targets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_adapter_target_injector_consumption_plan(
    *,
    trainer_preflight: Mapping[str, Any],
    injector_capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preflight = dict(trainer_preflight)
    capability = dict(injector_capability or {})
    target_modules = tuple(str(item) for item in preflight.get("target_modules", ()) if str(item).strip())
    rank_keys = tuple(str(item) for item in preflight.get("rank_map_keys", ()) if str(item).strip())
    rank_map = _rank_map(capability.get("rank_map"), rank_keys)
    supported_modules = {str(item) for item in capability.get("supported_target_modules", ()) if str(item)}
    blockers: list[str] = []

    if preflight.get("scorecard") != "adapter_target_trainer_preflight_v0":
        blockers.append("unexpected_trainer_preflight")
    if not bool(preflight.get("preflight_ready", preflight.get("ok", False))):
        blockers.append("trainer_preflight_not_ready")
    if _unsafe_flags(preflight, capability):
        blockers.append("unsafe_child_flag")
    if not target_modules:
        blockers.append("target_modules_missing")
    if not rank_map:
        blockers.append("rank_map_missing")
    if supported_modules:
        unsupported = sorted(set(target_modules) - supported_modules)
        blockers.extend(f"unsupported_target_module:{name}" for name in unsupported)
    if not bool(capability.get("metadata_writer_available", False)):
        blockers.append("metadata_writer_missing")
    if not bool(capability.get("merge_refusal_available", False)):
        blockers.append("merge_refusal_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_injector_consumption_plan_v0",
        "ok": ready,
        "injector_consumption_plan_ready": ready,
        "real_injector_consumption_allowed": False,
        "injector_contract": str(preflight.get("injector_contract") or ""),
        "target_modules": list(target_modules),
        "rank_map": rank_map,
        "selected_count": int(preflight.get("selected_count") or len(target_modules)),
        "metadata_stamp": {
            "adapter_target_policy": "profiled",
            "selected_modules": list(target_modules),
            "rank_map": rank_map,
            "injector_contract": str(preflight.get("injector_contract") or ""),
            "merge_policy": "refuse_merge_without_matching_adapter_target_metadata",
        },
        "merge_policy": {
            "merge_allowed": False,
            "requires_matching_metadata": True,
            "fallback_to_all_targets_allowed": False,
        },
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run selected-target injector execution audit against real trainer evidence"
            if ready
            else "complete adapter-target injector capability contract before consumption"
        ),
    }


def build_adapter_target_injector_consumption_audit(
    *,
    consumption_plan: Mapping[str, Any],
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(consumption_plan)
    payload = dict(observed or {})
    expected_modules = tuple(str(item) for item in plan.get("target_modules", ()) if str(item).strip())
    observed_modules = tuple(str(item) for item in payload.get("injected_modules", ()) if str(item).strip())
    expected_rank_map = {str(key): int(value) for key, value in dict(plan.get("rank_map") or {}).items()}
    observed_rank_map = {str(key): int(value) for key, value in dict(payload.get("rank_map") or {}).items()}
    blockers: list[str] = []

    if plan.get("scorecard") != "adapter_target_injector_consumption_plan_v0":
        blockers.append("unexpected_consumption_plan")
    if not bool(plan.get("injector_consumption_plan_ready", plan.get("ok", False))):
        blockers.append("consumption_plan_not_ready")
    if _unsafe_flags(plan, payload):
        blockers.append("unsafe_child_flag")
    if tuple(sorted(expected_modules)) != tuple(sorted(observed_modules)):
        blockers.append("injected_modules_mismatch")
    if expected_rank_map != observed_rank_map:
        blockers.append("rank_map_mismatch")
    if not bool(payload.get("metadata_stamp_observed", False)):
        blockers.append("metadata_stamp_missing")
    if not bool(payload.get("merge_policy_observed", False)):
        blockers.append("merge_policy_missing")
    if bool(payload.get("fallback_to_all_targets_used", False)):
        blockers.append("fallback_to_all_targets_used")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_injector_consumption_audit_v0",
        "ok": ready,
        "injector_consumption_audit_ready": ready,
        "selected_count": int(plan.get("selected_count") or len(expected_modules)),
        "observed_module_count": len(observed_modules),
        "metadata_stamp_observed": bool(payload.get("metadata_stamp_observed", False)),
        "merge_policy_observed": bool(payload.get("merge_policy_observed", False)),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "feed selected-target execution audit and quality comparison into review gate"
            if ready
            else "fix selected-target injector consumption evidence before quality review"
        ),
    }


def _rank_map(value: Any, fallback_keys: Sequence[str]) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(key): max(int(rank), 1) for key, rank in value.items() if str(key).strip()}
    return {key: 1 for key in fallback_keys}


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "request_fields_emitted",
        "request_adapter_registered",
        "training_launch_allowed",
        "training_launch_executed",
        "runs_dispatched",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enabled",
        "fallback_to_all_targets_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_adapter_target_injector_consumption_audit",
    "build_adapter_target_injector_consumption_plan",
]
