"""Request-adapter registration boundary for T-LoRA A/B request fields."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_MAPPING_FIELDS = ("tlora_ab_case_id", "tlora_ab_arm", "adapter_type", "tlora_rank_schedule")


def build_tlora_ab_request_adapter_registration_boundary(
    *,
    config_adapter_replay: Mapping[str, Any],
    registration_plan: Mapping[str, Any],
) -> dict[str, Any]:
    replay = dict(config_adapter_replay)
    plan = dict(registration_plan)
    mapping_fields = _field_names(plan.get("mapping_fields") or plan.get("request_field_names"))
    blockers: list[str] = []

    if replay.get("scorecard") != "tlora_ab_request_field_config_adapter_replay_v0":
        blockers.append("unexpected_config_adapter_replay")
    if not bool(replay.get("config_adapter_replay_ready", replay.get("ok", False))):
        blockers.append("config_adapter_replay_not_ready")
    if _unsafe_flags(replay, plan):
        blockers.append("unsafe_child_flag")
    if not str(plan.get("adapter_id") or "").strip():
        blockers.append("adapter_id_missing")
    if not str(plan.get("owner") or "").strip():
        blockers.append("owner_missing")
    if not str(plan.get("registration_scope") or "").strip():
        blockers.append("registration_scope_missing")
    if not str(plan.get("rollback_plan") or "").strip():
        blockers.append("rollback_plan_missing")
    if not str(plan.get("activation_policy") or "").strip():
        blockers.append("activation_policy_missing")
    if not mapping_fields:
        blockers.append("mapping_fields_missing")
    for name in REQUIRED_MAPPING_FIELDS:
        if name not in mapping_fields:
            blockers.append(f"mapping_field_missing:{name}")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_later_request_submission_contract", False)):
        blockers.append("later_request_submission_contract_missing")
    if not bool(plan.get("acknowledge_no_request_adapter_registration", False)):
        blockers.append("request_adapter_no_registration_ack_missing")
    if plan.get("request_adapter_registered") is not False:
        blockers.append("request_adapter_registered_must_be_false")
    if plan.get("registration_applied") is not False:
        blockers.append("registration_applied_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_request_adapter_registration_boundary_v0",
        "ok": ready,
        "request_adapter_registration_boundary_ready": ready,
        "registration_inventory_recorded": ready,
        "request_adapter_registered": False,
        "registration_applied": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "adapter_id": str(plan.get("adapter_id") or ""),
        "mapping_fields": list(mapping_fields),
        "config_adapter_replay_ready": bool(replay.get("config_adapter_replay_ready", replay.get("ok", False))),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare request submission boundary while keeping T-LoRA request adapter unregistered"
            if ready
            else "complete default-off T-LoRA request-adapter registration boundary"
        ),
    }


def _field_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "request_fields_emitted",
        "request_adapter_registered",
        "registration_applied",
        "runtime_activation_enabled",
        "runtime_activation_allowed",
        "training_launch_allowed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_tlora_ab_request_adapter_registration_boundary"]
