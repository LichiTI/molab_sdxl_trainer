"""ConfigAdapter replay boundary for T-LoRA A/B request-field contracts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_FIELD_NAMES = ("tlora_ab_case_id", "tlora_ab_arm", "adapter_type", "tlora_rank_schedule")


def build_tlora_ab_request_field_config_adapter_replay(
    *,
    request_field_contract: Mapping[str, Any],
    replay_result: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(request_field_contract)
    replay = dict(replay_result)
    expected_fields = set(_field_names(contract.get("field_names")))
    parsed = dict(replay.get("parsed_fields") or replay.get("normalized_fields") or {})
    blockers: list[str] = []

    if contract.get("scorecard") != "tlora_ab_request_field_emission_contract_v0":
        blockers.append("unexpected_request_field_contract")
    if not bool(contract.get("request_field_emission_contract_ready", contract.get("ok", False))):
        blockers.append("request_field_contract_not_ready")
    if _unsafe_flags(contract, replay):
        blockers.append("unsafe_child_flag")
    if not bool(replay.get("ok", False)):
        blockers.append("config_adapter_replay_failed")
    if not bool(replay.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(replay.get("dry_run_only", False)):
        blockers.append("dry_run_boundary_missing")
    if not bool(replay.get("request_normalization_preserved", False)):
        blockers.append("request_normalization_not_preserved")
    for name in REQUIRED_FIELD_NAMES:
        if name not in expected_fields:
            blockers.append(f"expected_field_missing:{name}")
        if name not in parsed:
            blockers.append(f"parsed_field_missing:{name}")
    if str(parsed.get("adapter_type") or parsed.get("lora_type") or "") != "tlora":
        blockers.append("tlora_adapter_type_not_preserved")
    if str(parsed.get("tlora_ab_arm") or "") not in {"tlora", "baseline"}:
        blockers.append("tlora_ab_arm_not_preserved")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_request_field_config_adapter_replay_v0",
        "ok": ready,
        "config_adapter_replay_ready": ready,
        "request_normalization_preserved": ready,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "runtime_activation_enabled": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "parsed_field_keys": sorted(parsed.keys()),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare request-adapter registration boundary while keeping T-LoRA default-off"
            if ready
            else "fix T-LoRA request normalization replay before request-adapter registration"
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
        "runtime_activation_enabled",
        "runtime_activation_allowed",
        "training_launch_allowed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_tlora_ab_request_field_config_adapter_replay"]
