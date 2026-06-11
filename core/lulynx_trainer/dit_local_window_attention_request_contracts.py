"""Default-off request-field contracts for local-window DiT attention."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


LOCAL_WINDOW_ATTENTION_REQUEST_FIELDS = (
    "dit_local_window_attention_enabled",
    "dit_local_window_attention_grid_h",
    "dit_local_window_attention_grid_w",
    "dit_local_window_attention_window_h",
    "dit_local_window_attention_window_w",
    "dit_local_window_attention_one_sided",
    "dit_local_window_attention_shift_h",
    "dit_local_window_attention_shift_w",
    "dit_local_window_attention_ab_contract",
)


def build_local_window_attention_request_field_emission_contract(
    *,
    activation_review: Mapping[str, Any],
    field_plan: Mapping[str, Any],
) -> dict[str, Any]:
    review = dict(activation_review)
    plan = dict(field_plan)
    field_names = _items(plan.get("field_names") or plan.get("request_field_names"))
    sample_payload = dict(plan.get("sample_payload") or plan.get("request_payload") or {})
    blockers: list[str] = []
    if review.get("scorecard") != "dit_local_window_attention_runtime_activation_review_v0":
        blockers.append("unexpected_activation_review")
    if not bool(review.get("runtime_activation_review_ready", review.get("ok", False))):
        blockers.append("activation_review_not_ready")
    if _unsafe(review, plan):
        blockers.append("unsafe_child_flag")
    if not field_names:
        blockers.append("request_field_names_missing")
    for name in LOCAL_WINDOW_ATTENTION_REQUEST_FIELDS:
        if name not in field_names:
            blockers.append(f"required_field_missing:{name}")
    if not sample_payload:
        blockers.append("sample_payload_missing")
    for name in LOCAL_WINDOW_ATTENTION_REQUEST_FIELDS:
        if sample_payload and name not in sample_payload:
            blockers.append(f"sample_payload_field_missing:{name}")
    if sample_payload.get("dit_local_window_attention_enabled") is not False:
        blockers.append("sample_payload_enabled_must_be_false")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_config_adapter_replay", False)):
        blockers.append("config_adapter_replay_requirement_missing")
    if not bool(plan.get("acknowledge_no_request_fields_emitted", False)):
        blockers.append("request_field_no_emit_ack_missing")
    if plan.get("request_fields_emitted") is not False:
        blockers.append("request_fields_emitted_must_be_false")
    if plan.get("request_adapter_registered") is not False:
        blockers.append("request_adapter_registered_must_be_false")
    if plan.get("trainer_wiring_allowed") is not False:
        blockers.append("trainer_wiring_allowed_must_be_false")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_request_field_emission_contract_v0",
        "ok": ready,
        "request_field_emission_contract_ready": ready,
        "request_field_inventory_recorded": ready,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "field_names": list(field_names),
        "sample_payload_keys": sorted(sample_payload.keys()),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "replay local-window attention request fields through ConfigAdapter before registration planning"
            if ready
            else "complete default-off local-window attention request-field emission contract"
        ),
    }


def build_local_window_attention_request_field_config_adapter_replay(
    *,
    request_field_contract: Mapping[str, Any],
    replay_result: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(request_field_contract)
    replay = dict(replay_result)
    expected = set(_items(contract.get("field_names")))
    parsed = dict(replay.get("parsed_fields") or replay.get("normalized_fields") or {})
    blockers: list[str] = []
    if contract.get("scorecard") != "dit_local_window_attention_request_field_emission_contract_v0":
        blockers.append("unexpected_request_field_contract")
    if not bool(contract.get("request_field_emission_contract_ready", contract.get("ok", False))):
        blockers.append("request_field_contract_not_ready")
    if _unsafe(contract, replay):
        blockers.append("unsafe_child_flag")
    if not bool(replay.get("ok", False)):
        blockers.append("config_adapter_replay_failed")
    if not bool(replay.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(replay.get("dry_run_only", False)):
        blockers.append("dry_run_boundary_missing")
    if not bool(replay.get("request_normalization_preserved", False)):
        blockers.append("request_normalization_not_preserved")
    for name in LOCAL_WINDOW_ATTENTION_REQUEST_FIELDS:
        if name not in expected:
            blockers.append(f"expected_field_missing:{name}")
        if name not in parsed:
            blockers.append(f"parsed_field_missing:{name}")
    if parsed.get("dit_local_window_attention_enabled") is not False:
        blockers.append("parsed_enabled_must_remain_false")
    for name in (
        "dit_local_window_attention_grid_h",
        "dit_local_window_attention_grid_w",
        "dit_local_window_attention_window_h",
        "dit_local_window_attention_window_w",
    ):
        if int(parsed.get(name) or 0) <= 0:
            blockers.append(f"{name}_not_preserved")
    if not str(parsed.get("dit_local_window_attention_ab_contract") or "").strip():
        blockers.append("ab_contract_not_preserved")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_request_field_config_adapter_replay_v0",
        "ok": ready,
        "config_adapter_replay_ready": ready,
        "request_normalization_preserved": ready,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "parsed_field_keys": sorted(parsed.keys()),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare local-window attention request-adapter registration boundary while keeping it default-off"
            if ready
            else "fix local-window attention request normalization replay before registration planning"
        ),
    }


def build_local_window_attention_request_adapter_registration_boundary(
    *,
    config_adapter_replay: Mapping[str, Any],
    registration_plan: Mapping[str, Any],
) -> dict[str, Any]:
    replay = dict(config_adapter_replay)
    plan = dict(registration_plan)
    mapping_fields = _items(plan.get("mapping_fields") or plan.get("request_field_names"))
    blockers: list[str] = []
    if replay.get("scorecard") != "dit_local_window_attention_request_field_config_adapter_replay_v0":
        blockers.append("unexpected_config_adapter_replay")
    if not bool(replay.get("config_adapter_replay_ready", replay.get("ok", False))):
        blockers.append("config_adapter_replay_not_ready")
    if _unsafe(replay, plan):
        blockers.append("unsafe_child_flag")
    for name in ("adapter_id", "owner", "registration_scope", "rollback_plan", "activation_policy"):
        if not str(plan.get(name) or "").strip():
            blockers.append(f"{name}_missing")
    if not mapping_fields:
        blockers.append("mapping_fields_missing")
    for name in LOCAL_WINDOW_ATTENTION_REQUEST_FIELDS:
        if name not in mapping_fields:
            blockers.append(f"mapping_field_missing:{name}")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_later_request_submission_contract", False)):
        blockers.append("requires_later_request_submission_contract_missing")
    if not bool(plan.get("acknowledge_no_request_adapter_registration", False)):
        blockers.append("request_adapter_no_registration_ack_missing")
    if plan.get("request_adapter_registered") is not False:
        blockers.append("request_adapter_registered_must_be_false")
    if plan.get("registration_applied") is not False:
        blockers.append("registration_applied_must_be_false")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_request_adapter_registration_boundary_v0",
        "ok": ready,
        "request_adapter_registration_boundary_ready": ready,
        "registration_inventory_recorded": ready,
        "request_adapter_registered": False,
        "registration_applied": False,
        "request_fields_emitted": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "adapter_id": str(plan.get("adapter_id") or ""),
        "mapping_fields": list(mapping_fields),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold local-window attention before request submission until an explicit route decision"
            if ready
            else "complete default-off local-window attention request-adapter registration boundary"
        ),
    }


def _items(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _unsafe(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "request_fields_emitted",
        "request_adapter_registered",
        "registration_applied",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "ab_dispatch_allowed",
        "ab_execution_allowed",
        "training_launch_allowed",
        "runs_dispatched",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "LOCAL_WINDOW_ATTENTION_REQUEST_FIELDS",
    "build_local_window_attention_request_adapter_registration_boundary",
    "build_local_window_attention_request_field_config_adapter_replay",
    "build_local_window_attention_request_field_emission_contract",
]
