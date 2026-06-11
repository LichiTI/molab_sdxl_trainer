"""Request-field emission boundary for T-LoRA A/B activation reviews."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_FIELD_NAMES = ("tlora_ab_case_id", "tlora_ab_arm", "adapter_type", "tlora_rank_schedule")


def build_tlora_ab_request_field_emission_contract(
    *,
    activation_review: Mapping[str, Any],
    field_plan: Mapping[str, Any],
) -> dict[str, Any]:
    review = dict(activation_review)
    plan = dict(field_plan)
    field_names = _field_names(plan.get("field_names") or plan.get("request_field_names"))
    sample_payload = dict(plan.get("sample_payload") or plan.get("request_payload") or {})
    blockers: list[str] = []

    if review.get("scorecard") != "tlora_ab_runtime_activation_review_v0":
        blockers.append("unexpected_activation_review")
    if not bool(review.get("runtime_activation_review_ready", review.get("ok", False))):
        blockers.append("activation_review_not_ready")
    if not bool(review.get("activation_review_signed", False)):
        blockers.append("activation_review_not_signed")
    if _unsafe_flags(review, plan):
        blockers.append("unsafe_child_flag")
    if not field_names:
        blockers.append("request_field_names_missing")
    for name in REQUIRED_FIELD_NAMES:
        if name not in field_names:
            blockers.append(f"required_field_missing:{name}")
    if not sample_payload:
        blockers.append("sample_payload_missing")
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

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_request_field_emission_contract_v0",
        "ok": ready,
        "request_field_emission_contract_ready": ready,
        "request_field_inventory_recorded": ready,
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
        "field_names": list(field_names),
        "sample_payload_keys": sorted(sample_payload.keys()),
        "activation_review_ready": bool(review.get("runtime_activation_review_ready", review.get("ok", False))),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "replay T-LoRA request fields through ConfigAdapter before any runtime registration"
            if ready
            else "complete default-off T-LoRA request-field emission contract before request wiring"
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
        "runtime_activation_enabled",
        "runtime_activation_allowed",
        "request_fields_emitted",
        "request_adapter_registered",
        "training_launch_allowed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_tlora_ab_request_field_emission_contract"]
