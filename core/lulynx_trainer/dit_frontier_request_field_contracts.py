"""Default-off request-field boundaries for DiT frontier features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ALLOWED_POLICIES = frozenset({"manual_selected", "quality_gated", "ab_passed"})


@dataclass(frozen=True)
class DiTFrontierRequestFieldSpec:
    feature_id: str
    label: str
    runtime_review_scorecard: str
    emission_scorecard: str
    replay_scorecard: str
    required_fields: tuple[str, ...]
    policy_field: str
    detail_field: str
    thresholds_field: str
    contract_field: str


FRONTIER_REQUEST_FIELD_SPECS: dict[str, DiTFrontierRequestFieldSpec] = {
    "sra2_haste": DiTFrontierRequestFieldSpec(
        feature_id="sra2_haste",
        label="SRA2/HASTE",
        runtime_review_scorecard="sra2_haste_runtime_activation_review_v0",
        emission_scorecard="sra2_haste_request_field_emission_contract_v0",
        replay_scorecard="sra2_haste_request_field_config_adapter_replay_v0",
        required_fields=(
            "sra2_haste_policy",
            "sra2_haste_layers",
            "sra2_haste_thresholds",
            "sra2_haste_ab_contract",
        ),
        policy_field="sra2_haste_policy",
        detail_field="sra2_haste_layers",
        thresholds_field="sra2_haste_thresholds",
        contract_field="sra2_haste_ab_contract",
    ),
    "cdm_qta_lora": DiTFrontierRequestFieldSpec(
        feature_id="cdm_qta_lora",
        label="CDM-QTA LoRA",
        runtime_review_scorecard="cdm_qta_lora_runtime_activation_review_v0",
        emission_scorecard="cdm_qta_lora_request_field_emission_contract_v0",
        replay_scorecard="cdm_qta_lora_request_field_config_adapter_replay_v0",
        required_fields=(
            "cdm_qta_lora_policy",
            "cdm_qta_lora_quant_bits",
            "cdm_qta_lora_thresholds",
            "cdm_qta_lora_ab_contract",
        ),
        policy_field="cdm_qta_lora_policy",
        detail_field="cdm_qta_lora_quant_bits",
        thresholds_field="cdm_qta_lora_thresholds",
        contract_field="cdm_qta_lora_ab_contract",
    ),
    "diffcr": DiTFrontierRequestFieldSpec(
        feature_id="diffcr",
        label="DiffCR",
        runtime_review_scorecard="diffcr_runtime_activation_review_v0",
        emission_scorecard="diffcr_request_field_emission_contract_v0",
        replay_scorecard="diffcr_request_field_config_adapter_replay_v0",
        required_fields=(
            "diffcr_policy",
            "diffcr_compression_plan",
            "diffcr_thresholds",
            "diffcr_ab_contract",
        ),
        policy_field="diffcr_policy",
        detail_field="diffcr_compression_plan",
        thresholds_field="diffcr_thresholds",
        contract_field="diffcr_ab_contract",
    ),
    "dit_blockskip": DiTFrontierRequestFieldSpec(
        feature_id="dit_blockskip",
        label="DiT-BlockSkip",
        runtime_review_scorecard="dit_blockskip_runtime_activation_review_v0",
        emission_scorecard="dit_blockskip_request_field_emission_contract_v0",
        replay_scorecard="dit_blockskip_request_field_config_adapter_replay_v0",
        required_fields=(
            "dit_blockskip_policy",
            "dit_blockskip_schedule",
            "dit_blockskip_thresholds",
            "dit_blockskip_ab_contract",
        ),
        policy_field="dit_blockskip_policy",
        detail_field="dit_blockskip_schedule",
        thresholds_field="dit_blockskip_thresholds",
        contract_field="dit_blockskip_ab_contract",
    ),
    "dit_local_window_attention": DiTFrontierRequestFieldSpec(
        feature_id="dit_local_window_attention",
        label="DiT local/window attention",
        runtime_review_scorecard="dit_local_window_attention_runtime_activation_review_v0",
        emission_scorecard="dit_local_window_attention_request_field_emission_contract_v0",
        replay_scorecard="dit_local_window_attention_request_field_config_adapter_replay_v0",
        required_fields=(
            "dit_local_window_attention_enabled",
            "dit_local_window_attention_grid_h",
            "dit_local_window_attention_grid_w",
            "dit_local_window_attention_window_h",
            "dit_local_window_attention_window_w",
            "dit_local_window_attention_one_sided",
            "dit_local_window_attention_shift_h",
            "dit_local_window_attention_shift_w",
            "dit_local_window_attention_ab_contract",
        ),
        policy_field="dit_local_window_attention_enabled",
        detail_field="dit_local_window_attention_window_h",
        thresholds_field="dit_local_window_attention_window_w",
        contract_field="dit_local_window_attention_ab_contract",
    ),
}


def build_dit_frontier_request_field_emission_contract(
    *,
    feature_id: str,
    activation_review: Mapping[str, Any],
    field_plan: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    review = dict(activation_review)
    plan = dict(field_plan)
    field_names = _items(plan.get("field_names") or plan.get("request_field_names"))
    sample_payload = dict(plan.get("sample_payload") or plan.get("request_payload") or {})
    blockers: list[str] = []

    if review.get("scorecard") != spec.runtime_review_scorecard:
        blockers.append("unexpected_activation_review")
    if not bool(review.get("runtime_activation_review_ready", review.get("ok", False))):
        blockers.append("activation_review_not_ready")
    if not bool(review.get("activation_review_signed", False)):
        blockers.append("activation_review_not_signed")
    if _unsafe_flags(review, plan):
        blockers.append("unsafe_child_flag")
    if not field_names:
        blockers.append("request_field_names_missing")
    if not sample_payload:
        blockers.append("sample_payload_missing")
    for name in spec.required_fields:
        if name not in field_names:
            blockers.append(f"required_field_missing:{name}")
        if sample_payload and name not in sample_payload:
            blockers.append(f"sample_payload_field_missing:{name}")
    if not bool(plan.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(plan.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(plan.get("requires_config_adapter_replay", False)):
        blockers.append("config_adapter_replay_requirement_missing")
    if not bool(plan.get("acknowledge_no_request_fields_emitted", False)):
        blockers.append("request_field_no_emit_ack_missing")
    for key in ("request_fields_emitted", "request_adapter_registered", "trainer_wiring_allowed"):
        if plan.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": spec.emission_scorecard,
        "feature_id": spec.feature_id,
        "ok": ready,
        "request_field_emission_contract_ready": ready,
        "request_field_inventory_recorded": ready,
        "required_fields": list(spec.required_fields),
        "field_names": list(field_names),
        "sample_payload_keys": sorted(sample_payload.keys()),
        "passed_case_count": int(review.get("passed_case_count") or 0),
        "passed_cases": list(review.get("passed_cases", ()) or ()),
        "activation_review_ready": bool(review.get("runtime_activation_review_ready", review.get("ok", False))),
        **_default_off_flags(),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"replay {spec.label} request fields through ConfigAdapter before any request-adapter registration"
            if ready
            else f"complete default-off {spec.label} request-field emission contract before request wiring"
        ),
    }


def build_dit_frontier_request_field_config_adapter_replay(
    *,
    feature_id: str,
    request_field_contract: Mapping[str, Any],
    replay_result: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _spec(feature_id)
    contract = dict(request_field_contract)
    replay = dict(replay_result)
    expected_fields = set(_items(contract.get("field_names")))
    parsed = dict(replay.get("parsed_fields") or replay.get("normalized_fields") or {})
    thresholds = parsed.get(spec.thresholds_field)
    blockers: list[str] = []

    if contract.get("scorecard") != spec.emission_scorecard:
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
    for name in spec.required_fields:
        if name not in expected_fields:
            blockers.append(f"expected_field_missing:{name}")
        if name not in parsed:
            blockers.append(f"parsed_field_missing:{name}")
    if not _policy_preserved(spec, parsed):
        blockers.append(f"{spec.policy_field}_not_preserved")
    if not _value_present(parsed.get(spec.detail_field)):
        blockers.append(f"{spec.detail_field}_not_preserved")
    if not _thresholds_preserved(spec, thresholds):
        blockers.append(f"{spec.thresholds_field}_not_preserved")
    if not str(parsed.get(spec.contract_field) or "").strip():
        blockers.append(f"{spec.contract_field}_not_preserved")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": spec.replay_scorecard,
        "feature_id": spec.feature_id,
        "ok": ready,
        "config_adapter_replay_ready": ready,
        "request_normalization_preserved": ready,
        "required_fields": list(spec.required_fields),
        "parsed_field_keys": sorted(parsed.keys()),
        "parsed_threshold_keys": sorted(thresholds.keys()) if isinstance(thresholds, Mapping) else [],
        **_default_off_flags(),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            f"prepare request-adapter registration boundary while keeping {spec.label} default-off"
            if ready
            else f"fix {spec.label} request normalization replay before request-adapter registration"
        ),
    }


def supported_dit_frontier_request_field_specs() -> tuple[DiTFrontierRequestFieldSpec, ...]:
    return tuple(FRONTIER_REQUEST_FIELD_SPECS.values())


def _spec(feature_id: str) -> DiTFrontierRequestFieldSpec:
    try:
        return FRONTIER_REQUEST_FIELD_SPECS[feature_id]
    except KeyError as exc:
        raise ValueError(f"unsupported DiT frontier feature: {feature_id}") from exc


def _default_off_flags() -> dict[str, bool]:
    return {
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_allowed": False,
        "runtime_activation_enabled": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "ab_dispatch_allowed": False,
        "ab_execution_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
    }


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
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "ab_execution_allowed",
        "training_launch_allowed",
        "training_launch_executed",
        "runs_dispatched",
        "default_rollout_allowed",
        "auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _items(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _value_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    if isinstance(value, (int, float)):
        return value > 0
    return value is not None


def _policy_preserved(spec: DiTFrontierRequestFieldSpec, parsed: Mapping[str, Any]) -> bool:
    value = parsed.get(spec.policy_field)
    if spec.feature_id == "dit_local_window_attention":
        return value is False
    return str(value or "").strip() in ALLOWED_POLICIES


def _thresholds_preserved(spec: DiTFrontierRequestFieldSpec, value: Any) -> bool:
    if spec.feature_id == "dit_local_window_attention":
        return _value_present(value)
    return isinstance(value, Mapping) and bool(value)


__all__ = [
    "DiTFrontierRequestFieldSpec",
    "FRONTIER_REQUEST_FIELD_SPECS",
    "build_dit_frontier_request_field_config_adapter_replay",
    "build_dit_frontier_request_field_emission_contract",
    "supported_dit_frontier_request_field_specs",
]
