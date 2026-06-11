"""Default-off optimizer-state contract for CDM-QTA LoRA training."""

from __future__ import annotations

from typing import Any, Mapping


def build_cdm_qta_lora_optimizer_state_contract(
    *,
    probe_scorecard: Mapping[str, Any],
    optimizer_capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scorecard = dict(probe_scorecard)
    capability = dict(optimizer_capability or {})
    config = dict(scorecard.get("config") or {})
    blockers: list[str] = []

    if scorecard.get("scorecard") != "cdm_qta_lora_quant_train_probe_v0":
        blockers.append("unexpected_probe_scorecard")
    if not bool(scorecard.get("probe_ready", scorecard.get("ok", False))):
        blockers.append("probe_not_ready")
    if _unsafe_flags(scorecard, capability):
        blockers.append("unsafe_child_flag")
    if not bool(config.get("enabled", False)):
        blockers.append("qta_probe_not_enabled")
    if int(config.get("quant_bits") or 0) not in {4, 8}:
        blockers.append("unsupported_quant_bits")
    if not bool(capability.get("fp32_master_params", False)):
        blockers.append("fp32_master_params_missing")
    if not bool(capability.get("optimizer_state_tracks_master_params", False)):
        blockers.append("optimizer_state_master_tracking_missing")
    if not bool(capability.get("state_dict_roundtrip_supported", False)):
        blockers.append("state_dict_roundtrip_missing")
    if not bool(capability.get("resume_parity_required", False)):
        blockers.append("resume_parity_requirement_missing")
    if bool(capability.get("quantized_optimizer_state_required", False)):
        blockers.append("quantized_optimizer_state_not_allowed")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "cdm_qta_lora_optimizer_state_contract_v0",
        "ok": ready,
        "optimizer_state_contract_ready": ready,
        "runtime_optimizer_state_integration_allowed": False,
        "quant_bits": int(config.get("quant_bits") or 0),
        "rank": int(config.get("rank") or 0),
        "optimizer_state_source": "fp32_lora_master_params",
        "forward_weight_source": "fake_quantized_lora_branch",
        "state_requirements": {
            "fp32_master_params": bool(capability.get("fp32_master_params", False)),
            "optimizer_state_tracks_master_params": bool(
                capability.get("optimizer_state_tracks_master_params", False)
            ),
            "state_dict_roundtrip_supported": bool(capability.get("state_dict_roundtrip_supported", False)),
            "resume_parity_required": bool(capability.get("resume_parity_required", False)),
            "quantized_optimizer_state_required": False,
        },
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
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
            "run CDM-QTA LoRA optimizer state roundtrip and resume parity evidence"
            if ready
            else "complete CDM-QTA LoRA optimizer-state capability contract"
        ),
    }


def build_cdm_qta_lora_optimizer_state_parity_gate(
    *,
    optimizer_state_contract: Mapping[str, Any],
    parity_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(optimizer_state_contract)
    report = dict(parity_report or {})
    blockers: list[str] = []

    if contract.get("scorecard") != "cdm_qta_lora_optimizer_state_contract_v0":
        blockers.append("unexpected_optimizer_state_contract")
    if not bool(contract.get("optimizer_state_contract_ready", contract.get("ok", False))):
        blockers.append("optimizer_state_contract_not_ready")
    if _unsafe_flags(contract, report):
        blockers.append("unsafe_child_flag")
    if not bool(report.get("state_dict_roundtrip_passed", False)):
        blockers.append("state_dict_roundtrip_not_passed")
    if not bool(report.get("resume_next_step_parity_passed", False)):
        blockers.append("resume_next_step_parity_not_passed")
    if not bool(report.get("fp32_master_params_preserved", False)):
        blockers.append("fp32_master_params_not_preserved")
    if not bool(report.get("quantized_forward_rebuilt", False)):
        blockers.append("quantized_forward_not_rebuilt")
    if float(report.get("max_loss_delta", 0.0) or 0.0) > float(report.get("max_allowed_loss_delta", 1e-6) or 1e-6):
        blockers.append("loss_delta_above_threshold")
    if float(report.get("max_param_delta", 0.0) or 0.0) > float(report.get("max_allowed_param_delta", 1e-6) or 1e-6):
        blockers.append("param_delta_above_threshold")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "cdm_qta_lora_optimizer_state_parity_gate_v0",
        "ok": ready,
        "optimizer_state_parity_ready": ready,
        "optimizer_state_contract_ready": bool(contract.get("optimizer_state_contract_ready", contract.get("ok", False))),
        "max_loss_delta": float(report.get("max_loss_delta", 0.0) or 0.0),
        "max_param_delta": float(report.get("max_param_delta", 0.0) or 0.0),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
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
            "feed optimizer-state parity evidence into CDM-QTA LoRA A/B result review"
            if ready
            else "fix CDM-QTA LoRA optimizer-state parity blockers"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "request_fields_emitted",
        "request_adapter_registered",
        "ab_dispatch_allowed",
        "training_launch_allowed",
        "training_launch_executed",
        "runs_dispatched",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_cdm_qta_lora_optimizer_state_contract",
    "build_cdm_qta_lora_optimizer_state_parity_gate",
]
