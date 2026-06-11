"""Default-off reconciliation contract for AdaLN/Modulation Guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ModulationGuidanceObservation:
    existing_adaln_bias_route: bool = True
    existing_route_name: str = "adaln_guidance"
    pooled_text_projection_contract: bool = False
    dedicated_distill_loop: bool = False
    inference_projection_contract: bool = False
    save_metadata_stamp: bool = False
    request_config_replay: bool = False
    ui_exposure_requested: bool = False
    default_behavior_changed: bool = False

    def validate(self) -> None:
        if not str(self.existing_route_name or "").strip():
            raise ValueError("existing_route_name must not be empty")


def build_modulation_guidance_metadata(
    observation: ModulationGuidanceObservation | Mapping[str, Any] | None = None,
) -> dict[str, str]:
    obs = _coerce_observation(observation)
    obs.validate()
    return {
        "ss_feature_type": "modulation_guidance",
        "ss_modulation_guidance_reconciliation_version": "1",
        "ss_modulation_guidance_existing_route": obs.existing_route_name,
        "ss_modulation_guidance_existing_adaln_bias_route": _bool_text(obs.existing_adaln_bias_route),
        "ss_modulation_guidance_pooled_text_projection_contract": _bool_text(obs.pooled_text_projection_contract),
        "ss_modulation_guidance_dedicated_distill_loop": _bool_text(obs.dedicated_distill_loop),
        "ss_modulation_guidance_inference_projection_contract": _bool_text(obs.inference_projection_contract),
        "ss_modulation_guidance_save_metadata_stamp": _bool_text(obs.save_metadata_stamp),
        "ss_modulation_guidance_request_config_replay": _bool_text(obs.request_config_replay),
        "ss_ui_exposure_requested": _bool_text(obs.ui_exposure_requested),
        "ss_training_path_enabled": "false",
        "ss_default_behavior_changed": _bool_text(obs.default_behavior_changed),
    }


def build_modulation_guidance_reconciliation(
    observation: ModulationGuidanceObservation | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    obs = _coerce_observation(observation)
    blockers: list[str] = []
    try:
        obs.validate()
    except ValueError as exc:
        blockers.append(f"invalid_observation:{exc}")
    if not obs.existing_adaln_bias_route:
        blockers.append("existing_adaln_bias_route_missing")
    if not obs.pooled_text_projection_contract:
        blockers.append("pooled_text_projection_contract_missing")
    if not obs.dedicated_distill_loop:
        blockers.append("dedicated_distill_loop_missing")
    if not obs.inference_projection_contract:
        blockers.append("inference_projection_contract_missing")
    if not obs.save_metadata_stamp:
        blockers.append("save_metadata_stamp_missing")
    if not obs.request_config_replay:
        blockers.append("request_config_replay_missing")
    if obs.ui_exposure_requested:
        blockers.append("ui_exposure_requested_before_full_contract")
    if obs.default_behavior_changed:
        blockers.append("default_behavior_changed")
    full_route_ready = not blockers
    return {
        "schema_version": 1,
        "contract": "modulation_guidance_reconciliation_v0",
        "ok": True,
        "reconciliation_complete": True,
        "full_route_ready": full_route_ready,
        "existing_route_name": obs.existing_route_name,
        "existing_adaln_bias_route": bool(obs.existing_adaln_bias_route),
        "pooled_text_projection_contract": bool(obs.pooled_text_projection_contract),
        "dedicated_distill_loop": bool(obs.dedicated_distill_loop),
        "inference_projection_contract": bool(obs.inference_projection_contract),
        "save_metadata_stamp": bool(obs.save_metadata_stamp),
        "request_config_replay": bool(obs.request_config_replay),
        "ui_exposure_allowed": full_route_ready and not obs.ui_exposure_requested,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add request/config replay and trainer preflight for pooled-text modulation guidance"
            if full_route_ready
            else "keep current AdaLN guidance marked partial until distill/projection/metadata contracts exist"
        ),
    }


def build_modulation_guidance_scorecard(
    *,
    reconciliation: Mapping[str, Any] | None = None,
    metadata_roundtrip_ok: bool = False,
    existing_route_smoke_ok: bool = False,
    ui_blocked_ok: bool = False,
    required_blockers: Sequence[str] = (
        "pooled_text_projection_contract_missing",
        "dedicated_distill_loop_missing",
        "inference_projection_contract_missing",
    ),
) -> dict[str, Any]:
    rec = dict(reconciliation or build_modulation_guidance_reconciliation())
    blockers: list[str] = []
    if not rec.get("reconciliation_complete"):
        blockers.append("reconciliation_missing")
    if not metadata_roundtrip_ok:
        blockers.append("metadata_roundtrip_missing")
    if not existing_route_smoke_ok:
        blockers.append("existing_route_smoke_missing")
    if not ui_blocked_ok:
        blockers.append("ui_block_missing")
    rec_blockers = set(str(item) for item in (rec.get("blocked_reasons") or ()))
    for required in required_blockers:
        if required not in rec_blockers and not rec.get("full_route_ready"):
            blockers.append(f"expected_blocker_missing:{required}")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "modulation_guidance_reconciliation_v0",
        "ok": ready,
        "reconciliation_ready": ready,
        "full_route_ready": bool(rec.get("full_route_ready")),
        "existing_route_name": rec.get("existing_route_name"),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "reconciliation": rec,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "schedule pooled-text modulation guidance trainer preflight"
            if ready
            else "complete reconciliation evidence before any modulation guidance registry or UI exposure"
        ),
    }


def _coerce_observation(
    observation: ModulationGuidanceObservation | Mapping[str, Any] | None,
) -> ModulationGuidanceObservation:
    if isinstance(observation, ModulationGuidanceObservation):
        return observation
    values = dict(observation or {})
    return ModulationGuidanceObservation(
        existing_adaln_bias_route=_boolish(values.get("existing_adaln_bias_route", True)),
        existing_route_name=str(values.get("existing_route_name", "adaln_guidance")),
        pooled_text_projection_contract=_boolish(values.get("pooled_text_projection_contract", False)),
        dedicated_distill_loop=_boolish(values.get("dedicated_distill_loop", False)),
        inference_projection_contract=_boolish(values.get("inference_projection_contract", False)),
        save_metadata_stamp=_boolish(values.get("save_metadata_stamp", False)),
        request_config_replay=_boolish(values.get("request_config_replay", False)),
        ui_exposure_requested=_boolish(values.get("ui_exposure_requested", False)),
        default_behavior_changed=_boolish(values.get("default_behavior_changed", False)),
    )


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


__all__ = [
    "ModulationGuidanceObservation",
    "build_modulation_guidance_metadata",
    "build_modulation_guidance_reconciliation",
    "build_modulation_guidance_scorecard",
]
