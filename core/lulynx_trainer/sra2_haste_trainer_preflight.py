"""Trainer-wiring preflight for SRA2 + HASTE alignment loss."""

from __future__ import annotations

from typing import Any, Mapping


SUPPORTED_VAE_FEATURE_SOURCES = {"batch", "cache", "sidecar"}


def build_sra2_haste_trainer_preflight(
    facade_scorecard: Mapping[str, Any],
    trainer_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scorecard = dict(facade_scorecard)
    contract = dict(trainer_contract or {})
    blockers: list[str] = []
    layers = tuple(str(item) for item in contract.get("hidden_capture_layers", ()) or ())
    vae_source = str(contract.get("vae_feature_source") or "").strip().lower()
    schedule = dict(contract.get("schedule") or {})

    if scorecard.get("scorecard") != "sra2_haste_alignment_facade_v0":
        blockers.append("unexpected_facade_scorecard")
    if not bool(scorecard.get("facade_ready", scorecard.get("ok", False))):
        blockers.append("facade_not_ready")
    if bool(scorecard.get("training_path_enabled", False)) or bool(scorecard.get("promotion_ready", False)):
        blockers.append("unsafe_facade_flag")
    if not layers:
        blockers.append("hidden_capture_layers_missing")
    if vae_source not in SUPPORTED_VAE_FEATURE_SOURCES:
        blockers.append("vae_feature_source_missing")
    if bool(contract.get("default_enabled", False)):
        blockers.append("default_enabled_not_allowed")
    if bool(contract.get("training_path_enabled", False)):
        blockers.append("trainer_path_must_remain_disabled")
    if int(schedule.get("stop_step", 1)) == 0:
        blockers.append("stop_step_zero_blocks_useful_schedule")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "sra2_haste_trainer_preflight_v0",
        "ok": ready,
        "preflight_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "hidden_capture_layers": list(layers),
        "vae_feature_source": vae_source,
        "contract": {
            "default_enabled": bool(contract.get("default_enabled", False)),
            "schedule": schedule,
            "quality_gate": dict(contract.get("quality_gate") or {}),
        },
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "wire behind default-off config after quality gate thresholds are approved"
            if ready
            else "complete layer capture, VAE feature source, and default-off contract before trainer wiring"
        ),
    }


__all__ = ["build_sra2_haste_trainer_preflight"]
