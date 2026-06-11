"""Trainer preflight for profiled adapter target selection."""

from __future__ import annotations

from typing import Any, Mapping


SUPPORTED_INJECTOR_CONTRACTS = {"lora_injector", "anima_dit_lora_injector", "newbie_dit_lora_injector"}


def build_adapter_target_trainer_preflight(
    *,
    request_patch_scorecard: Mapping[str, Any],
    config_adapter_replay: Mapping[str, Any],
    injector_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    patch = dict(request_patch_scorecard)
    replay = dict(config_adapter_replay)
    contract = dict(injector_contract or {})
    target_modules = tuple(str(item) for item in contract.get("target_modules", ()) or ())
    rank_map = dict(contract.get("rank_map") or {})
    blockers: list[str] = []

    if patch.get("scorecard") != "adapter_target_request_patch_plan_v0":
        blockers.append("unexpected_request_patch_scorecard")
    if not bool(patch.get("request_fields_emitted", patch.get("ok", False))):
        blockers.append("request_fields_not_emitted")
    if not bool(patch.get("dry_run_only", False)):
        blockers.append("request_patch_dry_run_boundary_missing")
    if replay.get("scorecard") != "adapter_target_config_adapter_replay_v0":
        blockers.append("unexpected_config_adapter_replay")
    if not bool(replay.get("replay_ready", replay.get("ok", False))):
        blockers.append("config_adapter_replay_not_ready")
    if _unsafe_flags(patch, replay, contract):
        blockers.append("unsafe_child_flag")
    if str(contract.get("injector_contract") or "") not in SUPPORTED_INJECTOR_CONTRACTS:
        blockers.append("injector_contract_missing")
    if not target_modules:
        blockers.append("target_modules_missing")
    if not rank_map:
        blockers.append("rank_map_missing")
    if bool(contract.get("default_enabled", False)):
        blockers.append("default_enabled_not_allowed")
    if not bool(contract.get("save_metadata_stamped", False)):
        blockers.append("save_metadata_stamp_missing")
    if not bool(contract.get("merge_policy_recorded", False)):
        blockers.append("merge_policy_missing")
    if not bool(contract.get("quality_gate_defined", False)):
        blockers.append("quality_gate_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_trainer_preflight_v0",
        "ok": ready,
        "preflight_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "selected_count": int(patch.get("selected_count") or replay.get("selected_count") or len(target_modules)),
        "injector_contract": str(contract.get("injector_contract") or ""),
        "target_modules": list(target_modules),
        "rank_map_keys": sorted(str(key) for key in rank_map),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "wire selected target policy behind default-off trainer config"
            if ready
            else "complete injector metadata, merge policy, and quality gate before trainer wiring"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    return any(
        bool(payload.get("training_path_enabled", False))
        or bool(payload.get("default_behavior_changed", False))
        or bool(payload.get("promotion_ready", False))
        for payload in payloads
    )


__all__ = ["build_adapter_target_trainer_preflight", "SUPPORTED_INJECTOR_CONTRACTS"]
