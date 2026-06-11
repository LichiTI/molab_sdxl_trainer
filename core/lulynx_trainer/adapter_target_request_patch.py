"""Dry-run request patch for profiled adapter target selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_adapter_target_request_patch_plan(
    profile_plan_report: Mapping[str, Any],
    *,
    base_request: Mapping[str, Any] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    report = dict(profile_plan_report)
    plan = dict(report.get("plan") or {})
    selected_names = tuple(str(item) for item in plan.get("selected_names", ()) or ())
    rank_by_name = {str(key): int(value) for key, value in dict(plan.get("rank_by_name") or {}).items()}
    blockers: list[str] = []
    if report.get("contract") != "adapter_target_policy_profile_plan_v0":
        blockers.append("unexpected_profile_plan_contract")
    if not bool(report.get("profile_ready", False)):
        blockers.append("profile_not_ready")
    if not selected_names:
        blockers.append("adapter_targets_missing")
    if not dry_run:
        blockers.append("dry_run_required")
    if bool(report.get("training_path_enabled", False)) or bool(plan.get("default_behavior_changed", False)):
        blockers.append("unsafe_profile_plan_flag")

    patch = {
        "adapter_target_policy": str(plan.get("policy") or "profiled"),
        "adapter_target_modules": ",".join(selected_names),
        "adapter_target_rank_map": dict(sorted(rank_by_name.items())),
        "adapter_target_profile_contract": str(report.get("contract") or ""),
        "adapter_target_profile_dry_run": bool(dry_run),
    }
    if base_request:
        patch = {**dict(base_request), **patch}

    ready = not blockers
    return {
        "schema_version": 1,
        "plan": "adapter_target_request_patch_plan_v0",
        "ok": ready,
        "request_fields_emitted": ready,
        "dry_run_only": bool(dry_run),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "selected_count": len(selected_names),
        "patch": patch,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "replay patch through ConfigAdapter before trainer wiring"
            if ready
            else "complete profiler plan before emitting adapter target request patch"
        ),
    }


def build_adapter_target_request_patch_scorecard(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(plan)
    blockers = list(payload.get("blocked_reasons") or [])
    if not bool(payload.get("request_fields_emitted", False)):
        blockers.append("request_fields_not_emitted")
    if not bool(payload.get("dry_run_only", False)):
        blockers.append("dry_run_boundary_missing")
    if bool(payload.get("training_path_enabled", False)) or bool(payload.get("default_behavior_changed", False)):
        blockers.append("unsafe_patch_plan_flag")
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_request_patch_plan_v0",
        "ok": not blockers,
        "request_fields_emitted": bool(payload.get("request_fields_emitted", False)),
        "dry_run_only": bool(payload.get("dry_run_only", False)),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "selected_count": int(payload.get("selected_count") or 0),
        "patch_keys": sorted(dict(payload.get("patch") or {}).keys()),
        "blocked_reasons": blockers,
    }


def build_adapter_target_config_adapter_replay(
    request_patch_plan: Mapping[str, Any],
    replay_result: Mapping[str, Any],
) -> dict[str, Any]:
    plan = dict(request_patch_plan)
    patch = dict(plan.get("patch") or {})
    replay = dict(replay_result)
    parsed = dict(replay.get("parsed_fields") or {})
    expected_modules = _split_modules(patch.get("adapter_target_modules"))
    parsed_modules = _split_modules(parsed.get("adapter_target_modules"))
    expected_ranks = {str(key): int(value) for key, value in dict(patch.get("adapter_target_rank_map") or {}).items()}
    parsed_ranks = {str(key): int(value) for key, value in dict(parsed.get("adapter_target_rank_map") or {}).items()}
    blockers: list[str] = []
    if plan.get("plan") != "adapter_target_request_patch_plan_v0":
        blockers.append("unexpected_request_patch_plan")
    if not bool(plan.get("request_fields_emitted", False)):
        blockers.append("request_fields_not_emitted")
    if not bool(replay.get("ok", False)):
        blockers.append("config_adapter_replay_failed")
    if bool(replay.get("training_path_enabled", False)):
        blockers.append("unsafe_training_path_enabled")
    if expected_modules != parsed_modules:
        blockers.append("adapter_target_modules_not_preserved")
    if expected_ranks != parsed_ranks:
        blockers.append("adapter_target_rank_map_not_preserved")
    if parsed.get("adapter_target_profile_dry_run") is not True:
        blockers.append("dry_run_marker_not_preserved")
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_config_adapter_replay_v0",
        "ok": not blockers,
        "replay_ready": not blockers,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "selected_count": len(expected_modules),
        "blocked_reasons": blockers,
    }


def _split_modules(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


__all__ = [
    "build_adapter_target_config_adapter_replay",
    "build_adapter_target_request_patch_plan",
    "build_adapter_target_request_patch_scorecard",
]
