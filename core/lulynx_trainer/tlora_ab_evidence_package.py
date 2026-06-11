"""Evidence package gate for T-LoRA A/B dry-run and result review."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_tlora_ab_evidence_package(
    *,
    manifest: Mapping[str, Any],
    request_patch_scorecard: Mapping[str, Any],
    result_gate: Mapping[str, Any] | None = None,
    config_adapter_replay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_payload = dict(manifest)
    request_payload = dict(request_patch_scorecard)
    result_payload = dict(result_gate or {})
    replay_payload = dict(config_adapter_replay or {})
    blockers: list[str] = []
    if not bool(manifest_payload.get("runner_ready", manifest_payload.get("ok", False))):
        blockers.append("manifest_not_runner_ready")
    if not bool(request_payload.get("request_fields_emitted")):
        blockers.append("request_fields_not_emitted")
    if not bool(request_payload.get("dry_run_only")):
        blockers.append("dry_run_boundary_missing")
    if _unsafe_flags(manifest_payload, request_payload, result_payload, replay_payload):
        blockers.append("unsafe_child_training_or_default_flag")
    if result_gate is None:
        blockers.append("ab_result_gate_missing")
    elif not bool(result_payload.get("ab_result_ready", result_payload.get("ok", False))):
        blockers.append("ab_result_gate_not_ready")
    if config_adapter_replay is None:
        blockers.append("config_adapter_replay_missing")
    else:
        blockers.extend(_config_replay_blockers(replay_payload))
    dry_run_ready = not any(
        reason in blockers
        for reason in (
            "manifest_not_runner_ready",
            "request_fields_not_emitted",
            "dry_run_boundary_missing",
            "unsafe_child_training_or_default_flag",
        )
    )
    result_ready = result_gate is not None and bool(result_payload.get("ab_result_ready", result_payload.get("ok", False)))
    replay_ready = config_adapter_replay is not None and not _config_replay_blockers(replay_payload)
    package_ready = dry_run_ready and result_ready and replay_ready
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_evidence_package_v0",
        "ok": package_ready,
        "package_ready": package_ready,
        "dry_run_ready": dry_run_ready,
        "result_ready": result_ready,
        "config_adapter_replay_ready": replay_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "real_dispatch_allowed": False,
        "case_count": int(manifest_payload.get("case_count") or result_payload.get("case_count") or 0),
        "patch_count": int((request_payload.get("plan") or {}).get("patch_count") or request_payload.get("patch_count") or 0),
        "manifest": _summary(manifest_payload, ("manifest", "runner_ready", "case_count", "blocked_reasons")),
        "request_patch": _summary(request_payload, ("scorecard", "request_fields_emitted", "dry_run_only", "blocked_reasons")),
        "result_gate": _summary(result_payload, ("scorecard", "ab_result_ready", "case_count", "result_count", "blocked_reasons")),
        "config_adapter_replay": _summary(replay_payload, ("replay", "ok", "checked_count", "blocked_reasons")),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "manual review can dispatch representative T-LoRA A/B cases"
            if package_ready
            else "complete dry-run request replay and benchmark result evidence before dispatch"
        ),
    }


def build_tlora_ab_config_adapter_replay(
    request_patch_plan: Mapping[str, Any],
    replay_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    patches = [dict(item) for item in request_patch_plan.get("patches", []) if isinstance(item, Mapping)]
    expected = {(str(item.get("case_id")), str(item.get("arm"))) for item in patches}
    observed = {(str(item.get("case_id")), str(item.get("arm"))) for item in replay_results if item.get("case_id") and item.get("arm")}
    blockers: list[str] = []
    for case_id, arm in sorted(expected - observed):
        blockers.append(f"{case_id}:{arm}:replay_missing")
    for case_id, arm in sorted(observed - expected):
        blockers.append(f"{case_id}:{arm}:unexpected_replay")
    for item in replay_results:
        case_id = str(item.get("case_id") or "")
        arm = str(item.get("arm") or "")
        if not bool(item.get("ok", False)):
            blockers.append(f"{case_id}:{arm}:config_adapter_replay_failed")
        if bool(item.get("training_path_enabled", False)):
            blockers.append(f"{case_id}:{arm}:unsafe_training_path_enabled")
        parsed = dict(item.get("parsed_fields") or {})
        if arm == "tlora" and parsed.get("network_module") != "networks.tlora":
            blockers.append(f"{case_id}:{arm}:network_module_not_tlora")
        if arm == "baseline" and parsed.get("network_module") == "networks.tlora":
            blockers.append(f"{case_id}:{arm}:baseline_uses_tlora")
    return {
        "schema_version": 1,
        "replay": "tlora_ab_config_adapter_replay_v0",
        "ok": not blockers and bool(expected),
        "checked_count": len(observed),
        "expected_count": len(expected),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
    }


def _config_replay_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = list(payload.get("blocked_reasons") or [])
    if not bool(payload.get("ok", False)):
        blockers.append("config_adapter_replay_not_ok")
    if bool(payload.get("training_path_enabled", False)):
        blockers.append("config_adapter_replay_unsafe_training_path")
    return blockers


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    return any(
        bool(payload.get("training_path_enabled", False))
        or bool(payload.get("default_behavior_changed", False))
        or bool(payload.get("promotion_ready", False))
        for payload in payloads
    )


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = [
    "build_tlora_ab_config_adapter_replay",
    "build_tlora_ab_evidence_package",
]
