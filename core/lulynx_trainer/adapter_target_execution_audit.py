"""Execution audit gate for profiled adapter target trainer wiring."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_adapter_target_execution_audit(
    *,
    trainer_preflight: Mapping[str, Any],
    execution_report: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = dict(trainer_preflight)
    report = dict(execution_report)
    expected_targets = tuple(str(item) for item in preflight.get("target_modules", ()) or ())
    observed_targets = tuple(str(item) for item in report.get("injected_target_modules", ()) or ())
    updated_targets = tuple(str(item) for item in report.get("updated_target_modules", ()) or ())
    blockers: list[str] = []

    if preflight.get("scorecard") != "adapter_target_trainer_preflight_v0":
        blockers.append("unexpected_trainer_preflight")
    if not bool(preflight.get("preflight_ready", preflight.get("ok", False))):
        blockers.append("trainer_preflight_not_ready")
    if _unsafe_flags(preflight, report):
        blockers.append("unsafe_child_flag")
    if not bool(report.get("execution_report_present", report.get("ok", False))):
        blockers.append("execution_report_missing")
    if set(observed_targets) != set(expected_targets):
        blockers.append("injected_targets_mismatch")
    if set(updated_targets) != set(expected_targets):
        blockers.append("updated_targets_mismatch")
    if not bool(report.get("save_metadata_stamped", False)):
        blockers.append("save_metadata_not_stamped")
    if not bool(report.get("merge_policy_observed", False)):
        blockers.append("merge_policy_not_observed")
    if not bool(report.get("quality_gate_passed", False)):
        blockers.append("quality_gate_not_passed")
    if bool(report.get("unexpected_full_target_injection", False)):
        blockers.append("unexpected_full_target_injection")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "adapter_target_execution_audit_v0",
        "ok": ready,
        "execution_audit_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "expected_target_count": len(expected_targets),
        "observed_target_count": len(observed_targets),
        "updated_target_count": len(updated_targets),
        "missing_injected_targets": sorted(set(expected_targets) - set(observed_targets)),
        "unexpected_injected_targets": sorted(set(observed_targets) - set(expected_targets)),
        "missing_updated_targets": sorted(set(expected_targets) - set(updated_targets)),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "compare profiler-selected adapter target run against all-target baseline"
            if ready
            else "fix selected-target injection/update evidence before quality comparison"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    return any(
        bool(payload.get("training_path_enabled", False))
        or bool(payload.get("default_behavior_changed", False))
        or bool(payload.get("promotion_ready", False))
        for payload in payloads
    )


__all__ = ["build_adapter_target_execution_audit"]
