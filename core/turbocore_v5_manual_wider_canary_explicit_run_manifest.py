"""Explicit-run manifest for V5 manual wider canary.

This module is contract-only. It records what a signed manual wider canary
request would look like and what evidence a single explicit run must produce,
but it never starts training or enables default rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_owner_review_evidence_package import load_json
from core.turbocore_v5_owner_review_request_adapter_replay import (
    request_fields_from_owner_review_package,
)


VALID_MODES = {"off", "observe", "manual_wider_canary", "auto"}


def build_v5_manual_wider_canary_explicit_run_manifest(
    *,
    owner_review_package: Mapping[str, Any] | None = None,
    native_training_mode: str = "manual_wider_canary",
) -> dict[str, Any]:
    package = _as_dict(owner_review_package)
    mode = _mode(native_training_mode)
    request_fields = request_fields_from_owner_review_package(package)
    request_ready = _request_ready(request_fields)
    route = _route(mode=mode, request_ready=request_ready)
    rollback = _rollback_policy(package)
    audit = _audit_skeleton(package=package, route=route)
    gates = {
        "signed_owner_review_package": request_ready,
        "route_decision_present": bool(route.get("decision")),
        "request_fields_present": _request_fields_present(request_fields),
        "default_and_auto_blocked": _default_blocked(route),
        "rollback_policy_ready": _rollback_ready(rollback),
        "audit_skeleton_present": bool(audit.get("required_runtime_evidence")),
        "default_behavior_unchanged": True,
    }
    manifest_ready = bool(route.get("explicit_run_manifest_ready", False)) and all(gates.values())
    blocked = _blockers(gates, route)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_manual_wider_canary_explicit_run_manifest_v0",
        "gate": "v5_manual_wider_canary_explicit_run_manifest",
        "ok": manifest_ready,
        "milestone_completed": manifest_ready,
        "explicit_run_manifest_ready": manifest_ready,
        "manual_wider_canary_explicit_run_allowed": bool(route.get("explicit_run_allowed", False)),
        "native_training_mode": mode,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_dispatch_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "requires_explicit_signed_request": True,
        "route_decision": route,
        "manifest_request_fields": request_fields if request_ready else {},
        "rollback_policy": rollback,
        "audit_skeleton": audit,
        "progress_gates": gates,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(route, manifest_ready),
        "notes": [
            "This manifest does not dispatch training.",
            "The request fields are valid only for an explicit signed manual wider canary request.",
            "Default and auto rollout remain disabled.",
        ],
    }


def _route(*, mode: str, request_ready: bool) -> dict[str, Any]:
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
        explicit_allowed = False
    elif mode == "auto":
        decision = "auto_blocked_for_v5"
        reason = "auto_rollout_blocked_for_v5"
        explicit_allowed = False
    elif not request_ready:
        decision = "blocked_until_signed_owner_review"
        reason = "signed_owner_review_package_missing"
        explicit_allowed = False
    elif mode == "observe":
        decision = "observe_manifest_ready_but_no_training_dispatch"
        reason = "observe_mode_records_contract_only"
        explicit_allowed = False
    else:
        decision = "manual_wider_canary_explicit_run_ready_but_disabled"
        reason = "explicit_signed_request_required"
        explicit_allowed = True
    return {
        "schema_version": 1,
        "feature": "v5_manual_wider_canary_native_adamw_update",
        "native_training_mode": mode,
        "requested_scope": "manual_wider_canary",
        "decision": decision,
        "reason": reason,
        "explicit_run_manifest_ready": decision in {
            "observe_manifest_ready_but_no_training_dispatch",
            "manual_wider_canary_explicit_run_ready_but_disabled",
        },
        "explicit_run_allowed": explicit_allowed,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_dispatch_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "missing_before_explicit_run": [] if request_ready else ["signed_owner_review_package"],
        "missing_before_auto": [
            "longer_replicate_matrix",
            "manual_wider_canary_run_audit",
            "owner_rollout_review",
        ],
    }


def _audit_skeleton(*, package: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    performance = _as_dict(package.get("performance_matrix_summary"))
    return {
        "schema_version": 1,
        "audit": "v5_manual_wider_canary_explicit_run_audit_skeleton_v0",
        "route_decision": str(route.get("decision", "") or ""),
        "owner_review_package_present": bool(package),
        "source_owner_review_package": str(package.get("_source_path") or ""),
        "representative_native_case": str(performance.get("representative_native_case", "") or ""),
        "representative_end_to_end_speedup": performance.get("representative_end_to_end_speedup"),
        "ctx_sync_free_speedup_vs_context_sync_native": performance.get(
            "ctx_sync_free_speedup_vs_context_sync_native"
        ),
        "required_runtime_evidence": [
            "native_dispatch_requested",
            "native_dispatch_executed",
            "native_dispatch_training_executor_timing_present",
            "native_dispatch_update_report_present",
            "native_dispatch_owner_native_report_present",
            "native_dispatch_probe_cache_retained",
            "native_dispatch_owner_native_runtime_synchronization",
            "native_dispatch_training_executor_last_error_empty",
            "fallback_state_sync_on_close_or_recovery",
            "checkpoint_resume_native_state_boundary",
        ],
        "required_report_fields": [
            "native_dispatch_training_executor_elapsed_ms_mean",
            "native_dispatch_update_executor_elapsed_ms_mean",
            "native_dispatch_update_executor_grad_sync_ms_mean",
            "native_dispatch_update_executor_copyback_ms_mean",
            "native_dispatch_owner_native_runtime_stream_binding",
            "native_dispatch_owner_native_stream_lifetime_bound",
        ],
        "rollback_triggers": [
            "native_error",
            "state_sync_failure",
            "checkpoint_resume_mismatch",
            "config_mismatch",
            "non_finite",
            "performance_regression",
        ],
    }


def _rollback_policy(package: Mapping[str, Any]) -> dict[str, Any]:
    review_gate = _as_dict(package.get("manual_review_gate"))
    rollback = _as_dict(review_gate.get("rollback_policy"))
    if rollback:
        return {
            **rollback,
            "default_training_path_enabled": False,
        }
    return {
        "schema_version": 1,
        "policy": "v5_manual_wider_canary_rollback_policy_v0",
        "fallback_authoritative": True,
        "fallback_backend": "pytorch_adamw",
        "disable_for_run_on_native_error": True,
        "disable_for_run_on_state_sync_failure": True,
        "disable_for_run_on_checkpoint_resume_mismatch": True,
        "disable_for_run_on_config_mismatch": True,
        "disable_for_run_on_non_finite": True,
        "rollback_on_resume_mismatch": True,
        "rollback_on_performance_regression": True,
        "default_training_path_enabled": False,
    }


def _request_ready(request_fields: Mapping[str, Any]) -> bool:
    return bool(
        request_fields.get("optimizerType") == "AdamW"
        and request_fields.get("turbocoreNativeUpdateCanaryScope") == "manual_wider_canary"
        and request_fields.get("turbocoreNativeUpdateManualWiderCanaryReviewReady") is True
    )


def _request_fields_present(request_fields: Mapping[str, Any]) -> bool:
    return _request_ready(request_fields)


def _default_blocked(route: Mapping[str, Any]) -> bool:
    return bool(
        route.get("default_training_path_enabled") is False
        and route.get("training_path_enabled") is False
        and route.get("default_dispatch_allowed") is False
        and route.get("default_rollout_allowed") is False
        and route.get("auto_rollout_allowed") is False
    )


def _rollback_ready(rollback: Mapping[str, Any]) -> bool:
    return bool(
        rollback.get("fallback_authoritative", False)
        and rollback.get("disable_for_run_on_native_error", False)
        and rollback.get("disable_for_run_on_state_sync_failure", False)
        and rollback.get("disable_for_run_on_checkpoint_resume_mismatch", False)
        and rollback.get("rollback_on_resume_mismatch", False)
        and rollback.get("rollback_on_performance_regression", False)
        and rollback.get("default_training_path_enabled") is False
    )


def _blockers(gates: Mapping[str, bool], route: Mapping[str, Any]) -> list[str]:
    blocked = [f"v5_p21_{name}_missing" for name, ok in gates.items() if not ok]
    decision = str(route.get("decision", "") or "")
    if decision == "off":
        blocked.append("v5_p21_native_training_mode_off")
    elif decision == "auto_blocked_for_v5":
        blocked.append("v5_p21_auto_rollout_blocked")
    elif decision == "blocked_until_signed_owner_review":
        blocked.append("v5_p21_signed_owner_review_package_missing")
    return _dedupe(blocked)


def _recommended_next_step(route: Mapping[str, Any], ready: bool) -> str:
    decision = str(route.get("decision", "") or "")
    if ready and decision == "manual_wider_canary_explicit_run_ready_but_disabled":
        return "run an explicit signed manual wider canary only under developer control"
    if ready and decision == "observe_manifest_ready_but_no_training_dispatch":
        return "record observe-only manifest evidence; training dispatch remains off"
    if decision == "auto_blocked_for_v5":
        return "keep auto rollout disabled until explicit canary run audits pass"
    return "complete signed owner review package before explicit run manifest"


def _mode(value: str) -> str:
    normalized = str(value or "manual_wider_canary").strip().lower().replace("-", "_")
    return normalized if normalized in VALID_MODES else "manual_wider_canary"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 manual wider canary explicit-run manifest.")
    parser.add_argument("--owner-review-package", default="", help="Signed owner review package JSON.")
    parser.add_argument("--native-training-mode", default="manual_wider_canary", choices=sorted(VALID_MODES))
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_manual_wider_canary_explicit_run_manifest(
        owner_review_package=load_json(args.owner_review_package) if args.owner_review_package else None,
        native_training_mode=str(args.native_training_mode),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_manual_wider_canary_explicit_run_manifest"]
