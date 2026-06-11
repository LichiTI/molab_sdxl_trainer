"""Owner rollout-review package for V5 explicit manual wider canary runs.

This module is contract-only. It turns a P22 explicit-run audit into review
material for a human rollout decision, but it never emits request-adapter fields
or enables native dispatch by default.
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


def build_v5_manual_wider_canary_run_review_package(
    *,
    explicit_run_audit: Mapping[str, Any] | None = None,
    explicit_run_manifest: Mapping[str, Any] | None = None,
    rollback_history: Any | None = None,
    owner_review_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = _as_dict(explicit_run_audit)
    manifest = _as_dict(explicit_run_manifest)
    owner_package = _as_dict(owner_review_package)
    run_summary = _run_audit_summary(audit)
    runtime_summary = _runtime_evidence_summary(audit, manifest)
    report_summary = _report_field_summary(audit, manifest)
    rollback_summary = _rollback_summary(audit)
    history_summary = _rollback_history_summary(rollback_history)
    owner_summary = _owner_review_summary(owner_package)
    blocked = _blockers(
        audit=audit,
        run_summary=run_summary,
        runtime_summary=runtime_summary,
        report_summary=report_summary,
        rollback_summary=rollback_summary,
        history_summary=history_summary,
        owner_summary=owner_summary,
    )
    ready = not blocked
    decision = "hold_for_owner_rollout_review" if ready else "rollback_required_or_hold"
    return {
        "schema_version": 1,
        "package": "turbocore_v5_manual_wider_canary_run_review_package_v0",
        "gate": "v5_manual_wider_canary_explicit_run_review_package",
        "ok": ready,
        "run_review_package_ready": ready,
        "ready_for_owner_rollout_review": ready,
        "rollout_review_decision": decision,
        "manual_review_required": True,
        "owner_rollout_review_required": True,
        "rollback_required": not ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_review_request_fields": {},
        "run_audit_summary": run_summary,
        "runtime_evidence_summary": runtime_summary,
        "report_field_summary": report_summary,
        "missing_report_fields": report_summary["missing_report_fields"],
        "performance_summary": run_summary["performance"],
        "rollback_summary": rollback_summary,
        "rollback_history_summary": history_summary,
        "owner_review_package_summary": owner_summary,
        "rollback_policy": _rollback_policy(manifest, owner_package),
        "rollout_review_template": _rollout_review_template(run_summary, history_summary),
        "review_checklist": _review_checklist(
            run_summary=run_summary,
            runtime_summary=runtime_summary,
            report_summary=report_summary,
            rollback_summary=rollback_summary,
            history_summary=history_summary,
        ),
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(ready, rollback_summary, history_summary),
        "notes": [
            "This package is review material only; it does not enable native dispatch.",
            "A ready package still requires a separate human rollout review.",
            "Request-adapter fields are intentionally not emitted by P23.",
        ],
    }


def _run_audit_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    run_result = _as_dict(audit.get("run_result_summary"))
    performance = _as_dict(run_result.get("performance")) or _as_dict(audit.get("performance_summary"))
    return {
        "present": bool(audit),
        "source_path": str(audit.get("_source_path") or audit.get("source_path") or ""),
        "run_audit_ready": bool(audit.get("run_audit_ready", False)),
        "keep_candidate_allowed": bool(audit.get("keep_candidate_allowed", False)),
        "rollback_required": bool(audit.get("rollback_required", False)),
        "decision": str(audit.get("decision") or ""),
        "native_case": str(performance.get("native_case") or run_result.get("native_case") or ""),
        "performance": {
            "representative_end_to_end_speedup": performance.get("representative_end_to_end_speedup"),
            "representative_steps": performance.get("representative_steps"),
            "native_case": str(performance.get("native_case") or run_result.get("native_case") or ""),
        },
        "blocked_reasons": _string_list(audit.get("blocked_reasons")),
        "promotion_blockers": _string_list(audit.get("promotion_blockers")),
    }


def _runtime_evidence_summary(
    audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    required = _required_runtime_evidence(audit, manifest)
    missing = _string_list(audit.get("missing_runtime_evidence"))
    present = [name for name in required if name not in missing]
    return {
        "required_runtime_evidence": required,
        "present_runtime_evidence": present,
        "missing_runtime_evidence": missing,
        "required_count": len(required),
        "present_count": len(present),
        "all_required_evidence_present": bool(required) and not missing,
    }


def _report_field_summary(
    audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    required = _required_report_fields(manifest)
    report_fields = _as_dict(_as_dict(audit.get("run_result_summary")).get("report_fields"))
    missing = [field for field in required if field not in report_fields]
    return {
        "required_report_fields": required,
        "present_report_fields": [field for field in required if field in report_fields],
        "missing_report_fields": missing,
        "all_required_report_fields_present": not missing,
        "report_fields": report_fields,
    }


def _rollback_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    run_result = _as_dict(audit.get("run_result_summary"))
    events = _dedupe(_string_list(audit.get("rollback_events")) + _string_list(run_result.get("rollback_events")))
    return {
        "rollback_events": events,
        "rollback_event_count": len(events),
        "has_rollback_event": bool(events),
        "rollback_required_by_audit": bool(audit.get("rollback_required", False)),
    }


def _rollback_history_summary(value: Any) -> dict[str, Any]:
    events = _history_items(value)
    normalized = [_normalize_history_item(item) for item in events]
    open_events = [item for item in normalized if item["open"]]
    return {
        "present": bool(events),
        "event_count": len(normalized),
        "open_rollback_count": len(open_events),
        "has_open_rollback": bool(open_events),
        "recent_events": normalized[-5:],
    }


def _owner_review_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(package),
        "source_path": str(package.get("_source_path") or ""),
        "promotion_review_ready": bool(package.get("promotion_review_ready", False)),
        "manual_wider_canary_allowed": bool(package.get("manual_wider_canary_allowed", False)),
        "promotion_decision": str(package.get("promotion_decision") or ""),
        "default_training_path_enabled": bool(package.get("default_training_path_enabled", False)),
        "training_path_enabled": bool(package.get("training_path_enabled", False)),
        "default_rollout_allowed": bool(package.get("default_rollout_allowed", False)),
        "auto_rollout_allowed": bool(package.get("auto_rollout_allowed", False)),
    }


def _blockers(
    *,
    audit: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    runtime_summary: Mapping[str, Any],
    report_summary: Mapping[str, Any],
    rollback_summary: Mapping[str, Any],
    history_summary: Mapping[str, Any],
    owner_summary: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if not audit:
        blocked.append("v5_p23_explicit_run_audit_missing")
    if not bool(run_summary.get("run_audit_ready", False)):
        blocked.append("v5_p23_explicit_run_audit_not_ready")
    if not bool(run_summary.get("keep_candidate_allowed", False)):
        blocked.append("v5_p23_keep_candidate_not_allowed")
    if bool(run_summary.get("rollback_required", False)):
        blocked.append("v5_p23_run_audit_requires_rollback")
    blocked.extend(_string_list(run_summary.get("blocked_reasons")))
    blocked.extend(_string_list(run_summary.get("promotion_blockers")))
    if not bool(runtime_summary.get("all_required_evidence_present", False)):
        blocked.append("v5_p23_required_runtime_evidence_missing")
        blocked.extend(f"missing:{name}" for name in _string_list(runtime_summary.get("missing_runtime_evidence")))
    if not bool(report_summary.get("all_required_report_fields_present", False)):
        blocked.append("v5_p23_required_report_fields_missing")
        blocked.extend(f"missing_report_field:{name}" for name in report_summary["missing_report_fields"])
    if bool(rollback_summary.get("has_rollback_event", False)):
        blocked.append("v5_p23_rollback_event_present")
        blocked.extend(f"rollback:{name}" for name in _string_list(rollback_summary.get("rollback_events")))
    if bool(history_summary.get("has_open_rollback", False)):
        blocked.append("v5_p23_open_rollback_history")
    if bool(owner_summary.get("default_training_path_enabled", False)):
        blocked.append("v5_p23_owner_package_default_training_path_enabled")
    if bool(owner_summary.get("default_rollout_allowed", False)):
        blocked.append("v5_p23_owner_package_default_rollout_allowed")
    if bool(owner_summary.get("auto_rollout_allowed", False)):
        blocked.append("v5_p23_owner_package_auto_rollout_allowed")
    return _dedupe(blocked)


def _required_runtime_evidence(
    audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    required = _string_list(audit.get("required_runtime_evidence"))
    if required:
        return required
    audit_skeleton = _as_dict(manifest.get("audit_skeleton"))
    required = _string_list(audit_skeleton.get("required_runtime_evidence"))
    if required:
        return required
    return [
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
    ]


def _required_report_fields(manifest: Mapping[str, Any]) -> list[str]:
    audit_skeleton = _as_dict(manifest.get("audit_skeleton"))
    required = _string_list(audit_skeleton.get("required_report_fields"))
    if required:
        return required
    return [
        "native_dispatch_training_executor_elapsed_ms_mean",
        "native_dispatch_update_executor_elapsed_ms_mean",
        "native_dispatch_update_executor_grad_sync_ms_mean",
        "native_dispatch_update_executor_copyback_ms_mean",
        "native_dispatch_owner_native_runtime_stream_binding",
        "native_dispatch_owner_native_stream_lifetime_bound",
    ]


def _rollback_policy(
    manifest: Mapping[str, Any],
    owner_package: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_policy = _as_dict(manifest.get("rollback_policy"))
    if manifest_policy:
        return {**manifest_policy, "default_training_path_enabled": False}
    review_gate = _as_dict(owner_package.get("manual_review_gate"))
    owner_policy = _as_dict(review_gate.get("rollback_policy"))
    if owner_policy:
        return {**owner_policy, "default_training_path_enabled": False}
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


def _rollout_review_template(
    run_summary: Mapping[str, Any],
    history_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": "manual_wider_canary_run_keep_review",
        "approve_keep_manual_wider_canary_evidence": False,
        "approve_default_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "acknowledge_no_request_adapter_mapping": False,
        "acknowledge_runtime_evidence_complete": False,
        "acknowledge_rollback_history_clear": not bool(history_summary.get("has_open_rollback", False)),
        "acknowledged_native_case": str(run_summary.get("native_case") or ""),
        "acknowledged_representative_end_to_end_speedup": _as_dict(
            run_summary.get("performance")
        ).get("representative_end_to_end_speedup"),
    }


def _review_checklist(
    *,
    run_summary: Mapping[str, Any],
    runtime_summary: Mapping[str, Any],
    report_summary: Mapping[str, Any],
    rollback_summary: Mapping[str, Any],
    history_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "p22_run_audit_ready",
            "ok": bool(run_summary.get("run_audit_ready", False))
            and bool(run_summary.get("keep_candidate_allowed", False)),
            "summary": "P22 explicit-run audit allows evidence keep review.",
        },
        {
            "id": "runtime_evidence_complete",
            "ok": bool(runtime_summary.get("all_required_evidence_present", False)),
            "summary": "All required runtime evidence from P21/P22 is present.",
        },
        {
            "id": "report_fields_complete",
            "ok": bool(report_summary.get("all_required_report_fields_present", False)),
            "summary": "Training executor timing and stream-lifetime report fields are present.",
        },
        {
            "id": "no_rollback_events",
            "ok": not bool(rollback_summary.get("has_rollback_event", False)),
            "summary": "The explicit run did not report rollback triggers.",
        },
        {
            "id": "rollback_history_clear",
            "ok": not bool(history_summary.get("has_open_rollback", False)),
            "summary": "No open rollback history is attached to this review package.",
        },
        {
            "id": "default_and_auto_off",
            "ok": True,
            "summary": "Default training path, default rollout, and auto rollout remain disabled.",
        },
        {
            "id": "request_adapter_not_mapped",
            "ok": True,
            "summary": "P23 does not emit request-adapter fields.",
        },
    ]


def _recommended_next_step(
    ready: bool,
    rollback_summary: Mapping[str, Any],
    history_summary: Mapping[str, Any],
) -> str:
    if ready:
        return "send explicit-run evidence to owner rollout review; default rollout remains off"
    if bool(rollback_summary.get("has_rollback_event", False)):
        return "rollback explicit manual wider canary and keep PyTorch AdamW authoritative"
    if bool(history_summary.get("has_open_rollback", False)):
        return "resolve rollback history before owner rollout review"
    return "collect missing explicit-run evidence before owner rollout review"


def _history_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("events", "history", "records", "rollback_events"):
            items = value.get(key)
            if isinstance(items, list):
                return items
        return [value] if value else []
    return []


def _normalize_history_item(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        event = str(value.get("event") or value.get("type") or value.get("reason") or value.get("status") or "")
        status = str(value.get("status") or "")
        source = str(value.get("source_task_id") or value.get("source") or "")
    else:
        event = str(value or "")
        status = "open"
        source = ""
    closed = status.lower() in {"resolved", "closed", "restored", "cleared", "ok", "success"}
    return {
        "event": event,
        "status": status,
        "source": source,
        "open": bool(event) and not closed,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _load_json_any(path: str | Path) -> Any:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.setdefault("_source_path", str(source))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 manual wider canary run review package.")
    parser.add_argument("--explicit-run-audit", default="", help="P22 explicit-run audit JSON.")
    parser.add_argument("--explicit-run-manifest", default="", help="P21 explicit-run manifest JSON.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON.")
    parser.add_argument("--owner-review-package", default="", help="Optional owner review package JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_manual_wider_canary_run_review_package(
        explicit_run_audit=load_json(args.explicit_run_audit) if args.explicit_run_audit else None,
        explicit_run_manifest=load_json(args.explicit_run_manifest) if args.explicit_run_manifest else None,
        rollback_history=_load_json_any(args.rollback_history) if args.rollback_history else None,
        owner_review_package=load_json(args.owner_review_package) if args.owner_review_package else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_manual_wider_canary_run_review_package"]
