"""Ingest explicit-run audit evidence for V5 manual wider canary."""

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


def build_v5_manual_wider_canary_run_audit(
    *,
    explicit_run_manifest: Mapping[str, Any] | None = None,
    run_result: Mapping[str, Any] | None = None,
    checkpoint_resume_boundary_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _as_dict(explicit_run_manifest)
    boundary = _boundary_evidence_summary(_as_dict(checkpoint_resume_boundary_evidence))
    result = _apply_boundary_evidence(_normalize_result(_as_dict(run_result)), boundary)
    manifest_ready = bool(manifest.get("explicit_run_manifest_ready", False))
    required = _required_evidence(manifest)
    missing = [name for name in required if not bool(_as_dict(result.get("evidence")).get(name, False))]
    rollback_events = [str(item) for item in list(result.get("rollback_events", []) or []) if str(item)]
    blockers: list[str] = []
    if not manifest_ready:
        blockers.append("v5_p22_explicit_run_manifest_not_ready")
    if not result.get("present", False):
        blockers.append("v5_p22_run_result_missing")
    if result.get("success") is not True:
        blockers.append("v5_p22_run_result_not_successful")
    if missing:
        blockers.append("v5_p22_required_runtime_evidence_missing")
        blockers.extend(f"missing:{name}" for name in missing)
    if rollback_events:
        blockers.append("v5_p22_rollback_event_present")
        blockers.extend(f"rollback:{name}" for name in rollback_events)
    ready = not blockers
    decision = "keep_manual_wider_canary_evidence" if ready else "rollback_required_or_hold"
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_manual_wider_canary_run_audit_v0",
        "gate": "v5_manual_wider_canary_explicit_run_audit",
        "ok": ready,
        "run_audit_ready": ready,
        "decision": decision,
        "keep_candidate_allowed": ready,
        "rollback_required": not ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "manifest_summary": {
            "explicit_run_manifest_ready": manifest_ready,
            "route_decision": _as_dict(manifest.get("route_decision")).get("decision"),
            "manual_wider_canary_explicit_run_allowed": bool(
                manifest.get("manual_wider_canary_explicit_run_allowed", False)
            ),
        },
        "checkpoint_resume_boundary_evidence_summary": boundary,
        "run_result_summary": result,
        "required_runtime_evidence": required,
        "missing_runtime_evidence": missing,
        "rollback_events": rollback_events,
        "blocked_reasons": _dedupe(blockers),
        "promotion_blockers": _dedupe(blockers),
        "recommended_next_step": _recommended_next_step(ready, rollback_events, missing),
        "notes": [
            "This ingester does not enable default rollout.",
            "A keep decision only means the explicit run evidence can be reviewed.",
            "Any missing required evidence or rollback trigger keeps the route blocked.",
        ],
    }


def _normalize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not result:
        return {"present": False}
    if isinstance(result.get("evidence"), Mapping):
        return {
            "present": True,
            "source_path": str(result.get("_source_path") or result.get("source_path") or ""),
            "success": bool(result.get("success", False)),
            "evidence": _as_dict(result.get("evidence")),
            "report_fields": _as_dict(result.get("report_fields")),
            "rollback_events": list(result.get("rollback_events", []) or []),
            "performance": _as_dict(result.get("performance")),
        }
    matrix_summary = _as_dict(result.get("summary"))
    performance_report = _as_dict(result.get("native_update_performance_report"))
    native_case = _native_case_name(performance_report)
    native_summary = _matrix_case_summary(result, native_case)
    training_matrix = _training_matrix(performance_report)
    evidence = {
        "native_dispatch_requested": bool(native_summary.get("native_dispatch_requested", False)),
        "native_dispatch_executed": bool(native_summary.get("native_dispatch_executed", False)),
        "native_dispatch_training_executor_timing_present": bool(
            native_summary.get("native_dispatch_training_executor_timing_present", False)
        ),
        "native_dispatch_update_report_present": bool(
            native_summary.get("native_dispatch_update_report_present", False)
        ),
        "native_dispatch_owner_native_report_present": bool(
            native_summary.get("native_dispatch_owner_native_report_present", False)
        ),
        "native_dispatch_probe_cache_retained": bool(
            native_summary.get("native_dispatch_probe_cache_retained", False)
        ),
        "native_dispatch_owner_native_runtime_synchronization": bool(
            native_summary.get("native_dispatch_owner_native_runtime_synchronization")
        ),
        "native_dispatch_training_executor_last_error_empty": not bool(
            native_summary.get("native_dispatch_training_executor_last_error", "")
        ),
        "fallback_state_sync_on_close_or_recovery": bool(
            native_summary.get("native_dispatch_training_dispatch_recovery_ready", False)
        ),
        "checkpoint_resume_native_state_boundary": bool(
            native_summary.get("checkpoint_roundtrip_ok", False)
        ),
    }
    return {
        "present": True,
        "source_path": str(result.get("_source_path") or result.get("matrix_summary_path") or ""),
        "success": bool(matrix_summary.get("all_success", False)),
        "native_case": native_case,
        "evidence": evidence,
        "report_fields": _report_fields(native_summary),
        "rollback_events": _matrix_rollback_events(native_summary),
        "performance": {
            "representative_end_to_end_speedup": training_matrix.get("end_to_end_speedup"),
            "representative_steps": training_matrix.get("representative_steps"),
            "native_case": native_case,
        },
    }


def _required_evidence(manifest: Mapping[str, Any]) -> list[str]:
    audit = _as_dict(manifest.get("audit_skeleton"))
    required = [str(item) for item in list(audit.get("required_runtime_evidence", []) or []) if str(item)]
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
    ]


def _boundary_evidence_summary(boundary: Mapping[str, Any]) -> dict[str, Any]:
    patch = _as_dict(boundary.get("runtime_evidence_patch"))
    checkpoint_ready = bool(
        patch.get("checkpoint_resume_native_state_boundary", False)
        or boundary.get("checkpoint_resume_native_state_boundary", False)
        or boundary.get("checkpoint_resume_native_state_boundary_ready", False)
    )
    return {
        "present": bool(boundary),
        "source_path": str(boundary.get("_source_path") or boundary.get("source_path") or ""),
        "checkpoint_resume_native_state_boundary_ready": checkpoint_ready,
        "checkpoint_roundtrip_ok": (
            boundary.get("checkpoint_roundtrip_ok")
            if "checkpoint_roundtrip_ok" in boundary
            else patch.get("checkpoint_resume_native_state_boundary")
        ),
        "default_training_path_enabled": bool(boundary.get("default_training_path_enabled", False)),
        "default_rollout_allowed": bool(boundary.get("default_rollout_allowed", False)),
        "auto_rollout_allowed": bool(boundary.get("auto_rollout_allowed", False)),
        "blocked_reasons": _string_list(boundary.get("blocked_reasons")),
    }


def _apply_boundary_evidence(result: dict[str, Any], boundary: Mapping[str, Any]) -> dict[str, Any]:
    if not result or not bool(boundary.get("checkpoint_resume_native_state_boundary_ready", False)):
        return result
    patched = dict(result)
    evidence = _as_dict(patched.get("evidence"))
    evidence["checkpoint_resume_native_state_boundary"] = True
    patched["evidence"] = evidence
    notes = _string_list(patched.get("patch_notes"))
    notes.append("checkpoint_resume_native_state_boundary supplied by P24 boundary evidence")
    patched["patch_notes"] = _dedupe(notes)
    return patched


def _native_case_name(report: Mapping[str, Any]) -> str:
    training = _training_matrix(report)
    return str(training.get("native_case", "") or "native_update_dispatch_promotion_perf")


def _training_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    gate = _as_dict(report.get("performance_gate"))
    evidence = _as_dict(gate.get("evidence"))
    return _as_dict(evidence.get("training_matrix"))


def _matrix_case_summary(payload: Mapping[str, Any], case_name: str) -> dict[str, Any]:
    for item in payload.get("cases", []) if isinstance(payload.get("cases"), list) else []:
        entry = _as_dict(item)
        meta = _as_dict(entry.get("case"))
        if str(meta.get("name", "") or "") == case_name:
            return _as_dict(entry.get("summary"))
    return {}


def _report_fields(native_summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "native_dispatch_training_executor_elapsed_ms_mean",
        "native_dispatch_update_executor_elapsed_ms_mean",
        "native_dispatch_update_executor_grad_sync_ms_mean",
        "native_dispatch_update_executor_copyback_ms_mean",
        "native_dispatch_owner_native_runtime_stream_binding",
        "native_dispatch_owner_native_stream_lifetime_bound",
    )
    return {key: native_summary.get(key) for key in keys if key in native_summary}


def _matrix_rollback_events(native_summary: Mapping[str, Any]) -> list[str]:
    events: list[str] = []
    if native_summary.get("native_dispatch_training_executor_last_error"):
        events.append("native_error")
    if native_summary.get("native_dispatch_disabled_for_run"):
        events.append("native_dispatch_disabled_for_run")
    if native_summary.get("gate_blocked_reasons"):
        events.append("gate_blocked")
    return events


def _recommended_next_step(ready: bool, rollback_events: list[str], missing: list[str]) -> str:
    if ready:
        return "review explicit run evidence for manual keep; default rollout remains off"
    if rollback_events:
        return "rollback explicit manual wider canary and keep PyTorch AdamW authoritative"
    if missing:
        return "collect missing explicit-run runtime evidence before keep review"
    return "complete explicit-run manifest and run result before audit"


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest V5 manual wider canary explicit-run audit result.")
    parser.add_argument("--manifest", default="", help="Explicit-run manifest JSON.")
    parser.add_argument("--run-result", default="", help="Run result or matrix summary JSON.")
    parser.add_argument(
        "--checkpoint-resume-boundary-evidence",
        default="",
        help="Optional P24 checkpoint/resume native-state boundary evidence JSON.",
    )
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_manual_wider_canary_run_audit(
        explicit_run_manifest=load_json(args.manifest) if args.manifest else None,
        run_result=load_json(args.run_result) if args.run_result else None,
        checkpoint_resume_boundary_evidence=(
            load_json(args.checkpoint_resume_boundary_evidence)
            if args.checkpoint_resume_boundary_evidence
            else None
        ),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_manual_wider_canary_run_audit"]
