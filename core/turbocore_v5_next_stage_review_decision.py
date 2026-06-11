"""Signed next-stage review decision for TurboCore V5-P27.

This contract records a human decision after the P26 longer-replicate gate.
It remains report-only: no request-adapter fields are emitted and no default
training or rollout behavior is enabled.
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


APPROVED_P26_DECISION = "longer_replicate_failure_history_review_ready"
APPROVED_P27_DECISION = "signed_next_stage_review_recorded_default_off"
REJECTED_P27_DECISION = "signed_next_stage_review_rejected_default_off"
HOLD_P27_DECISION = "hold_for_signed_next_stage_review"
BLOCKED_P27_DECISION = "rollback_required_or_hold"
P27_SCOPE = "manual_next_stage_review"


def build_v5_next_stage_review_decision(
    *,
    p26_gate: Mapping[str, Any] | None = None,
    next_stage_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate = _as_dict(p26_gate)
    review = _as_dict(next_stage_review)
    progress = _progress_gates(gate, review)
    decision = _decision(progress, review)
    blocked = _blocked_reasons(progress, decision, gate)
    ready = not blocked and bool(review)
    approved = ready and decision == APPROVED_P27_DECISION
    rejected = ready and decision == REJECTED_P27_DECISION
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_next_stage_review_decision_v0",
        "gate": "v5_next_stage_review_decision",
        "ok": ready,
        "decision_record_ready": ready,
        "next_stage_review_recorded": ready,
        "next_stage_review_signed": ready,
        "signed_next_stage_review_recorded": ready,
        "signed_next_stage_review_signed": ready,
        "approved_for_next_stage_manual_experiment": approved,
        "approved_for_next_contract_stage": approved,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "next_stage_review_decision": decision,
        "rollout_review_decision": decision,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_review_request_fields": {},
        "p26_gate_summary": _p26_summary(gate),
        "next_stage_review": _review_summary(review),
        "next_stage_review_template": _review_template(gate),
        "progress_gates": progress,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(approved, rejected, decision),
        "notes": [
            "This P27 contract records a signed next-stage review only.",
            "Approval does not enable default rollout or request-adapter mapping.",
            "Any rejected review keeps PyTorch AdamW authoritative and default-off.",
        ],
    }


def _progress_gates(gate: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    p25 = _as_dict(gate.get("p25_decision_summary"))
    replicate = _as_dict(gate.get("longer_replicate_summary"))
    failure = _as_dict(gate.get("failure_history_summary"))
    rollback = _as_dict(gate.get("rollback_history_summary"))
    return {
        "p26_gate_present": bool(gate),
        "p26_gate_ready": bool(gate.get("ok", False))
        and bool(gate.get("longer_replicate_failure_history_gate_ready", False))
        and str(gate.get("decision") or gate.get("gate_decision") or gate.get("rollout_review_decision") or "")
        == APPROVED_P26_DECISION,
        "p26_manual_next_stage_allowed": bool(gate.get("manual_next_stage_review_allowed", False)),
        "p26_no_blockers": not _string_list(gate.get("blocked_reasons"))
        and not _string_list(gate.get("promotion_blockers")),
        "p26_default_off": _default_off_confirmed(gate),
        "p26_request_adapter_off": gate.get("request_adapter_mapping_allowed") is False
        and gate.get("request_fields_emitted") is False,
        "p26_post_fields_empty": not bool(_as_dict(gate.get("post_gate_request_fields"))),
        "p26_p25_approved_default_off": bool(p25.get("approved_default_off_decision", False))
        and bool(p25.get("approved_for_next_stage", False))
        and bool(p25.get("owner_rollout_review_recorded", False))
        and bool(p25.get("owner_rollout_review_signed", False))
        and not bool(p25.get("rollback_required", False)),
        "p26_longer_replicate_summary_ready": bool(replicate.get("ready", False))
        and not _string_list(replicate.get("blocked_reasons")),
        "p26_failure_history_clear": bool(failure.get("clear_for_p26", False)),
        "p26_rollback_history_clear": bool(rollback.get("clear_for_p26", False)),
        "signed_next_stage_review_present": bool(review),
        "requested_scope_valid": str(review.get("requested_scope") or "") == P27_SCOPE,
        "defaults_confirmed_off": _review_defaults_off(review),
        "review_request_adapter_not_requested": not bool(review.get("approve_request_adapter_mapping_allowed", False))
        and not bool(review.get("approve_request_fields_emitted", False)),
        "p26_gate_acknowledged": bool(review.get("acknowledge_p26_gate_ready", False)),
        "request_adapter_acknowledged_off": bool(review.get("acknowledge_no_request_adapter_mapping", False)),
        "runtime_evidence_acknowledged": bool(review.get("acknowledge_runtime_evidence_complete", False)),
        "manual_review_only_acknowledged": bool(review.get("acknowledge_manual_review_only", False)),
        "longer_replicate_acknowledged": bool(review.get("acknowledge_longer_replicate_evidence", False)),
        "failure_history_acknowledged_clear": bool(review.get("acknowledge_failure_history_clear", False)),
        "rollback_history_acknowledged_clear": bool(review.get("acknowledge_rollback_history_clear", False)),
    }


def _decision(progress: Mapping[str, bool], review: Mapping[str, Any]) -> str:
    if not bool(progress.get("signed_next_stage_review_present", False)):
        return HOLD_P27_DECISION
    required = (
        "p26_gate_ready",
        "p26_manual_next_stage_allowed",
        "p26_no_blockers",
        "p26_post_fields_empty",
        "p26_p25_approved_default_off",
        "p26_longer_replicate_summary_ready",
        "p26_failure_history_clear",
        "p26_rollback_history_clear",
        "requested_scope_valid",
        "defaults_confirmed_off",
        "review_request_adapter_not_requested",
        "p26_gate_acknowledged",
        "request_adapter_acknowledged_off",
        "runtime_evidence_acknowledged",
        "manual_review_only_acknowledged",
        "longer_replicate_acknowledged",
        "failure_history_acknowledged_clear",
        "rollback_history_acknowledged_clear",
    )
    if any(not bool(progress.get(name, False)) for name in required):
        return BLOCKED_P27_DECISION
    if bool(review.get("approve_next_stage_manual_experiment", False)):
        return APPROVED_P27_DECISION
    return REJECTED_P27_DECISION


def _blocked_reasons(progress: Mapping[str, bool], decision: str, gate: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not bool(progress.get("p26_gate_present", False)):
        blocked.append("v5_p27_p26_gate_missing")
    if not bool(progress.get("p26_gate_ready", False)):
        blocked.append("v5_p27_p26_gate_not_ready")
        blocked.extend(_string_list(gate.get("blocked_reasons")))
        blocked.extend(_string_list(gate.get("promotion_blockers")))
    if not bool(progress.get("p26_manual_next_stage_allowed", False)):
        blocked.append("v5_p27_p26_manual_next_stage_not_allowed")
    if not bool(progress.get("p26_no_blockers", False)):
        blocked.append("v5_p27_p26_blockers_present")
        blocked.extend(_string_list(gate.get("blocked_reasons")))
        blocked.extend(_string_list(gate.get("promotion_blockers")))
    if not bool(progress.get("p26_default_off", False)):
        blocked.append("v5_p27_p26_default_off_violation")
    if not bool(progress.get("p26_request_adapter_off", False)):
        blocked.append("v5_p27_p26_request_adapter_violation")
    if not bool(progress.get("p26_post_fields_empty", False)):
        blocked.append("v5_p27_p26_post_fields_present")
    if not bool(progress.get("p26_p25_approved_default_off", False)):
        blocked.append("v5_p27_p26_p25_summary_not_approved_default_off")
    if not bool(progress.get("p26_longer_replicate_summary_ready", False)):
        blocked.append("v5_p27_p26_longer_replicate_summary_not_ready")
    if not bool(progress.get("p26_failure_history_clear", False)):
        blocked.append("v5_p27_p26_failure_history_not_clear")
    if not bool(progress.get("p26_rollback_history_clear", False)):
        blocked.append("v5_p27_p26_rollback_history_not_clear")
    if decision == HOLD_P27_DECISION:
        blocked.append("v5_p27_signed_next_stage_review_missing")
    if decision == BLOCKED_P27_DECISION:
        if not bool(progress.get("requested_scope_valid", False)):
            blocked.append("v5_p27_requested_scope_invalid")
        if not bool(progress.get("defaults_confirmed_off", False)):
            blocked.append("v5_p27_default_off_confirmation_missing")
        if not bool(progress.get("review_request_adapter_not_requested", False)):
            blocked.append("v5_p27_review_request_adapter_requested")
        if not bool(progress.get("p26_gate_acknowledged", False)):
            blocked.append("v5_p27_p26_gate_ack_missing")
        if not bool(progress.get("request_adapter_acknowledged_off", False)):
            blocked.append("v5_p27_request_adapter_ack_missing")
        if not bool(progress.get("runtime_evidence_acknowledged", False)):
            blocked.append("v5_p27_runtime_evidence_ack_missing")
        if not bool(progress.get("manual_review_only_acknowledged", False)):
            blocked.append("v5_p27_manual_review_only_ack_missing")
        if not bool(progress.get("longer_replicate_acknowledged", False)):
            blocked.append("v5_p27_longer_replicate_ack_missing")
        if not bool(progress.get("failure_history_acknowledged_clear", False)):
            blocked.append("v5_p27_failure_history_clear_ack_missing")
        if not bool(progress.get("rollback_history_acknowledged_clear", False)):
            blocked.append("v5_p27_rollback_history_clear_ack_missing")
    return _dedupe(blocked)


def _p26_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    replicate = _as_dict(gate.get("longer_replicate_summary"))
    return {
        "present": bool(gate),
        "source_path": str(gate.get("_source_path") or gate.get("source_path") or ""),
        "ok": bool(gate.get("ok", False)),
        "longer_replicate_failure_history_gate_ready": bool(
            gate.get("longer_replicate_failure_history_gate_ready", False)
        ),
        "decision": str(gate.get("decision") or gate.get("gate_decision") or gate.get("rollout_review_decision") or ""),
        "default_rollout_allowed": bool(gate.get("default_rollout_allowed", False)),
        "auto_rollout_allowed": bool(gate.get("auto_rollout_allowed", False)),
        "request_adapter_mapping_allowed": bool(gate.get("request_adapter_mapping_allowed", False)),
        "request_fields_emitted": bool(gate.get("request_fields_emitted", False)),
        "post_gate_request_fields_present": bool(_as_dict(gate.get("post_gate_request_fields"))),
        "run_count": replicate.get("run_count"),
        "min_speedup": replicate.get("min_speedup"),
        "speedup_spread_ratio": replicate.get("speedup_spread_ratio"),
        "blocked_reasons": _string_list(gate.get("blocked_reasons")),
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_next_stage_manual_experiment": bool(
            review.get("approve_next_stage_manual_experiment", False)
        ),
        "approve_default_training_path_enabled": bool(review.get("approve_default_training_path_enabled", False)),
        "approve_default_rollout_allowed": bool(review.get("approve_default_rollout_allowed", False)),
        "approve_auto_rollout_allowed": bool(review.get("approve_auto_rollout_allowed", False)),
        "approve_request_adapter_mapping_allowed": bool(
            review.get("approve_request_adapter_mapping_allowed", False)
        ),
        "approve_request_fields_emitted": bool(review.get("approve_request_fields_emitted", False)),
        "acknowledge_p26_gate_ready": bool(review.get("acknowledge_p26_gate_ready", False)),
        "acknowledge_no_request_adapter_mapping": bool(review.get("acknowledge_no_request_adapter_mapping", False)),
        "acknowledge_runtime_evidence_complete": bool(review.get("acknowledge_runtime_evidence_complete", False)),
        "acknowledge_manual_review_only": bool(review.get("acknowledge_manual_review_only", False)),
        "acknowledge_longer_replicate_evidence": bool(
            review.get("acknowledge_longer_replicate_evidence", False)
        ),
        "acknowledge_failure_history_clear": bool(review.get("acknowledge_failure_history_clear", False)),
        "acknowledge_rollback_history_clear": bool(review.get("acknowledge_rollback_history_clear", False)),
    }


def _review_template(gate: Mapping[str, Any]) -> dict[str, Any]:
    replicate = _as_dict(gate.get("longer_replicate_summary"))
    return {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": P27_SCOPE,
        "approve_next_stage_manual_experiment": False,
        "approve_default_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "acknowledge_p26_gate_ready": False,
        "acknowledge_no_request_adapter_mapping": False,
        "acknowledge_runtime_evidence_complete": False,
        "acknowledge_manual_review_only": False,
        "acknowledge_longer_replicate_evidence": False,
        "acknowledge_failure_history_clear": False,
        "acknowledge_rollback_history_clear": False,
        "acknowledged_longer_replicate_run_count": replicate.get("run_count"),
        "acknowledged_min_speedup": replicate.get("min_speedup"),
        "acknowledged_speedup_spread_ratio": replicate.get("speedup_spread_ratio"),
    }


def _recommended_next_step(approved: bool, rejected: bool, decision: str) -> str:
    if approved:
        return "record P27 approval for manual next-stage experiments; default rollout remains off"
    if rejected:
        return "record P27 rejection and keep PyTorch AdamW authoritative"
    if decision == HOLD_P27_DECISION:
        return "collect a signed P27 next-stage owner review"
    return "hold P27 until P26 evidence and review acknowledgements are complete"


def _default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def _review_defaults_off(review: Mapping[str, Any]) -> bool:
    return bool(
        review.get("approve_default_training_path_enabled") is False
        and review.get("approve_default_rollout_allowed") is False
        and review.get("approve_auto_rollout_allowed") is False
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
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
    parser = argparse.ArgumentParser(description="Build V5 P27 next-stage review decision contract.")
    parser.add_argument("--p26-gate", default="", help="P26 longer-replicate gate JSON.")
    parser.add_argument("--next-stage-review", default="", help="Signed P27 next-stage review JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_next_stage_review_decision(
        p26_gate=load_json(args.p26_gate) if args.p26_gate else None,
        next_stage_review=load_json(args.next_stage_review) if args.next_stage_review else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_next_stage_review_decision"]
