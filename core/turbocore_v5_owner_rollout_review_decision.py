"""Signed owner-rollout review decision contract for V5 manual wider canary.

This module records the outcome of a human rollout review over a P23 review
package. It does not emit request-adapter fields and never enables default
training or rollout behavior.
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


def build_v5_owner_rollout_review_decision(
    *,
    run_review_package: Mapping[str, Any] | None = None,
    owner_rollout_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package = _as_dict(run_review_package)
    review = _as_dict(owner_rollout_review)
    gates = _progress_gates(package, review)
    decision = _decision(gates, review)
    blocked = _blocked_reasons(gates, decision, package)
    ready = not blocked and bool(review)
    approved = ready and str(decision) == "owner_rollout_review_recorded_default_off"
    rejected = ready and str(decision) == "owner_rollout_review_rejected_default_off"
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_owner_rollout_review_decision_v0",
        "gate": "v5_owner_rollout_review_decision",
        "ok": ready,
        "decision_record_ready": ready,
        "owner_rollout_review_recorded": ready,
        "owner_rollout_review_signed": ready,
        "approved_for_next_stage": approved,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "rollout_decision": decision,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "run_review_package_summary": _package_summary(package),
        "owner_rollout_review": _review_summary(review),
        "owner_rollout_review_template": _review_template(package),
        "progress_gates": gates,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(ready, approved, rejected),
        "notes": [
            "This decision contract records a signed owner review but keeps default behavior off.",
            "Even an approved review does not emit request-adapter fields.",
            "A rejected review is still a valid recorded decision and keeps PyTorch authoritative.",
        ],
    }


def _progress_gates(package: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "run_review_package_present": bool(package),
        "run_review_package_ready": bool(package.get("run_review_package_ready", False))
        and bool(package.get("ready_for_owner_rollout_review", False)),
        "run_review_default_off": _package_default_off(package),
        "run_review_request_adapter_off": not bool(package.get("request_adapter_mapping_allowed", True))
        and not bool(package.get("request_fields_emitted", True)),
        "signed_owner_rollout_review_present": bool(review),
        "requested_scope_valid": _scope_ok(review),
        "defaults_confirmed_off": _defaults_confirmed_off(review),
        "request_adapter_acknowledged_off": bool(review.get("acknowledge_no_request_adapter_mapping", False)),
        "runtime_evidence_acknowledged": bool(review.get("acknowledge_runtime_evidence_complete", False)),
        "manual_review_only_acknowledged": bool(review.get("acknowledge_manual_review_only", False)),
    }


def _decision(gates: Mapping[str, bool], review: Mapping[str, Any]) -> str:
    if not bool(gates.get("signed_owner_rollout_review_present", False)):
        return "hold_for_signed_owner_rollout_review"
    if not bool(gates.get("run_review_package_ready", False)):
        return "rollback_required_or_hold"
    if not bool(gates.get("requested_scope_valid", False)):
        return "rollback_required_or_hold"
    if not bool(gates.get("defaults_confirmed_off", False)):
        return "rollback_required_or_hold"
    if not bool(gates.get("request_adapter_acknowledged_off", False)):
        return "rollback_required_or_hold"
    if not bool(gates.get("runtime_evidence_acknowledged", False)):
        return "rollback_required_or_hold"
    if not bool(gates.get("manual_review_only_acknowledged", False)):
        return "rollback_required_or_hold"
    if bool(review.get("approve_keep_manual_wider_canary_evidence", False)):
        return "owner_rollout_review_recorded_default_off"
    return "owner_rollout_review_rejected_default_off"


def _blocked_reasons(gates: Mapping[str, bool], decision: str, package: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if not bool(gates.get("run_review_package_present", False)):
        blocked.append("v5_p25_run_review_package_missing")
    if not bool(gates.get("run_review_package_ready", False)):
        blocked.append("v5_p25_run_review_package_not_ready")
        blocked.extend(_string_list(package.get("blocked_reasons")))
    if not bool(gates.get("run_review_default_off", False)):
        blocked.append("v5_p25_run_review_default_off_violation")
    if not bool(gates.get("run_review_request_adapter_off", False)):
        blocked.append("v5_p25_run_review_request_adapter_violation")
    if decision == "hold_for_signed_owner_rollout_review":
        blocked.append("v5_p25_signed_owner_rollout_review_missing")
    if decision == "rollback_required_or_hold":
        if not bool(gates.get("requested_scope_valid", False)):
            blocked.append("v5_p25_requested_scope_invalid")
        if not bool(gates.get("defaults_confirmed_off", False)):
            blocked.append("v5_p25_default_off_confirmation_missing")
        if not bool(gates.get("request_adapter_acknowledged_off", False)):
            blocked.append("v5_p25_request_adapter_ack_missing")
        if not bool(gates.get("runtime_evidence_acknowledged", False)):
            blocked.append("v5_p25_runtime_evidence_ack_missing")
        if not bool(gates.get("manual_review_only_acknowledged", False)):
            blocked.append("v5_p25_manual_review_only_ack_missing")
    return _dedupe(blocked)


def _package_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(package),
        "source_path": str(package.get("_source_path") or ""),
        "run_review_package_ready": bool(package.get("run_review_package_ready", False)),
        "ready_for_owner_rollout_review": bool(package.get("ready_for_owner_rollout_review", False)),
        "rollout_review_decision": str(package.get("rollout_review_decision") or ""),
        "request_adapter_mapping_allowed": bool(package.get("request_adapter_mapping_allowed", False)),
        "request_fields_emitted": bool(package.get("request_fields_emitted", False)),
        "default_rollout_allowed": bool(package.get("default_rollout_allowed", False)),
        "auto_rollout_allowed": bool(package.get("auto_rollout_allowed", False)),
        "blocked_reasons": _string_list(package.get("blocked_reasons")),
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_keep_manual_wider_canary_evidence": bool(
            review.get("approve_keep_manual_wider_canary_evidence", False)
        ),
        "approve_default_training_path_enabled": bool(review.get("approve_default_training_path_enabled", False)),
        "approve_default_rollout_allowed": bool(review.get("approve_default_rollout_allowed", False)),
        "approve_auto_rollout_allowed": bool(review.get("approve_auto_rollout_allowed", False)),
        "acknowledge_no_request_adapter_mapping": bool(
            review.get("acknowledge_no_request_adapter_mapping", False)
        ),
        "acknowledge_runtime_evidence_complete": bool(
            review.get("acknowledge_runtime_evidence_complete", False)
        ),
        "acknowledge_manual_review_only": bool(review.get("acknowledge_manual_review_only", False)),
    }


def _review_template(package: Mapping[str, Any]) -> dict[str, Any]:
    run_summary = _as_dict(package.get("run_audit_summary"))
    performance = _as_dict(run_summary.get("performance"))
    return {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": "manual_wider_canary_owner_rollout_review",
        "approve_keep_manual_wider_canary_evidence": False,
        "approve_default_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "acknowledge_no_request_adapter_mapping": False,
        "acknowledge_runtime_evidence_complete": False,
        "acknowledge_manual_review_only": False,
        "acknowledged_native_case": str(run_summary.get("native_case") or ""),
        "acknowledged_representative_end_to_end_speedup": performance.get("representative_end_to_end_speedup"),
    }


def _recommended_next_step(ready: bool, approved: bool, rejected: bool) -> str:
    if approved:
        return "record signed owner rollout review; default rollout remains off"
    if rejected:
        return "record rejection and keep PyTorch AdamW authoritative"
    if ready:
        return "owner rollout review was recorded"
    return "collect a signed owner rollout review over the P23 package"


def _scope_ok(review: Mapping[str, Any]) -> bool:
    return str(review.get("requested_scope", "") or "") == "manual_wider_canary_owner_rollout_review"


def _defaults_confirmed_off(review: Mapping[str, Any]) -> bool:
    return bool(
        review.get("approve_default_training_path_enabled") is False
        and review.get("approve_default_rollout_allowed") is False
        and review.get("approve_auto_rollout_allowed") is False
    )


def _package_default_off(package: Mapping[str, Any]) -> bool:
    return bool(
        package.get("default_training_path_enabled") is False
        and package.get("training_path_enabled") is False
        and package.get("default_rollout_allowed") is False
        and package.get("auto_rollout_allowed") is False
    )


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
    parser = argparse.ArgumentParser(description="Build V5 owner rollout review decision contract.")
    parser.add_argument("--run-review-package", default="", help="P23 owner rollout review package JSON.")
    parser.add_argument("--owner-rollout-review", default="", help="Signed owner rollout review JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_owner_rollout_review_decision(
        run_review_package=load_json(args.run_review_package) if args.run_review_package else None,
        owner_rollout_review=load_json(args.owner_rollout_review) if args.owner_rollout_review else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_owner_rollout_review_decision"]
