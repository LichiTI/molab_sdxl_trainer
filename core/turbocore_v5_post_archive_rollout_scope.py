"""Post-archive rollout scope classifier for TurboCore V5-P36.

P36 consumes the P35 archive replay verification plus optional readiness/audit
evidence and records what can be claimed next. It is deliberately a classifier:
it does not launch training, emit request-adapter fields, expose UI controls, or
turn archived canary evidence into default rollout permission.
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


P35_READY_DECISION = "p32_final_archive_replay_verification_ready_default_off"
P36_READY_DECISION = "post_archive_rollout_scope_classified_default_off"
P36_BLOCKED_DECISION = "post_archive_rollout_scope_blocked_default_off"
P36_SCOPE = "post_archive_rollout_scope_classification"


def build_v5_post_archive_rollout_scope(
    *,
    p35_archive_replay_verification: Mapping[str, Any] | None = None,
    readiness_report: Mapping[str, Any] | None = None,
    roadmap_audit: Mapping[str, Any] | None = None,
    rollout_policy_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify post-P35 rollout scope while keeping every active path off."""

    p35 = _as_dict(p35_archive_replay_verification)
    readiness = _as_dict(readiness_report)
    audit = _as_dict(roadmap_audit)
    review = _as_dict(rollout_policy_review)
    p35_summary = _p35_summary(p35)
    readiness_summary = _readiness_summary(readiness)
    audit_summary = _audit_summary(audit)
    review_summary = _review_summary(review)
    classification = _classify_scopes(p35_summary, readiness_summary, audit_summary, review_summary)
    classification_blockers = _classification_blockers(p35_summary, readiness_summary, audit_summary, review_summary)
    rollout_blockers = _rollout_blockers(classification)
    ready = not classification_blockers
    decision = P36_READY_DECISION if ready else P36_BLOCKED_DECISION
    broader_route_evidence_ready = bool(
        ready
        and classification["broader_real_route_coverage"]["state"] == "ready"
        and classification["packaging_observability"]["state"] == "ready"
        and classification["controlled_rollout_policy"]["state"] == "recorded"
    )
    return {
        "schema_version": 1,
        "package": "turbocore_v5_post_archive_rollout_scope_v0",
        "gate": "v5_post_archive_rollout_scope",
        "ok": ready,
        "scope_classification_ready": ready,
        "post_archive_rollout_scope_classified": ready,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "ui_exposure_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_scope_request_fields": {},
        "p35_archive_replay_summary": p35_summary,
        "readiness_summary": readiness_summary,
        "roadmap_audit_summary": audit_summary,
        "rollout_policy_review": review_summary,
        "scope_classification": classification,
        "controlled_rollout_policy_recorded": bool(
            ready and review_summary.get("present") and review_summary.get("valid")
        ),
        "broader_route_evidence_claim_ready": broader_route_evidence_ready,
        "broader_rollout_claim_ready": False,
        "rollout_authorization_allowed": False,
        "blocked_reasons": classification_blockers,
        "classification_blockers": classification_blockers,
        "rollout_blockers": rollout_blockers,
        "promotion_blockers": _dedupe(classification_blockers + rollout_blockers),
        "allowed_next_actions": _allowed_next_actions(ready, classification, classification_blockers),
        "recommended_next_step": _recommended_next_step(ready, classification_blockers, rollout_blockers),
        "notes": [
            "P36 classifies scope after P35; it does not approve a broader rollout.",
            "Internal canary evidence and product/UI exposure are tracked separately.",
            "Any actual rollout still needs a later explicit default-off policy contract.",
        ],
    }


def _p35_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "ok": bool(report.get("ok", False)),
        "archive_replay_verification_ready": bool(report.get("archive_replay_verification_ready", False)),
        "archived_ledger_matches_replay": bool(report.get("archived_ledger_matches_replay", False)),
        "decision": str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or ""),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "training_launch_allowed": bool(report.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(report.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(report.get("runs_dispatched", True)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", True)),
        "manual_review_required": bool(report.get("manual_review_required", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "post_fields_empty": not bool(_as_dict(report.get("post_replay_verification_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "ready": _p35_ready(report),
    }


def _p35_ready(report: Mapping[str, Any]) -> bool:
    return bool(
        report
        and report.get("ok") is True
        and report.get("archive_replay_verification_ready") is True
        and report.get("archived_ledger_matches_replay") is True
        and str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
        == P35_READY_DECISION
        and report.get("training_launch_allowed") is False
        and report.get("auto_launch_allowed") is False
        and report.get("runs_dispatched") is False
        and report.get("default_behavior_changed") is False
        and report.get("manual_review_required") is True
        and not bool(report.get("ui_exposure_allowed", False))
        and _default_off_confirmed(report)
        and _request_adapter_off(report)
        and not _as_dict(report.get("post_replay_verification_request_fields"))
        and not _string_list(report.get("blocked_reasons"))
    )


def _readiness_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    sections = _as_dict(report.get("sections"))
    native_update = _as_dict(sections.get("native_update_promotion_scorecard"))
    lora = _as_dict(sections.get("lora_promotion_scorecard"))
    primary_blockers = _string_list(summary.get("native_update_promotion_blockers"))
    evidence_blockers = _readiness_evidence_blockers(report, summary, primary_blockers)
    evidence_valid = bool(report) and not evidence_blockers
    return {
        "present": bool(report),
        "summary_ok": bool(summary.get("ok", False)),
        "ready_for_ui": bool(summary.get("ready_for_ui", False)),
        "native_training_path_locked": bool(summary.get("native_training_path_locked", True)),
        "native_update_promotion_ready": bool(
            summary.get("native_update_promotion_ready", native_update.get("promotion_ready", False))
        ),
        "lora_promotion_ready": bool(summary.get("lora_promotion_ready", lora.get("promotion_ready", False))),
        "native_stub_schema_complete": bool(summary.get("native_stub_schema_complete", False)),
        "workspace_lifecycle_ok": bool(summary.get("workspace_data_pipeline_lifecycle_ok", False)),
        "broader_real_route_coverage_ready": bool(summary.get("broader_real_route_coverage_ready", False)),
        "packaging_observability_ready": bool(summary.get("packaging_observability_ready", False)),
        "controlled_rollout_policy_ready": bool(summary.get("controlled_rollout_policy_ready", False)),
        "recommended_next_step": str(summary.get("recommended_next_step") or ""),
        "primary_blockers": primary_blockers,
        "evidence_valid": evidence_valid,
        "evidence_blockers": evidence_blockers,
        "ready": evidence_valid,
    }


def _audit_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    required = _as_dict(report.get("required_promotions"))
    summary = _as_dict(report.get("summary"))
    required_ready = bool(required) and all(bool(value) for value in required.values())
    remaining_blockers = _string_list(report.get("remaining_blockers"))
    evidence_valid = bool(
        report
        and report.get("ok") is True
        and required_ready
        and bool(report.get("post_p5_milestones_completed", False))
        and not remaining_blockers
    )
    return {
        "present": bool(report),
        "ok": bool(report.get("ok", False)),
        "roadmap_completed": bool(report.get("roadmap_completed", False)),
        "required_promotions_ready": required_ready,
        "post_p5_milestones_completed": bool(report.get("post_p5_milestones_completed", False)),
        "ready_gate_count": int(summary.get("ready_gate_count", 0) or 0),
        "required_gate_count": int(summary.get("required_gate_count", 0) or 0),
        "post_p5_ready_gate_count": int(summary.get("post_p5_ready_gate_count", 0) or 0),
        "post_p5_milestone_gate_count": int(summary.get("post_p5_milestone_gate_count", 0) or 0),
        "remaining_blockers": remaining_blockers,
        "recommended_next_step": str(summary.get("recommended_next_step") or ""),
        "evidence_valid": evidence_valid,
        "evidence_blockers": _audit_evidence_blockers(report, required_ready, remaining_blockers),
        "internal_canary_scope_ready": evidence_valid,
        "broader_real_route_coverage_ready": bool(evidence_valid and report.get("broader_real_route_coverage_ready", False)),
        "packaging_observability_ready": bool(evidence_valid and report.get("packaging_observability_ready", False)),
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    valid = _review_valid(review)
    unsafe = _review_unsafe_requests(review)
    return {
        "present": bool(review),
        "valid": valid,
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_scope_classification_record": bool(review.get("approve_scope_classification_record", False)),
        "unsafe_requests": unsafe,
        "acknowledgements": {
            "p35_replay_ready": bool(review.get("acknowledge_p35_archive_replay_ready", False)),
            "default_off": bool(review.get("acknowledge_default_off_boundary", False)),
            "no_ui_exposure": bool(review.get("acknowledge_no_ui_exposure", False)),
            "no_request_adapter": bool(review.get("acknowledge_no_request_adapter_mapping", False)),
            "no_auto_launch": bool(review.get("acknowledge_no_auto_launch", False)),
            "broader_coverage_required": bool(review.get("acknowledge_broader_route_coverage_required", False)),
        },
    }


def _review_valid(review: Mapping[str, Any]) -> bool:
    return bool(
        review
        and bool(str(review.get("reviewer") or "").strip())
        and bool(str(review.get("reviewed_at") or "").strip())
        and str(review.get("requested_scope") or "") == P36_SCOPE
        and bool(review.get("approve_scope_classification_record", False))
        and bool(review.get("acknowledge_p35_archive_replay_ready", False))
        and bool(review.get("acknowledge_default_off_boundary", False))
        and bool(review.get("acknowledge_no_ui_exposure", False))
        and bool(review.get("acknowledge_no_request_adapter_mapping", False))
        and bool(review.get("acknowledge_no_auto_launch", False))
        and bool(review.get("acknowledge_broader_route_coverage_required", False))
        and not _review_unsafe_requests(review)
    )


def _review_unsafe_requests(review: Mapping[str, Any]) -> list[str]:
    unsafe: list[str] = []
    for field in (
        "approve_training_launch_allowed",
        "approve_auto_launch_allowed",
        "approve_runs_dispatched",
        "approve_default_training_path_enabled",
        "approve_training_path_enabled",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
        "approve_ui_exposure_allowed",
        "approve_request_adapter_mapping_allowed",
        "approve_request_fields_emitted",
    ):
        if bool(review.get(field, False)):
            unsafe.append(field)
    return unsafe


def _classify_scopes(
    p35: Mapping[str, Any],
    readiness: Mapping[str, Any],
    audit: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    canary_ready = bool(
        bool(readiness.get("evidence_valid", False))
        and (readiness.get("native_update_promotion_ready", False) or readiness.get("lora_promotion_ready", False))
        or audit.get("internal_canary_scope_ready", False)
    )
    broader_ready = bool(
        bool(readiness.get("evidence_valid", False))
        and readiness.get("broader_real_route_coverage_ready", False)
        or audit.get("broader_real_route_coverage_ready", False)
    )
    packaging_ready = bool(
        bool(readiness.get("evidence_valid", False))
        and readiness.get("packaging_observability_ready", False)
        and readiness.get("workspace_lifecycle_ok", False)
        or audit.get("packaging_observability_ready", False)
    )
    return {
        "archive_replay_evidence": _scope("ready" if p35.get("ready") else "blocked", p35.get("blocked_reasons")),
        "internal_canary_scope": _scope(
            "ready" if canary_ready else "insufficient_evidence",
            [] if canary_ready else ["v5_p36_internal_canary_readiness_evidence_missing"],
        ),
        "broader_real_route_coverage": _scope(
            "ready" if broader_ready else "blocked",
            [] if broader_ready else ["v5_p36_broader_real_route_coverage_missing"],
        ),
        "packaging_observability": _scope(
            "ready" if packaging_ready else "partial",
            [] if packaging_ready else ["v5_p36_packaging_observability_not_complete"],
        ),
        "controlled_rollout_policy": _scope(
            "recorded" if review.get("valid") else "pending_review",
            [] if review.get("valid") else ["v5_p36_signed_scope_classification_review_missing"],
        ),
        "product_ui_exposure": _scope("blocked", ["v5_p36_ui_exposure_requires_later_policy_contract"]),
        "default_or_auto_rollout": _scope("blocked", ["v5_p36_default_or_auto_rollout_requires_later_policy_contract"]),
    }


def _scope(state: str, blockers: Any) -> dict[str, Any]:
    return {"state": state, "blockers": _string_list(blockers)}


def _classification_blockers(
    p35: Mapping[str, Any],
    readiness: Mapping[str, Any],
    audit: Mapping[str, Any],
    review: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p35.get("present", False)):
        blocked.append("v5_p36_p35_archive_replay_missing")
    elif not bool(p35.get("ready", False)):
        blocked.append("v5_p36_p35_archive_replay_not_ready")
        blocked.extend(_string_list(p35.get("blocked_reasons")))
    if bool(readiness.get("ready_for_ui", False)):
        blocked.append("v5_p36_readiness_claims_ui_ready_before_policy")
    if bool(readiness.get("present", False)) and not bool(readiness.get("evidence_valid", False)):
        blocked.extend(_string_list(readiness.get("evidence_blockers")))
    if bool(audit.get("present", False)) and not bool(audit.get("evidence_valid", False)):
        blocked.extend(_string_list(audit.get("evidence_blockers")))
    if bool(review.get("present", False)) and not bool(review.get("valid", False)):
        blocked.append("v5_p36_scope_classification_review_invalid")
        if not str(review.get("reviewer") or "").strip():
            blocked.append("v5_p36_scope_classification_review_reviewer_missing")
        if not str(review.get("reviewed_at") or "").strip():
            blocked.append("v5_p36_scope_classification_review_date_missing")
    if review.get("present") and not review.get("valid"):
        for field in _string_list(review.get("unsafe_requests")):
            blocked.append(f"v5_p36_review_requested_unsafe_field:{field}")
    return _dedupe(blocked)


def _rollout_blockers(classification: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    for scope, payload in classification.items():
        if scope in {"archive_replay_evidence"}:
            continue
        for reason in _string_list(payload.get("blockers")):
            blocked.append(f"{scope}:{reason}")
    return _dedupe(blocked)


def _allowed_next_actions(
    ready: bool,
    classification: Mapping[str, Mapping[str, Any]],
    blockers: list[str],
) -> list[str]:
    if not ready:
        if any("readiness_claims_ui_ready" in item for item in blockers):
            return ["remove_premature_ui_readiness_claim_before_scope_classification"]
        if any("review_requested_unsafe_field" in item or "review_invalid" in item for item in blockers):
            return ["repair_scope_classification_review_without_rollout_or_ui_approval"]
        if any("readiness" in item for item in blockers):
            return ["repair_readiness_evidence_before_scope_classification"]
        if any("roadmap" in item for item in blockers):
            return ["repair_roadmap_audit_evidence_before_scope_classification"]
        return ["repair_p35_archive_replay_before_scope_classification"]
    actions = ["archive_p36_scope_classification"]
    if classification["broader_real_route_coverage"]["state"] != "ready":
        actions.append("collect_broader_real_route_coverage")
    if classification["packaging_observability"]["state"] != "ready":
        actions.append("finish_packaging_observability_evidence")
    if classification["controlled_rollout_policy"]["state"] != "recorded":
        actions.append("collect_signed_scope_classification_review")
    return actions


def _recommended_next_step(ready: bool, blockers: list[str], rollout_blockers: list[str]) -> str:
    if not ready:
        if any("p35" in item for item in blockers):
            return "repair or provide the P35 archive replay verification before classifying rollout scope"
        if any("readiness" in item for item in blockers):
            return "repair readiness evidence before classifying rollout scope"
        if any("roadmap" in item for item in blockers):
            return "repair roadmap audit evidence before classifying rollout scope"
        return "repair invalid scope review or unsafe policy claims before classifying rollout scope"
    if rollout_blockers:
        return "keep default-off; close broader route coverage, observability, and signed policy blockers before rollout"
    return "archive the P36 classification; any rollout authorization still needs a later explicit policy contract"


def _readiness_evidence_blockers(
    report: Mapping[str, Any],
    summary: Mapping[str, Any],
    primary_blockers: list[str],
) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if not summary:
        blocked.append("v5_p36_readiness_summary_missing")
    if summary and summary.get("ok") is not True:
        blocked.append("v5_p36_readiness_summary_not_ok")
    if summary and summary.get("native_training_path_locked") is not True:
        blocked.append("v5_p36_readiness_training_path_not_locked")
    for reason in primary_blockers:
        blocked.append(f"v5_p36_readiness_primary_blocker:{reason}")
    return _dedupe(blocked)


def _audit_evidence_blockers(
    report: Mapping[str, Any],
    required_ready: bool,
    remaining_blockers: list[str],
) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("ok") is not True:
        blocked.append("v5_p36_roadmap_audit_not_ok")
    if not required_ready:
        blocked.append("v5_p36_roadmap_required_promotions_not_ready")
    if report.get("post_p5_milestones_completed") is not True:
        blocked.append("v5_p36_roadmap_post_p5_milestones_not_complete")
    for reason in remaining_blockers:
        blocked.append(f"v5_p36_roadmap_remaining_blocker:{reason}")
    return _dedupe(blocked)


def _default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def _request_adapter_off(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("request_adapter_mapping_allowed") is False
        and value.get("request_fields_emitted") is False
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
    parser = argparse.ArgumentParser(description="Classify V5 post-archive rollout scope.")
    parser.add_argument("--p35-archive-replay-verification", default="", help="P35 replay verification JSON.")
    parser.add_argument("--readiness-report", default="", help="Optional TurboCore readiness report JSON.")
    parser.add_argument("--roadmap-audit", default="", help="Optional native roadmap audit JSON.")
    parser.add_argument("--rollout-policy-review", default="", help="Optional signed P36 scope review JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_post_archive_rollout_scope(
        p35_archive_replay_verification=(
            load_json(args.p35_archive_replay_verification) if args.p35_archive_replay_verification else None
        ),
        readiness_report=load_json(args.readiness_report) if args.readiness_report else None,
        roadmap_audit=load_json(args.roadmap_audit) if args.roadmap_audit else None,
        rollout_policy_review=load_json(args.rollout_policy_review) if args.rollout_policy_review else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_post_archive_rollout_scope"]
