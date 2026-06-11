"""Manual dispatch review gate for T-LoRA A/B runs.

This gate turns a complete dry-run evidence package into an explicit owner
review artifact. It still does not run training; it only records whether a
separate dispatcher may submit the representative A/B cases.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


APPROVED_DECISIONS = {"approved", "approve", "approved_for_dispatch"}


def build_tlora_ab_manual_dispatch_review(
    evidence_package: Mapping[str, Any],
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package = dict(evidence_package)
    payload = dict(review or {})
    blockers: list[str] = []
    case_count = int(package.get("case_count") or 0)
    reviewed_case_ids = tuple(str(item) for item in payload.get("reviewed_case_ids", ()) or ())
    decision = str(payload.get("decision") or "").strip().lower()

    if package.get("scorecard") != "tlora_ab_evidence_package_v0":
        blockers.append("unexpected_evidence_package")
    if not bool(package.get("package_ready", package.get("ok", False))):
        blockers.append("evidence_package_not_ready")
    if bool(package.get("training_path_enabled", False)) or bool(package.get("promotion_ready", False)):
        blockers.append("unsafe_evidence_package_flag")
    if decision not in APPROVED_DECISIONS:
        blockers.append("manual_decision_not_approved")
    if not str(payload.get("reviewer") or "").strip():
        blockers.append("reviewer_missing")
    if not str(payload.get("package_digest") or payload.get("artifact_digest") or "").strip():
        blockers.append("package_digest_missing")
    if not bool(payload.get("risk_acknowledged", False)):
        blockers.append("risk_acknowledgement_missing")
    if case_count > 0 and reviewed_case_ids and len(set(reviewed_case_ids)) != case_count:
        blockers.append("reviewed_case_count_mismatch")

    review_ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_manual_dispatch_review_v0",
        "ok": review_ready,
        "review_ready": review_ready,
        "real_dispatch_allowed": review_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "case_count": case_count,
        "reviewed_case_count": len(set(reviewed_case_ids)),
        "review": _summary(payload, ("decision", "reviewer", "package_digest", "artifact_digest", "risk_acknowledged")),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "submit reviewed representative A/B cases through the normal request dispatcher"
            if review_ready
            else "complete owner review before any real T-LoRA A/B dispatch"
        ),
    }


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = ["build_tlora_ab_manual_dispatch_review"]
