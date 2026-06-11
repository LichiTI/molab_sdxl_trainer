"""Report-only limited probe plan after internal-gate manual review.

This module prepares a constrained planning package for a future non-release
probe after manual review approval. It never enables the internal gate, starts
training, or authorizes release claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_PLAN = (
    "lulynx_internal_gate_limited_non_release_probe_plan_v0"
)
APPROVED_DECISION = "internal_gate_enablement_review_recorded_default_off"


def build_lulynx_internal_gate_limited_non_release_probe_plan(
    *,
    internal_gate_enablement_review_decision: Mapping[str, Any] | None = None,
    internal_gate_enablement_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed limited probe planning package."""

    decision = dict(_mapping(internal_gate_enablement_review_decision))
    review = dict(_mapping(internal_gate_enablement_review))
    review_checks = _mapping(review.get("checks"))
    review_summary = _mapping(decision.get("review_package_summary"))
    release_claim_leaks = _release_claim_leaks(decision=decision, review=review)

    checks = {
        "decision_record_ready": bool(decision.get("decision_record_ready")),
        "decision_approved_for_probe_planning": bool(
            decision.get("approved_for_limited_non_release_probe_planning")
        )
        and str(decision.get("decision") or "") == APPROVED_DECISION,
        "internal_gate_stays_disabled": bool(decision.get("internal_gate_stays_disabled")),
        "internal_gate_enablement_not_allowed": not bool(decision.get("internal_gate_enablement_allowed")),
        "review_package_ready": bool(review.get("passed"))
        and str(review.get("status") or "") == "ready_for_manual_internal_gate_review",
        "review_keeps_execution_path_disabled": bool(
            review_checks.get("behavior_equivalent_execution_path_not_enabled")
        ),
        "review_keeps_batch2_release_probe_blocked": bool(
            review_checks.get("batch2_release_probe_still_blocked")
        ),
        "release_claim_closed": not release_claim_leaks,
    }
    blockers = _blockers_from_checks(checks)
    if not decision:
        blockers.append("internal_gate_enablement_review_decision_missing")
    if not review:
        blockers.append("internal_gate_enablement_review_missing")
    blockers.extend(
        f"internal_gate_enablement_review_decision:{item}"
        for item in _string_list(decision.get("blocked_reasons"))
    )
    blockers.extend(
        f"internal_gate_enablement_review:{item}"
        for item in _string_list(review.get("blockers"))
    )
    blockers.extend(f"release_claim_leak:{item}" for item in release_claim_leaks)
    blockers = _dedupe(blockers)
    ready = not blockers

    probe_plan = {
        "probe_scope": "behavior_equivalent_internal_gate_non_release_probe",
        "probe_batch_contract": "real_gpu_batch1_only",
        "probe_release_policy": "non_release_only",
        "internal_gate_must_remain_disabled_until_explicit_execution_contract": True,
        "requires_explicit_execution_contract": True,
        "allowed_probe_surface": [
            "manual_plan_review",
            "batch1_non_release_probe_design",
            "evidence_package_refresh",
        ],
        "blocked_probe_surface": [
            "turn_internal_gate_on_now",
            "new_training_entrypoint",
            "batch2_4_8_release_probe",
            "release_claim",
            "default_enablement",
        ],
        "candidate_work_items": [
            "define_limited_batch1_non_release_probe_inputs",
            "record_probe_guardrails_and_stop_conditions",
            "prepare_followup_execution_contract_without_enabling_internal_gate",
        ],
    }

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_PLAN,
        "status": "ready_for_limited_non_release_probe_planning" if ready else "blocked",
        "passed": ready,
        "manual_review_required": True,
        "safe_to_auto_start": False,
        "internal_gate_enablement_allowed": False,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "checks": checks,
        "blockers": blockers,
        "deferred_release_probe_blockers": _string_list(review.get("deferred_release_probe_blockers")),
        "decision_summary": _decision_summary(decision),
        "review_summary": {
            "present": bool(review),
            "status": str(review.get("status") or ""),
            "passed": bool(review.get("passed")),
            "deferred_release_probe_blocker_count": len(
                _string_list(review.get("deferred_release_probe_blockers"))
            ),
            "review_package_status": str(review_summary.get("status") or ""),
        },
        "probe_plan": probe_plan,
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _decision_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "ok": bool(report.get("ok")),
        "decision": str(report.get("decision") or ""),
        "approved_for_limited_non_release_probe_planning": bool(
            report.get("approved_for_limited_non_release_probe_planning")
        ),
        "internal_gate_stays_disabled": bool(report.get("internal_gate_stays_disabled")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "draft_limited_non_release_probe_execution_contract",
            "keep_internal_gate_disabled_until_explicit_execution_contract",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_limited_non_release_probe_planning_prerequisites"]
    if any("decision" in item for item in blockers):
        actions.append("collect_signed_internal_gate_enablement_review_decision")
    if any("review" in item for item in blockers):
        actions.append("refresh_internal_gate_enablement_review_package")
    if any("release_claim_leak" in item for item in blockers):
        actions.append("close_release_claim_leaks_before_probe_planning")
    return _dedupe(actions)


def _release_claim_leaks(*, decision: Mapping[str, Any], review: Mapping[str, Any]) -> list[str]:
    leaks: list[str] = []
    for name, payload in (
        ("internal_gate_enablement_review_decision", decision),
        ("internal_gate_enablement_review", review),
    ):
        if bool(_mapping(payload).get("release_claim_allowed")):
            leaks.append(name)
    return leaks


def _blockers_from_checks(checks: Mapping[str, bool]) -> list[str]:
    return [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_PLAN",
    "build_lulynx_internal_gate_limited_non_release_probe_plan",
]
