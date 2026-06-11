"""Report-only execution contract for a limited non-release internal-gate probe.

This contract stays entirely on the planning side. It can record a signed
manual execution-contract review for a future batch1-only probe, but it never
turns the internal gate on, starts training, or opens release claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_EXECUTION_CONTRACT = (
    "lulynx_internal_gate_limited_non_release_probe_execution_contract_v0"
)
READY_PLAN_STATUS = "ready_for_limited_non_release_probe_planning"
SCOPE = "training_step_orchestrator_internal_gate_limited_non_release_probe_execution_contract"
HOLD_DECISION = (
    "internal_gate_limited_non_release_probe_execution_contract_hold_for_signed_review_default_off"
)
APPROVED_DECISION = (
    "internal_gate_limited_non_release_probe_execution_contract_recorded_default_off"
)
REJECTED_DECISION = (
    "internal_gate_limited_non_release_probe_execution_contract_rejected_default_off"
)


def build_lulynx_internal_gate_limited_non_release_probe_execution_contract(
    *,
    internal_gate_limited_non_release_probe_plan: Mapping[str, Any] | None = None,
    signed_execution_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a default-off execution contract package for a later manual probe."""

    plan = dict(_mapping(internal_gate_limited_non_release_probe_plan))
    review = dict(_mapping(signed_execution_review))
    checks = _checks(plan=plan, review=review)
    decision = _decision(checks=checks, review=review)
    blockers = _blockers(plan=plan, checks=checks, decision=decision)
    ready = decision == APPROVED_DECISION and not blockers

    return {
        "schema_version": 1,
        "report": LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_EXECUTION_CONTRACT,
        "status": "ready_for_limited_non_release_probe_execution_contract" if ready else "blocked",
        "passed": ready,
        "manual_review_required": True,
        "safe_to_auto_start": False,
        "decision": decision,
        "internal_gate_enablement_allowed": False,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work": True,
        "does_not_start_dataloader_iteration": True,
        "checks": checks,
        "blockers": blockers,
        "probe_plan_summary": _probe_plan_summary(plan),
        "signed_execution_review": _review_summary(review),
        "signed_execution_review_template": _review_template(),
        "execution_contract": {
            "probe_scope": "behavior_equivalent_internal_gate_batch1_non_release_probe_contract",
            "probe_batch_contract": "real_gpu_batch1_only",
            "probe_release_policy": "non_release_only",
            "allowed_contract_surface": [
                "explicit_execution_contract_record",
                "batch1_only_probe_guardrails",
                "manual_stop_conditions",
                "before_after_evidence_refresh",
            ],
            "blocked_execution_surface": [
                "turn_internal_gate_on_now",
                "start_training_now",
                "new_training_entrypoint",
                "batch2_4_8_release_probe",
                "release_claim",
                "default_enablement",
                "background_auto_enablement",
            ],
            "required_guardrails": [
                "internal_gate_default_off",
                "batch1_only_probe",
                "manual_start_only",
                "non_release_only",
                "before_after_evidence_required",
            ],
            "required_stop_conditions": [
                "loss_regression",
                "throughput_regression",
                "vram_regression",
                "unexpected_runtime_path_diff",
                "missing_manifest_evidence",
            ],
        },
        "recommended_next_actions": _recommended_next_actions(ready=ready, blockers=blockers),
    }


def _checks(*, plan: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    probe_plan = _mapping(plan.get("probe_plan"))
    blocked_surface = set(_string_list(probe_plan.get("blocked_probe_surface")))
    candidate_items = set(_string_list(probe_plan.get("candidate_work_items")))
    return {
        "probe_plan_present": bool(plan),
        "probe_plan_ready": bool(plan.get("passed"))
        and str(plan.get("status") or "") == READY_PLAN_STATUS,
        "internal_gate_stays_disabled": bool(plan.get("checks", {}).get("internal_gate_stays_disabled")),
        "internal_gate_enablement_not_allowed": not bool(
            plan.get("internal_gate_enablement_allowed")
        ),
        "release_claim_closed": not bool(plan.get("release_claim_allowed")),
        "probe_batch1_only": str(probe_plan.get("probe_batch_contract") or "") == "real_gpu_batch1_only",
        "probe_non_release_only": str(probe_plan.get("probe_release_policy") or "") == "non_release_only",
        "requires_explicit_execution_contract": bool(
            probe_plan.get("requires_explicit_execution_contract")
        ),
        "plan_blocks_gate_enablement": "turn_internal_gate_on_now" in blocked_surface,
        "plan_blocks_new_training_entrypoint": "new_training_entrypoint" in blocked_surface,
        "plan_blocks_batch2_4_8_release_probe": "batch2_4_8_release_probe" in blocked_surface,
        "plan_blocks_release_claim": "release_claim" in blocked_surface,
        "plan_blocks_default_enablement": "default_enablement" in blocked_surface,
        "plan_candidate_execution_contract_present": (
            "prepare_followup_execution_contract_without_enabling_internal_gate"
            in candidate_items
        ),
        "signed_execution_review_present": bool(review),
        "requested_scope_valid": str(review.get("requested_scope") or "") == SCOPE,
        "manual_review_only_ack": bool(review.get("acknowledge_manual_review_only")),
        "probe_plan_ready_ack": bool(review.get("acknowledge_probe_plan_ready")),
        "default_off_ack": bool(review.get("acknowledge_internal_gate_stays_disabled")),
        "batch1_only_ack": bool(review.get("acknowledge_batch1_non_release_probe_only")),
        "no_new_training_entrypoint_ack": bool(
            review.get("acknowledge_no_new_training_entrypoint")
        ),
        "no_batch2_release_probe_ack": bool(
            review.get("acknowledge_no_batch2_4_8_release_probe")
        ),
        "no_release_claim_ack": bool(review.get("acknowledge_no_release_claim")),
        "stop_conditions_ack": bool(review.get("acknowledge_stop_conditions_defined")),
        "gate_enable_not_requested": not bool(review.get("approve_turn_internal_gate_on_now")),
        "training_entrypoint_not_requested": not bool(
            review.get("approve_new_training_entrypoint")
        ),
        "batch2_release_probe_not_requested": not bool(
            review.get("approve_batch2_4_8_release_probe")
        ),
        "release_claim_not_requested": not bool(review.get("approve_release_claims")),
        "default_enablement_not_requested": not bool(review.get("approve_default_enablement")),
        "probe_start_not_requested": not bool(review.get("approve_start_probe_now")),
    }


def _decision(*, checks: Mapping[str, bool], review: Mapping[str, Any]) -> str:
    if not bool(checks.get("signed_execution_review_present")):
        return HOLD_DECISION
    required = (
        "probe_plan_ready",
        "internal_gate_stays_disabled",
        "internal_gate_enablement_not_allowed",
        "release_claim_closed",
        "probe_batch1_only",
        "probe_non_release_only",
        "requires_explicit_execution_contract",
        "plan_blocks_gate_enablement",
        "plan_blocks_new_training_entrypoint",
        "plan_blocks_batch2_4_8_release_probe",
        "plan_blocks_release_claim",
        "plan_blocks_default_enablement",
        "plan_candidate_execution_contract_present",
        "requested_scope_valid",
        "manual_review_only_ack",
        "probe_plan_ready_ack",
        "default_off_ack",
        "batch1_only_ack",
        "no_new_training_entrypoint_ack",
        "no_batch2_release_probe_ack",
        "no_release_claim_ack",
        "stop_conditions_ack",
        "gate_enable_not_requested",
        "training_entrypoint_not_requested",
        "batch2_release_probe_not_requested",
        "release_claim_not_requested",
        "default_enablement_not_requested",
        "probe_start_not_requested",
    )
    if any(not bool(checks.get(name, False)) for name in required):
        return HOLD_DECISION
    if bool(review.get("approve_record_execution_only_contract")):
        return APPROVED_DECISION
    return REJECTED_DECISION


def _blockers(
    *,
    plan: Mapping[str, Any],
    checks: Mapping[str, bool],
    decision: str,
) -> list[str]:
    blockers = [f"{name}_failed" for name, passed in checks.items() if not bool(passed)]
    if not plan:
        blockers.append("internal_gate_limited_non_release_probe_plan_missing")
    blockers.extend(
        f"internal_gate_limited_non_release_probe_plan:{item}"
        for item in _string_list(plan.get("blockers"))
    )
    if decision == HOLD_DECISION and not bool(checks.get("signed_execution_review_present")):
        blockers.append("signed_limited_non_release_probe_execution_contract_review_missing")
    if decision == REJECTED_DECISION:
        blockers.append("limited_non_release_probe_execution_contract_not_approved")
    return _dedupe(blockers)


def _probe_plan_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    probe_plan = _mapping(report.get("probe_plan"))
    return {
        "present": bool(report),
        "status": str(report.get("status") or ""),
        "passed": bool(report.get("passed")),
        "internal_gate_enablement_allowed": bool(report.get("internal_gate_enablement_allowed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "probe_scope": str(probe_plan.get("probe_scope") or ""),
        "probe_batch_contract": str(probe_plan.get("probe_batch_contract") or ""),
        "probe_release_policy": str(probe_plan.get("probe_release_policy") or ""),
        "blocked_probe_surface": _string_list(probe_plan.get("blocked_probe_surface")),
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review),
        "requested_scope": str(review.get("requested_scope") or ""),
        "reviewer": str(review.get("reviewer") or ""),
        "approve_record_execution_only_contract": bool(
            review.get("approve_record_execution_only_contract")
        ),
        "approve_turn_internal_gate_on_now": bool(review.get("approve_turn_internal_gate_on_now")),
        "approve_new_training_entrypoint": bool(review.get("approve_new_training_entrypoint")),
        "approve_batch2_4_8_release_probe": bool(
            review.get("approve_batch2_4_8_release_probe")
        ),
        "approve_release_claims": bool(review.get("approve_release_claims")),
        "approve_default_enablement": bool(review.get("approve_default_enablement")),
        "approve_start_probe_now": bool(review.get("approve_start_probe_now")),
    }


def _review_template() -> dict[str, Any]:
    return {
        "requested_scope": SCOPE,
        "reviewer": "",
        "acknowledge_probe_plan_ready": False,
        "acknowledge_internal_gate_stays_disabled": False,
        "acknowledge_batch1_non_release_probe_only": False,
        "acknowledge_no_new_training_entrypoint": False,
        "acknowledge_no_batch2_4_8_release_probe": False,
        "acknowledge_no_release_claim": False,
        "acknowledge_stop_conditions_defined": False,
        "acknowledge_manual_review_only": False,
        "approve_record_execution_only_contract": False,
        "approve_turn_internal_gate_on_now": False,
        "approve_new_training_entrypoint": False,
        "approve_batch2_4_8_release_probe": False,
        "approve_release_claims": False,
        "approve_default_enablement": False,
        "approve_start_probe_now": False,
        "review_notes": "",
    }


def _recommended_next_actions(*, ready: bool, blockers: Sequence[str]) -> list[str]:
    if ready:
        return [
            "draft_manual_batch1_non_release_probe_runbook",
            "refresh_before_after_evidence_templates_without_enabling_internal_gate",
            "keep_batch2_4_8_release_probe_blocked_until_long_window_matrix_passes",
        ]
    actions = ["finish_limited_non_release_probe_execution_contract_prerequisites"]
    if any("signed_execution_review" in item for item in blockers):
        actions.append("collect_signed_limited_non_release_probe_execution_contract_review")
    if any("internal_gate_limited_non_release_probe_plan" in item for item in blockers):
        actions.append("refresh_internal_gate_limited_non_release_probe_plan")
    if any("release_claim" in item for item in blockers):
        actions.append("close_release_claim_leaks_before_execution_contract_review")
    return _dedupe(actions)


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
    "APPROVED_DECISION",
    "HOLD_DECISION",
    "LULYNX_INTERNAL_GATE_LIMITED_NON_RELEASE_PROBE_EXECUTION_CONTRACT",
    "REJECTED_DECISION",
    "SCOPE",
    "build_lulynx_internal_gate_limited_non_release_probe_execution_contract",
]
