"""Dispatch manifest contract for reviewed T-LoRA A/B cases.

The manifest is the handoff from owner review to a separate request dispatcher.
It emits auditable payloads but does not submit jobs or execute training.
"""

from __future__ import annotations

from typing import Any, Mapping


def build_tlora_ab_dispatch_manifest(
    *,
    manual_review: Mapping[str, Any],
    request_patch_plan: Mapping[str, Any],
    dispatcher_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    review = dict(manual_review)
    plan = dict(request_patch_plan)
    contract = dict(dispatcher_contract or {})
    patches = [dict(item) for item in plan.get("patches", []) if isinstance(item, Mapping)]
    blockers: list[str] = []

    if review.get("scorecard") != "tlora_ab_manual_dispatch_review_v0":
        blockers.append("unexpected_manual_review")
    if not bool(review.get("review_ready", review.get("ok", False))):
        blockers.append("manual_review_not_ready")
    if not bool(review.get("real_dispatch_allowed", False)):
        blockers.append("manual_review_dispatch_not_allowed")
    if plan.get("plan") != "tlora_ab_request_patch_plan_v0":
        blockers.append("unexpected_request_patch_plan")
    if not bool(plan.get("request_fields_emitted", False)):
        blockers.append("request_fields_not_emitted")
    if not bool(plan.get("dry_run_only", False)):
        blockers.append("request_patch_dry_run_boundary_missing")
    if bool(review.get("training_path_enabled", False)) or bool(plan.get("training_path_enabled", False)):
        blockers.append("unsafe_child_training_path_enabled")
    if bool(contract.get("auto_submit", False)):
        blockers.append("auto_submit_not_allowed")
    if not str(contract.get("dispatcher") or "").strip():
        blockers.append("dispatcher_missing")
    if not patches:
        blockers.append("dispatch_patches_missing")

    payloads = [_dispatch_payload(item, contract) for item in patches]
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_dispatch_manifest_v0",
        "ok": ready,
        "dispatch_manifest_ready": ready,
        "dispatch_payloads_emitted": ready,
        "execution_performed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "real_dispatch_allowed": ready,
        "case_count": int(plan.get("case_count") or review.get("case_count") or 0),
        "payload_count": len(payloads),
        "dispatcher": str(contract.get("dispatcher") or ""),
        "payloads": payloads if ready else [],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hand payloads to the normal request dispatcher for representative A/B runs"
            if ready
            else "complete manual review and dispatcher contract before dispatch manifest handoff"
        ),
    }


def _dispatch_payload(row: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(row.get("request_patch") or {})
    request["dry_run"] = False
    request["tlora_ab_dispatch_reviewed"] = True
    request["tlora_ab_dispatcher"] = str(contract.get("dispatcher") or "")
    return {
        "case_id": str(row.get("case_id") or ""),
        "arm": str(row.get("arm") or ""),
        "family": str(row.get("family") or ""),
        "expected_result_path": str(row.get("expected_result_path") or ""),
        "request": request,
    }


__all__ = ["build_tlora_ab_dispatch_manifest"]
