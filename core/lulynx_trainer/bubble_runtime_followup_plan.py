"""Follow-up planning for bubble runtime evidence packs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


FOLLOWUP_PLAN_REPORT = "bubble_runtime_followup_plan_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _priority(category: str, status: str) -> int:
    if status == "missing":
        return 10
    priorities = {
        "cache_readiness": 20,
        "action_boundary": 30,
        "loss_guardrail": 40,
        "vram_guardrail": 50,
        "release_claim_gate": 60,
        "evidence_schema": 70,
        "missing_evidence": 80,
    }
    return priorities.get(category, 90)


def _summary_status(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "no_followup_needed"
    categories = {str(item.get("category") or "") for item in items}
    if "cache_readiness" in categories or "missing_evidence" in categories:
        return "blocked_collect_evidence"
    if categories <= {"release_claim_gate"}:
        return "blocked_rebuild_release_wording"
    return "blocked_review_required"


def build_bubble_runtime_followup_plan(
    natural_load_canary: Mapping[str, Any],
    release_claims: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact next-action plan from natural-load gate output."""

    families = [
        _mapping(item)
        for item in natural_load_canary.get("families", [])
        if str(_mapping(item).get("status") or "") != "ready"
    ]
    items: list[dict[str, Any]] = []
    for family in families:
        family_name = str(family.get("family") or "")
        status = str(family.get("status") or "")
        categories = _string_list(family.get("blocking_categories"))
        next_actions = _string_list(family.get("next_actions"))
        reasons = _string_list(family.get("blocked_reasons"))
        gpu_rerun_plan = dict(_mapping(family.get("gpu_rerun_plan")))
        if not categories and reasons:
            categories = ["other"]
        for category in categories or ["missing_evidence"]:
            items.append(
                {
                    "priority": _priority(category, status),
                    "family": family_name,
                    "status": status,
                    "category": category,
                    "next_actions": next_actions or ["inspect_blocked_natural_load_canary_evidence"],
                    "reasons": reasons,
                    "candidate_count": family.get("candidate_count"),
                    "accepted_candidate_count": family.get("accepted_candidate_count"),
                    "gpu_rerun_plan": gpu_rerun_plan,
                }
            )

    items.sort(key=lambda item: (int(item["priority"]), str(item["family"]), str(item["category"])))
    blocked_natural_claims = [
        dict(_mapping(item))
        for item in release_claims.get("blocked_claims", [])
        if str(_mapping(item).get("status") or "") == "blocked_pending_natural_load_canary"
    ]
    return {
        "schema_version": 1,
        "report": FOLLOWUP_PLAN_REPORT,
        "status": _summary_status(items),
        "natural_load_canary_status": str(natural_load_canary.get("status") or ""),
        "release_candidate_allowed": bool(natural_load_canary.get("release_candidate_allowed")),
        "item_count": len(items),
        "blocked_natural_claim_count": len(blocked_natural_claims),
        "items": items,
        "blocker_summary": dict(_mapping(natural_load_canary.get("blocker_summary"))),
        "gpu_rerun_plan": dict(_mapping(natural_load_canary.get("gpu_rerun_plan"))),
    }


__all__ = ["FOLLOWUP_PLAN_REPORT", "build_bubble_runtime_followup_plan"]
