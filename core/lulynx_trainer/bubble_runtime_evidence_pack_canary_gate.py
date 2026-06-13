"""Natural-load canary helpers for bubble evidence packs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


NATURAL_CLAIM_IDS = {
    "natural_data_wait_dataloader_rebuild_observed",
    "natural_data_wait_next_run_workers_prefetch_gain_observed",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _sync_release_gate_fields(report: dict[str, Any]) -> None:
    claim_allowed = str(report.get("release_readiness") or "") == "ready_with_case_specific_wording"
    report["publishable"] = claim_allowed
    report["release_claim_allowed"] = claim_allowed
    report["not_release_evidence"] = not claim_allowed
    report["safe_to_auto_start"] = False
    report["claim_publication_scope"] = (
        "case_specific_benchmark_claims" if claim_allowed else "non_release_benchmark_claims"
    )
    report["supported_benchmark_claims"] = list(report.get("publishable_claims", []))


def natural_load_canary_review_items(canary: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for family in canary.get("families", []):
        mapped = _mapping(family)
        status = str(mapped.get("status") or "")
        if status == "ready":
            continue
        items.append(
            {
                "severity": "medium" if status == "blocked" else "high",
                "type": "natural_load_canary_gate",
                "family": mapped.get("family"),
                "status": status,
                "candidate_count": mapped.get("candidate_count"),
                "accepted_candidate_count": mapped.get("accepted_candidate_count"),
                "reasons": _string_list(mapped.get("blocked_reasons")),
                "blocking_categories": _string_list(mapped.get("blocking_categories")),
                "next_actions": _string_list(mapped.get("next_actions")),
            }
        )
    return items


def gate_natural_claims(release_claims: Mapping[str, Any], canary: Mapping[str, Any]) -> dict[str, Any]:
    gated = dict(release_claims)
    gated["natural_load_canary_release_candidate_allowed"] = bool(canary.get("release_candidate_allowed"))
    gated["natural_load_canary_status"] = str(canary.get("status") or "")
    gated["natural_load_canary_blocker_summary"] = dict(_mapping(canary.get("blocker_summary")))
    if gated["natural_load_canary_release_candidate_allowed"]:
        _sync_release_gate_fields(gated)
        return gated

    publishable: list[dict[str, Any]] = []
    blocked = [dict(_mapping(item)) for item in gated.get("blocked_claims", [])]
    moved = 0
    for claim in gated.get("publishable_claims", []):
        mapped = dict(_mapping(claim))
        if mapped.get("id") not in NATURAL_CLAIM_IDS:
            publishable.append(mapped)
            continue
        moved += 1
        mapped["status"] = "blocked_pending_natural_load_canary"
        mapped["reason"] = "SDXL/Anima/Newbie natural-load canary is not fully ready"
        blocked.append(mapped)

    gated["publishable_claims"] = publishable
    gated["blocked_claims"] = blocked
    gaps = [dict(_mapping(item)) for item in gated.get("evidence_gaps", [])]
    if not any(str(item.get("id") or "") == "natural_load_canary_pending" for item in gaps):
        gaps.append(
            {
                "id": "natural_load_canary_pending",
                "reason": "natural data-wait claims require all required canary families to pass",
                "blocker_summary": dict(_mapping(canary.get("blocker_summary"))),
                "missing_families": _string_list(canary.get("missing_families")),
                "blocked_families": _string_list(canary.get("blocked_families")),
            }
        )
    gated["evidence_gaps"] = gaps
    gated["release_readiness"] = "blocked_pending_evidence"
    _sync_release_gate_fields(gated)
    return gated


__all__ = ["gate_natural_claims", "natural_load_canary_review_items"]
