"""Phase 6 DiT training frontier readiness aggregator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# Keep Phase 6 readiness focused on representative hard gates; detailed
# boundary artifacts are recorded as extra evidence instead of blocking release.
DEFAULT_PHASE6_REQUIRED_EVIDENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "primitive_contracts": (
        "tread_token_route_fixture_v0",
        "sra2_haste_alignment_facade_v0",
        "soft_tokens_adapter_primitive_v0",
        "modulation_guidance_reconciliation_v0",
        "chimera_hydra_adapter_primitive_v0",
    ),
    "ab_evidence": (
        "dit_frontier_ab_result_summary_from_artifacts_v0",
        "dit_compute_reducer_trainer_ab_case_summary_from_artifacts_v0",
    ),
    "quality_review": (
        "adapter_target_quality_review_decision_v0",
        "dit_frontier_ab_quality_review_bridge_v0",
        "dit_compute_reducer_quality_review_decision_v0",
    ),
    "safe_default_off_boundary": (),
}

DEFAULT_PHASE6_REQUIRED_EVIDENCE: tuple[str, ...] = tuple(
    evidence_id
    for group in DEFAULT_PHASE6_REQUIRED_EVIDENCE_GROUPS.values()
    for evidence_id in group
)


@dataclass(frozen=True)
class DiTFrontierEvidenceRow:
    evidence_id: str
    present: bool
    ok: bool
    training_path_enabled: bool
    default_behavior_changed: bool
    promotion_ready: bool
    blocked_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "present": bool(self.present),
            "ok": bool(self.ok),
            "training_path_enabled": bool(self.training_path_enabled),
            "default_behavior_changed": bool(self.default_behavior_changed),
            "promotion_ready": bool(self.promotion_ready),
            "blocked_reasons": list(self.blocked_reasons),
        }


def build_dit_training_frontier_readiness(
    evidence: Sequence[Mapping[str, Any]],
    *,
    required_evidence: Sequence[str] = DEFAULT_PHASE6_REQUIRED_EVIDENCE,
) -> dict[str, Any]:
    by_id = {_evidence_id(item): dict(item) for item in evidence if _evidence_id(item)}
    rows = tuple(_row(required, by_id.get(required)) for required in required_evidence)
    extra_ids = sorted(key for key in by_id if key not in set(required_evidence))
    missing = [row.evidence_id for row in rows if not row.present]
    unsafe_training = [row.evidence_id for row in rows if row.training_path_enabled]
    unsafe_default = [row.evidence_id for row in rows if row.default_behavior_changed]
    unsafe_promotion = [row.evidence_id for row in rows if row.promotion_ready]
    blockers: list[str] = []
    blockers.extend(f"missing:{item}" for item in missing)
    blockers.extend(f"unsafe_training_path_enabled:{item}" for item in unsafe_training)
    blockers.extend(f"default_behavior_changed:{item}" for item in unsafe_default)
    blockers.extend(f"unexpected_promotion_ready:{item}" for item in unsafe_promotion)
    primitive_ready = not missing and not unsafe_training and not unsafe_default and not unsafe_promotion
    return {
        "schema_version": 1,
        "scorecard": "dit_training_frontier_phase6_readiness_v0",
        "ok": primitive_ready,
        "primitive_or_note_ready": primitive_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "gate_count": len(DEFAULT_PHASE6_REQUIRED_EVIDENCE_GROUPS),
        "gate_ids": list(DEFAULT_PHASE6_REQUIRED_EVIDENCE_GROUPS),
        "required_count": len(tuple(required_evidence)),
        "present_count": sum(1 for row in rows if row.present),
        "extra_evidence_ids": extra_ids,
        "rows": [row.as_dict() for row in rows],
        "blocked_reasons": blockers,
        "recommended_next_step": "wire selected default-off trainer contracts only after real A/B quality gates",
    }


def _row(evidence_id: str, payload: Mapping[str, Any] | None) -> DiTFrontierEvidenceRow:
    if payload is None:
        return DiTFrontierEvidenceRow(
            evidence_id=evidence_id,
            present=False,
            ok=False,
            training_path_enabled=False,
            default_behavior_changed=False,
            promotion_ready=False,
            blocked_reasons=("missing_evidence",),
        )
    return DiTFrontierEvidenceRow(
        evidence_id=evidence_id,
        present=True,
        ok=bool(payload.get("ok", payload.get("facade_ready", payload.get("profile_ready", True)))),
        training_path_enabled=bool(payload.get("training_path_enabled", False)),
        default_behavior_changed=bool(payload.get("default_behavior_changed", False)),
        promotion_ready=bool(payload.get("promotion_ready", False)),
        blocked_reasons=tuple(str(item) for item in payload.get("blocked_reasons", ()) or ()),
    )


def _evidence_id(payload: Mapping[str, Any]) -> str:
    for key in ("scorecard", "plan", "contract", "facade", "note", "evidence_id"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


__all__ = [
    "DEFAULT_PHASE6_REQUIRED_EVIDENCE_GROUPS",
    "DEFAULT_PHASE6_REQUIRED_EVIDENCE",
    "DiTFrontierEvidenceRow",
    "build_dit_training_frontier_readiness",
]
