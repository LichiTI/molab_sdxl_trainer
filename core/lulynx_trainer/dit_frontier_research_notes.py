"""Structured gates for remaining DiT frontier research notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DiTFrontierCandidate:
    key: str
    group: str
    year: int
    goal: str
    dependency: str
    near_term_after_dependency: bool = False
    requires_new_model_family: bool = False
    trainer_relevant: bool = True
    inference_relevant: bool = False

    def normalized(self) -> "DiTFrontierCandidate":
        return DiTFrontierCandidate(
            key=_slug(self.key),
            group=_slug(self.group),
            year=int(self.year),
            goal=str(self.goal or "unknown").strip() or "unknown",
            dependency=_slug(self.dependency),
            near_term_after_dependency=bool(self.near_term_after_dependency),
            requires_new_model_family=bool(self.requires_new_model_family),
            trainer_relevant=bool(self.trainer_relevant),
            inference_relevant=bool(self.inference_relevant),
        )


DEFAULT_FRONTIER_CANDIDATES: tuple[DiTFrontierCandidate, ...] = (
    DiTFrontierCandidate(
        key="ahpa",
        group="alignment",
        year=2026,
        goal="hierarchical VAE-prior alignment for stronger DiT representations",
        dependency="sra2_real_quality_gate",
        near_term_after_dependency=True,
    ),
    DiTFrontierCandidate(
        key="reg",
        group="alignment",
        year=2025,
        goal="global representation guidance and class-token entanglement",
        dependency="external_representation_target_route",
    ),
    DiTFrontierCandidate(
        key="sara",
        group="alignment",
        year=2025,
        goal="structure-aware/adversarial representation alignment",
        dependency="baseline_repa_or_sra2_ab",
    ),
    DiTFrontierCandidate(
        key="nami_progressive_flow",
        group="progressive_flow",
        year=2025,
        goal="coarse-to-fine rectified-flow stages for faster convergence",
        dependency="spd_multiresolution_route",
        near_term_after_dependency=True,
        inference_relevant=True,
    ),
    DiTFrontierCandidate(
        key="adaln_rotation_modulation",
        group="modulation",
        year=2025,
        goal="magnitude-preserving modulation and rotation-style conditioning",
        dependency="compatible_model_family_route",
        requires_new_model_family=True,
    ),
)


def build_dit_frontier_research_notes(
    candidates: Sequence[DiTFrontierCandidate | Mapping[str, Any]] | None = None,
    *,
    satisfied_dependencies: Sequence[str] | None = None,
    owned_model_family: bool = False,
) -> dict[str, Any]:
    satisfied = {_slug(item) for item in (satisfied_dependencies or ())}
    rows = [_classify(_candidate(candidate), satisfied, owned_model_family) for candidate in (candidates or DEFAULT_FRONTIER_CANDIDATES)]
    ready = [row for row in rows if row["ready_for_next_spike"]]
    return {
        "schema_version": 1,
        "note": "dit_frontier_research_notes_v0",
        "candidate_count": len(rows),
        "ready_for_next_spike_count": len(ready),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "owned_model_family": bool(owned_model_family),
        "satisfied_dependencies": sorted(satisfied),
        "groups": sorted({row["group"] for row in rows}),
        "candidates": rows,
        "blocked_reasons": sorted({reason for row in rows for reason in row["blocked_reasons"]}),
        "recommended_next_step": "satisfy prerequisite quality/runtime gates before implementing the next frontier spike",
    }


def build_dit_frontier_research_scorecard(note: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(note)
    blockers = list(payload.get("blocked_reasons") or [])
    if int(payload.get("ready_for_next_spike_count") or 0) == 0 and "no_frontier_dependency_satisfied" not in blockers:
        blockers.append("no_frontier_dependency_satisfied")
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_research_notes_v0",
        "ok": bool(payload.get("candidate_count")),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "note": payload,
        "blocked_reasons": blockers,
        "recommended_next_step": payload.get("recommended_next_step"),
    }


def _classify(candidate: DiTFrontierCandidate, satisfied: set[str], owned_model_family: bool) -> dict[str, Any]:
    blockers: list[str] = []
    if candidate.dependency not in satisfied:
        blockers.append(f"dependency_missing:{candidate.dependency}")
    if candidate.requires_new_model_family and not owned_model_family:
        blockers.append("new_model_family_required")
    ready = not blockers
    if ready and candidate.near_term_after_dependency:
        track = "next_spike_candidate"
    elif ready:
        track = "research_backlog"
    else:
        track = "blocked_note"
    return {
        "key": candidate.key,
        "group": candidate.group,
        "year": candidate.year,
        "goal": candidate.goal,
        "dependency": candidate.dependency,
        "trainer_relevant": candidate.trainer_relevant,
        "inference_relevant": candidate.inference_relevant,
        "requires_new_model_family": candidate.requires_new_model_family,
        "near_term_after_dependency": candidate.near_term_after_dependency,
        "ready_for_next_spike": ready,
        "recommended_track": track,
        "blocked_reasons": blockers,
    }


def _candidate(candidate: DiTFrontierCandidate | Mapping[str, Any]) -> DiTFrontierCandidate:
    if isinstance(candidate, Mapping):
        return DiTFrontierCandidate(**candidate).normalized()
    return candidate.normalized()


def _slug(value: str) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_") or "unknown"
