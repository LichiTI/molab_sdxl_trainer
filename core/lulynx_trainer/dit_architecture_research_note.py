"""Structured note for non-drop-in DiT architecture research candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DiTArchitectureCandidate:
    key: str
    year: int
    primary_goal: str
    mechanism: str
    checkpoint_compatible: bool = False
    adapter_only: bool = False
    trainer_speed_relevant: bool = True
    quality_relevant: bool = True
    requires_new_model_family: bool = True

    def normalized(self) -> "DiTArchitectureCandidate":
        return DiTArchitectureCandidate(
            key=_slug(self.key),
            year=int(self.year),
            primary_goal=str(self.primary_goal or "unknown").strip() or "unknown",
            mechanism=str(self.mechanism or "unknown").strip() or "unknown",
            checkpoint_compatible=bool(self.checkpoint_compatible),
            adapter_only=bool(self.adapter_only),
            trainer_speed_relevant=bool(self.trainer_speed_relevant),
            quality_relevant=bool(self.quality_relevant),
            requires_new_model_family=bool(self.requires_new_model_family),
        )


DEFAULT_DIT_ARCHITECTURE_CANDIDATES: tuple[DiTArchitectureCandidate, ...] = (
    DiTArchitectureCandidate(
        key="dit_air",
        year=2025,
        primary_goal="compact DiT training and protocol efficiency",
        mechanism="layer-wise sharing and DiT/MMDiT/PixArt design ablation",
    ),
    DiTArchitectureCandidate(
        key="edit_linear_compressed_attention",
        year=2025,
        primary_goal="high-resolution attention compute reduction",
        mechanism="linear/compressed image-token attention while preserving prompt interactions",
    ),
    DiTArchitectureCandidate(
        key="lit_linear_dit",
        year=2025,
        primary_goal="lightweight deployment and simplified training",
        mechanism="linear transformer path for diffusion transformers",
    ),
    DiTArchitectureCandidate(
        key="ec_dit",
        year=2025,
        primary_goal="MoE routing for convergence and alignment",
        mechanism="expert-choice routing over text/image patch complexity",
    ),
    DiTArchitectureCandidate(
        key="skip_dit",
        year=2025,
        primary_goal="stable efficient DiT training",
        mechanism="long skip connections with spectral constraints",
    ),
    DiTArchitectureCandidate(
        key="ledit",
        year=2025,
        primary_goal="high-resolution extrapolation",
        mechanism="length-extrapolatable DiT without position encoding",
    ),
    DiTArchitectureCandidate(
        key="ddt",
        year=2025,
        primary_goal="faster convergence and detail quality",
        mechanism="decoupled semantic extraction and high-frequency denoising",
    ),
    DiTArchitectureCandidate(
        key="rae_lv_rae",
        year=2026,
        primary_goal="better latent space for DiT training",
        mechanism="representation autoencoder latent/tokenizer replacement",
    ),
)


def build_dit_architecture_research_note(
    candidates: Sequence[DiTArchitectureCandidate | Mapping[str, Any]] | None = None,
    *,
    owned_model_family: bool = False,
    checkpoint_compatible_required: bool = True,
) -> dict[str, Any]:
    rows = [
        _classify_candidate(_candidate(candidate), owned_model_family=owned_model_family, checkpoint_required=checkpoint_compatible_required)
        for candidate in (candidates or DEFAULT_DIT_ARCHITECTURE_CANDIDATES)
    ]
    trainer_candidates = [row for row in rows if row["trainer_speed_relevant"] or row["quality_relevant"]]
    blocked = [row for row in rows if row["blocked_reasons"]]
    return {
        "schema_version": 1,
        "note": "dit_architecture_research_note_v0",
        "candidate_count": len(rows),
        "trainer_relevant_count": len(trainer_candidates),
        "drop_in_candidate_count": sum(1 for row in rows if row["drop_in_ready"]),
        "design_only_count": sum(1 for row in rows if row["recommended_track"] == "design_only"),
        "owned_model_family": bool(owned_model_family),
        "checkpoint_compatible_required": bool(checkpoint_compatible_required),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "candidates": rows,
        "recommended_next_step": "keep architecture replacements design-only until Lulynx owns a new DiT family route",
        "blocked_reasons": sorted({reason for row in blocked for reason in row["blocked_reasons"]}),
    }


def build_dit_architecture_research_scorecard(note: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(note)
    drop_in_count = int(payload.get("drop_in_candidate_count") or 0)
    blockers = list(payload.get("blocked_reasons") or [])
    if drop_in_count == 0 and "no_drop_in_architecture_candidate" not in blockers:
        blockers.append("no_drop_in_architecture_candidate")
    if not bool(payload.get("owned_model_family")) and "new_model_family_decision_missing" not in blockers:
        blockers.append("new_model_family_decision_missing")
    return {
        "schema_version": 1,
        "scorecard": "dit_architecture_research_note_v0",
        "ok": bool(payload.get("candidate_count")) and bool(payload.get("trainer_relevant_count")),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "note": payload,
        "blocked_reasons": blockers,
        "recommended_next_step": payload.get("recommended_next_step"),
    }


def _classify_candidate(
    candidate: DiTArchitectureCandidate,
    *,
    owned_model_family: bool,
    checkpoint_required: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if checkpoint_required and not candidate.checkpoint_compatible:
        blockers.append("not_checkpoint_compatible")
    if candidate.requires_new_model_family and not owned_model_family:
        blockers.append("new_model_family_required")
    if candidate.adapter_only:
        track = "adapter_reference"
    elif candidate.checkpoint_compatible and not candidate.requires_new_model_family:
        track = "trainer_probe"
    else:
        track = "design_only"
    drop_in_ready = not blockers and track != "design_only"
    return {
        "key": candidate.key,
        "year": candidate.year,
        "primary_goal": candidate.primary_goal,
        "mechanism": candidate.mechanism,
        "checkpoint_compatible": candidate.checkpoint_compatible,
        "adapter_only": candidate.adapter_only,
        "trainer_speed_relevant": candidate.trainer_speed_relevant,
        "quality_relevant": candidate.quality_relevant,
        "requires_new_model_family": candidate.requires_new_model_family,
        "recommended_track": track,
        "drop_in_ready": drop_in_ready,
        "blocked_reasons": blockers,
    }


def _candidate(candidate: DiTArchitectureCandidate | Mapping[str, Any]) -> DiTArchitectureCandidate:
    if isinstance(candidate, Mapping):
        return DiTArchitectureCandidate(**candidate).normalized()
    return candidate.normalized()


def _slug(value: str) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_") or "unknown"
