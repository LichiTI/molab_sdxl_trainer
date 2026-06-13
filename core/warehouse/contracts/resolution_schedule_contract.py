"""
Y-2 - Mixed-Resolution & Staged Training - Interface Contract

Defines the plan data model and planner/runner protocols for multi-phase
resolution training (e.g. 512->768->1024).

This module contains NO behavioral implementation.  It specifies:
- What a resolution phase plan looks like (frozen data model).
- What a phase planner must compute (batch scaling, epoch alignment).
- What a staged runner must orchestrate (sequential phase execution).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PhaseStatus(enum.Enum):
    """Lifecycle status of a single resolution phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResumeStrategy(enum.Enum):
    """How the runner should handle resume state across phases."""
    FRESH = "fresh"                   # No resume; start from scratch.
    EXPLICIT = "explicit"             # Resume from user-supplied path.
    AUTO_PREVIOUS_PHASE = "auto_prev" # Resume from the previous phase's saved state.
    PLAN_MATCHED = "plan_matched"     # Resume from a state that matches the current plan_id.


# ---------------------------------------------------------------------------
# Data Models - Phase plan (frozen, pure data, no behavior)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolutionPhase:
    """A single phase within a mixed-resolution training plan.

    Describes the resolution target, batch scaling, and step budget
    for this phase.  All values are pre-computed by the planner.
    """
    phase_index: int
    label: str                          # e.g. "512x512", "768x768"
    stage_side: int                     # target max side length
    resolution: tuple[int, int]         # (width, height)
    ratio_percent: float                # fraction of total training budget

    # Batch configuration (already scaled for this resolution)
    batch_size_global: int
    batch_size_per_device: int
    world_size: int
    gradient_accumulation_steps: int

    # Step/epoch budget
    steps_per_epoch: int
    raw_epochs: int
    actual_epochs: int                  # after alignment rounding
    epoch_rounding_multiple: int
    epoch_scale_factor: float           # relative to base resolution

    # Save/sample cadence for this phase
    save_every_n_epochs: int | None
    sample_every_n_epochs: int | None

    # Cumulative positioning (relative to the full plan)
    phase_steps: int
    start_step: int
    cumulative_steps: int
    start_epoch: int
    cumulative_epochs: int

    # Display helpers
    loop_epoch_base: int
    epoch_display_offset: int


@dataclass(frozen=True)
class ResolutionPlan:
    """Complete mixed-resolution training plan.

    Produced by a PhasePlanner.  Consumed by a StagedRunner.
    Immutable once created.
    """
    enabled: bool
    plan_id: str                        # stable hash of the plan parameters
    world_size: int
    total_samples: int
    base_resolution: tuple[int, int]
    base_batch_size: int
    base_batch_size_global: int
    base_batch_size_per_device: int
    base_gradient_accumulation_steps: int
    base_save_every_n_epochs: int | None
    base_sample_every_n_epochs: int | None
    alignment_epochs: int
    total_ratio_percent: float
    total_mixed_epochs: int
    total_mixed_steps: int
    phases: tuple[ResolutionPhase, ...]
    warnings: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PhaseRunConfig:
    """Per-phase configuration that a runner passes to the underlying trainer.

    Wraps the original training config with phase-specific overrides
    (resolution, batch size, max steps, resume path, etc.).
    """
    phase: ResolutionPhase
    config_overrides: dict[str, Any]    # key/value pairs to apply
    resume_strategy: ResumeStrategy
    resume_path: str | None


@dataclass(frozen=True)
class PhaseResult:
    """Outcome of executing one phase."""
    phase_index: int
    status: PhaseStatus
    exit_code: int
    saved_state_path: str | None
    final_step: int | None
    final_epoch: int | None
    error_message: str | None = None


@dataclass(frozen=True)
class RunSummary:
    """Final outcome of a complete staged-resolution run."""
    plan_id: str
    total_phases: int
    completed_phases: int
    failed_phase_index: int | None
    final_model_path: str | None
    final_state_path: str | None
    phase_results: tuple[PhaseResult, ...]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class PhasePlannerProtocol(Protocol):
    """Computes a ResolutionPlan from training config + dataset metadata.

    Implementations MUST:
    - Validate that mixed-resolution is applicable to the training type.
    - Scale batch sizes proportionally to resolution area.
    - Align epoch counts to save/sample cadence LCM.
    - Produce a deterministic plan_id from the plan parameters.
    - Emit warnings for any unsupported combinations.
    """

    def build_plan(
        self,
        config: dict[str, Any],
        *,
        training_type: str,
        total_samples: int,
        world_size: int,
    ) -> ResolutionPlan:
        """Build a complete phase plan from the training config."""
        ...

    def build_phase_run_configs(
        self,
        plan: ResolutionPlan,
        base_config: dict[str, Any],
    ) -> Sequence[PhaseRunConfig]:
        """Generate per-phase run configs from an existing plan."""
        ...


@runtime_checkable
class StagedRunnerProtocol(Protocol):
    """Orchestrates sequential execution of resolution phases.

    Implementations MUST:
    - Execute phases in order (index 0..N).
    - Handle inter-phase resume (state from phase N -> resume in phase N+1).
    - Clear dataset caches when resolution changes between phases.
    - Return a RunSummary with per-phase results.
    """

    def run(
        self,
        plan: ResolutionPlan,
        phase_configs: Sequence[PhaseRunConfig],
        *,
        base_config: dict[str, Any],
    ) -> RunSummary:
        """Execute all phases sequentially. Blocks until complete or failure."""
        ...

    def infer_resume(
        self,
        plan: ResolutionPlan,
        phase_configs: Sequence[PhaseRunConfig],
    ) -> tuple[int, ResumeStrategy, str | None]:
        """Determine where to resume from (start phase index, strategy, path).

        Returns (start_phase_index, resume_strategy, resume_state_path).
        """
        ...


@runtime_checkable
class CacheInvalidatorProtocol(Protocol):
    """Handles dataset cache invalidation when resolution changes between phases."""

    def should_invalidate(
        self,
        previous_resolution: tuple[int, int] | None,
        next_resolution: tuple[int, int],
    ) -> bool:
        """Return True if caches should be cleared for this resolution change."""
        ...

    def invalidate(self, config: dict[str, Any]) -> None:
        """Clear dataset npz/metadata caches for the given config."""
        ...
