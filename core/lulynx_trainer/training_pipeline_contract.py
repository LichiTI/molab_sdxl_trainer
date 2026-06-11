"""Contracts for the planned Lulynx staged training runtime.

The current trainer already has data loading, batch collation, forward, loss,
backward, optimizer, and telemetry phases, but much of that orchestration lives
inside the training loop. This module defines a small internal contract for the
release-before-refactor pipeline without copying any reference implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LulynxHookAccess(str, Enum):
    READONLY = "readonly"
    ADVISORY = "advisory"
    TRANSFORM = "transform"
    CONTROL = "control"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class LulynxTrainingStage:
    id: str
    order: int
    hook_access: tuple[LulynxHookAccess, ...]
    mutable_fields: tuple[str, ...] = ()
    required_contracts: tuple[str, ...] = ()


LULYNX_TRAINING_STAGES: tuple[LulynxTrainingStage, ...] = (
    LulynxTrainingStage(
        "dataset_scan",
        10,
        (LulynxHookAccess.READONLY, LulynxHookAccess.ADVISORY),
        required_contracts=("dataset_descriptor",),
    ),
    LulynxTrainingStage(
        "bucket_plan",
        20,
        (LulynxHookAccess.READONLY, LulynxHookAccess.ADVISORY, LulynxHookAccess.TRANSFORM),
        mutable_fields=("bucket_policy", "per_bucket_batch_size"),
        required_contracts=("bucket_uniform_batch",),
    ),
    LulynxTrainingStage(
        "batch_collate",
        30,
        (LulynxHookAccess.READONLY, LulynxHookAccess.TRANSFORM),
        mutable_fields=("batch",),
        required_contracts=("collate_shape_contract",),
    ),
    LulynxTrainingStage(
        "batch_contract",
        40,
        (LulynxHookAccess.READONLY, LulynxHookAccess.ADVISORY, LulynxHookAccess.CONTROL),
        mutable_fields=("execution_strategy",),
        required_contracts=("lulynx_multi_batch_training_batch_contract_v0",),
    ),
    LulynxTrainingStage(
        "host_to_device",
        50,
        (LulynxHookAccess.READONLY, LulynxHookAccess.ADVISORY, LulynxHookAccess.CONTROL),
        mutable_fields=("transfer_policy",),
        required_contracts=("h2d_transfer_profile",),
    ),
    LulynxTrainingStage(
        "noise_timestep",
        60,
        (LulynxHookAccess.READONLY, LulynxHookAccess.TRANSFORM),
        mutable_fields=("noise", "timesteps", "target"),
        required_contracts=("batch_dim_preserved",),
    ),
    LulynxTrainingStage(
        "conditioning",
        70,
        (LulynxHookAccess.READONLY, LulynxHookAccess.TRANSFORM),
        mutable_fields=("encoder_hidden_states", "attention_mask", "added_cond_kwargs"),
        required_contracts=("batch_dim_preserved",),
    ),
    LulynxTrainingStage(
        "forward",
        80,
        (LulynxHookAccess.READONLY, LulynxHookAccess.ADVISORY, LulynxHookAccess.CONTROL),
        mutable_fields=("attention_backend", "compile_boundary", "execution_strategy"),
        required_contracts=("forward_batch_dim_preserved",),
    ),
    LulynxTrainingStage(
        "loss",
        90,
        (LulynxHookAccess.READONLY, LulynxHookAccess.TRANSFORM),
        mutable_fields=("loss", "loss_weights"),
        required_contracts=("loss_scalar_contract",),
    ),
    LulynxTrainingStage(
        "backward",
        100,
        (LulynxHookAccess.READONLY, LulynxHookAccess.CONTROL),
        mutable_fields=("gradient_policy",),
        required_contracts=("backward_stability_contract",),
    ),
    LulynxTrainingStage(
        "optimizer_step",
        110,
        (LulynxHookAccess.READONLY, LulynxHookAccess.ADVISORY, LulynxHookAccess.CONTROL),
        mutable_fields=("optimizer_policy",),
        required_contracts=("optimizer_step_contract",),
    ),
    LulynxTrainingStage(
        "telemetry",
        120,
        (LulynxHookAccess.READONLY, LulynxHookAccess.ADVISORY),
        required_contracts=("phase_profile", "bubble_profile"),
    ),
)


def lulynx_training_stage_ids() -> list[str]:
    return [stage.id for stage in sorted(LULYNX_TRAINING_STAGES, key=lambda item: item.order)]


def build_lulynx_training_pipeline_contract() -> dict[str, Any]:
    stages = sorted(LULYNX_TRAINING_STAGES, key=lambda item: item.order)
    return {
        "schema_version": 1,
        "contract": "lulynx_training_pipeline_contract_v0",
        "status": "planned_internal_refactor_contract",
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "agpl_risk_policy": "behavior_contract_only_reimplemented_in_house",
        "stage_count": len(stages),
        "stages": [
            {
                "id": stage.id,
                "order": stage.order,
                "hook_access": [access.value for access in stage.hook_access],
                "mutable_fields": list(stage.mutable_fields),
                "required_contracts": list(stage.required_contracts),
            }
            for stage in stages
        ],
        "plugin_hook_tiers": [access.value for access in LulynxHookAccess],
        "required_invariants": [
            "runtime_request_boundary_preserved",
            "no_new_training_entrypoint",
            "launcher_does_not_host_webui_or_backend_training_logic",
            "physical_batch_size_separate_from_effective_batch_size",
            "bucket_uniform_batches_before_native_batch_forward",
            "plugins_must_declare_hook_access_tier",
            "diagnostic_hooks_do_not_create_release_claims",
        ],
    }


def validate_lulynx_pipeline_hook_request(
    *,
    stage_id: str,
    requested_access: str,
    requested_mutations: Sequence[str] | None = None,
) -> dict[str, Any]:
    stage_by_id = {stage.id: stage for stage in LULYNX_TRAINING_STAGES}
    stage = stage_by_id.get(str(stage_id))
    requested = str(requested_access or "").strip().lower()
    mutations = [str(item) for item in (requested_mutations or [])]
    if stage is None:
        return {
            "ok": False,
            "reason": "unknown_stage",
            "stage_id": str(stage_id),
            "requested_access": requested,
        }
    allowed_access = {access.value for access in stage.hook_access}
    allowed_mutations = set(stage.mutable_fields)
    mutation_errors = [item for item in mutations if item not in allowed_mutations]
    ok = requested in allowed_access and not mutation_errors
    reasons: list[str] = []
    if requested not in allowed_access:
        reasons.append("access_tier_not_allowed_for_stage")
    if mutation_errors:
        reasons.append("mutation_not_allowed_for_stage")
    return {
        "ok": ok,
        "stage_id": stage.id,
        "requested_access": requested,
        "allowed_access": sorted(allowed_access),
        "requested_mutations": mutations,
        "allowed_mutations": sorted(allowed_mutations),
        "mutation_errors": mutation_errors,
        "reasons": reasons,
    }


def build_lulynx_pipeline_refactor_roadmap_item() -> dict[str, Any]:
    return {
        "id": "p_pre_release_training_pipeline_refactor",
        "priority": "before_multi_batch_promotion",
        "status": "planned",
        "rationale": (
            "Refactor the internal training flow into staged contracts before release, so native batch 2/4/8, "
            "bubble-aware scheduling, compile boundaries, and plugin hooks share one stable boundary."
        ),
        "phases": [
            "stage_contracts_and_hook_tiers",
            "batch_contract_and_bucket_batch_policy",
            "train_step_stage_extraction",
            "native_batch_microbatch_single_item_strategy",
            "phase_telemetry_and_plugin_hook_gate",
            "long_window_batch2_4_8_stability_matrix",
        ],
    }


__all__ = [
    "LULYNX_TRAINING_STAGES",
    "LulynxHookAccess",
    "LulynxTrainingStage",
    "build_lulynx_pipeline_refactor_roadmap_item",
    "build_lulynx_training_pipeline_contract",
    "lulynx_training_stage_ids",
    "validate_lulynx_pipeline_hook_request",
]
