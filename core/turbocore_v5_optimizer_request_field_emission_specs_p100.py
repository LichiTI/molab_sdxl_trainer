"""Stage spec for optimizer request-field emission contract P100."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_late_stage_specs_p95_p99 import P99_SPEC


COMMON_SECTIONS = (
    "request_field_inventory_boundary",
    "field_mapping_boundary",
    "generation_request_boundary",
    "request_schema_boundary",
    "config_adapter_boundary",
    "request_payload_boundary",
    "validation_boundary",
    "backend_router_boundary",
    "router_submission_boundary",
    "training_request_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_request_field_emission_boundary",
    "no_request_payload_materialization_boundary",
    "no_generation_request_patch_boundary",
    "no_request_schema_patch_boundary",
    "no_config_adapter_patch_boundary",
    "no_backend_router_registration_boundary",
    "no_router_submission_boundary",
    "no_training_request_submit_boundary",
    "no_training_step_or_launch_boundary",
    "no_default_rollout_boundary",
)


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P100_SPEC = OptimizerLateStageSpec(
    stage_id=100,
    token="optimizer_request_field_emission",
    scope="optimizer_request_field_emission_contract",
    title="optimizer request-field emission",
    previous_token="optimizer_request_adapter_activation",
    previous_label="P99 optimizer request-adapter activation contract",
    previous_ready_decision=P99_SPEC.ready_decision,
    previous_ready_field="optimizer_request_adapter_activation_contract_ready",
    previous_evidence_field="optimizer_request_adapter_activation_evidence_recorded",
    previous_signed_field="optimizer_request_adapter_activation_signed",
    previous_post_fields="post_p99_request_fields",
    previous_ack="acknowledge_p99_optimizer_request_adapter_activation_contract_recorded",
    package_ready_field="optimizer_request_field_emission_package_ready",
    policy_ready_field="request_field_emission_policy_ready",
    row_keys=("optimizer_request_field_emission_rows", "optimizer_request_field_rows"),
    row_ready_field="request_field_emission_review_ready",
    later_field="later_optimizer_request_submission_contract_required",
    later_ack="acknowledge_later_optimizer_request_submission_contract_required",
    next_contract="optimizer_request_submission",
    allowed_intents=frozenset(
        {
            "request_field_emission_candidate",
            "hold_for_more_evidence",
            "reject_request_field_emission",
        }
    ),
    required_sections=_sections(
        "p99_optimizer_request_adapter_activation_contract_reference",
        "optimizer_request_field_emission_package",
        "per_optimizer_request_field_emission_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_request_field_emission_applied",
        "optimizer_request_field_emission_enabled",
        "optimizer_request_field_emission_executed",
        "request_field_emission_allowed",
        "request_field_emission_enabled",
        "request_field_emission_executed",
        "request_field_generated",
        "request_field_materialized",
        "request_fields_emitted",
        "request_payload_generated",
        "request_payload_materialized",
        "request_payload_built",
        "request_payload_submitted",
        "generation_request_patched",
        "request_schema_patched",
        "config_adapter_patched",
        "backend_router_registered",
        "router_submission_allowed",
        "router_submission_executed",
        "request_submitted",
        "training_request_submitted",
        "training_launch_allowed",
        "training_job_created",
        "run_dispatched",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_request_field_emission_payload",
        "request_field_payload",
        "request_fields",
        "request_payload",
        "generation_request_patch_payload",
        "request_schema_patch_payload",
        "config_adapter_patch_payload",
        "backend_router_payload",
        "router_submission_payload",
        "training_request_payload",
        "training_launch_payload",
    ),
    inherited_unsafe_true_fields=P99_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P99_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P100_SPEC"]
