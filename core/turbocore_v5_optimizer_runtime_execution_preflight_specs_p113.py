"""Stage spec for optimizer runtime-execution preflight contract P113."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_training_launch_execution_specs_p112 import P112_SPEC


COMMON_SECTIONS = (
    "runtime_execution_plan_inventory",
    "runtime_execution_precondition_inventory",
    "runtime_adapter_boundary",
    "runtime_state_boundary",
    "runtime_dispatch_boundary",
    "native_dispatch_boundary",
    "kernel_launch_boundary",
    "parity_boundary",
    "tensor_transfer_boundary",
    "training_step_boundary",
    "operator_runtime_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_runtime_execution_boundary",
    "no_runtime_state_refresh_boundary",
    "no_runtime_adapter_enabled_boundary",
    "no_runtime_dispatch_execution_boundary",
    "no_native_dispatch_execution_boundary",
    "no_kernel_launch_execution_boundary",
    "no_parity_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_training_step_execution_boundary",
    "no_training_launch_execution_boundary",
    "no_training_runtime_start_boundary",
    "no_run_dispatch_boundary",
    "no_run_record_write_boundary",
    "no_scheduler_dispatch_boundary",
    "no_request_execution_boundary",
    "no_job_creation_boundary",
    "no_queue_enqueue_boundary",
    "no_request_submission_boundary",
    "no_schema_config_router_ui_patch_boundary",
    "no_default_rollout_boundary",
)


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P113_SPEC = OptimizerLateStageSpec(
    stage_id=113,
    token="optimizer_runtime_execution_preflight",
    scope="optimizer_runtime_execution_preflight_contract",
    title="optimizer runtime-execution preflight",
    previous_token="optimizer_training_launch_execution",
    previous_label="P112 optimizer training-launch execution contract",
    previous_ready_decision=P112_SPEC.ready_decision,
    previous_ready_field="optimizer_training_launch_execution_contract_ready",
    previous_evidence_field="optimizer_training_launch_execution_evidence_recorded",
    previous_signed_field="optimizer_training_launch_execution_signed",
    previous_post_fields="post_p112_request_fields",
    previous_ack="acknowledge_p112_optimizer_training_launch_execution_contract_recorded",
    package_ready_field="optimizer_runtime_execution_preflight_package_ready",
    policy_ready_field="runtime_execution_preflight_policy_ready",
    row_keys=("optimizer_runtime_execution_preflight_rows", "optimizer_runtime_preflight_rows"),
    row_ready_field="runtime_execution_preflight_review_ready",
    later_field="later_optimizer_runtime_dispatch_execution_contract_required",
    later_ack="acknowledge_later_optimizer_runtime_dispatch_execution_contract_required",
    next_contract="optimizer_runtime_dispatch_execution",
    allowed_intents=frozenset(
        {
            "runtime_execution_preflight_candidate",
            "hold_for_more_evidence",
            "reject_runtime_execution_preflight",
        }
    ),
    required_sections=_sections(
        "p112_optimizer_training_launch_execution_contract_reference",
        "optimizer_runtime_execution_preflight_package",
        "per_optimizer_runtime_execution_preflight_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_runtime_execution_preflight_applied",
        "optimizer_runtime_execution_preflight_enabled",
        "optimizer_runtime_execution_preflight_executed",
        "runtime_execution_approved",
        "runtime_execution_allowed",
        "runtime_execution_enabled",
        "runtime_execution_executed",
        "runtime_execution_started",
        "runtime_request_executed",
        "runtime_state_refresh_allowed",
        "runtime_state_refreshed",
        "runtime_adapter_enabled",
        "runtime_dispatch_allowed",
        "runtime_dispatch_enabled",
        "runtime_dispatch_executed",
        "native_dispatch_allowed",
        "native_dispatch_enabled",
        "native_dispatch_executed",
        "kernel_launch_allowed",
        "kernel_launch_executed",
        "parity_executed",
        "tensor_transfer_executed",
        "training_step_executed",
        "training_launch_executed",
        "training_runtime_started",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_runtime_execution_preflight_payload",
        "runtime_execution_preflight_payload",
        "runtime_execution_payload",
        "runtime_adapter_payload",
        "runtime_state_payload",
        "runtime_dispatch_payload",
        "native_dispatch_payload",
        "kernel_launch_payload",
        "parity_payload",
        "tensor_transfer_payload",
        "training_step_payload",
    ),
    inherited_unsafe_true_fields=P112_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P112_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P113_SPEC"]
