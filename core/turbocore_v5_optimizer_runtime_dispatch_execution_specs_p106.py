"""Stage spec for optimizer runtime-dispatch execution contract P106."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_runtime_execution_preflight_specs_p105 import P105_SPEC


COMMON_SECTIONS = (
    "runtime_dispatch_execution_plan_inventory",
    "runtime_dispatch_execution_precondition_inventory",
    "runtime_dispatch_authorization_boundary",
    "runtime_adapter_lock_boundary",
    "runtime_state_lock_boundary",
    "runtime_dispatch_handoff_boundary",
    "native_dispatch_boundary",
    "kernel_launch_boundary",
    "parity_boundary",
    "tensor_transfer_boundary",
    "training_step_boundary",
    "operator_dispatch_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_runtime_dispatch_execution_boundary",
    "no_native_dispatch_execution_boundary",
    "no_kernel_launch_execution_boundary",
    "no_parity_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_training_step_execution_boundary",
    "no_runtime_execution_boundary",
    "no_runtime_state_refresh_boundary",
    "no_runtime_adapter_enabled_boundary",
    "no_training_launch_execution_boundary",
    "no_training_runtime_start_boundary",
    "no_run_dispatch_boundary",
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


P106_SPEC = OptimizerLateStageSpec(
    stage_id=106,
    token="optimizer_runtime_dispatch_execution",
    scope="optimizer_runtime_dispatch_execution_contract",
    title="optimizer runtime-dispatch execution",
    previous_token="optimizer_runtime_execution_preflight",
    previous_label="P105 optimizer runtime-execution preflight contract",
    previous_ready_decision=P105_SPEC.ready_decision,
    previous_ready_field="optimizer_runtime_execution_preflight_contract_ready",
    previous_evidence_field="optimizer_runtime_execution_preflight_evidence_recorded",
    previous_signed_field="optimizer_runtime_execution_preflight_signed",
    previous_post_fields="post_p105_request_fields",
    previous_ack="acknowledge_p105_optimizer_runtime_execution_preflight_contract_recorded",
    package_ready_field="optimizer_runtime_dispatch_execution_package_ready",
    policy_ready_field="runtime_dispatch_execution_policy_ready",
    row_keys=("optimizer_runtime_dispatch_execution_rows", "optimizer_runtime_dispatch_rows"),
    row_ready_field="runtime_dispatch_execution_review_ready",
    later_field="later_optimizer_native_dispatch_execution_contract_required",
    later_ack="acknowledge_later_optimizer_native_dispatch_execution_contract_required",
    next_contract="optimizer_native_dispatch_execution",
    allowed_intents=frozenset(
        {
            "runtime_dispatch_execution_candidate",
            "hold_for_more_evidence",
            "reject_runtime_dispatch_execution",
        }
    ),
    required_sections=_sections(
        "p105_optimizer_runtime_execution_preflight_contract_reference",
        "optimizer_runtime_dispatch_execution_package",
        "per_optimizer_runtime_dispatch_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_runtime_dispatch_execution_applied",
        "optimizer_runtime_dispatch_execution_enabled",
        "optimizer_runtime_dispatch_execution_executed",
        "runtime_dispatch_approved",
        "runtime_dispatch_allowed",
        "runtime_dispatch_enabled",
        "runtime_dispatch_executed",
        "runtime_dispatch_started",
        "runtime_dispatch_request_executed",
        "runtime_native_dispatch_handoff_enabled",
        "native_dispatch_allowed",
        "native_dispatch_enabled",
        "native_dispatch_executed",
        "native_dispatch_started",
        "kernel_launch_allowed",
        "kernel_launch_executed",
        "parity_executed",
        "tensor_transfer_executed",
        "training_step_executed",
        "runtime_execution_executed",
        "runtime_state_refreshed",
        "runtime_adapter_enabled",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_runtime_dispatch_execution_payload",
        "runtime_dispatch_execution_payload",
        "runtime_dispatch_authorization_payload",
        "runtime_adapter_lock_payload",
        "runtime_state_lock_payload",
        "runtime_dispatch_handoff_payload",
        "native_dispatch_payload",
        "kernel_launch_payload",
        "parity_payload",
        "tensor_transfer_payload",
        "training_step_payload",
    ),
    inherited_unsafe_true_fields=P105_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P105_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P106_SPEC"]
