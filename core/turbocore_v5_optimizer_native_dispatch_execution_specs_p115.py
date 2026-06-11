"""Stage spec for optimizer native-dispatch execution contract P115."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_runtime_dispatch_execution_specs_p114 import P114_SPEC


COMMON_SECTIONS = (
    "native_dispatch_execution_plan_inventory",
    "native_dispatch_execution_precondition_inventory",
    "native_dispatch_authorization_boundary",
    "native_runtime_handoff_boundary",
    "native_dispatch_adapter_boundary",
    "native_kernel_handoff_boundary",
    "kernel_launch_boundary",
    "parity_boundary",
    "tensor_transfer_boundary",
    "training_step_boundary",
    "operator_native_dispatch_boundary",
    "observability_boundary",
    "rollback_policy_summary",
    "no_native_dispatch_execution_boundary",
    "no_kernel_launch_execution_boundary",
    "no_parity_execution_boundary",
    "no_tensor_transfer_execution_boundary",
    "no_training_step_execution_boundary",
    "no_runtime_dispatch_execution_boundary",
    "no_runtime_execution_boundary",
    "no_runtime_state_refresh_boundary",
    "no_runtime_adapter_enabled_boundary",
    "no_training_launch_execution_boundary",
    "no_run_dispatch_boundary",
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


P115_SPEC = OptimizerLateStageSpec(
    stage_id=115,
    token="optimizer_native_dispatch_execution",
    scope="optimizer_native_dispatch_execution_contract",
    title="optimizer native-dispatch execution",
    previous_token="optimizer_runtime_dispatch_execution",
    previous_label="P114 optimizer runtime-dispatch execution contract",
    previous_ready_decision=P114_SPEC.ready_decision,
    previous_ready_field="optimizer_runtime_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_runtime_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_runtime_dispatch_execution_signed",
    previous_post_fields="post_p114_request_fields",
    previous_ack="acknowledge_p114_optimizer_runtime_dispatch_execution_contract_recorded",
    package_ready_field="optimizer_native_dispatch_execution_package_ready",
    policy_ready_field="native_dispatch_execution_policy_ready",
    row_keys=("optimizer_native_dispatch_execution_rows", "optimizer_native_dispatch_rows"),
    row_ready_field="native_dispatch_execution_review_ready",
    later_field="later_optimizer_kernel_launch_execution_contract_required",
    later_ack="acknowledge_later_optimizer_kernel_launch_execution_contract_required",
    next_contract="optimizer_kernel_launch_execution",
    allowed_intents=frozenset(
        {
            "native_dispatch_execution_candidate",
            "hold_for_more_evidence",
            "reject_native_dispatch_execution",
        }
    ),
    required_sections=_sections(
        "p114_optimizer_runtime_dispatch_execution_contract_reference",
        "optimizer_native_dispatch_execution_package",
        "per_optimizer_native_dispatch_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_native_dispatch_execution_applied",
        "optimizer_native_dispatch_execution_enabled",
        "optimizer_native_dispatch_execution_executed",
        "native_dispatch_approved",
        "native_dispatch_allowed",
        "native_dispatch_enabled",
        "native_dispatch_executed",
        "native_dispatch_started",
        "native_dispatch_request_executed",
        "native_dispatch_execution_allowed",
        "native_dispatch_execution_enabled",
        "native_runtime_handoff_enabled",
        "native_runtime_handoff_executed",
        "native_adapter_handoff_enabled",
        "native_adapter_handoff_executed",
        "native_kernel_handoff_enabled",
        "kernel_launch_requested",
        "kernel_launch_allowed",
        "kernel_launch_executed",
        "kernel_launch_started",
        "parity_allowed",
        "parity_executed",
        "tensor_transfer_allowed",
        "tensor_transfer_executed",
        "training_step_allowed",
        "training_step_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_native_dispatch_execution_payload",
        "native_dispatch_execution_payload",
        "native_dispatch_authorization_payload",
        "native_runtime_handoff_payload",
        "native_adapter_handoff_payload",
        "native_kernel_handoff_payload",
        "kernel_launch_payload",
        "parity_payload",
        "tensor_transfer_payload",
        "training_step_payload",
    ),
    inherited_unsafe_true_fields=P114_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P114_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P115_SPEC"]
