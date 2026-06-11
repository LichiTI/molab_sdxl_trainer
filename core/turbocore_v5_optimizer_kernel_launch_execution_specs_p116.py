"""Stage spec for optimizer kernel-launch execution contract P116."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p108 import COMMON_SECTIONS
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p115 import P115_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P116_SPEC = OptimizerLateStageSpec(
    stage_id=116,
    token="optimizer_kernel_launch_execution",
    scope="optimizer_kernel_launch_execution_contract",
    title="optimizer kernel-launch execution",
    previous_token="optimizer_native_dispatch_execution",
    previous_label="P115 optimizer native-dispatch execution contract",
    previous_ready_decision=P115_SPEC.ready_decision,
    previous_ready_field="optimizer_native_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_native_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_native_dispatch_execution_signed",
    previous_post_fields="post_p115_request_fields",
    previous_ack="acknowledge_p115_optimizer_native_dispatch_execution_contract_recorded",
    package_ready_field="optimizer_kernel_launch_execution_package_ready",
    policy_ready_field="kernel_launch_execution_policy_ready",
    row_keys=("optimizer_kernel_launch_execution_rows", "optimizer_kernel_execution_rows"),
    row_ready_field="kernel_launch_execution_review_ready",
    later_field="later_optimizer_parity_execution_contract_required",
    later_ack="acknowledge_later_optimizer_parity_execution_contract_required",
    next_contract="optimizer_parity_execution",
    allowed_intents=frozenset(
        {
            "kernel_launch_execution_candidate",
            "hold_for_more_evidence",
            "reject_kernel_launch_execution",
        }
    ),
    required_sections=_sections(
        "p115_optimizer_native_dispatch_execution_contract_reference",
        "optimizer_kernel_launch_execution_package",
        "per_optimizer_kernel_launch_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_kernel_launch_execution_applied",
        "optimizer_kernel_launch_execution_enabled",
        "optimizer_kernel_launch_execution_executed",
        "kernel_launch_approved",
        "kernel_launch_allowed",
        "kernel_launch_enabled",
        "kernel_launch_requested",
        "kernel_launch_executed",
        "kernel_launch_started",
        "kernel_artifact_loaded",
        "kernel_parameters_materialized",
        "kernel_stream_bound",
        "kernel_event_chain_bound",
        "parity_allowed",
        "parity_executed",
        "tensor_transfer_allowed",
        "tensor_transfer_executed",
        "training_step_allowed",
        "training_step_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_kernel_launch_execution_payload",
        "kernel_launch_execution_payload",
        "kernel_launch_authorization_payload",
        "kernel_artifact_payload",
        "kernel_parameter_payload",
        "kernel_stream_event_payload",
        "kernel_launch_handoff_payload",
        "parity_payload",
        "tensor_transfer_payload",
        "training_step_payload",
    ),
    inherited_unsafe_true_fields=P115_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P115_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P116_SPEC"]
