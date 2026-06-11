"""Stage spec for optimizer tensor-transfer execution contract P118."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_parity_execution_specs_p117 import P117_SPEC
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p110 import COMMON_SECTIONS


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


def _unsafe(*fields: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fields))


P118_SPEC = OptimizerLateStageSpec(
    stage_id=118,
    token="optimizer_tensor_transfer_execution",
    scope="optimizer_tensor_transfer_execution_contract",
    title="optimizer tensor-transfer execution",
    previous_token="optimizer_parity_execution",
    previous_label="P117 optimizer parity execution contract",
    previous_ready_decision=P117_SPEC.ready_decision,
    previous_ready_field="optimizer_parity_execution_contract_ready",
    previous_evidence_field="optimizer_parity_execution_evidence_recorded",
    previous_signed_field="optimizer_parity_execution_signed",
    previous_post_fields="post_p117_request_fields",
    previous_ack="acknowledge_p117_optimizer_parity_execution_contract_recorded",
    package_ready_field="optimizer_tensor_transfer_execution_package_ready",
    policy_ready_field="tensor_transfer_execution_policy_ready",
    row_keys=("optimizer_tensor_transfer_execution_rows", "optimizer_transfer_execution_rows"),
    row_ready_field="tensor_transfer_execution_review_ready",
    later_field="later_optimizer_training_step_execution_contract_required",
    later_ack="acknowledge_later_optimizer_training_step_execution_contract_required",
    next_contract="optimizer_training_step_execution",
    allowed_intents=frozenset(
        {
            "tensor_transfer_execution_candidate",
            "hold_for_more_evidence",
            "reject_tensor_transfer_execution",
        }
    ),
    required_sections=_sections(
        "p117_optimizer_parity_execution_contract_reference",
        "optimizer_tensor_transfer_execution_package",
        "per_optimizer_tensor_transfer_execution_rows",
    ),
    unsafe_true_fields=_unsafe(
        "optimizer_tensor_transfer_execution_applied",
        "optimizer_tensor_transfer_execution_enabled",
        "optimizer_tensor_transfer_execution_executed",
        "tensor_transfer_approved",
        "tensor_transfer_allowed",
        "tensor_transfer_enabled",
        "tensor_transfer_requested",
        "tensor_transfer_executed",
        "tensor_transfer_started",
        "device_tensor_read_executed",
        "device_tensor_write_executed",
        "host_device_sync_executed",
        "stream_sync_executed",
        "training_step_allowed",
        "training_step_executed",
    ),
    unsafe_non_empty_fields=_unsafe(
        "optimizer_tensor_transfer_execution_payload",
        "tensor_transfer_execution_payload",
        "tensor_transfer_authorization_payload",
        "device_tensor_payload",
        "host_device_sync_payload",
        "stream_sync_payload",
        "training_step_payload",
    ),
    inherited_unsafe_true_fields=P117_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P117_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P118_SPEC"]
