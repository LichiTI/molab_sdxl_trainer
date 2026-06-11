"""Stage spec for optimizer training-step execution contract P135."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p134 import P134_SPEC
from core.turbocore_v5_optimizer_training_step_execution_specs_p127 import COMMON_SECTIONS, P127_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P135_SPEC = OptimizerLateStageSpec(
    stage_id=135,
    token=P127_SPEC.token,
    scope=P127_SPEC.scope,
    title=P127_SPEC.title,
    previous_token="optimizer_tensor_transfer_execution",
    previous_label="P134 optimizer tensor-transfer execution contract",
    previous_ready_decision=P134_SPEC.ready_decision,
    previous_ready_field="optimizer_tensor_transfer_execution_contract_ready",
    previous_evidence_field="optimizer_tensor_transfer_execution_evidence_recorded",
    previous_signed_field="optimizer_tensor_transfer_execution_signed",
    previous_post_fields="post_p134_request_fields",
    previous_ack="acknowledge_p134_optimizer_tensor_transfer_execution_contract_recorded",
    package_ready_field=P127_SPEC.package_ready_field,
    policy_ready_field=P127_SPEC.policy_ready_field,
    row_keys=P127_SPEC.row_keys,
    row_ready_field=P127_SPEC.row_ready_field,
    later_field=P127_SPEC.later_field,
    later_ack=P127_SPEC.later_ack,
    next_contract=P127_SPEC.next_contract,
    allowed_intents=P127_SPEC.allowed_intents,
    required_sections=_sections(
        "p134_optimizer_tensor_transfer_execution_contract_reference",
        "optimizer_training_step_execution_package",
        "per_optimizer_training_step_execution_rows",
    ),
    unsafe_true_fields=P127_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P127_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P134_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P134_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P135_SPEC"]

