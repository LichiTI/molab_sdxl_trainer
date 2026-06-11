"""Stage spec for optimizer training-step execution contract P143."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p142 import P142_SPEC
from core.turbocore_v5_optimizer_training_step_execution_specs_p135 import COMMON_SECTIONS, P135_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P143_SPEC = OptimizerLateStageSpec(
    stage_id=143,
    token=P135_SPEC.token,
    scope=P135_SPEC.scope,
    title=P135_SPEC.title,
    previous_token="optimizer_tensor_transfer_execution",
    previous_label="P142 optimizer tensor-transfer execution contract",
    previous_ready_decision=P142_SPEC.ready_decision,
    previous_ready_field="optimizer_tensor_transfer_execution_contract_ready",
    previous_evidence_field="optimizer_tensor_transfer_execution_evidence_recorded",
    previous_signed_field="optimizer_tensor_transfer_execution_signed",
    previous_post_fields="post_p142_request_fields",
    previous_ack="acknowledge_p142_optimizer_tensor_transfer_execution_contract_recorded",
    package_ready_field=P135_SPEC.package_ready_field,
    policy_ready_field=P135_SPEC.policy_ready_field,
    row_keys=P135_SPEC.row_keys,
    row_ready_field=P135_SPEC.row_ready_field,
    later_field=P135_SPEC.later_field,
    later_ack=P135_SPEC.later_ack,
    next_contract=P135_SPEC.next_contract,
    allowed_intents=P135_SPEC.allowed_intents,
    required_sections=_sections(
        "p142_optimizer_tensor_transfer_execution_contract_reference",
        "optimizer_training_step_execution_package",
        "per_optimizer_training_step_execution_rows",
    ),
    unsafe_true_fields=P135_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P135_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P142_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P142_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P143_SPEC"]

