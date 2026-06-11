"""Stage spec for optimizer training-step execution contract P151."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p150 import P150_SPEC
from core.turbocore_v5_optimizer_training_step_execution_specs_p143 import COMMON_SECTIONS, P143_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P151_SPEC = OptimizerLateStageSpec(
    stage_id=151,
    token=P143_SPEC.token,
    scope=P143_SPEC.scope,
    title=P143_SPEC.title,
    previous_token="optimizer_tensor_transfer_execution",
    previous_label="P150 optimizer tensor-transfer execution contract",
    previous_ready_decision=P150_SPEC.ready_decision,
    previous_ready_field="optimizer_tensor_transfer_execution_contract_ready",
    previous_evidence_field="optimizer_tensor_transfer_execution_evidence_recorded",
    previous_signed_field="optimizer_tensor_transfer_execution_signed",
    previous_post_fields="post_p150_request_fields",
    previous_ack="acknowledge_p150_optimizer_tensor_transfer_execution_contract_recorded",
    package_ready_field=P143_SPEC.package_ready_field,
    policy_ready_field=P143_SPEC.policy_ready_field,
    row_keys=P143_SPEC.row_keys,
    row_ready_field=P143_SPEC.row_ready_field,
    later_field=P143_SPEC.later_field,
    later_ack=P143_SPEC.later_ack,
    next_contract=P143_SPEC.next_contract,
    allowed_intents=P143_SPEC.allowed_intents,
    required_sections=_sections(
        "p150_optimizer_tensor_transfer_execution_contract_reference",
        "optimizer_training_step_execution_package",
        "per_optimizer_training_step_execution_rows",
    ),
    unsafe_true_fields=P143_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P143_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P150_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P150_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P151_SPEC"]

