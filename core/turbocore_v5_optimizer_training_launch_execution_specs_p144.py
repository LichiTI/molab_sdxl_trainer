"""Stage spec for optimizer training-launch execution contract P144."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_training_launch_execution_specs_p136 import COMMON_SECTIONS, P136_SPEC
from core.turbocore_v5_optimizer_training_step_execution_specs_p143 import P143_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P144_SPEC = OptimizerLateStageSpec(
    stage_id=144,
    token=P136_SPEC.token,
    scope=P136_SPEC.scope,
    title=P136_SPEC.title,
    previous_token="optimizer_training_step_execution",
    previous_label="P143 optimizer training-step execution contract",
    previous_ready_decision=P143_SPEC.ready_decision,
    previous_ready_field="optimizer_training_step_execution_contract_ready",
    previous_evidence_field="optimizer_training_step_execution_evidence_recorded",
    previous_signed_field="optimizer_training_step_execution_signed",
    previous_post_fields="post_p143_request_fields",
    previous_ack="acknowledge_p143_optimizer_training_step_execution_contract_recorded",
    package_ready_field=P136_SPEC.package_ready_field,
    policy_ready_field=P136_SPEC.policy_ready_field,
    row_keys=P136_SPEC.row_keys,
    row_ready_field=P136_SPEC.row_ready_field,
    later_field=P136_SPEC.later_field,
    later_ack=P136_SPEC.later_ack,
    next_contract=P136_SPEC.next_contract,
    allowed_intents=P136_SPEC.allowed_intents,
    required_sections=_sections(
        "p143_optimizer_training_step_execution_contract_reference",
        "optimizer_training_launch_execution_package",
        "per_optimizer_training_launch_execution_rows",
    ),
    unsafe_true_fields=P136_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P136_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P143_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P143_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P144_SPEC"]

