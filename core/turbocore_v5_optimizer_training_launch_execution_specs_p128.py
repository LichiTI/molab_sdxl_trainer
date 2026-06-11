"""Stage spec for optimizer training-launch execution contract P128."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_training_launch_execution_specs_p120 import COMMON_SECTIONS, P120_SPEC
from core.turbocore_v5_optimizer_training_step_execution_specs_p127 import P127_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P128_SPEC = OptimizerLateStageSpec(
    stage_id=128,
    token=P120_SPEC.token,
    scope=P120_SPEC.scope,
    title=P120_SPEC.title,
    previous_token="optimizer_training_step_execution",
    previous_label="P127 optimizer training-step execution contract",
    previous_ready_decision=P127_SPEC.ready_decision,
    previous_ready_field="optimizer_training_step_execution_contract_ready",
    previous_evidence_field="optimizer_training_step_execution_evidence_recorded",
    previous_signed_field="optimizer_training_step_execution_signed",
    previous_post_fields="post_p127_request_fields",
    previous_ack="acknowledge_p127_optimizer_training_step_execution_contract_recorded",
    package_ready_field=P120_SPEC.package_ready_field,
    policy_ready_field=P120_SPEC.policy_ready_field,
    row_keys=P120_SPEC.row_keys,
    row_ready_field=P120_SPEC.row_ready_field,
    later_field=P120_SPEC.later_field,
    later_ack=P120_SPEC.later_ack,
    next_contract=P120_SPEC.next_contract,
    allowed_intents=P120_SPEC.allowed_intents,
    required_sections=_sections(
        "p127_optimizer_training_step_execution_contract_reference",
        "optimizer_training_launch_execution_package",
        "per_optimizer_training_launch_execution_rows",
    ),
    unsafe_true_fields=P120_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P120_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P127_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P127_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P128_SPEC"]

