"""Stage spec for optimizer training-launch execution contract P120."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_training_launch_execution_specs_p112 import COMMON_SECTIONS, P112_SPEC
from core.turbocore_v5_optimizer_training_step_execution_specs_p119 import P119_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P120_SPEC = OptimizerLateStageSpec(
    stage_id=120,
    token=P112_SPEC.token,
    scope=P112_SPEC.scope,
    title=P112_SPEC.title,
    previous_token="optimizer_training_step_execution",
    previous_label="P119 optimizer training-step execution contract",
    previous_ready_decision=P119_SPEC.ready_decision,
    previous_ready_field="optimizer_training_step_execution_contract_ready",
    previous_evidence_field="optimizer_training_step_execution_evidence_recorded",
    previous_signed_field="optimizer_training_step_execution_signed",
    previous_post_fields="post_p119_request_fields",
    previous_ack="acknowledge_p119_optimizer_training_step_execution_contract_recorded",
    package_ready_field=P112_SPEC.package_ready_field,
    policy_ready_field=P112_SPEC.policy_ready_field,
    row_keys=P112_SPEC.row_keys,
    row_ready_field=P112_SPEC.row_ready_field,
    later_field=P112_SPEC.later_field,
    later_ack=P112_SPEC.later_ack,
    next_contract=P112_SPEC.next_contract,
    allowed_intents=P112_SPEC.allowed_intents,
    required_sections=_sections(
        "p119_optimizer_training_step_execution_contract_reference",
        "optimizer_training_launch_execution_package",
        "per_optimizer_training_launch_execution_rows",
    ),
    unsafe_true_fields=P112_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P112_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P119_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P119_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P120_SPEC"]
