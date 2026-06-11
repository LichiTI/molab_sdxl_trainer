"""Stage spec for optimizer runtime-execution preflight contract P121."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_runtime_execution_preflight_specs_p113 import COMMON_SECTIONS, P113_SPEC
from core.turbocore_v5_optimizer_training_launch_execution_specs_p120 import P120_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P121_SPEC = OptimizerLateStageSpec(
    stage_id=121,
    token=P113_SPEC.token,
    scope=P113_SPEC.scope,
    title=P113_SPEC.title,
    previous_token="optimizer_training_launch_execution",
    previous_label="P120 optimizer training-launch execution contract",
    previous_ready_decision=P120_SPEC.ready_decision,
    previous_ready_field="optimizer_training_launch_execution_contract_ready",
    previous_evidence_field="optimizer_training_launch_execution_evidence_recorded",
    previous_signed_field="optimizer_training_launch_execution_signed",
    previous_post_fields="post_p120_request_fields",
    previous_ack="acknowledge_p120_optimizer_training_launch_execution_contract_recorded",
    package_ready_field=P113_SPEC.package_ready_field,
    policy_ready_field=P113_SPEC.policy_ready_field,
    row_keys=P113_SPEC.row_keys,
    row_ready_field=P113_SPEC.row_ready_field,
    later_field=P113_SPEC.later_field,
    later_ack=P113_SPEC.later_ack,
    next_contract=P113_SPEC.next_contract,
    allowed_intents=P113_SPEC.allowed_intents,
    required_sections=_sections(
        "p120_optimizer_training_launch_execution_contract_reference",
        "optimizer_runtime_execution_preflight_package",
        "per_optimizer_runtime_execution_preflight_rows",
    ),
    unsafe_true_fields=P113_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P113_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P120_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P120_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P121_SPEC"]
