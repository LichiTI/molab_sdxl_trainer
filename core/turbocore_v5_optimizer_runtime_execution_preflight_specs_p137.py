"""Stage spec for optimizer runtime-execution preflight contract P137."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_runtime_execution_preflight_specs_p129 import COMMON_SECTIONS, P129_SPEC
from core.turbocore_v5_optimizer_training_launch_execution_specs_p136 import P136_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P137_SPEC = OptimizerLateStageSpec(
    stage_id=137,
    token=P129_SPEC.token,
    scope=P129_SPEC.scope,
    title=P129_SPEC.title,
    previous_token="optimizer_training_launch_execution",
    previous_label="P136 optimizer training-launch execution contract",
    previous_ready_decision=P136_SPEC.ready_decision,
    previous_ready_field="optimizer_training_launch_execution_contract_ready",
    previous_evidence_field="optimizer_training_launch_execution_evidence_recorded",
    previous_signed_field="optimizer_training_launch_execution_signed",
    previous_post_fields="post_p136_request_fields",
    previous_ack="acknowledge_p136_optimizer_training_launch_execution_contract_recorded",
    package_ready_field=P129_SPEC.package_ready_field,
    policy_ready_field=P129_SPEC.policy_ready_field,
    row_keys=P129_SPEC.row_keys,
    row_ready_field=P129_SPEC.row_ready_field,
    later_field=P129_SPEC.later_field,
    later_ack=P129_SPEC.later_ack,
    next_contract=P129_SPEC.next_contract,
    allowed_intents=P129_SPEC.allowed_intents,
    required_sections=_sections(
        "p136_optimizer_training_launch_execution_contract_reference",
        "optimizer_runtime_execution_preflight_package",
        "per_optimizer_runtime_execution_preflight_rows",
    ),
    unsafe_true_fields=P129_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P129_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P136_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P136_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P137_SPEC"]

