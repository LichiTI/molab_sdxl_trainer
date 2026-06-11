"""Stage spec for optimizer runtime-execution preflight contract P129."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_runtime_execution_preflight_specs_p121 import COMMON_SECTIONS, P121_SPEC
from core.turbocore_v5_optimizer_training_launch_execution_specs_p128 import P128_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P129_SPEC = OptimizerLateStageSpec(
    stage_id=129,
    token=P121_SPEC.token,
    scope=P121_SPEC.scope,
    title=P121_SPEC.title,
    previous_token="optimizer_training_launch_execution",
    previous_label="P128 optimizer training-launch execution contract",
    previous_ready_decision=P128_SPEC.ready_decision,
    previous_ready_field="optimizer_training_launch_execution_contract_ready",
    previous_evidence_field="optimizer_training_launch_execution_evidence_recorded",
    previous_signed_field="optimizer_training_launch_execution_signed",
    previous_post_fields="post_p128_request_fields",
    previous_ack="acknowledge_p128_optimizer_training_launch_execution_contract_recorded",
    package_ready_field=P121_SPEC.package_ready_field,
    policy_ready_field=P121_SPEC.policy_ready_field,
    row_keys=P121_SPEC.row_keys,
    row_ready_field=P121_SPEC.row_ready_field,
    later_field=P121_SPEC.later_field,
    later_ack=P121_SPEC.later_ack,
    next_contract=P121_SPEC.next_contract,
    allowed_intents=P121_SPEC.allowed_intents,
    required_sections=_sections(
        "p128_optimizer_training_launch_execution_contract_reference",
        "optimizer_runtime_execution_preflight_package",
        "per_optimizer_runtime_execution_preflight_rows",
    ),
    unsafe_true_fields=P121_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P121_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P128_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P128_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P129_SPEC"]

