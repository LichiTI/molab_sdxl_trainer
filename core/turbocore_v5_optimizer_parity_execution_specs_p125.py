"""Stage spec for optimizer parity execution contract P125."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p124 import P124_SPEC
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_parity_execution_specs_p117 import COMMON_SECTIONS, P117_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P125_SPEC = OptimizerLateStageSpec(
    stage_id=125,
    token=P117_SPEC.token,
    scope=P117_SPEC.scope,
    title=P117_SPEC.title,
    previous_token="optimizer_kernel_launch_execution",
    previous_label="P124 optimizer kernel-launch execution contract",
    previous_ready_decision=P124_SPEC.ready_decision,
    previous_ready_field="optimizer_kernel_launch_execution_contract_ready",
    previous_evidence_field="optimizer_kernel_launch_execution_evidence_recorded",
    previous_signed_field="optimizer_kernel_launch_execution_signed",
    previous_post_fields="post_p124_request_fields",
    previous_ack="acknowledge_p124_optimizer_kernel_launch_execution_contract_recorded",
    package_ready_field=P117_SPEC.package_ready_field,
    policy_ready_field=P117_SPEC.policy_ready_field,
    row_keys=P117_SPEC.row_keys,
    row_ready_field=P117_SPEC.row_ready_field,
    later_field=P117_SPEC.later_field,
    later_ack=P117_SPEC.later_ack,
    next_contract=P117_SPEC.next_contract,
    allowed_intents=P117_SPEC.allowed_intents,
    required_sections=_sections(
        "p124_optimizer_kernel_launch_execution_contract_reference",
        "optimizer_parity_execution_package",
        "per_optimizer_parity_execution_rows",
    ),
    unsafe_true_fields=P117_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P117_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P124_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P124_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P125_SPEC"]
