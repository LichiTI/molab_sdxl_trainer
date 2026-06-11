"""Stage spec for optimizer parity execution contract P141."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p140 import P140_SPEC
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_parity_execution_specs_p133 import COMMON_SECTIONS, P133_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P141_SPEC = OptimizerLateStageSpec(
    stage_id=141,
    token=P133_SPEC.token,
    scope=P133_SPEC.scope,
    title=P133_SPEC.title,
    previous_token="optimizer_kernel_launch_execution",
    previous_label="P140 optimizer kernel-launch execution contract",
    previous_ready_decision=P140_SPEC.ready_decision,
    previous_ready_field="optimizer_kernel_launch_execution_contract_ready",
    previous_evidence_field="optimizer_kernel_launch_execution_evidence_recorded",
    previous_signed_field="optimizer_kernel_launch_execution_signed",
    previous_post_fields="post_p140_request_fields",
    previous_ack="acknowledge_p140_optimizer_kernel_launch_execution_contract_recorded",
    package_ready_field=P133_SPEC.package_ready_field,
    policy_ready_field=P133_SPEC.policy_ready_field,
    row_keys=P133_SPEC.row_keys,
    row_ready_field=P133_SPEC.row_ready_field,
    later_field=P133_SPEC.later_field,
    later_ack=P133_SPEC.later_ack,
    next_contract=P133_SPEC.next_contract,
    allowed_intents=P133_SPEC.allowed_intents,
    required_sections=_sections(
        "p140_optimizer_kernel_launch_execution_contract_reference",
        "optimizer_parity_execution_package",
        "per_optimizer_parity_execution_rows",
    ),
    unsafe_true_fields=P133_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P133_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P140_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P140_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P141_SPEC"]

