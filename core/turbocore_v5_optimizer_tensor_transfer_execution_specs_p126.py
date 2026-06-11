"""Stage spec for optimizer tensor-transfer execution contract P126."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_parity_execution_specs_p125 import P125_SPEC
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p118 import COMMON_SECTIONS, P118_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P126_SPEC = OptimizerLateStageSpec(
    stage_id=126,
    token=P118_SPEC.token,
    scope=P118_SPEC.scope,
    title=P118_SPEC.title,
    previous_token="optimizer_parity_execution",
    previous_label="P125 optimizer parity execution contract",
    previous_ready_decision=P125_SPEC.ready_decision,
    previous_ready_field="optimizer_parity_execution_contract_ready",
    previous_evidence_field="optimizer_parity_execution_evidence_recorded",
    previous_signed_field="optimizer_parity_execution_signed",
    previous_post_fields="post_p125_request_fields",
    previous_ack="acknowledge_p125_optimizer_parity_execution_contract_recorded",
    package_ready_field=P118_SPEC.package_ready_field,
    policy_ready_field=P118_SPEC.policy_ready_field,
    row_keys=P118_SPEC.row_keys,
    row_ready_field=P118_SPEC.row_ready_field,
    later_field=P118_SPEC.later_field,
    later_ack=P118_SPEC.later_ack,
    next_contract=P118_SPEC.next_contract,
    allowed_intents=P118_SPEC.allowed_intents,
    required_sections=_sections(
        "p125_optimizer_parity_execution_contract_reference",
        "optimizer_tensor_transfer_execution_package",
        "per_optimizer_tensor_transfer_execution_rows",
    ),
    unsafe_true_fields=P118_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P118_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P125_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P125_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P126_SPEC"]
