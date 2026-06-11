"""Stage spec for optimizer tensor-transfer execution contract P142."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_parity_execution_specs_p141 import P141_SPEC
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p134 import COMMON_SECTIONS, P134_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P142_SPEC = OptimizerLateStageSpec(
    stage_id=142,
    token=P134_SPEC.token,
    scope=P134_SPEC.scope,
    title=P134_SPEC.title,
    previous_token="optimizer_parity_execution",
    previous_label="P141 optimizer parity execution contract",
    previous_ready_decision=P141_SPEC.ready_decision,
    previous_ready_field="optimizer_parity_execution_contract_ready",
    previous_evidence_field="optimizer_parity_execution_evidence_recorded",
    previous_signed_field="optimizer_parity_execution_signed",
    previous_post_fields="post_p141_request_fields",
    previous_ack="acknowledge_p141_optimizer_parity_execution_contract_recorded",
    package_ready_field=P134_SPEC.package_ready_field,
    policy_ready_field=P134_SPEC.policy_ready_field,
    row_keys=P134_SPEC.row_keys,
    row_ready_field=P134_SPEC.row_ready_field,
    later_field=P134_SPEC.later_field,
    later_ack=P134_SPEC.later_ack,
    next_contract=P134_SPEC.next_contract,
    allowed_intents=P134_SPEC.allowed_intents,
    required_sections=_sections(
        "p141_optimizer_parity_execution_contract_reference",
        "optimizer_tensor_transfer_execution_package",
        "per_optimizer_tensor_transfer_execution_rows",
    ),
    unsafe_true_fields=P134_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P134_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P141_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P141_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P142_SPEC"]

