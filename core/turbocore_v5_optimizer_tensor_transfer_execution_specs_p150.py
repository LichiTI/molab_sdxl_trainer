"""Stage spec for optimizer tensor-transfer execution contract P150."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_parity_execution_specs_p149 import P149_SPEC
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p142 import COMMON_SECTIONS, P142_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P150_SPEC = OptimizerLateStageSpec(
    stage_id=150,
    token=P142_SPEC.token,
    scope=P142_SPEC.scope,
    title=P142_SPEC.title,
    previous_token="optimizer_parity_execution",
    previous_label="P149 optimizer parity execution contract",
    previous_ready_decision=P149_SPEC.ready_decision,
    previous_ready_field="optimizer_parity_execution_contract_ready",
    previous_evidence_field="optimizer_parity_execution_evidence_recorded",
    previous_signed_field="optimizer_parity_execution_signed",
    previous_post_fields="post_p149_request_fields",
    previous_ack="acknowledge_p149_optimizer_parity_execution_contract_recorded",
    package_ready_field=P142_SPEC.package_ready_field,
    policy_ready_field=P142_SPEC.policy_ready_field,
    row_keys=P142_SPEC.row_keys,
    row_ready_field=P142_SPEC.row_ready_field,
    later_field=P142_SPEC.later_field,
    later_ack=P142_SPEC.later_ack,
    next_contract=P142_SPEC.next_contract,
    allowed_intents=P142_SPEC.allowed_intents,
    required_sections=_sections(
        "p149_optimizer_parity_execution_contract_reference",
        "optimizer_tensor_transfer_execution_package",
        "per_optimizer_tensor_transfer_execution_rows",
    ),
    unsafe_true_fields=P142_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P142_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P149_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P149_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P150_SPEC"]

