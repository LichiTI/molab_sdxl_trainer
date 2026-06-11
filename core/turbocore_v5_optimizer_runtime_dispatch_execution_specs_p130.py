"""Stage spec for optimizer runtime-dispatch execution contract P130."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_runtime_dispatch_execution_specs_p122 import COMMON_SECTIONS, P122_SPEC
from core.turbocore_v5_optimizer_runtime_execution_preflight_specs_p129 import P129_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P130_SPEC = OptimizerLateStageSpec(
    stage_id=130,
    token=P122_SPEC.token,
    scope=P122_SPEC.scope,
    title=P122_SPEC.title,
    previous_token="optimizer_runtime_execution_preflight",
    previous_label="P129 optimizer runtime-execution preflight contract",
    previous_ready_decision=P129_SPEC.ready_decision,
    previous_ready_field="optimizer_runtime_execution_preflight_contract_ready",
    previous_evidence_field="optimizer_runtime_execution_preflight_evidence_recorded",
    previous_signed_field="optimizer_runtime_execution_preflight_signed",
    previous_post_fields="post_p129_request_fields",
    previous_ack="acknowledge_p129_optimizer_runtime_execution_preflight_contract_recorded",
    package_ready_field=P122_SPEC.package_ready_field,
    policy_ready_field=P122_SPEC.policy_ready_field,
    row_keys=P122_SPEC.row_keys,
    row_ready_field=P122_SPEC.row_ready_field,
    later_field=P122_SPEC.later_field,
    later_ack=P122_SPEC.later_ack,
    next_contract=P122_SPEC.next_contract,
    allowed_intents=P122_SPEC.allowed_intents,
    required_sections=_sections(
        "p129_optimizer_runtime_execution_preflight_contract_reference",
        "optimizer_runtime_dispatch_execution_package",
        "per_optimizer_runtime_dispatch_execution_rows",
    ),
    unsafe_true_fields=P122_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P122_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P129_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P129_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P130_SPEC"]

