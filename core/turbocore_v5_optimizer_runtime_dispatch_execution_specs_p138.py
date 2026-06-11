"""Stage spec for optimizer runtime-dispatch execution contract P138."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_runtime_dispatch_execution_specs_p130 import COMMON_SECTIONS, P130_SPEC
from core.turbocore_v5_optimizer_runtime_execution_preflight_specs_p137 import P137_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P138_SPEC = OptimizerLateStageSpec(
    stage_id=138,
    token=P130_SPEC.token,
    scope=P130_SPEC.scope,
    title=P130_SPEC.title,
    previous_token="optimizer_runtime_execution_preflight",
    previous_label="P137 optimizer runtime-execution preflight contract",
    previous_ready_decision=P137_SPEC.ready_decision,
    previous_ready_field="optimizer_runtime_execution_preflight_contract_ready",
    previous_evidence_field="optimizer_runtime_execution_preflight_evidence_recorded",
    previous_signed_field="optimizer_runtime_execution_preflight_signed",
    previous_post_fields="post_p137_request_fields",
    previous_ack="acknowledge_p137_optimizer_runtime_execution_preflight_contract_recorded",
    package_ready_field=P130_SPEC.package_ready_field,
    policy_ready_field=P130_SPEC.policy_ready_field,
    row_keys=P130_SPEC.row_keys,
    row_ready_field=P130_SPEC.row_ready_field,
    later_field=P130_SPEC.later_field,
    later_ack=P130_SPEC.later_ack,
    next_contract=P130_SPEC.next_contract,
    allowed_intents=P130_SPEC.allowed_intents,
    required_sections=_sections(
        "p137_optimizer_runtime_execution_preflight_contract_reference",
        "optimizer_runtime_dispatch_execution_package",
        "per_optimizer_runtime_dispatch_execution_rows",
    ),
    unsafe_true_fields=P130_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P130_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P137_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P137_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P138_SPEC"]

