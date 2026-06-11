"""Stage spec for optimizer native-dispatch execution contract P139."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p131 import COMMON_SECTIONS, P131_SPEC
from core.turbocore_v5_optimizer_runtime_dispatch_execution_specs_p138 import P138_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P139_SPEC = OptimizerLateStageSpec(
    stage_id=139,
    token=P131_SPEC.token,
    scope=P131_SPEC.scope,
    title=P131_SPEC.title,
    previous_token="optimizer_runtime_dispatch_execution",
    previous_label="P138 optimizer runtime-dispatch execution contract",
    previous_ready_decision=P138_SPEC.ready_decision,
    previous_ready_field="optimizer_runtime_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_runtime_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_runtime_dispatch_execution_signed",
    previous_post_fields="post_p138_request_fields",
    previous_ack="acknowledge_p138_optimizer_runtime_dispatch_execution_contract_recorded",
    package_ready_field=P131_SPEC.package_ready_field,
    policy_ready_field=P131_SPEC.policy_ready_field,
    row_keys=P131_SPEC.row_keys,
    row_ready_field=P131_SPEC.row_ready_field,
    later_field=P131_SPEC.later_field,
    later_ack=P131_SPEC.later_ack,
    next_contract=P131_SPEC.next_contract,
    allowed_intents=P131_SPEC.allowed_intents,
    required_sections=_sections(
        "p138_optimizer_runtime_dispatch_execution_contract_reference",
        "optimizer_native_dispatch_execution_package",
        "per_optimizer_native_dispatch_execution_rows",
    ),
    unsafe_true_fields=P131_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P131_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P138_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P138_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P139_SPEC"]

