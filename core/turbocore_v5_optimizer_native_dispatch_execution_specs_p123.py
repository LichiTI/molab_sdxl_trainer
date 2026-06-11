"""Stage spec for optimizer native-dispatch execution contract P123."""

from __future__ import annotations

from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p115 import COMMON_SECTIONS, P115_SPEC
from core.turbocore_v5_optimizer_runtime_dispatch_execution_specs_p122 import P122_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P123_SPEC = OptimizerLateStageSpec(
    stage_id=123,
    token=P115_SPEC.token,
    scope=P115_SPEC.scope,
    title=P115_SPEC.title,
    previous_token="optimizer_runtime_dispatch_execution",
    previous_label="P122 optimizer runtime-dispatch execution contract",
    previous_ready_decision=P122_SPEC.ready_decision,
    previous_ready_field="optimizer_runtime_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_runtime_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_runtime_dispatch_execution_signed",
    previous_post_fields="post_p122_request_fields",
    previous_ack="acknowledge_p122_optimizer_runtime_dispatch_execution_contract_recorded",
    package_ready_field=P115_SPEC.package_ready_field,
    policy_ready_field=P115_SPEC.policy_ready_field,
    row_keys=P115_SPEC.row_keys,
    row_ready_field=P115_SPEC.row_ready_field,
    later_field=P115_SPEC.later_field,
    later_ack=P115_SPEC.later_ack,
    next_contract=P115_SPEC.next_contract,
    allowed_intents=P115_SPEC.allowed_intents,
    required_sections=_sections(
        "p122_optimizer_runtime_dispatch_execution_contract_reference",
        "optimizer_native_dispatch_execution_package",
        "per_optimizer_native_dispatch_execution_rows",
    ),
    unsafe_true_fields=P115_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P115_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P122_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P122_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P123_SPEC"]
