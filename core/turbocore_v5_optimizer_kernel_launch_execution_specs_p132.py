"""Stage spec for optimizer kernel-launch execution contract P132."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p124 import COMMON_SECTIONS, P124_SPEC
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p131 import P131_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P132_SPEC = OptimizerLateStageSpec(
    stage_id=132,
    token=P124_SPEC.token,
    scope=P124_SPEC.scope,
    title=P124_SPEC.title,
    previous_token="optimizer_native_dispatch_execution",
    previous_label="P131 optimizer native-dispatch execution contract",
    previous_ready_decision=P131_SPEC.ready_decision,
    previous_ready_field="optimizer_native_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_native_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_native_dispatch_execution_signed",
    previous_post_fields="post_p131_request_fields",
    previous_ack="acknowledge_p131_optimizer_native_dispatch_execution_contract_recorded",
    package_ready_field=P124_SPEC.package_ready_field,
    policy_ready_field=P124_SPEC.policy_ready_field,
    row_keys=P124_SPEC.row_keys,
    row_ready_field=P124_SPEC.row_ready_field,
    later_field=P124_SPEC.later_field,
    later_ack=P124_SPEC.later_ack,
    next_contract=P124_SPEC.next_contract,
    allowed_intents=P124_SPEC.allowed_intents,
    required_sections=_sections(
        "p131_optimizer_native_dispatch_execution_contract_reference",
        "optimizer_kernel_launch_execution_package",
        "per_optimizer_kernel_launch_execution_rows",
    ),
    unsafe_true_fields=P124_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P124_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P131_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P131_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P132_SPEC"]

