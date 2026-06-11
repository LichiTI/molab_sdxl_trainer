"""Stage spec for optimizer kernel-launch execution contract P124."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p116 import COMMON_SECTIONS, P116_SPEC
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p123 import P123_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P124_SPEC = OptimizerLateStageSpec(
    stage_id=124,
    token=P116_SPEC.token,
    scope=P116_SPEC.scope,
    title=P116_SPEC.title,
    previous_token="optimizer_native_dispatch_execution",
    previous_label="P123 optimizer native-dispatch execution contract",
    previous_ready_decision=P123_SPEC.ready_decision,
    previous_ready_field="optimizer_native_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_native_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_native_dispatch_execution_signed",
    previous_post_fields="post_p123_request_fields",
    previous_ack="acknowledge_p123_optimizer_native_dispatch_execution_contract_recorded",
    package_ready_field=P116_SPEC.package_ready_field,
    policy_ready_field=P116_SPEC.policy_ready_field,
    row_keys=P116_SPEC.row_keys,
    row_ready_field=P116_SPEC.row_ready_field,
    later_field=P116_SPEC.later_field,
    later_ack=P116_SPEC.later_ack,
    next_contract=P116_SPEC.next_contract,
    allowed_intents=P116_SPEC.allowed_intents,
    required_sections=_sections(
        "p123_optimizer_native_dispatch_execution_contract_reference",
        "optimizer_kernel_launch_execution_package",
        "per_optimizer_kernel_launch_execution_rows",
    ),
    unsafe_true_fields=P116_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P116_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P123_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P123_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P124_SPEC"]
