"""Stage spec for optimizer kernel-launch execution contract P140."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p132 import COMMON_SECTIONS, P132_SPEC
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p139 import P139_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P140_SPEC = OptimizerLateStageSpec(
    stage_id=140,
    token=P132_SPEC.token,
    scope=P132_SPEC.scope,
    title=P132_SPEC.title,
    previous_token="optimizer_native_dispatch_execution",
    previous_label="P139 optimizer native-dispatch execution contract",
    previous_ready_decision=P139_SPEC.ready_decision,
    previous_ready_field="optimizer_native_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_native_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_native_dispatch_execution_signed",
    previous_post_fields="post_p139_request_fields",
    previous_ack="acknowledge_p139_optimizer_native_dispatch_execution_contract_recorded",
    package_ready_field=P132_SPEC.package_ready_field,
    policy_ready_field=P132_SPEC.policy_ready_field,
    row_keys=P132_SPEC.row_keys,
    row_ready_field=P132_SPEC.row_ready_field,
    later_field=P132_SPEC.later_field,
    later_ack=P132_SPEC.later_ack,
    next_contract=P132_SPEC.next_contract,
    allowed_intents=P132_SPEC.allowed_intents,
    required_sections=_sections(
        "p139_optimizer_native_dispatch_execution_contract_reference",
        "optimizer_kernel_launch_execution_package",
        "per_optimizer_kernel_launch_execution_rows",
    ),
    unsafe_true_fields=P132_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P132_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P139_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P139_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P140_SPEC"]

