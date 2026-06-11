"""Stage spec for optimizer kernel-launch execution contract P148."""

from __future__ import annotations

from core.turbocore_v5_optimizer_kernel_launch_execution_specs_p140 import COMMON_SECTIONS, P140_SPEC
from core.turbocore_v5_optimizer_late_stage_contract_utils import OptimizerLateStageSpec
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p147 import P147_SPEC


def _sections(previous: str, package: str, rows: str) -> tuple[str, ...]:
    return (previous, package, rows, *COMMON_SECTIONS)


P148_SPEC = OptimizerLateStageSpec(
    stage_id=148,
    token=P140_SPEC.token,
    scope=P140_SPEC.scope,
    title=P140_SPEC.title,
    previous_token="optimizer_native_dispatch_execution",
    previous_label="P147 optimizer native-dispatch execution contract",
    previous_ready_decision=P147_SPEC.ready_decision,
    previous_ready_field="optimizer_native_dispatch_execution_contract_ready",
    previous_evidence_field="optimizer_native_dispatch_execution_evidence_recorded",
    previous_signed_field="optimizer_native_dispatch_execution_signed",
    previous_post_fields="post_p147_request_fields",
    previous_ack="acknowledge_p147_optimizer_native_dispatch_execution_contract_recorded",
    package_ready_field=P140_SPEC.package_ready_field,
    policy_ready_field=P140_SPEC.policy_ready_field,
    row_keys=P140_SPEC.row_keys,
    row_ready_field=P140_SPEC.row_ready_field,
    later_field=P140_SPEC.later_field,
    later_ack=P140_SPEC.later_ack,
    next_contract=P140_SPEC.next_contract,
    allowed_intents=P140_SPEC.allowed_intents,
    required_sections=_sections(
        "p147_optimizer_native_dispatch_execution_contract_reference",
        "optimizer_kernel_launch_execution_package",
        "per_optimizer_kernel_launch_execution_rows",
    ),
    unsafe_true_fields=P140_SPEC.unsafe_true_fields,
    unsafe_non_empty_fields=P140_SPEC.unsafe_non_empty_fields,
    inherited_unsafe_true_fields=P147_SPEC.all_unsafe_true_fields,
    inherited_unsafe_non_empty_fields=P147_SPEC.all_unsafe_non_empty_fields,
)


__all__ = ["P148_SPEC"]

