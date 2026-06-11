"""Report-only native canary aggregation for simple optimizer variants."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from core.configs import OptimizerType
from core.turbocore_plugin_schedulefree_radam_training_loop_canary_scorecard import (
    build_plugin_schedulefree_radam_training_loop_canary_scorecard,
)
from core.turbocore_plugin_schedulefree_sgd_training_loop_canary_scorecard import (
    build_plugin_schedulefree_sgd_training_loop_canary_scorecard,
)
from core.turbocore_simple_optimizer_quantized_variant_parity_scorecard import (
    build_simple_optimizer_quantized_variant_parity_scorecard,
)
from core.turbocore_simple_optimizer_quantized_native_scratch_scorecard import (
    build_simple_optimizer_quantized_native_scratch_scorecard,
)
from core.turbocore_simple_optimizer_quantized_runtime_canary_scorecard import (
    build_simple_optimizer_quantized_runtime_canary_scorecard,
)
from core.turbocore_simple_optimizer_quantized_training_loop_canary_scorecard import (
    build_simple_optimizer_quantized_training_loop_canary_scorecard,
)
from core.turbocore_simple_optimizer_quantized_e2e_no_regression_scorecard import (
    build_simple_optimizer_quantized_e2e_no_regression_scorecard,
)
from core.turbocore_simple_optimizer_quantized_product_state_sync_scorecard import (
    build_simple_optimizer_quantized_product_state_sync_scorecard,
)
from core.turbocore_simple_optimizer_quantized_rollout_policy_scorecard import (
    build_simple_optimizer_quantized_rollout_policy_scorecard,
)
from core.turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard import (
    build_simple_optimizer_quantized_dispatch_integration_review_scorecard,
)
from core.turbocore_simple_optimizer_quantized_owner_approval_hold_scorecard import (
    build_simple_optimizer_quantized_owner_approval_hold_scorecard,
)
from core.turbocore_simple_optimizer_variant_native_abi_scorecard import (
    build_simple_optimizer_variant_native_abi_scorecard,
)


SCHEDULE_FREE_CANARY_TARGETS = (
    OptimizerType.RADAM_SCHEDULE_FREE,
    OptimizerType.SGD_SCHEDULE_FREE,
)
QUANTIZED_PENDING_TARGETS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)


def build_simple_optimizer_variant_native_canary_scorecard(
    *,
    native_abi_report: Mapping[str, Any] | None = None,
    quantized_parity_report: Mapping[str, Any] | None = None,
    quantized_native_scratch_report: Mapping[str, Any] | None = None,
    quantized_runtime_canary_report: Mapping[str, Any] | None = None,
    quantized_training_loop_canary_report: Mapping[str, Any] | None = None,
    quantized_e2e_no_regression_report: Mapping[str, Any] | None = None,
    quantized_product_state_sync_report: Mapping[str, Any] | None = None,
    quantized_rollout_policy_report: Mapping[str, Any] | None = None,
    quantized_dispatch_integration_review_report: Mapping[str, Any] | None = None,
    quantized_owner_approval_hold_report: Mapping[str, Any] | None = None,
    schedule_free_reports: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map existing schedule-free native canaries to built-in simple variants."""

    abi = dict(native_abi_report or build_simple_optimizer_variant_native_abi_scorecard())
    quantized = dict(
        quantized_parity_report
        or build_simple_optimizer_quantized_variant_parity_scorecard(native_abi_report=abi)
    )
    native_scratch = dict(
        quantized_native_scratch_report
        or build_simple_optimizer_quantized_native_scratch_scorecard(quantized_parity_report=quantized)
    )
    runtime_canary = dict(
        quantized_runtime_canary_report
        or build_simple_optimizer_quantized_runtime_canary_scorecard(native_scratch_report=native_scratch)
    )
    training_loop = dict(
        quantized_training_loop_canary_report
        or build_simple_optimizer_quantized_training_loop_canary_scorecard(runtime_canary_report=runtime_canary)
    )
    e2e = dict(
        quantized_e2e_no_regression_report
        or build_simple_optimizer_quantized_e2e_no_regression_scorecard(
            training_loop_canary_report=training_loop
        )
    )
    state_sync = dict(
        quantized_product_state_sync_report
        or build_simple_optimizer_quantized_product_state_sync_scorecard(
            e2e_no_regression_report=e2e
        )
    )
    rollout = dict(
        quantized_rollout_policy_report
        or build_simple_optimizer_quantized_rollout_policy_scorecard(
            product_state_sync_report=state_sync
        )
    )
    dispatch_review = dict(
        quantized_dispatch_integration_review_report
        or build_simple_optimizer_quantized_dispatch_integration_review_scorecard(
            rollout_policy_report=rollout
        )
    )
    owner_hold = dict(
        quantized_owner_approval_hold_report
        or build_simple_optimizer_quantized_owner_approval_hold_scorecard(
            dispatch_review_report=dispatch_review
        )
    )
    schedule_reports = _schedule_free_reports(schedule_free_reports)
    rows = [_schedule_free_row(optimizer, abi, schedule_reports) for optimizer in SCHEDULE_FREE_CANARY_TARGETS]
    rows.extend(
        _quantized_pending_row(
            optimizer,
            abi,
            quantized,
            native_scratch,
            runtime_canary,
            training_loop,
            e2e,
            state_sync,
            rollout,
            dispatch_review,
            owner_hold,
        )
        for optimizer in QUANTIZED_PENDING_TARGETS
    )
    failed = [row for row in rows if row["variant_kind"] == "schedule_free_state_machine" and not row["native_canary_ready"]]
    blockers = _dedupe(reason for row in failed for reason in _strings(row.get("blocked_reasons")))
    schedule_ready_count = sum(1 for row in rows if row["variant_kind"] == "schedule_free_state_machine" and row["native_canary_ready"])
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_variant_native_canary_scorecard_v0",
        "gate": "simple_formula_variant_schedule_free_native_canary",
        "ok": not blockers,
        "promotion_ready": False,
        "variant_schedule_free_native_canary_ready": not blockers,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "native_kernel_ready": False,
        "product_native_dispatch_ready": False,
        "target_optimizer_types": [optimizer.value for optimizer in SCHEDULE_FREE_CANARY_TARGETS + QUANTIZED_PENDING_TARGETS],
        "rows": rows,
        "schedule_free_source_reports": {
            name: _compact_source_report(report) for name, report in schedule_reports.items()
        },
        "native_abi_summary": dict(abi.get("summary") or {}),
        "quantized_parity_summary": dict(quantized.get("summary") or {}),
        "quantized_native_scratch_summary": dict(native_scratch.get("summary") or {}),
        "quantized_runtime_canary_summary": dict(runtime_canary.get("summary") or {}),
        "quantized_training_loop_canary_summary": dict(training_loop.get("summary") or {}),
        "quantized_e2e_no_regression_summary": dict(e2e.get("summary") or {}),
        "quantized_product_state_sync_summary": dict(state_sync.get("summary") or {}),
        "quantized_rollout_policy_summary": dict(rollout.get("summary") or {}),
        "quantized_dispatch_integration_review_summary": dict(dispatch_review.get("summary") or {}),
        "quantized_owner_approval_hold_summary": dict(owner_hold.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(rows),
            "schedule_free_target_count": len(SCHEDULE_FREE_CANARY_TARGETS),
            "schedule_free_native_canary_ready_count": schedule_ready_count,
            "quantized_formula_parity_ready_count": int(
                dict(quantized.get("summary") or {}).get("quantized_formula_parity_ready_count", 0) or 0
            ),
            "quantized_native_scratch_kernel_ready_count": int(
                dict(native_scratch.get("summary") or {}).get("native_scratch_kernel_ready_count", 0) or 0
            ),
            "quantized_runtime_canary_manifest_ready_count": int(
                dict(runtime_canary.get("summary") or {}).get("runtime_canary_manifest_ready_count", 0) or 0
            ),
            "quantized_training_loop_canary_manifest_ready_count": int(
                dict(training_loop.get("summary") or {}).get("training_loop_canary_manifest_ready_count", 0) or 0
            ),
            "quantized_training_loop_canary_ready_count": int(
                dict(training_loop.get("summary") or {}).get("training_loop_canary_ready_count", 0) or 0
            ),
            "quantized_e2e_no_regression_ready_count": int(
                dict(e2e.get("summary") or {}).get("e2e_no_regression_ready_count", 0) or 0
            ),
            "quantized_product_state_sync_review_ready_count": int(
                dict(e2e.get("summary") or {}).get("product_state_sync_review_ready_count", 0) or 0
            ),
            "quantized_product_optimizer_state_sync_ready_count": int(
                dict(state_sync.get("summary") or {}).get("product_optimizer_state_sync_ready_count", 0) or 0
            ),
            "quantized_optimizer_state_sync_state_tensor_count": int(
                dict(state_sync.get("summary") or {}).get("optimizer_state_sync_state_tensor_count", 0) or 0
            ),
            "quantized_optimizer_state_sync_parameter_tensor_count": int(
                dict(state_sync.get("summary") or {}).get("optimizer_state_sync_parameter_tensor_count", 0) or 0
            ),
            "quantized_rollout_policy_ready_count": int(
                dict(rollout.get("summary") or {}).get("optimizer_count", 0) or 0
                if rollout.get("canary_rollout_policy_ready") is True
                else 0
            ),
            "quantized_dispatch_integration_review_ready_count": int(
                dict(dispatch_review.get("summary") or {}).get("optimizer_count", 0) or 0
                if dispatch_review.get("dispatch_integration_review") is True
                else 0
            ),
            "quantized_owner_approval_hold_ready_count": int(
                dict(owner_hold.get("summary") or {}).get("optimizer_count", 0) or 0
                if owner_hold.get("owner_approval_hold_ready") is True
                else 0
            ),
            "quantized_native_canary_pending_count": sum(
                1
                for row in rows
                if row["variant_kind"] == "quantized_state" and row["native_canary_ready"] is not True
            ),
            "native_abi_spec_ready_count": int(dict(abi.get("summary") or {}).get("native_abi_spec_ready_count", 0) or 0),
            "native_kernel_ready_count": int(
                dict(native_scratch.get("summary") or {}).get("native_scratch_kernel_ready_count", 0) or 0
            ),
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers
        + [
            "simple_variant_product_rollout_review_missing",
        ],
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold quantized product dispatch for explicit owner approval; keep schedule-free variants default-off pending rollout review"
            if not blockers
            else "fix schedule-free simple variant native canary blockers"
        ),
        "notes": [
            "This scorecard reuses existing schedule-free native canaries as route evidence for built-in variants.",
            "The source canaries keep training_path_enabled, runtime_dispatch_ready, and native_dispatch_allowed false at the scorecard boundary.",
            "Quantized Lion/SGD variants now have formula parity, native scratch kernels, runtime manifests, TrainingLoop canaries, e2e no-regression evidence, optimizer-state sync evidence, rollout policy evidence, dispatch review evidence, and owner-approval hold evidence, but remain product-dispatch pending.",
        ],
    }


def _schedule_free_reports(explicit: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if explicit is not None:
        return {str(name): dict(report) for name, report in explicit.items() if isinstance(report, Mapping)}
    return {
        OptimizerType.RADAM_SCHEDULE_FREE.value: build_plugin_schedulefree_radam_training_loop_canary_scorecard(),
        OptimizerType.SGD_SCHEDULE_FREE.value: build_plugin_schedulefree_sgd_training_loop_canary_scorecard(),
    }


def _schedule_free_row(
    optimizer: OptimizerType,
    abi_report: Mapping[str, Any],
    schedule_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source = dict(schedule_reports.get(optimizer.value) or {})
    source_ready = source.get("selected_native_canary_ready") is True
    abi_ready = _abi_ready_for(optimizer, abi_report)
    default_off = _default_off(source)
    ready = source_ready and abi_ready and default_off
    return {
        "optimizer_type": optimizer.value,
        "optimizer_kind": "schedulefree_radam" if optimizer == OptimizerType.RADAM_SCHEDULE_FREE else "schedulefree_sgd",
        "optimizer_family": "simple_formula",
        "variant_kind": "schedule_free_state_machine",
        "variant_status": "schedule_free_native_canary_ready" if ready else "schedule_free_native_canary_blocked",
        "native_abi_spec_ready": abi_ready,
        "native_canary_ready": ready,
        "source_scorecard": str(source.get("scorecard") or ""),
        "source_selected_optimizer_name": str(source.get("selected_optimizer_name") or ""),
        "native_step_count": int(dict(source.get("summary") or {}).get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(dict(source.get("summary") or {}).get("native_kernel_launch_count", 0) or 0),
        "native_kernel_ready": False,
        "runtime_canary_ready": ready,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": "default_off_rollout_review_and_e2e_shadow_matrix" if ready else "fix_schedule_free_native_canary",
        "blocked_reasons": [] if ready else _schedule_free_blockers(source_ready, abi_ready, default_off, source),
    }


def _quantized_pending_row(
    optimizer: OptimizerType,
    abi_report: Mapping[str, Any],
    quantized_report: Mapping[str, Any],
    native_scratch_report: Mapping[str, Any],
    runtime_canary_report: Mapping[str, Any],
    training_loop_report: Mapping[str, Any],
    e2e_report: Mapping[str, Any],
    state_sync_report: Mapping[str, Any],
    rollout_report: Mapping[str, Any],
    dispatch_review_report: Mapping[str, Any],
    owner_hold_report: Mapping[str, Any],
) -> dict[str, Any]:
    parity_ready = _quantized_parity_ready_for(optimizer, quantized_report)
    scratch_ready = _quantized_native_scratch_ready_for(optimizer, native_scratch_report)
    manifest_ready = _quantized_runtime_manifest_ready_for(optimizer, runtime_canary_report)
    loop_manifest_ready = _quantized_training_loop_manifest_ready_for(optimizer, training_loop_report)
    loop_ready = _quantized_training_loop_ready_for(optimizer, training_loop_report)
    e2e_ready = _quantized_e2e_ready_for(optimizer, e2e_report)
    state_sync_review_ready = _quantized_state_sync_review_ready_for(optimizer, e2e_report)
    product_state_sync_ready = _quantized_product_state_sync_ready_for(optimizer, state_sync_report)
    rollout_policy_ready = _quantized_rollout_policy_ready_for(optimizer, rollout_report)
    dispatch_review_ready = _quantized_dispatch_review_ready_for(optimizer, dispatch_review_report)
    owner_hold_ready = _quantized_owner_hold_ready_for(optimizer, owner_hold_report)
    return {
        "optimizer_type": optimizer.value,
        "optimizer_kind": _quantized_kind(optimizer),
        "optimizer_family": "simple_formula",
        "variant_kind": "quantized_state",
        "variant_status": (
            "quantized_owner_approval_hold_ready"
            if owner_hold_ready
            else "quantized_dispatch_integration_review_ready"
            if dispatch_review_ready
            else "quantized_rollout_policy_ready"
            if rollout_policy_ready
            else "quantized_product_state_sync_ready"
            if product_state_sync_ready
            else "quantized_e2e_no_regression_ready"
            if e2e_ready
            else "quantized_training_loop_canary_ready"
            if loop_ready
            else "quantized_training_loop_canary_manifest_ready"
            if loop_manifest_ready
            else "quantized_runtime_canary_manifest_ready"
            if manifest_ready
            else "quantized_native_scratch_kernel_ready"
            if scratch_ready
            else "quantized_formula_parity_ready"
            if parity_ready
            else "quantized_native_canary_pending"
        ),
        "native_abi_spec_ready": _abi_ready_for(optimizer, abi_report),
        "formula_parity_ready": parity_ready,
        "native_scratch_kernel_ready": scratch_ready,
        "runtime_canary_manifest_ready": manifest_ready,
        "training_loop_canary_manifest_ready": loop_manifest_ready,
        "training_loop_canary_ready": loop_ready,
        "e2e_no_regression_ready": e2e_ready,
        "product_state_sync_review_ready": state_sync_review_ready,
        "product_optimizer_state_sync_ready": product_state_sync_ready,
        "canary_rollout_policy_ready": rollout_policy_ready,
        "dispatch_integration_review_ready": dispatch_review_ready,
        "owner_approval_hold_ready": owner_hold_ready,
        "native_canary_ready": loop_ready,
        "native_kernel_ready": scratch_ready,
        "runtime_canary_ready": loop_ready,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": (
            "quantized_variant_explicit_owner_approval_record"
            if owner_hold_ready
            else "quantized_variant_owner_approval_hold"
            if dispatch_review_ready
            else "quantized_variant_dispatch_integration_review"
            if rollout_policy_ready
            else "quantized_variant_default_off_rollout_review"
            if product_state_sync_ready
            else "quantized_variant_product_state_sync_review"
            if e2e_ready
            else "quantized_variant_e2e_no_regression"
            if loop_ready
            else "quantized_variant_training_loop_executor"
            if loop_manifest_ready
            else "quantized_variant_training_loop_canary"
            if manifest_ready
            else "quantized_variant_runtime_canary"
            if scratch_ready
            else "quantized_variant_scratch_kernel"
        ),
        "blocked_reasons": [
            "quantized_simple_variant_owner_approval_missing"
            if owner_hold_ready or dispatch_review_ready
            else "quantized_simple_variant_dispatch_integration_review_missing"
            if rollout_policy_ready
            else "quantized_simple_variant_product_rollout_review_missing"
            if product_state_sync_ready
            else "quantized_simple_variant_product_state_sync_missing"
            if e2e_ready
            else "quantized_simple_variant_e2e_no_regression_missing"
            if loop_ready
            else "quantized_simple_variant_training_loop_executor_missing"
            if loop_manifest_ready
            else "quantized_simple_variant_training_loop_canary_missing"
            if manifest_ready
            else "quantized_simple_variant_runtime_canary_missing"
            if scratch_ready
            else "quantized_simple_variant_native_kernel_missing"
        ],
    }


def _abi_ready_for(optimizer: OptimizerType, abi_report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in abi_report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("native_abi_spec_ready") is True)


def _quantized_parity_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("formula_parity_ready") is True)


def _quantized_native_scratch_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("native_scratch_kernel_parity_ready") is True)


def _quantized_runtime_manifest_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("route_decisions", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("runtime_canary_manifest_ready") is True)


def _quantized_training_loop_manifest_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("training_loop_canary_manifest_ready") is True)


def _quantized_training_loop_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("training_loop_canary_ready") is True)


def _quantized_e2e_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("e2e_no_regression_ready") is True)


def _quantized_state_sync_review_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("product_state_sync_review_ready") is True)


def _quantized_product_state_sync_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("product_optimizer_state_sync_ready") is True)


def _quantized_rollout_policy_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    policy = report.get("policy")
    if not isinstance(policy, Mapping) or report.get("canary_rollout_policy_ready") is not True:
        return False
    optimizer_types = policy.get("optimizer_types")
    return isinstance(optimizer_types, list) and optimizer.value in {str(item) for item in optimizer_types}


def _quantized_dispatch_review_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    review = report.get("review_package")
    if not isinstance(review, Mapping) or report.get("dispatch_integration_review") is not True:
        return False
    optimizer_types = review.get("optimizer_types")
    return isinstance(optimizer_types, list) and optimizer.value in {str(item) for item in optimizer_types}


def _quantized_owner_hold_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    hold = report.get("hold_manifest")
    if not isinstance(hold, Mapping) or report.get("owner_approval_hold_ready") is not True:
        return False
    optimizer_types = hold.get("optimizer_types")
    return isinstance(optimizer_types, list) and optimizer.value in {str(item) for item in optimizer_types}


def _default_off(report: Mapping[str, Any]) -> bool:
    if not report:
        return False
    return (
        report.get("training_path_enabled") is False
        and report.get("runtime_dispatch_ready") is False
        and report.get("native_dispatch_allowed") is False
        and report.get("default_behavior_changed") is False
    )


def _schedule_free_blockers(
    source_ready: bool,
    abi_ready: bool,
    default_off: bool,
    source: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not source_ready:
        reasons.append("schedule_free_source_native_canary_not_ready")
    if not abi_ready:
        reasons.append("simple_variant_native_abi_spec_not_ready")
    if not default_off:
        reasons.append("schedule_free_source_scorecard_not_default_off")
    reasons.extend(_strings(source.get("blocked_reasons")))
    return _dedupe(reasons)


def _compact_source_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary") or {})
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "selected_native_canary_ready": report.get("selected_native_canary_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _quantized_kind(optimizer: OptimizerType) -> str:
    mapping: dict[OptimizerType, str] = {
        OptimizerType.LION_8BIT: "lion_8bit",
        OptimizerType.PAGED_LION_8BIT: "paged_lion_8bit",
        OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov_8bit",
    }
    return mapping[optimizer]


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_simple_optimizer_variant_native_canary_scorecard"]
