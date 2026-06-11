"""Report-only batch scorecard for selected closure/second-order plugin optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_closure_second_order_execution_matrix import (
    build_closure_second_order_execution_matrix,
    promote_closure_second_order_row,
)


TARGET_PLUGIN_OPTIMIZERS: tuple[str, ...] = (
    "adahessian",
    "alig",
    "bsam",
    "kron",
    "lbfgs",
)

UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "product_native_dispatch_ready",
    "product_native_ready",
    "native_kernel_ready",
)

_ABI_REQUIREMENTS_BY_OPTIMIZER: dict[str, dict[str, bool]] = {
    "adahessian": {
        "requires_closure": False,
        "requires_create_graph": True,
        "requires_hessian_or_hvp": True,
        "requires_second_order_gradients": True,
        "requires_training_loop_abi": True,
    },
    "alig": {
        "requires_closure": True,
        "requires_create_graph": False,
        "requires_hessian_or_hvp": False,
        "requires_second_order_gradients": False,
        "requires_training_loop_abi": True,
    },
    "bsam": {
        "requires_closure": True,
        "requires_create_graph": True,
        "requires_hessian_or_hvp": True,
        "requires_second_order_gradients": True,
        "requires_training_loop_abi": True,
    },
    "kron": {
        "requires_closure": True,
        "requires_create_graph": True,
        "requires_hessian_or_hvp": True,
        "requires_second_order_gradients": True,
        "requires_training_loop_abi": True,
    },
    "lbfgs": {
        "requires_closure": True,
        "requires_create_graph": False,
        "requires_hessian_or_hvp": False,
        "requires_second_order_gradients": False,
        "requires_training_loop_abi": True,
    },
}


def build_plugin_closure_second_order_family_batch_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate selected plugin closure/second-order status without dispatch."""

    selector = _as_dict(selector_report) if selector_report is not None else _call_selector()
    selector_rows = _selector_rows(selector)
    rows = [_row(name, selector_rows.get(name, {})) for name in TARGET_PLUGIN_OPTIMIZERS]
    execution_matrix = build_closure_second_order_execution_matrix(rows)
    rows = [promote_closure_second_order_row(row, execution_matrix) for row in rows]
    missing = [row["selected_optimizer_name"] for row in rows if row["selector_classified"] is not True]
    unsafe = _unsafe_claims({"selector": selector}, rows)
    ready = selector.get("ok") is True and execution_matrix.get("ok") is True and not missing and not unsafe

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_closure_second_order_family_batch_scorecard_v0",
        "gate": "plugin_closure_second_order_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_closure_second_order_family_batch_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": "closure_or_second_order",
        "selector_scorecard": _compact_selector(selector),
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
            "runtime_authority": "existing_pytorch_optimizer_plugin",
            "native_route_policy": "blocked_until_closure_create_graph_or_hessian_training_loop_abi",
        },
        "training_loop_abi": {
            "requires_reentrant_closure_support": True,
            "requires_create_graph_policy": True,
            "requires_hessian_or_hvp_policy": True,
            "requires_higher_order_gradient_lifetime_policy": True,
            "abi_spec_ready": ready,
            "abi_implementation_ready": execution_matrix.get("ok") is True,
            "adamw_step_kernel_compatible": False,
            "simple_formula_kernel_compatible": False,
            "can_reuse_exact_adamw_native_dispatch": False,
        },
        "execution_matrix": _compact_execution_matrix(execution_matrix),
        "rows": rows,
        "summary": {
            "selected_optimizer_count": len(rows),
            "selector_closure_or_second_order_count": int(
                _as_dict(_summary(selector).get("route_family_counts")).get("closure_or_second_order", 0) or 0
            ),
            "selector_classified_count": sum(1 for row in rows if row["selector_classified"] is True),
            "closure_required_count": sum(1 for row in rows if row["abi_requirements"]["requires_closure"] is True),
            "create_graph_required_count": sum(
                1 for row in rows if row["abi_requirements"]["requires_create_graph"] is True
            ),
            "hessian_or_hvp_required_count": sum(
                1 for row in rows if row["abi_requirements"]["requires_hessian_or_hvp"] is True
            ),
            "higher_order_training_loop_abi_required_count": sum(
                1 for row in rows if row["requires_higher_order_training_loop_abi"] is True
            ),
            "training_loop_abi_spec_ready_count": sum(
                1 for row in rows if row["training_loop_abi_spec_ready"] is True
            ),
            "training_loop_abi_implementation_ready_count": int(
                _summary(execution_matrix).get("training_loop_abi_implementation_ready_count", 0) or 0
            ),
            "closure_replay_contract_required_count": sum(
                1 for row in rows if row["training_loop_abi_contract"]["closure_replay_required"] is True
            ),
            "resume_parity_matrix_spec_ready_count": sum(
                1 for row in rows if row["resume_parity_matrix_plan"]["spec_ready"] is True
            ),
            "resume_parity_matrix_implementation_ready_count": int(
                _summary(execution_matrix).get("resume_parity_matrix_implementation_ready_count", 0) or 0
            ),
            "closure_resume_replay_artifact_ready_count": sum(
                1
                for row in rows
                if row["resume_parity_matrix_plan"]["closure_resume_replay_artifact"]["artifact_ready"] is True
            ),
            "closure_resume_replay_artifact_row_count": sum(
                len(row["resume_parity_matrix_plan"]["closure_resume_replay_artifact"]["rows"]) for row in rows
            ),
            "closure_resume_replay_artifact_implementation_ready_count": int(
                _summary(execution_matrix).get("closure_resume_replay_artifact_implementation_ready_count", 0) or 0
            ),
            "closure_resume_replay_row_implementation_ready_count": int(
                _summary(execution_matrix).get("closure_resume_replay_row_implementation_ready_count", 0) or 0
            ),
            "closure_replay_case_plan_ready_count": sum(
                1 for row in rows if row["resume_parity_matrix_plan"]["closure_replay_cases"]
            ),
            "closure_replay_case_planned_count": sum(
                len(row["resume_parity_matrix_plan"]["closure_replay_cases"]) for row in rows
            ),
            "create_graph_hvp_lifetime_case_plan_ready_count": sum(
                1 for row in rows if row["resume_parity_matrix_plan"]["create_graph_hvp_lifetime_cases"]
            ),
            "create_graph_hvp_lifetime_case_planned_count": sum(
                len(row["resume_parity_matrix_plan"]["create_graph_hvp_lifetime_cases"]) for row in rows
            ),
            "higher_order_graph_lifetime_required_count": sum(
                1 for row in rows if row["training_loop_abi_contract"]["higher_order_graph_lifetime"] != "none"
            ),
            "state_resume_adapter_required_count": sum(
                1 for row in rows if row["training_loop_abi_contract"]["state_resume_adapter_required"] is True
            ),
            "state_resume_adapter_scope_plan_ready_count": sum(
                1 for row in rows if row["resume_parity_matrix_plan"]["state_resume_adapter_scope"]
            ),
            "state_resume_adapter_implementation_ready_count": int(
                _summary(execution_matrix).get("state_resume_adapter_implementation_ready_count", 0) or 0
            ),
            "unsafe_native_reuse_blocker_plan_ready_count": sum(
                1 for row in rows if row["resume_parity_matrix_plan"]["unsafe_native_reuse_blockers"]
            ),
            "unsafe_native_reuse_blocker_planned_count": sum(
                len(row["resume_parity_matrix_plan"]["unsafe_native_reuse_blockers"]) for row in rows
            ),
            "native_kernel_precondition_plan_ready_count": sum(
                1 for row in rows if row["resume_parity_matrix_plan"]["native_kernel_preconditions"]
            ),
            "native_kernel_preconditions_implementation_ready_count": int(
                _summary(execution_matrix).get("native_kernel_preconditions_implementation_ready_count", 0) or 0
            ),
            "adamw_kernel_compatible_count": 0,
            "simple_kernel_compatible_count": 0,
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "plugin_selected_native_ready_count": 0,
            "exact_adamw_product_native_route_count_delta": 0,
            "unsafe_claim_count": len(unsafe),
            "missing_selector_classification_count": len(missing),
        },
        "promotion_blockers": _dedupe(
            unsafe
            + [f"selector_closure_or_second_order_missing:{name}" for name in missing]
            + [
                "adamw_native_simple_kernel_not_reusable",
                "native_kernel_implementation_missing",
                "runtime_dispatch_shadow_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(
            unsafe + [f"selector_closure_or_second_order_missing:{name}" for name in missing]
        ),
        "recommended_next_step": (
            "owner/release hold for implementation-ready closure/second-order ABI matrices with dispatch default-off"
            if ready
            else "fix selector closure/second-order blockers before ABI drafting"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Closure and second-order plugins require training-loop ABI work before kernel work.",
            "These optimizers cannot reuse exact AdamW or simple-formula native kernels.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _call_selector() -> dict[str, Any]:
    try:
        return dict(build_plugin_optimizer_selector_scorecard())
    except Exception as exc:
        return _failed_report("build_plugin_optimizer_selector_scorecard", exc)


def _selector_rows(selector: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = selector.get("rows", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("optimizer_name", "")).strip().lower(): row
        for row in rows
        if isinstance(row, Mapping) and row.get("native_route_family") == "closure_or_second_order"
    }


def _row(name: str, selector_row: Mapping[str, Any]) -> dict[str, Any]:
    requirements = dict(_ABI_REQUIREMENTS_BY_OPTIMIZER[name])
    classified = str(selector_row.get("native_route_family", "")) == "closure_or_second_order"
    abi_spec_ready = classified
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selector": str(selector_row.get("selector", OptimizerType.PYTORCH_OPTIMIZER.value)),
        "native_route_family": "closure_or_second_order",
        "selector_classified": classified,
        "resume_proven": selector_row.get("resume_proven") is True,
        "batch_status": "abi_required_report_only" if classified else "selector_classification_pending",
        "abi_requirements": requirements,
        "requires_higher_order_training_loop_abi": bool(
            requirements["requires_training_loop_abi"]
            and (
                requirements["requires_closure"]
                or requirements["requires_create_graph"]
                or requirements["requires_hessian_or_hvp"]
                or requirements["requires_second_order_gradients"]
            )
        ),
        "training_loop_abi_spec_ready": abi_spec_ready,
        "training_loop_abi_implementation_ready": False,
        "training_loop_abi_contract": _training_loop_abi_contract(name, requirements),
        "resume_parity_matrix_plan": _resume_parity_matrix_plan(name, requirements, abi_spec_ready),
        "runtime_authority": "existing_pytorch_optimizer_plugin",
        "native_route": "none_report_only",
        "adamw_kernel_compatible": False,
        "simple_formula_kernel_compatible": False,
        "can_reuse_exact_adamw_native_dispatch": False,
        "plugin_selected_native_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": "selected_plugin_closure_create_graph_hessian_training_loop_abi",
        "blocked_reasons": [
            "selected_plugin_closure_create_graph_training_loop_abi_implementation_missing",
            "selected_plugin_hessian_or_hvp_lifetime_policy_implementation_missing",
            "adamw_native_simple_kernel_not_reusable",
            "native_dispatch_gate_not_requested",
        ],
    }


def _resume_parity_matrix_plan(
    name: str,
    requirements: Mapping[str, bool],
    spec_ready: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "spec_ready": spec_ready,
        "implementation_ready": False,
        "closure_replay_cases": _closure_replay_cases(name, requirements),
        "create_graph_hvp_lifetime_cases": _create_graph_hvp_lifetime_cases(name, requirements),
        "state_resume_adapter_scope": _state_resume_scope(name),
        "state_resume_adapter_cases": _state_resume_adapter_cases(name),
        "closure_resume_replay_artifact": _closure_resume_replay_artifact(name, requirements, spec_ready),
        "unsafe_native_reuse_blockers": _unsafe_native_reuse_blockers(name, requirements),
        "native_kernel_preconditions": _native_kernel_preconditions(name, requirements),
        "evidence_status": "planned_report_only",
    }


def _training_loop_abi_contract(name: str, requirements: Mapping[str, bool]) -> dict[str, Any]:
    closure_required = requirements["requires_closure"]
    create_graph_required = requirements["requires_create_graph"]
    hessian_required = requirements["requires_hessian_or_hvp"]
    second_order_required = requirements["requires_second_order_gradients"]
    return {
        "schema_version": 1,
        "closure_replay_required": closure_required,
        "closure_call_site": "optimizer_step_closure" if closure_required else "gradient_phase_after_backward",
        "create_graph_policy": "required" if create_graph_required else "disabled_or_optimizer_default",
        "hessian_or_hvp_policy": "required" if hessian_required else "not_required",
        "higher_order_graph_lifetime": (
            "retain_until_optimizer_step_and_state_sync"
            if create_graph_required or hessian_required or second_order_required
            else "none"
        ),
        "gradient_payload_policy": (
            "second_order_gradients_or_hvp_payload"
            if hessian_required or second_order_required
            else "first_order_gradients_with_closure_loss_payload"
        ),
        "state_resume_adapter_required": True,
        "state_resume_adapter_scope": _state_resume_scope(name),
        "native_kernel_precondition": "training_loop_abi_and_resume_parity_matrix_ready",
    }


def _closure_replay_cases(name: str, requirements: Mapping[str, bool]) -> list[str]:
    if not requirements["requires_closure"]:
        return [
            "single_backward_no_closure_replay",
            "second_order_gradient_payload_after_standard_backward",
            "resume_after_step_without_saved_closure",
        ]
    cases = [
        "closure_called_exactly_once_per_attempt",
        "closure_loss_is_recomputed_after_resume",
        "closure_rng_and_grad_accumulation_are_restored",
        "failed_or_skipped_step_does_not_reuse_stale_closure_loss",
    ]
    if name == "lbfgs":
        cases.extend(["multi_eval_line_search_replays_closure", "history_buffers_match_after_resume"])
    if name == "bsam":
        cases.append("perturbation_closure_replay_restores_base_weights")
    return cases


def _create_graph_hvp_lifetime_cases(name: str, requirements: Mapping[str, bool]) -> list[str]:
    cases = ["create_graph_disabled_path_releases_first_order_graph"]
    if requirements["requires_create_graph"]:
        cases.extend(
            [
                "create_graph_true_retains_graph_until_optimizer_step",
                "higher_order_graph_is_released_after_state_sync",
            ]
        )
    if requirements["requires_hessian_or_hvp"]:
        cases.extend(["hvp_payload_lifetime_survives_native_boundary", "hessian_statistics_match_after_resume"])
    if name == "kron":
        cases.append("kron_expression_tree_rebuilds_without_dangling_hvp_refs")
    return cases


def _state_resume_adapter_cases(name: str) -> list[str]:
    common = ["state_dict_roundtrip_before_step", "state_dict_roundtrip_after_step"]
    if name == "lbfgs":
        return common + ["history_buffer_resume", "line_search_loss_resume"]
    if name == "kron":
        return common + ["factor_state_resume", "expression_tree_resume"]
    if name == "bsam":
        return common + ["perturbation_state_resume", "base_optimizer_state_resume"]
    if name == "adahessian":
        return common + ["hessian_statistics_resume"]
    return common + ["plugin_specific_scalar_state_resume"]


def _closure_resume_replay_artifact(name: str, requirements: Mapping[str, bool], spec_ready: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "closure_second_order_resume_replay_rows_v0",
        "artifact_ready": spec_ready,
        "artifact_status": "planned_report_only" if spec_ready else "selector_classification_pending",
        "implementation_ready": False,
        "rows": _closure_resume_replay_artifact_rows(name, requirements),
    }


def _closure_resume_replay_artifact_rows(name: str, requirements: Mapping[str, bool]) -> list[dict[str, Any]]:
    closure_boundary = "optimizer_step_closure_replay" if requirements["requires_closure"] else "standard_backward"
    rows = [
        _replay_artifact_row(
            "pre_step_resume",
            closure_boundary,
            ["optimizer_state_dict", "param_group_hparams", "rng_snapshot"],
            ["next_step_loss_replay_matches_reference", "stale_loss_not_reused"],
        ),
        _replay_artifact_row(
            "post_step_resume",
            "state_dict_after_optimizer_step",
            ["optimizer_state_dict", "step_index", _state_resume_scope(name)],
            ["state_resume_scope_roundtrips", "next_step_update_matches_reference"],
        ),
        _replay_artifact_row(
            "skip_or_failed_attempt_resume",
            closure_boundary,
            ["skip_reason", "grad_accumulation_boundary", "optimizer_state_dict"],
            ["failed_attempt_does_not_advance_state", "next_attempt_rebuilds_grad_payload"],
        ),
    ]
    if requirements["requires_create_graph"] or requirements["requires_hessian_or_hvp"]:
        rows.append(
            _replay_artifact_row(
                "higher_order_payload_resume",
                "create_graph_hvp_lifetime",
                ["create_graph_flag", "hvp_or_hessian_payload", "graph_release_token"],
                ["higher_order_payload_rebuilt_after_resume", "graph_lifetime_policy_observed"],
            )
        )
    if name == "lbfgs":
        rows.append(
            _replay_artifact_row(
                "line_search_history_resume",
                "multi_eval_line_search",
                ["closure_eval_index", "history_buffers", "line_search_loss"],
                ["line_search_replays_closure_order", "history_buffers_match_reference"],
            )
        )
    if name == "bsam":
        rows.append(
            _replay_artifact_row(
                "perturbation_resume",
                "sam_perturbation_restore",
                ["base_weight_snapshot", "perturbation_state", "base_optimizer_state"],
                ["base_weights_restored_before_replay", "perturbation_state_roundtrips"],
            )
        )
    return rows


def _replay_artifact_row(
    case_id: str, replay_boundary: str, required_payload: list[str], replay_assertions: list[str]
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "artifact_status": "planned_report_only",
        "replay_boundary": replay_boundary,
        "required_payload": required_payload,
        "replay_assertions": replay_assertions,
        "implementation_ready": False,
        "native_dispatch_allowed": False,
    }


def _unsafe_native_reuse_blockers(name: str, requirements: Mapping[str, bool]) -> list[str]:
    blockers = [
        "not_elementwise_adamw_update",
        "not_simple_formula_single_pass_update",
        "requires_training_loop_abi_before_kernel",
        "resume_parity_matrix_not_implemented",
    ]
    if requirements["requires_closure"]:
        blockers.append("closure_replay_crosses_optimizer_step_boundary")
    if requirements["requires_create_graph"]:
        blockers.append("create_graph_lifetime_crosses_native_boundary")
    if requirements["requires_hessian_or_hvp"]:
        blockers.append("hessian_or_hvp_payload_not_represented_by_adamw_state")
    if name == "lbfgs":
        blockers.append("multi_eval_line_search_state_machine")
    if name == "kron":
        blockers.append("kron_factor_expression_state_machine")
    return blockers


def _native_kernel_preconditions(name: str, requirements: Mapping[str, bool]) -> list[str]:
    preconditions = [
        "training_loop_abi_implementation_ready",
        "resume_parity_matrix_implementation_ready",
        "owner_release_hold_recorded",
        "default_off_product_gate_preserved",
    ]
    if requirements["requires_closure"]:
        preconditions.append("closure_replay_cases_pass")
    if requirements["requires_create_graph"] or requirements["requires_hessian_or_hvp"]:
        preconditions.append("create_graph_hvp_lifetime_cases_pass")
    preconditions.append(f"state_resume_adapter_scope:{_state_resume_scope(name)}")
    return preconditions


def _state_resume_scope(name: str) -> str:
    if name == "kron":
        return "kron_expression_tree_and_factor_state"
    if name == "lbfgs":
        return "lbfgs_history_buffers_and_closure_loss"
    if name == "adahessian":
        return "hessian_running_statistics"
    if name == "bsam":
        return "sam_perturbation_and_base_optimizer_state"
    return "optimizer_plugin_state_dict"


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    counts = _as_dict(summary.get("route_family_counts"))
    return {
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "selector_boundary_ready": report.get("selector_boundary_ready") is True,
        "all_discovered_plugins_resume_proven": report.get("all_discovered_plugins_resume_proven") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "closure_or_second_order_count": int(counts.get("closure_or_second_order", 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
    }


def _compact_execution_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "execution_matrix_ready": report.get("execution_matrix_ready") is True,
        "training_path_enabled": report.get("training_path_enabled") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "native_kernel_ready": report.get("native_kernel_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "training_loop_abi_implementation_ready_count": int(
            summary.get("training_loop_abi_implementation_ready_count", 0) or 0
        ),
        "resume_parity_matrix_implementation_ready_count": int(
            summary.get("resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "closure_resume_replay_row_implementation_ready_count": int(
            summary.get("closure_resume_replay_row_implementation_ready_count", 0) or 0
        ),
        "blocked_reasons": [str(reason) for reason in report.get("blocked_reasons", []) or [] if str(reason)],
    }


def _unsafe_claims(
    reports: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> list[str]:
    out: list[str] = []
    for name, report in reports.items():
        scorecard = str(report.get("scorecard", name))
        for field in UNSAFE_TRUE_FIELDS:
            if report.get(field) is True:
                out.append(f"unsafe_plugin_closure_second_order_source:{scorecard}:{field}")
    for row in rows:
        selected = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                out.append(f"unsafe_plugin_closure_second_order_row:{selected}:{field}")
        if row.get("adamw_kernel_compatible") is True or row.get("simple_formula_kernel_compatible") is True:
            out.append(f"unsafe_plugin_closure_second_order_row:{selected}:kernel_compatible")
    return _dedupe(out)


def _failed_report(builder_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "scorecard": builder_name,
        "error": f"{type(exc).__name__}: {exc}",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [f"builder_failed:{builder_name}"],
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_closure_second_order_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "TARGET_PLUGIN_OPTIMIZERS",
    "build_plugin_closure_second_order_family_batch_scorecard",
]
