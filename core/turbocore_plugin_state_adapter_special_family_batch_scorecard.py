"""Report-only batch scorecard for selected plugin state-adapter special routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_state_adapter_special_execution_matrix import (
    build_state_adapter_special_execution_matrix,
    promote_state_adapter_special_row,
)


STATE_ADAPTER_SPECIAL_ROUTE_FAMILY = "state_adapter_special"
STATE_ADAPTER_SPECIAL_OPTIMIZERS = ("demo", "sgdsai", "spam")
UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "product_native_ready",
    "product_native_dispatch_ready",
    "native_kernel_ready",
)


def build_plugin_state_adapter_special_family_batch_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate selected state-adapter special plugins without dispatch."""

    selector = _as_dict(selector_report) if selector_report is not None else _call_selector()
    selector_rows = _selected_state_adapter_rows(selector)
    rows = [_row(row) for row in selector_rows]
    execution_matrix = build_state_adapter_special_execution_matrix(rows)
    rows = [promote_state_adapter_special_row(row, execution_matrix) for row in rows]
    selector_count = int(
        _as_dict(_summary(selector).get("route_family_counts")).get(STATE_ADAPTER_SPECIAL_ROUTE_FAMILY, 0) or 0
    )
    missing = max(selector_count - len(rows), 0)
    unsafe = _unsafe_claims({"selector": selector}, rows)
    ready = selector.get("ok") is True and execution_matrix.get("ok") is True and bool(rows) and missing == 0 and not unsafe

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_state_adapter_special_family_batch_scorecard_v0",
        "gate": "plugin_state_adapter_special_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_state_adapter_special_family_batch_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": STATE_ADAPTER_SPECIAL_ROUTE_FAMILY,
        "selector_scorecard": _compact_selector(selector),
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": [row["selected_optimizer_name"] for row in rows],
            "runtime_authority": "existing_pytorch_optimizer_plugin_with_lulynx_bridge",
            "native_route_policy": "blocked_until_special_state_adapter_resume_and_param_ownership_abi",
        },
        "native_compatibility": {
            "requires_special_optimizer_state_adapter": True,
            "requires_resume_state_adapter": True,
            "requires_param_ownership_abi": True,
            "adapter_abi_spec_ready": ready,
            "adapter_abi_implementation_ready": execution_matrix.get("ok") is True,
            "adapter_resume_matrix_artifact_ready": ready,
            "adapter_resume_matrix_implementation_ready": execution_matrix.get("ok") is True,
            "adamw_state_schema_compatible": False,
            "adamw_step_kernel_compatible": False,
            "simple_formula_kernel_compatible": False,
            "can_reuse_exact_adamw_native_dispatch": False,
            "exact_adamw_product_native_route_count_delta": 0,
        },
        "execution_matrix": _compact_execution_matrix(execution_matrix),
        "rows": rows,
        "summary": {
            "selected_optimizer_count": len(rows),
            "selector_state_adapter_special_count": selector_count,
            "selector_classified_count": sum(1 for row in rows if row["selector_classified"] is True),
            "resume_proven_count": sum(1 for row in rows if row["resume_proven"] is True),
            "special_optimizer_state_adapter_required_count": sum(
                1 for row in rows if row["state_adapter_contract"]["requires_special_optimizer_state_adapter"] is True
            ),
            "resume_state_adapter_required_count": sum(
                1 for row in rows if row["state_adapter_contract"]["requires_resume_state_adapter"] is True
            ),
            "param_ownership_abi_required_count": sum(
                1 for row in rows if row["state_adapter_contract"]["requires_param_ownership_abi"] is True
            ),
            "adapter_abi_spec_ready_count": sum(1 for row in rows if row["adapter_abi_spec_ready"] is True),
            "adapter_abi_implementation_ready_count": int(
                _summary(execution_matrix).get("adapter_abi_implementation_ready_count", 0) or 0
            ),
            "param_ownership_abi_spec_ready_count": _spec_section_ready_count(rows, "param_ownership"),
            "state_adapter_role_spec_ready_count": _spec_section_ready_count(rows, "state_adapter_role"),
            "resume_translation_scope_spec_ready_count": _spec_section_ready_count(rows, "resume_translation_scope"),
            "quality_safety_guard_spec_ready_count": _spec_section_ready_count(rows, "quality_safety_guard"),
            "native_kernel_precondition_spec_ready_count": _spec_section_ready_count(
                rows, "native_kernel_preconditions"
            ),
            "adapter_resume_matrix_artifact_ready_count": sum(
                1 for row in rows if row["adapter_resume_matrix_artifact"]["spec_ready"] is True
            ),
            "adapter_resume_matrix_implementation_ready_count": int(
                _summary(execution_matrix).get("adapter_resume_matrix_implementation_ready_count", 0) or 0
            ),
            "adapter_resume_replay_case_planned_count": sum(
                len(row["adapter_resume_matrix_artifact"]["resume_replay_cases"]) for row in rows
            ),
            "adapter_resume_translation_case_planned_count": sum(
                len(row["adapter_resume_matrix_artifact"]["translation_cases"]) for row in rows
            ),
            "adapter_resume_replay_case_implementation_ready_count": int(
                _summary(execution_matrix).get("adapter_resume_replay_case_implementation_ready_count", 0) or 0
            ),
            "adapter_resume_translation_case_implementation_ready_count": int(
                _summary(execution_matrix).get("adapter_resume_translation_case_implementation_ready_count", 0) or 0
            ),
            "native_kernel_precondition_implementation_ready_count": int(
                _summary(execution_matrix).get("native_kernel_precondition_implementation_ready_count", 0) or 0
            ),
            "adamw_kernel_compatible_count": 0,
            "simple_kernel_compatible_count": 0,
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "product_native_dispatch_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "plugin_selected_native_ready_count": 0,
            "exact_adamw_product_native_route_count_delta": 0,
            "missing_selector_classification_count": missing,
            "unsafe_claim_count": len(unsafe),
        },
        "promotion_blockers": _dedupe(
            unsafe
            + (
                [f"selector_state_adapter_special_count_mismatch:{selector_count}:{len(rows)}"]
                if missing
                else []
            )
            + [
                "adamw_native_simple_kernel_not_reusable",
                "native_kernel_implementation_missing",
                "runtime_dispatch_shadow_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(
            unsafe
            + (
                [f"selector_state_adapter_special_count_mismatch:{selector_count}:{len(rows)}"]
                if missing
                else []
            )
        ),
        "recommended_next_step": (
            "owner/release hold for implementation-ready state-adapter ABI matrices with dispatch default-off"
            if ready
            else "fix selector state-adapter-special blockers before ABI drafting"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "State-adapter-special plugins require optimizer-specific state adapters and resume ABI.",
            "These plugins require a param ownership ABI and cannot reuse exact AdamW or simple kernels.",
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


def _selected_state_adapter_rows(selector: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = selector.get("rows", [])
    if not isinstance(rows, list):
        return []
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("native_route_family", "")) == STATE_ADAPTER_SPECIAL_ROUTE_FAMILY
    ]
    return sorted(selected, key=lambda row: str(row.get("optimizer_name", "")).strip().lower())


def _row(selector_row: Mapping[str, Any]) -> dict[str, Any]:
    name = str(selector_row.get("optimizer_name", "")).strip().lower()
    resume_proven = selector_row.get("resume_proven") is True
    abi_spec_ready = resume_proven and name in STATE_ADAPTER_SPECIAL_OPTIMIZERS
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selector": str(selector_row.get("selector", OptimizerType.PYTORCH_OPTIMIZER.value)),
        "native_route_family": STATE_ADAPTER_SPECIAL_ROUTE_FAMILY,
        "selector_classified": True,
        "resume_proven": resume_proven,
        "special_handling": str(selector_row.get("special_handling", "")),
        "batch_status": "state_adapter_contract_required_report_only",
        "state_adapter_contract": _state_adapter_contract(name),
        "adapter_abi_spec_ready": abi_spec_ready,
        "adapter_abi_implementation_ready": False,
        "state_adapter_abi_spec": _state_adapter_abi_spec(name),
        "adapter_resume_matrix_artifact": _adapter_resume_matrix_artifact(name, abi_spec_ready),
        "runtime_authority": "existing_pytorch_optimizer_plugin_with_lulynx_bridge",
        "native_route": "none_report_only",
        "adamw_state_schema_compatible": False,
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
        "next_gate": "selected_plugin_special_state_adapter_resume_param_ownership_abi",
        "blocked_reasons": [
            "selected_plugin_special_optimizer_state_adapter_abi_implementation_missing",
            "selected_plugin_special_resume_state_adapter_abi_implementation_missing",
            "selected_plugin_param_ownership_abi_implementation_missing",
            "selected_plugin_state_adapter_special_resume_parity_matrix_missing",
            "adamw_native_simple_kernel_not_reusable",
            "native_dispatch_gate_not_requested",
        ],
    }


def _state_adapter_contract(name: str) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "requires_special_optimizer_state_adapter": True,
        "requires_resume_state_adapter": True,
        "requires_param_ownership_abi": True,
        "requires_state_dict_key_translation": True,
        "requires_optimizer_step_side_effect_review": True,
        "adamw_state_schema_compatible": False,
        "simple_formula_kernel_compatible": False,
    }
    if name == "demo":
        contract.update(
            {
                "state_adapter_family": "distributed_demo_state_bridge",
                "requires_distributed_gather_fallback": True,
                "requires_external_demo_state_persistence": True,
            }
        )
    elif name == "sgdsai":
        contract.update(
            {
                "state_adapter_family": "warmup_flag_resume_bridge",
                "requires_non_state_dict_attribute_restore": True,
                "requires_warmup_phase_resume_contract": True,
            }
        )
    elif name == "spam":
        contract.update(
            {
                "state_adapter_family": "sparse_mask_resume_bridge",
                "requires_bool_mask_restore": True,
                "requires_sparse_state_density_contract": True,
            }
        )
    else:
        contract["state_adapter_family"] = "optimizer_specific_state_bridge"
    return contract


def _state_adapter_abi_spec(name: str) -> dict[str, Any]:
    common = {
        "schema_version": 1,
        "report_only": True,
        "implementation_ready": False,
        "native_dispatch_allowed": False,
        "adamw_state_schema_compatible": False,
        "simple_formula_kernel_compatible": False,
    }
    spec_by_name = {
        "demo": {
            "param_ownership": {
                "spec_ready": True,
                "owner": "optimizer.demo_state sidecar keyed by ordered param_groups params",
                "identity": "ordered_parameter_index",
                "bridge_payload": "lulynx_demo_state",
                "mutation_boundary": "demo_all_gather may update optimizer-owned sparse demo state",
            },
            "state_adapter_role": {
                "spec_ready": True,
                "role": "wrap demo_all_gather for no-process-group identity fallback and persist demo_state",
                "python_bridge": "optimizer_plugin_bridge._patch_demo_optimizer",
                "side_effects": "distributed gather fallback plus optimizer.demo_state sidecar persistence",
            },
            "resume_translation_scope": {
                "spec_ready": True,
                "source": "state_dict['lulynx_demo_state'] ordered sidecar list",
                "target": "optimizer.demo_state[param] restored after base load_state_dict",
                "tensor_policy": "clone tensors to target param device and floating dtype",
            },
            "quality_safety_guard": {
                "spec_ready": True,
                "guards": [
                    "ordered_param_count_match",
                    "no_process_group_identity_gather_parity",
                    "sidecar_tensor_device_dtype_clone",
                    "state_dict_roundtrip_resume_parity",
                ],
            },
            "native_kernel_preconditions": {
                "spec_ready": True,
                "required_before_kernel": [
                    "param_order_abi_implemented",
                    "demo_state_sidecar_resume_adapter_implemented",
                    "distributed_collective_policy_locked",
                    "resume_parity_matrix_passed",
                ],
            },
        },
        "sgdsai": {
            "param_ownership": {
                "spec_ready": True,
                "owner": "optimizer attribute has_warmup plus normal upstream state_dict state",
                "identity": "single_optimizer_instance_attribute",
                "bridge_payload": "lulynx_optimizer_attrs.has_warmup",
                "mutation_boundary": "warmup phase flag is outside upstream optimizer state_dict",
            },
            "state_adapter_role": {
                "spec_ready": True,
                "role": "capture and restore has_warmup around load_state_dict",
                "python_bridge": "optimizer_plugin_bridge state-attribute restore for sgdsai",
                "side_effects": "warmup/post-warmup branch selection during optimizer.step",
            },
            "resume_translation_scope": {
                "spec_ready": True,
                "source": "serialized non-state_dict optimizer attribute has_warmup",
                "target": "optimizer.has_warmup restored after base load_state_dict",
                "tensor_policy": "no tensor sidecar; bool attribute must keep exact Python bool value",
            },
            "quality_safety_guard": {
                "spec_ready": True,
                "guards": [
                    "warmup_phase_resume_parity",
                    "post_warmup_resume_parity",
                    "bool_attribute_type_guard",
                    "scheduler_and_lr_contract_unchanged",
                ],
            },
            "native_kernel_preconditions": {
                "spec_ready": True,
                "required_before_kernel": [
                    "has_warmup_attr_adapter_implemented",
                    "warmup_branch_native_formula_spec_ready",
                    "post_warmup_branch_native_formula_spec_ready",
                    "phase_transition_resume_parity_passed",
                ],
            },
        },
        "spam": {
            "param_ownership": {
                "spec_ready": True,
                "owner": "optimizer sparse mask state plus normal upstream state_dict state",
                "identity": "parameter_state_sparse_mask",
                "bridge_payload": "lulynx_sparse_masks",
                "mutation_boundary": "sparse bool masks gate which parameter elements receive updates",
            },
            "state_adapter_role": {
                "spec_ready": True,
                "role": "restore bool sparse masks and preserve deterministic density policy",
                "python_bridge": "optimizer_plugin_bridge state-attribute restore for spam",
                "side_effects": "mask density and bool tensor placement affect update coverage",
            },
            "resume_translation_scope": {
                "spec_ready": True,
                "source": "saved sparse mask entries from optimizer state/bridge sidecar",
                "target": "optimizer sparse mask tensors restored after base load_state_dict",
                "tensor_policy": "preserve bool dtype, shape, device, and density contract",
            },
            "quality_safety_guard": {
                "spec_ready": True,
                "guards": [
                    "bool_mask_dtype_guard",
                    "mask_shape_matches_param",
                    "density_contract_resume_parity",
                    "masked_update_coverage_parity",
                ],
            },
            "native_kernel_preconditions": {
                "spec_ready": True,
                "required_before_kernel": [
                    "sparse_mask_adapter_implemented",
                    "mask_shape_density_abi_locked",
                    "masked_update_native_formula_spec_ready",
                    "mask_resume_parity_matrix_passed",
                ],
            },
        },
    }
    return {**common, **spec_by_name.get(name, _generic_state_adapter_abi_spec())}


def _adapter_resume_matrix_artifact(name: str, spec_ready: bool) -> dict[str, Any]:
    common_resume_cases = [
        "state_dict_roundtrip_before_step",
        "state_dict_roundtrip_after_step",
        "resume_next_step_matches_plugin_reference",
    ]
    common_translation_cases = [
        "ordered_param_identity_preserved",
        "bridge_payload_serialized",
        "bridge_payload_restored_after_base_load_state_dict",
    ]
    cases_by_name = {
        "demo": {
            "resume_replay_cases": common_resume_cases
            + ["demo_state_sidecar_roundtrip", "no_process_group_identity_gather_resume"],
            "translation_cases": common_translation_cases
            + ["lulynx_demo_state_sidecar_to_optimizer_demo_state"],
            "quality_guard_cases": [
                "ordered_param_count_match",
                "sidecar_tensor_device_dtype_clone",
                "distributed_collective_policy_default_off",
            ],
        },
        "sgdsai": {
            "resume_replay_cases": common_resume_cases
            + ["warmup_phase_flag_roundtrip", "post_warmup_branch_resume"],
            "translation_cases": common_translation_cases + ["has_warmup_bool_attr_restore"],
            "quality_guard_cases": [
                "bool_attribute_type_guard",
                "warmup_phase_resume_parity",
                "scheduler_and_lr_contract_unchanged",
            ],
        },
        "spam": {
            "resume_replay_cases": common_resume_cases
            + ["sparse_mask_state_roundtrip", "masked_update_coverage_resume"],
            "translation_cases": common_translation_cases + ["lulynx_sparse_masks_to_optimizer_mask_state"],
            "quality_guard_cases": [
                "bool_mask_dtype_guard",
                "mask_shape_matches_param",
                "density_contract_resume_parity",
            ],
        },
    }
    matrix = cases_by_name.get(
        name,
        {
            "resume_replay_cases": common_resume_cases,
            "translation_cases": common_translation_cases,
            "quality_guard_cases": ["optimizer_specific_state_adapter_review"],
        },
    )
    return {
        "schema_version": 1,
        "artifact_kind": "selected_plugin_state_adapter_special_resume_matrix",
        "report_only": True,
        "selected_optimizer_name": name,
        "spec_ready": spec_ready,
        "implementation_ready": False,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "resume_replay_cases": matrix["resume_replay_cases"],
        "translation_cases": matrix["translation_cases"],
        "quality_guard_cases": matrix["quality_guard_cases"],
        "blocked_until": [
            "special_state_adapter_resume_matrix_implemented",
            "state_translation_roundtrip_parity_passed",
            "owner_release_hold",
        ],
        "evidence_status": "planned_report_only",
    }


def _generic_state_adapter_abi_spec() -> dict[str, Any]:
    return {
        "param_ownership": {"spec_ready": False, "owner": "unknown"},
        "state_adapter_role": {"spec_ready": False, "role": "unknown"},
        "resume_translation_scope": {"spec_ready": False, "source": "unknown", "target": "unknown"},
        "quality_safety_guard": {"spec_ready": False, "guards": []},
        "native_kernel_preconditions": {"spec_ready": False, "required_before_kernel": []},
    }


def _compact_execution_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "execution_matrix_ready": report.get("execution_matrix_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "adapter_abi_implementation_ready_count": int(
            summary.get("adapter_abi_implementation_ready_count", 0) or 0
        ),
        "adapter_resume_matrix_implementation_ready_count": int(
            summary.get("adapter_resume_matrix_implementation_ready_count", 0) or 0
        ),
        "adapter_resume_replay_case_implementation_ready_count": int(
            summary.get("adapter_resume_replay_case_implementation_ready_count", 0) or 0
        ),
        "adapter_resume_translation_case_implementation_ready_count": int(
            summary.get("adapter_resume_translation_case_implementation_ready_count", 0) or 0
        ),
        "native_kernel_precondition_implementation_ready_count": int(
            summary.get("native_kernel_precondition_implementation_ready_count", 0) or 0
        ),
        "execution_failed_count": int(summary.get("execution_failed_count", 0) or 0),
        "unsafe_claim_count": int(summary.get("unsafe_claim_count", 0) or 0),
    }


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    counts = _as_dict(summary.get("route_family_counts"))
    return {
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "selector_boundary_ready": report.get("selector_boundary_ready") is True,
        "all_discovered_plugins_resume_proven": report.get("all_discovered_plugins_resume_proven") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "state_adapter_special_count": int(counts.get(STATE_ADAPTER_SPECIAL_ROUTE_FAMILY, 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "runtime_dispatch_ready": report.get("runtime_dispatch_ready") is True,
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
                out.append(f"unsafe_plugin_state_adapter_special_source:{scorecard}:{field}")
    for row in rows:
        selected = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                out.append(f"unsafe_plugin_state_adapter_special_row:{selected}:{field}")
        if row.get("can_reuse_exact_adamw_native_dispatch") is True:
            out.append(f"unsafe_plugin_state_adapter_special_row:{selected}:adamw_dispatch_reuse")
        if row.get("adamw_kernel_compatible") is True or row.get("simple_formula_kernel_compatible") is True:
            out.append(f"unsafe_plugin_state_adapter_special_row:{selected}:kernel_compatible")
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
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [f"builder_failed:{builder_name}"],
    }


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_state_adapter_special_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _spec_section_ready_count(rows: list[Mapping[str, Any]], section: str) -> int:
    return sum(
        1
        for row in rows
        if _as_dict(_as_dict(row.get("state_adapter_abi_spec")).get(section)).get("spec_ready") is True
    )


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "STATE_ADAPTER_SPECIAL_ROUTE_FAMILY",
    "build_plugin_state_adapter_special_family_batch_scorecard",
]
