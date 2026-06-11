"""Report-only batch scorecard for selected plugin model/shape-aware routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.lulynx_trainer.optimizer_plugin_support import (
    PLUGIN_MUON_FAMILY_OPTIMIZERS,
    PLUGIN_SPECIAL_HANDLING,
)
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard
from core.turbocore_plugin_model_shape_aware_execution_matrix import (
    build_model_shape_aware_execution_matrix,
    promote_model_shape_aware_row,
)


TARGET_PLUGIN_OPTIMIZERS: tuple[str, ...] = tuple(
    sorted({"adammini", "alice", "distributedmuon", "spectralsphere"} | set(PLUGIN_MUON_FAMILY_OPTIMIZERS))
)
UNSAFE_TRUE_FIELDS = (
    "training_path_enabled",
    "default_behavior_changed",
    "runtime_dispatch_ready",
    "native_dispatch_allowed",
    "native_kernel_ready",
    "product_native_ready",
    "product_native_dispatch_ready",
)


def build_plugin_model_shape_aware_family_batch_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate selected model/shape-aware plugin status without enabling dispatch."""

    selector = _as_dict(selector_report) if selector_report is not None else _call_selector()
    selector_rows = _selector_rows(selector)
    rows = [_row(name, selector_rows.get(name, {})) for name in TARGET_PLUGIN_OPTIMIZERS]
    execution_matrix = build_model_shape_aware_execution_matrix(rows)
    rows = [promote_model_shape_aware_row(row, execution_matrix) for row in rows]
    missing = [row["selected_optimizer_name"] for row in rows if row["selector_classified"] is not True]
    unsafe = _unsafe_claims({"selector": selector}, rows)
    ready = selector.get("ok") is True and execution_matrix.get("ok") is True and not missing and not unsafe
    selector_count = int(_as_dict(_summary(selector).get("route_family_counts")).get("model_or_shape_aware", 0) or 0)

    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_model_shape_aware_family_batch_scorecard_v0",
        "gate": "plugin_model_shape_aware_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_model_shape_aware_family_batch_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": "model_or_shape_aware",
        "request_contract": {
            "optimizer_type": OptimizerType.PYTORCH_OPTIMIZER.value,
            "selected_optimizer_source": "optimizer_args.name",
            "selected_optimizer_names": list(TARGET_PLUGIN_OPTIMIZERS),
            "runtime_authority": "existing_python_or_third_party_plugin_optimizer",
            "native_route_policy": "blocked_until_explicit_model_shape_param_group_abi",
        },
        "native_compatibility": {
            "adamw_native_simple_kernel_reusable": False,
            "simple_formula_kernel_reusable": False,
            "requires_model_structure_or_shape_contract": True,
            "requires_param_group_semantics_contract": True,
            "param_group_abi_spec_ready": ready,
            "param_group_abi_implementation_ready": execution_matrix.get("ok") is True,
            "exact_adamw_product_native_route_count_delta": 0,
        },
        "selector_scorecard": _compact_selector(selector),
        "execution_matrix": _compact_execution_matrix(execution_matrix),
        "rows": rows,
        "summary": {
            "selected_optimizer_count": len(rows),
            "selector_model_or_shape_aware_count": selector_count,
            "model_structure_dependent_count": sum(
                1 for row in rows if row["dependency_contract"]["requires_model_structure"] is True
            ),
            "parameter_shape_dependent_count": sum(
                1 for row in rows if row["dependency_contract"]["requires_parameter_shapes"] is True
            ),
            "param_group_semantics_dependent_count": sum(
                1 for row in rows if row["dependency_contract"]["requires_param_group_semantics"] is True
            ),
            "param_group_abi_spec_ready_count": sum(
                1 for row in rows if row["param_group_abi_spec_ready"] is True
            ),
            "param_group_abi_implementation_ready_count": int(
                _summary(execution_matrix).get("param_group_abi_implementation_ready_count", 0) or 0
            ),
            "param_group_resume_replay_matrix_artifact_ready_count": sum(
                1 for row in rows if row["param_group_resume_replay_matrix_artifact"]["artifact_ready"] is True
            ),
            "param_group_resume_replay_matrix_row_count": sum(
                len(row["param_group_resume_replay_matrix_artifact"]["rows"]) for row in rows
            ),
            "param_group_resume_replay_matrix_implementation_ready_count": int(
                _summary(execution_matrix).get(
                    "param_group_resume_replay_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "param_group_resume_replay_row_implementation_ready_count": int(
                _summary(execution_matrix).get("param_group_resume_replay_row_implementation_ready_count", 0) or 0
            ),
            "model_structure_contract_count": sum(
                1 for row in rows if row["param_group_abi_contract"]["model_structure_binding"] != "none"
            ),
            "shape_partition_contract_count": sum(
                1 for row in rows if row["param_group_abi_contract"]["shape_partition_policy"] != "none"
            ),
            "distributed_collective_contract_count": sum(
                1 for row in rows if row["param_group_abi_contract"]["distributed_collective_policy"] != "none"
            ),
            "selected_plugin_native_ready_count": 0,
            "product_native_ready_count": 0,
            "product_native_dispatch_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "exact_adamw_product_native_route_count_delta": 0,
            "missing_selector_count": len(missing),
            "unsafe_claim_count": len(unsafe),
        },
        "promotion_blockers": _dedupe(
            unsafe
            + [f"selector_model_or_shape_aware_missing:{name}" for name in missing]
            + [
                "adamw_or_simple_native_kernel_not_reusable",
                "native_kernel_implementation_missing",
                "runtime_dispatch_shadow_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(unsafe + [f"selector_model_or_shape_aware_missing:{name}" for name in missing]),
        "recommended_next_step": (
            "owner/release hold for implementation-ready model/shape-aware param-group ABI matrices with dispatch default-off"
            if ready
            else "fix selector or unsafe native-readiness claims for model/shape-aware plugins"
        ),
        "notes": [
            "This batch is report-only and never enables native dispatch.",
            "Rows depend on model structure, parameter shape, layer/name hierarchy, or proprietary param-group semantics.",
            "These selected plugins cannot reuse the exact AdamW or native simple-formula kernel contract.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _call_selector() -> dict[str, Any]:
    try:
        return dict(build_plugin_optimizer_selector_scorecard())
    except Exception as exc:
        return {
            "schema_version": 1,
            "scorecard": "build_plugin_optimizer_selector_scorecard",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
            "default_behavior_changed": False,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "blocked_reasons": ["builder_failed:build_plugin_optimizer_selector_scorecard"],
        }


def _row(name: str, selector_row: Mapping[str, Any]) -> dict[str, Any]:
    classified = selector_row.get("native_route_family") == "model_or_shape_aware"
    resume_proven = selector_row.get("resume_proven") is True
    contract = _dependency_contract(name)
    abi_spec_ready = classified and resume_proven
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selector": OptimizerType.PYTORCH_OPTIMIZER.value,
        "native_route_family": "model_or_shape_aware",
        "model_shape_aware_family": _family(name),
        "batch_status": "report_only_contract_ready" if classified and resume_proven else "selector_or_resume_pending",
        "selector_classified": classified,
        "resume_proven": resume_proven,
        "special_handling": str(selector_row.get("special_handling") or PLUGIN_SPECIAL_HANDLING.get(name, "")),
        "dependency_contract": contract,
        "param_group_abi_spec_ready": abi_spec_ready,
        "param_group_abi_implementation_ready": False,
        "param_group_abi_contract": _param_group_abi_contract(name, contract),
        "param_group_resume_replay_matrix_artifact": _param_group_resume_replay_matrix_artifact(
            name,
            contract,
            abi_spec_ready,
        ),
        "adamw_native_simple_kernel_compatible": False,
        "native_simple_kernel_reusable": False,
        "runtime_authority": "existing_pytorch_optimizer_plugin",
        "native_route": "none_report_only",
        "plugin_selected_native_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": _next_gate(name),
        "blocked_reasons": [
            "selected_plugin_model_shape_aware_param_group_abi_implementation_missing",
            "adamw_or_simple_formula_kernel_not_reusable",
            "native_dispatch_gate_not_requested",
        ],
    }


def _param_group_abi_contract(name: str, dependency: Mapping[str, bool]) -> dict[str, str | int]:
    return {
        "schema_version": 1,
        "model_structure_binding": (
            "named_module_or_parameter_hierarchy"
            if dependency["requires_model_structure"] or dependency["requires_layer_or_name_hierarchy"]
            else "none"
        ),
        "shape_partition_policy": (
            _shape_partition_policy(name) if dependency["requires_parameter_shapes"] else "none"
        ),
        "param_group_semantics_policy": (
            _param_group_semantics_policy(name) if dependency["requires_param_group_semantics"] else "none"
        ),
        "distributed_collective_policy": (
            "explicit_process_group_and_rank_partition"
            if dependency["requires_distributed_collective_semantics"]
            else "none"
        ),
        "state_resume_scope": _state_resume_scope(name),
        "native_kernel_precondition": "param_group_abi_and_batch_parity_ready",
    }


def _param_group_resume_replay_matrix_artifact(
    name: str,
    dependency: Mapping[str, bool],
    spec_ready: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "model_shape_param_group_resume_replay_matrix_v0",
        "artifact_ready": spec_ready,
        "artifact_status": "planned_report_only" if spec_ready else "selector_or_resume_pending",
        "implementation_ready": False,
        "rows": _param_group_resume_replay_rows(name, dependency),
    }


def _param_group_resume_replay_rows(name: str, dependency: Mapping[str, bool]) -> list[dict[str, Any]]:
    rows = [
        _matrix_row(
            "param_group_inventory_resume",
            "param_group_identity",
            ["group_index", "param_names_or_shape_keys", "group_hparams"],
            ["group_identity_stable_after_resume", "native_flattening_still_blocked"],
        ),
        _matrix_row(
            "state_dict_roundtrip_resume",
            _state_resume_scope(name),
            ["optimizer_state_dict", "param_group_hparams", "shape_partition_signature"],
            ["state_scope_roundtrips", "next_step_reference_replay_required"],
        ),
        _matrix_row(
            "blocked_native_reuse_review",
            "owner_release_hold",
            ["adamw_kernel_incompatibility_reason", "simple_kernel_incompatibility_reason"],
            ["dispatch_remains_default_off", "native_ready_not_claimed"],
        ),
    ]
    if dependency["requires_model_structure"] or dependency["requires_layer_or_name_hierarchy"]:
        rows.append(
            _matrix_row(
                "model_hierarchy_replay",
                "named_module_or_parameter_hierarchy",
                ["module_path", "parameter_name", "group_assignment"],
                ["hierarchy_binding_rebuilt_after_resume", "missing_names_fail_closed"],
            )
        )
    if dependency["requires_parameter_shapes"]:
        rows.append(
            _matrix_row(
                "shape_partition_replay",
                _shape_partition_policy(name),
                ["shape_tuple", "rank_or_matrix_partition", "group_assignment"],
                ["shape_grouping_rebuilt_after_resume", "shape_mismatch_fails_closed"],
            )
        )
    if dependency["requires_distributed_collective_semantics"]:
        rows.append(
            _matrix_row(
                "distributed_collective_replay",
                "explicit_process_group_and_rank_partition",
                ["world_size", "rank", "process_group_id", "partition_signature"],
                ["collective_metadata_roundtrips", "rank_mismatch_fails_closed"],
            )
        )
    return rows


def _matrix_row(
    case_id: str,
    replay_boundary: str,
    required_payload: list[str],
    replay_assertions: list[str],
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


def _shape_partition_policy(name: str) -> str:
    if name == "alice":
        return "low_rank_basis_shape_split"
    if name == "spectralsphere":
        return "spectral_shape_projection_split"
    if "muon" in name:
        return "matrix_shape_muon_grouping"
    return "optimizer_specific_shape_grouping"


def _param_group_semantics_policy(name: str) -> str:
    if name == "adammini":
        return "layer_name_grouped_hyperparameters"
    if name == "distributedmuon":
        return "muon_grouped_hyperparameters_with_collectives"
    if "muon" in name:
        return "muon_grouped_hyperparameters"
    return "optimizer_specific_param_groups"


def _state_resume_scope(name: str) -> str:
    if name == "adammini":
        return "named_group_state_and_layer_metadata"
    if name == "distributedmuon":
        return "shape_group_state_with_collective_metadata"
    if "muon" in name:
        return "muon_shape_group_state"
    return "shape_aware_optimizer_state"


def _dependency_contract(name: str) -> dict[str, bool]:
    if name == "adammini":
        return {
            "requires_model_structure": True,
            "requires_parameter_shapes": False,
            "requires_layer_or_name_hierarchy": True,
            "requires_param_group_semantics": True,
            "requires_distributed_collective_semantics": False,
        }
    if name == "alice":
        return {
            "requires_model_structure": False,
            "requires_parameter_shapes": True,
            "requires_layer_or_name_hierarchy": False,
            "requires_param_group_semantics": True,
            "requires_distributed_collective_semantics": False,
        }
    if name == "spectralsphere":
        return {
            "requires_model_structure": False,
            "requires_parameter_shapes": True,
            "requires_layer_or_name_hierarchy": False,
            "requires_param_group_semantics": True,
            "requires_distributed_collective_semantics": False,
        }
    if name == "distributedmuon":
        return {
            "requires_model_structure": False,
            "requires_parameter_shapes": True,
            "requires_layer_or_name_hierarchy": False,
            "requires_param_group_semantics": True,
            "requires_distributed_collective_semantics": True,
        }
    return {
        "requires_model_structure": False,
        "requires_parameter_shapes": True,
        "requires_layer_or_name_hierarchy": False,
        "requires_param_group_semantics": True,
        "requires_distributed_collective_semantics": False,
    }


def _family(name: str) -> str:
    if name == "adammini":
        return "model_named_parameter_grouping"
    if name == "alice":
        return "shape_split_low_rank_basis"
    if name == "spectralsphere":
        return "shape_split_spectral_fallback"
    if name == "distributedmuon":
        return "distributed_muon_shape_grouping"
    return "muon_shape_grouping"


def _next_gate(name: str) -> str:
    if name == "adammini":
        return "selected_plugin_adammini_model_grouping_abi_review"
    if name == "distributedmuon":
        return "selected_plugin_distributedmuon_shape_grouping_collective_abi_review"
    if name in PLUGIN_MUON_FAMILY_OPTIMIZERS:
        return "selected_plugin_muon_shape_grouping_param_group_abi_review"
    return f"selected_plugin_{name}_shape_aware_param_group_abi_review"


def _selector_rows(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("rows", []) if isinstance(report, Mapping) else []
    return {
        str(row.get("optimizer_name", "")).strip().lower(): row
        for row in rows
        if isinstance(row, Mapping)
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
        "model_or_shape_aware_count": int(counts.get("model_or_shape_aware", 0) or 0),
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
        "param_group_abi_implementation_ready_count": int(
            summary.get("param_group_abi_implementation_ready_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_implementation_ready_count": int(
            summary.get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "param_group_resume_replay_row_implementation_ready_count": int(
            summary.get("param_group_resume_replay_row_implementation_ready_count", 0) or 0
        ),
        "execution_failed_count": int(summary.get("execution_failed_count", 0) or 0),
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
                out.append(f"unsafe_plugin_model_shape_source:{scorecard}:{field}")
    for row in rows:
        selected = str(row.get("selected_optimizer_name", "unknown"))
        for field in UNSAFE_TRUE_FIELDS:
            if row.get(field) is True:
                out.append(f"unsafe_plugin_model_shape_row:{selected}:{field}")
    return _dedupe(out)


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = Path(__file__).resolve().parents[2] / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_model_shape_aware_family_batch_scorecard.json"
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


__all__ = ["TARGET_PLUGIN_OPTIMIZERS", "build_plugin_model_shape_aware_family_batch_scorecard"]
