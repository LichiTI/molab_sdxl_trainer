"""Report-only batch scorecard for selected plugin adaptive-LR optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_adaptive_lr_state_machine_batch_scorecard import (
    build_adaptive_lr_state_machine_batch_scorecard,
)
from core.turbocore_plugin_adaptivelr_execution_matrix import (
    build_adaptivelr_execution_matrix,
    promote_adaptivelr_row,
)
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_PLUGIN_OPTIMIZERS: tuple[str, ...] = (
    "dadaptadagrad",
    "dadaptadam",
    "dadaptadan",
    "dadaptlion",
    "dadaptsgd",
    "prodigy",
)

_BUILTIN_REFERENCE_BY_PLUGIN = {
    "dadaptadagrad": OptimizerType.DADAPT_ADAGRAD,
    "dadaptadam": OptimizerType.DADAPT_ADAM,
    "dadaptadan": OptimizerType.DADAPT_ADAN,
    "dadaptlion": OptimizerType.DADAPT_LION,
    "dadaptsgd": OptimizerType.DADAPT_SGD,
    "prodigy": OptimizerType.PRODIGY,
}


def build_plugin_adaptivelr_family_batch_scorecard(*, write_artifact: bool = False) -> dict[str, Any]:
    """Aggregate selected plugin adaptive-LR state-machine evidence without dispatch."""

    selector = build_plugin_optimizer_selector_scorecard()
    adaptive = build_adaptive_lr_state_machine_batch_scorecard()
    rows = [_row(name, selector, adaptive) for name in TARGET_PLUGIN_OPTIMIZERS]
    execution_matrix = build_adaptivelr_execution_matrix(rows)
    rows = [promote_adaptivelr_row(row, execution_matrix) for row in rows]
    unsafe = _unsafe_claims(selector, adaptive)
    not_ready = [
        row
        for row in rows
        if row["selected_state_machine_reference_ready"] is not True
        or row["selected_state_machine_abi_spec_ready"] is not True
    ]
    ready = (
        selector.get("ok") is True
        and adaptive.get("ok") is True
        and execution_matrix.get("ok") is True
        and not not_ready
        and not unsafe
    )
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_adaptivelr_family_batch_scorecard_v0",
        "gate": "plugin_adaptivelr_selected_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "selected_adaptivelr_family_batch_ready": ready,
        "report_only": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "plugin_selected_native_ready_count": 0,
        "selected_optimizer_family": "adaptive_lr_state_machine",
        "selector_scorecard": _compact_selector(selector),
        "adaptive_lr_reference_batch": _compact_adaptive_reference(adaptive),
        "selected_state_machine_abi": _compact_selected_state_machine_abi(rows),
        "selected_state_machine_replay_matrix": _compact_state_machine_replay_matrix(rows),
        "execution_matrix": _compact_execution_matrix(execution_matrix),
        "rows": rows,
        "summary": {
            "selected_optimizer_count": len(rows),
            "selected_adaptivelr_family_batch_ready": ready,
            "selector_adaptive_lr_state_machine_count": int(
                _as_dict(_summary(selector).get("route_family_counts")).get("adaptive_lr_state_machine", 0) or 0
            ),
            "selected_state_machine_reference_ready_count": sum(
                1 for row in rows if row["selected_state_machine_reference_ready"] is True
            ),
            "selected_state_machine_abi_spec_ready_count": sum(
                1 for row in rows if row["selected_state_machine_abi_spec_ready"] is True
            ),
            "selected_dynamic_lr_scalar_state_spec_ready_count": _count_selected_spec(
                rows, "dynamic_lr_scalar_state"
            ),
            "selected_d_estimator_global_state_spec_ready_count": _count_selected_spec(
                rows, "d_estimator_global_state"
            ),
            "selected_per_step_quality_guard_spec_ready_count": _count_selected_spec(
                rows, "per_step_quality_guard"
            ),
            "selected_resume_scope_spec_ready_count": _count_selected_spec(rows, "resume_scope"),
            "selected_native_kernel_preconditions_spec_ready_count": _count_selected_spec(
                rows, "native_kernel_preconditions"
            ),
            "selected_state_machine_replay_matrix_artifact_ready_count": sum(
                1 for row in rows if row["state_machine_replay_matrix_artifact"]["spec_ready"] is True
            ),
            "selected_state_machine_replay_matrix_implementation_ready_count": int(
                _summary(execution_matrix).get("state_machine_replay_matrix_implementation_ready_count", 0) or 0
            ),
            "selected_state_machine_replay_case_planned_count": sum(
                len(row["state_machine_replay_matrix_artifact"]["replay_cases"]) for row in rows
            ),
            "selected_state_machine_replay_resume_case_planned_count": sum(
                len(row["state_machine_replay_matrix_artifact"]["resume_replay_cases"]) for row in rows
            ),
            "selected_state_machine_replay_case_implementation_ready_count": int(
                _summary(execution_matrix).get("state_machine_replay_case_implementation_ready_count", 0) or 0
            ),
            "selected_state_machine_replay_resume_case_implementation_ready_count": int(
                _summary(execution_matrix).get(
                    "state_machine_replay_resume_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "selected_state_machine_abi_implementation_ready_count": int(
                _summary(execution_matrix).get("state_machine_abi_implementation_ready_count", 0) or 0
            ),
            "selected_native_kernel_preconditions_implementation_ready_count": int(
                _summary(execution_matrix).get("native_kernel_preconditions_implementation_ready_count", 0) or 0
            ),
            "prodigy_reference_count": sum(1 for row in rows if row["adaptive_lr_family"] == "adaptive_lr_prodigy"),
            "dadapt_reference_count": sum(1 for row in rows if row["adaptive_lr_family"] == "adaptive_lr_dadapt"),
            "native_ready_count": 0,
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "default_behavior_changed_count": 0,
            "exact_adamw_product_native_route_count_delta": 0,
        },
        "promotion_blockers": _dedupe(
            unsafe
            + [f"plugin_adaptivelr_reference_pending:{row['selected_optimizer_name']}" for row in not_ready]
            + [
                "native_kernel_implementation_missing",
                "selected_plugin_adaptivelr_runtime_dispatch_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": _dedupe(
            unsafe + [f"plugin_adaptivelr_reference_pending:{row['selected_optimizer_name']}" for row in not_ready]
        ),
        "recommended_next_step": (
            "owner/release hold for implementation-ready adaptive-LR state-machine matrices with dispatch default-off"
            if ready
            else "fix selected plugin adaptive-LR reference blockers"
        ),
        "notes": [
            "This batch maps selected pytorch_optimizer adaptive-LR plugins onto existing reference-only state models.",
            "It does not add product native-ready optimizers or change the exact AdamW dispatch count.",
            "Native kernel, runtime dispatch, and default behavior remain disabled.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _row(name: str, selector: Mapping[str, Any], adaptive: Mapping[str, Any]) -> dict[str, Any]:
    selector_row = _selector_rows(selector).get(name, {})
    reference_optimizer = _BUILTIN_REFERENCE_BY_PLUGIN[name]
    reference_row = _adaptive_rows(adaptive).get(reference_optimizer.value, {})
    classified = selector_row.get("native_route_family") == "adaptive_lr_state_machine"
    reference_ready = (
        classified
        and selector_row.get("resume_proven") is True
        and reference_row.get("state_machine_reference_ready") is True
        and reference_row.get("native_ready") is not True
    )
    abi_spec = _selected_abi_spec(name, reference_row)
    abi_spec_ready = reference_ready and _abi_spec_ready(abi_spec)
    family = _family(name)
    return {
        "schema_version": 1,
        "selected_optimizer_name": name,
        "selector": OptimizerType.PYTORCH_OPTIMIZER.value,
        "native_route_family": "adaptive_lr_state_machine",
        "adaptive_lr_family": family,
        "selector_classified": classified,
        "resume_proven": selector_row.get("resume_proven") is True,
        "builtin_reference_optimizer_type": reference_optimizer.value,
        "state_machine_status": "abi_spec_ready_report_only" if abi_spec_ready else "reference_pending",
        "selected_state_machine_reference_ready": reference_ready,
        "selected_state_machine_abi_spec_ready": abi_spec_ready,
        "state_machine_abi_implementation_ready": False,
        "native_kernel_preconditions_spec_ready": abi_spec_ready,
        "native_kernel_preconditions_implementation_ready": False,
        "batch_reference_ready": reference_ready,
        "state_model": _state_model(family),
        "state_machine_abi_spec": abi_spec,
        "state_machine_replay_matrix_artifact": _state_machine_replay_matrix_artifact(name, family, abi_spec_ready),
        "runtime_authority": "existing_pytorch_optimizer_plugin",
        "native_route": "none_report_only",
        "plugin_selected_native_ready": False,
        "product_native_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "next_gate": _next_gate(family),
        "blocked_reasons": [
            "selected_plugin_adaptivelr_native_state_machine_abi_implementation_missing",
            "selected_plugin_adaptivelr_batch_resume_parity_not_validated",
            "native_dispatch_gate_not_requested",
        ],
    }


def _family(name: str) -> str:
    if name == "prodigy":
        return "adaptive_lr_prodigy"
    return "adaptive_lr_dadapt"


def _next_gate(family: str) -> str:
    if family == "adaptive_lr_prodigy":
        return "selected_plugin_prodigy_native_state_machine_abi_review"
    return "selected_plugin_dadapt_native_state_machine_abi_review"


def _state_model(family: str) -> dict[str, Any]:
    if family == "adaptive_lr_prodigy":
        return {
            "requires_dynamic_lr_estimate": True,
            "requires_global_distance_state": True,
            "requires_schedule_free_mode_review": False,
            "adamw_kernel_compatible": False,
        }
    return {
        "requires_dynamic_lr_estimate": True,
        "requires_global_d_adaptation_state": True,
        "requires_variant_specific_accumulators": True,
        "adamw_kernel_compatible": False,
    }


def _selected_abi_spec(name: str, reference_row: Mapping[str, Any]) -> dict[str, Any]:
    spec = _as_dict(reference_row.get("state_machine_abi_spec"))
    if not spec:
        return {
            "schema_version": 1,
            "report_only": True,
            "selected_optimizer_name": name,
            "source_reference_optimizer_type": reference_row.get("optimizer_type"),
            "spec_ready": False,
            "blocked_reasons": ["adaptive_lr_reference_state_machine_abi_spec_missing"],
        }
    selected = dict(spec)
    selected.update(
        {
            "selected_optimizer_name": name,
            "source_reference_optimizer_type": reference_row.get("optimizer_type"),
            "spec_ready": _abi_spec_ready(spec),
            "implementation_ready": False,
            "plugin_binding": {
                "spec_ready": True,
                "implementation_ready": False,
                "optimizer_selector": OptimizerType.PYTORCH_OPTIMIZER.value,
                "plugin_name": name,
                "runtime_authority": "existing_pytorch_optimizer_plugin",
                "native_runtime_authority": "none_report_only",
            },
        }
    )
    return selected


def _state_machine_replay_matrix_artifact(name: str, family: str, spec_ready: bool) -> dict[str, Any]:
    replay_cases = [
        "dynamic_lr_scalar_recomputed_from_saved_state",
        "d_estimator_global_state_replayed_before_param_update",
        "per_step_quality_guard_replay_blocks_bad_d_estimate",
        "lr_scalar_materialized_before_native_boundary",
    ]
    resume_cases = [
        "state_dict_roundtrip_before_step",
        "state_dict_roundtrip_after_step",
        "resume_next_step_matches_python_reference",
    ]
    if family == "adaptive_lr_prodigy":
        replay_cases.extend(["prodigy_global_distance_state_replay", "prodigy_growth_guard_replay"])
        resume_cases.append("prodigy_distance_buffer_resume")
    else:
        replay_cases.extend(["dadapt_variant_accumulator_replay", "dadapt_growth_clip_guard_replay"])
        resume_cases.append("dadapt_variant_accumulator_resume")
    return {
        "schema_version": 1,
        "artifact_kind": "selected_plugin_adaptivelr_state_machine_replay_matrix",
        "report_only": True,
        "selected_optimizer_name": name,
        "adaptive_lr_family": family,
        "spec_ready": spec_ready,
        "implementation_ready": False,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "replay_cases": replay_cases,
        "resume_replay_cases": resume_cases,
        "blocked_until": [
            "state_machine_replay_matrix_implemented",
            "resume_next_step_parity_passed",
            "owner_release_hold",
        ],
        "evidence_status": "planned_report_only",
    }


def _abi_spec_ready(spec: Mapping[str, Any]) -> bool:
    required = (
        "dynamic_lr_scalar_state",
        "d_estimator_global_state",
        "per_step_quality_guard",
        "resume_scope",
        "native_kernel_preconditions",
    )
    return all(_as_dict(spec.get(key)).get("spec_ready") is True for key in required)


def _compact_selected_state_machine_abi(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "report_only": True,
        "selected_optimizer_count": len(rows),
        "selected_state_machine_abi_spec_ready_count": sum(
            1 for row in rows if row.get("selected_state_machine_abi_spec_ready") is True
        ),
        "selected_state_machine_abi_implementation_ready_count": sum(
            1 for row in rows if row.get("state_machine_abi_implementation_ready") is True
        ),
        "selected_native_kernel_preconditions_implementation_ready_count": sum(
            1 for row in rows if row.get("native_kernel_preconditions_implementation_ready") is True
        ),
        "runtime_dispatch_ready_count": 0,
        "native_dispatch_allowed_count": 0,
        "training_path_enabled_count": 0,
    }


def _compact_state_machine_replay_matrix(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "report_only": True,
        "selected_optimizer_count": len(rows),
        "artifact_ready_count": sum(
            1 for row in rows if _as_dict(row.get("state_machine_replay_matrix_artifact")).get("spec_ready") is True
        ),
        "implementation_ready_count": sum(
            1
            for row in rows
            if _as_dict(row.get("state_machine_replay_matrix_artifact")).get("implementation_ready") is True
        ),
        "replay_case_planned_count": sum(
            len(_as_dict(row.get("state_machine_replay_matrix_artifact")).get("replay_cases", [])) for row in rows
        ),
        "resume_replay_case_planned_count": sum(
            len(_as_dict(row.get("state_machine_replay_matrix_artifact")).get("resume_replay_cases", []))
            for row in rows
        ),
        "replay_case_implementation_ready_count": sum(
            len(_as_dict(row.get("state_machine_replay_matrix_artifact")).get("replay_case_status", {}))
            for row in rows
        ),
        "resume_replay_case_implementation_ready_count": sum(
            len(_as_dict(row.get("state_machine_replay_matrix_artifact")).get("resume_replay_case_status", {}))
            for row in rows
        ),
        "training_path_enabled_count": 0,
        "runtime_dispatch_ready_count": 0,
        "native_dispatch_allowed_count": 0,
    }


def _compact_execution_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "execution_matrix_ready": report.get("execution_matrix_ready") is True,
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "state_machine_replay_matrix_implementation_ready_count": int(
            summary.get("state_machine_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "state_machine_replay_case_implementation_ready_count": int(
            summary.get("state_machine_replay_case_implementation_ready_count", 0) or 0
        ),
        "state_machine_replay_resume_case_implementation_ready_count": int(
            summary.get("state_machine_replay_resume_case_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "execution_failed_count": int(summary.get("execution_failed_count", 0) or 0),
        "unsafe_claim_count": int(summary.get("unsafe_claim_count", 0) or 0),
    }


def _count_selected_spec(rows: list[Mapping[str, Any]], key: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("selected_state_machine_abi_spec_ready") is True
        and _as_dict(row.get("state_machine_abi_spec")).get(key, {}).get("spec_ready") is True
    )


def _selector_rows(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_name", "")).strip().lower(): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }


def _adaptive_rows(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("optimizer_type", "")).strip(): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }


def _compact_selector(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "plugin_selector_classification_ready": report.get("plugin_selector_classification_ready") is True,
        "all_discovered_plugins_resume_proven": report.get("all_discovered_plugins_resume_proven") is True,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "adaptive_lr_state_machine_count": int(
            _as_dict(summary.get("route_family_counts")).get("adaptive_lr_state_machine", 0) or 0
        ),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
    }


def _compact_adaptive_reference(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "ok": report.get("ok") is True,
        "report_only": report.get("report_only") is True,
        "state_machine_abi_spec_ready": report.get("state_machine_abi_spec_ready") is True,
        "native_ready": report.get("native_ready") is True,
        "native_dispatch_allowed": report.get("native_dispatch_allowed") is True,
        "target_count": int(summary.get("target_count", 0) or 0),
        "state_machine_reference_ready_count": int(summary.get("state_machine_reference_ready_count", 0) or 0),
        "state_machine_abi_spec_ready_count": int(summary.get("state_machine_abi_spec_ready_count", 0) or 0),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
    }


def _unsafe_claims(*reports: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for report in reports:
        scorecard = str(report.get("scorecard", "unknown_scorecard"))
        for field in ("training_path_enabled", "default_behavior_changed", "runtime_dispatch_ready", "native_dispatch_allowed"):
            if report.get(field) is True:
                out.append(f"{scorecard}:{field}")
        if report.get("native_ready") is True or report.get("product_native_ready") is True:
            out.append(f"{scorecard}:native_ready")
    return _dedupe(out)


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_adaptivelr_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(report.get("summary"))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


__all__ = ["TARGET_PLUGIN_OPTIMIZERS", "build_plugin_adaptivelr_family_batch_scorecard"]
