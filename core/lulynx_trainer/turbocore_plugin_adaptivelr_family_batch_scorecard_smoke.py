"""Smoke checks for selected plugin adaptive-LR family batch scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_plugin_adaptivelr_family_batch_scorecard import (  # noqa: E402
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_adaptivelr_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adaptivelr_family_batch_scorecard(write_artifact=False)
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]

    assert report["scorecard"] == "turbocore_plugin_adaptivelr_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_adaptivelr_family_batch_ready"] is True, report
    assert report["report_only"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report
    assert report["execution_matrix"]["execution_matrix_ready"] is True, report

    assert set(rows) == set(TARGET_PLUGIN_OPTIMIZERS), rows
    assert summary["selected_optimizer_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_adaptivelr_family_batch_ready"] is True, summary
    assert summary["selector_adaptive_lr_state_machine_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_state_machine_reference_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_state_machine_abi_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_dynamic_lr_scalar_state_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_d_estimator_global_state_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_per_step_quality_guard_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_resume_scope_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_native_kernel_preconditions_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_state_machine_replay_matrix_artifact_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_state_machine_replay_matrix_implementation_ready_count"] == len(
        TARGET_PLUGIN_OPTIMIZERS
    ), summary
    assert summary["selected_state_machine_replay_case_planned_count"] == 36, summary
    assert summary["selected_state_machine_replay_case_implementation_ready_count"] == 36, summary
    assert summary["selected_state_machine_replay_resume_case_planned_count"] == 24, summary
    assert summary["selected_state_machine_replay_resume_case_implementation_ready_count"] == 24, summary
    assert summary["selected_state_machine_abi_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_native_kernel_preconditions_implementation_ready_count"] == len(
        TARGET_PLUGIN_OPTIMIZERS
    ), summary
    assert summary["prodigy_reference_count"] == 1, summary
    assert summary["dadapt_reference_count"] == 5, summary
    assert summary["native_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["exact_adamw_product_native_route_count_delta"] == 0, summary

    for name, row in rows.items():
        assert row["native_route_family"] == "adaptive_lr_state_machine", row
        assert row["selector_classified"] is True, row
        assert row["resume_proven"] is True, row
        assert row["state_machine_status"] == "abi_replay_ready_report_only", row
        assert row["selected_state_machine_reference_ready"] is True, row
        assert row["selected_state_machine_abi_spec_ready"] is True, row
        assert row["state_machine_abi_implementation_ready"] is True, row
        assert row["native_kernel_preconditions_spec_ready"] is True, row
        assert row["native_kernel_preconditions_implementation_ready"] is True, row
        assert row["batch_reference_ready"] is True, row
        _assert_state_machine_abi_spec(name, row)
        _assert_state_machine_replay_matrix(name, row)
        assert row["native_route"] == "none_report_only", row
        assert row["plugin_selected_native_ready"] is False, row
        assert row["product_native_ready"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["default_behavior_changed"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["native_kernel_ready"] is False, row
        assert row["blocked_reasons"] == [
            "native_kernel_implementation_missing",
            "runtime_dispatch_shadow_missing",
            "owner_release_hold_missing",
        ], row
        if name == "prodigy":
            assert row["adaptive_lr_family"] == "adaptive_lr_prodigy", row
        else:
            assert row["adaptive_lr_family"] == "adaptive_lr_dadapt", row

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adaptivelr_family_batch_scorecard_smoke",
        "ok": True,
        "artifact_written": False,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _assert_state_machine_abi_spec(name: str, row: dict[str, Any]) -> None:
    spec = row["state_machine_abi_spec"]
    assert spec["report_only"] is True, spec
    assert spec["spec_ready"] is True, spec
    assert spec["implementation_ready"] is True, spec
    assert spec["selected_optimizer_name"] == name, spec
    assert spec["plugin_binding"]["spec_ready"] is True, spec
    assert spec["plugin_binding"]["implementation_ready"] is True, spec
    assert spec["plugin_binding"]["native_runtime_authority"] == "none_report_only", spec
    for key in (
        "dynamic_lr_scalar_state",
        "d_estimator_global_state",
        "per_step_quality_guard",
        "resume_scope",
        "native_kernel_preconditions",
    ):
        block = spec[key]
        assert block["spec_ready"] is True, (key, spec)
        assert block["implementation_ready"] is True, (key, spec)
    assert spec["dynamic_lr_scalar_state"]["kernel_input_policy"] == "materialize_scalars_before_launch", spec
    assert spec["resume_scope"]["resume_parity_gate"] == "step_state_dict_load_state_dict_next_step", spec
    assert "owner_release_hold" in spec["native_kernel_preconditions"]["blocked_until"], spec


def _assert_state_machine_replay_matrix(name: str, row: dict[str, Any]) -> None:
    matrix = row["state_machine_replay_matrix_artifact"]
    assert matrix["artifact_kind"] == "selected_plugin_adaptivelr_state_machine_replay_matrix", matrix
    assert matrix["report_only"] is True, matrix
    assert matrix["selected_optimizer_name"] == name, matrix
    assert matrix["spec_ready"] is True, matrix
    assert matrix["implementation_ready"] is True, matrix
    assert matrix["training_path_enabled"] is False, matrix
    assert matrix["runtime_dispatch_ready"] is False, matrix
    assert matrix["native_dispatch_allowed"] is False, matrix
    assert matrix["native_kernel_ready"] is False, matrix
    assert "dynamic_lr_scalar_recomputed_from_saved_state" in matrix["replay_cases"], matrix
    assert "resume_next_step_matches_python_reference" in matrix["resume_replay_cases"], matrix
    assert "owner_release_hold" in matrix["blocked_until"], matrix
    assert matrix["evidence_status"] == "planned_report_only", matrix
    assert all(status == "implementation_ready" for status in matrix["replay_case_status"].values()), matrix
    assert all(status == "implementation_ready" for status in matrix["resume_replay_case_status"].values()), matrix
    if name == "prodigy":
        assert "prodigy_global_distance_state_replay" in matrix["replay_cases"], matrix
        assert "prodigy_distance_buffer_resume" in matrix["resume_replay_cases"], matrix
    else:
        assert "dadapt_variant_accumulator_replay" in matrix["replay_cases"], matrix
        assert "dadapt_variant_accumulator_resume" in matrix["resume_replay_cases"], matrix


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
