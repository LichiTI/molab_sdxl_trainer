"""Smoke checks for selected plugin model/shape-aware family batch scorecard."""

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

from core.turbocore_plugin_model_shape_aware_family_batch_scorecard import (  # noqa: E402
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_model_shape_aware_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_model_shape_aware_family_batch_scorecard()
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]

    assert report["scorecard"] == "turbocore_plugin_model_shape_aware_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_model_shape_aware_family_batch_ready"] is True, report
    assert report["report_only"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report

    assert set(rows) == set(TARGET_PLUGIN_OPTIMIZERS), rows
    assert summary["selected_optimizer_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selector_model_or_shape_aware_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selected_plugin_native_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["product_native_dispatch_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["default_behavior_changed_count"] == 0, summary
    assert summary["param_group_abi_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["param_group_abi_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["param_group_resume_replay_matrix_artifact_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["param_group_resume_replay_matrix_row_count"] == 29, summary
    assert summary["param_group_resume_replay_matrix_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["param_group_resume_replay_row_implementation_ready_count"] == 29, summary
    assert summary["model_structure_contract_count"] >= 1, summary
    assert summary["shape_partition_contract_count"] >= 1, summary
    assert summary["distributed_collective_contract_count"] == 1, summary
    assert summary["exact_adamw_product_native_route_count_delta"] == 0, summary

    for name, row in rows.items():
        assert row["native_route_family"] == "model_or_shape_aware", row
        assert row["selector_classified"] is True, row
        assert row["resume_proven"] is True, row
        assert row["batch_status"] == "param_group_abi_replay_ready_report_only", row
        assert row["native_route"] == "none_report_only", row
        assert row["plugin_selected_native_ready"] is False, row
        assert row["product_native_ready"] is False, row
        assert row["product_native_dispatch_ready"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["default_behavior_changed"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["native_kernel_ready"] is False, row
        assert row["adamw_native_simple_kernel_compatible"] is False, row
        assert row["native_simple_kernel_reusable"] is False, row
        assert row["param_group_abi_spec_ready"] is True, row
        assert row["param_group_abi_implementation_ready"] is True, row
        assert row["param_group_abi_contract"]["native_kernel_precondition"] == (
            "param_group_abi_and_batch_parity_ready"
        ), row
        artifact = row["param_group_resume_replay_matrix_artifact"]
        assert artifact["artifact_kind"] == "model_shape_param_group_resume_replay_matrix_v0", row
        assert artifact["artifact_ready"] is True, row
        assert artifact["artifact_status"] == "implementation_ready", row
        assert artifact["implementation_ready"] is True, row
        assert artifact["rows"], row
        for artifact_row in artifact["rows"]:
            assert artifact_row["artifact_status"] == "implementation_ready", artifact_row
            assert artifact_row["required_payload"], artifact_row
            assert artifact_row["replay_assertions"], artifact_row
            assert artifact_row["implementation_ready"] is True, artifact_row
            assert artifact_row["native_dispatch_allowed"] is False, artifact_row
        assert row["blocked_reasons"], row
        dependency = row["dependency_contract"]
        assert (
            dependency["requires_model_structure"]
            or dependency["requires_parameter_shapes"]
            or dependency["requires_layer_or_name_hierarchy"]
            or dependency["requires_param_group_semantics"]
        ), row
        if name == "adammini":
            assert row["model_shape_aware_family"] == "model_named_parameter_grouping", row
        elif name == "alice":
            assert row["model_shape_aware_family"] == "shape_split_low_rank_basis", row
        elif name == "spectralsphere":
            assert row["model_shape_aware_family"] == "shape_split_spectral_fallback", row
        elif name == "distributedmuon":
            assert row["model_shape_aware_family"] == "distributed_muon_shape_grouping", row
        else:
            assert row["model_shape_aware_family"] == "muon_shape_grouping", row

    unsafe = build_plugin_model_shape_aware_family_batch_scorecard(
        selector_report=_selector_fixture(runtime_dispatch_ready=True),
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["unsafe_claim_count"] == 1, unsafe
    assert unsafe["plugin_selected_native_ready_count"] == 0, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_model_shape_aware_family_batch_scorecard_smoke",
        "ok": True,
        "artifact_written": False,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _selector_fixture(*, runtime_dispatch_ready: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "selector_fixture",
        "ok": True,
        "plugin_selector_classification_ready": True,
        "selector_boundary_ready": True,
        "all_discovered_plugins_resume_proven": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": runtime_dispatch_ready,
        "native_dispatch_allowed": False,
        "summary": {
            "plugin_optimizer_count": len(TARGET_PLUGIN_OPTIMIZERS),
            "missing_resume_count": 0,
            "route_family_counts": {"model_or_shape_aware": len(TARGET_PLUGIN_OPTIMIZERS)},
        },
        "rows": [_selector_row(name) for name in TARGET_PLUGIN_OPTIMIZERS],
    }


def _selector_row(name: str) -> dict[str, Any]:
    return {
        "optimizer_name": name,
        "selector": "PytorchOptimizer",
        "native_route_family": "model_or_shape_aware",
        "resume_proven": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
