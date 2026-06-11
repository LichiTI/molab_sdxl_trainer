"""Smoke checks for built-in Muon model/shape-aware family batch scorecard."""

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

from core.turbocore_muon_model_shape_aware_family_batch_scorecard import (  # noqa: E402
    TARGET_OPTIMIZER,
    build_muon_model_shape_aware_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_muon_model_shape_aware_family_batch_scorecard(write_artifact=True)
    summary = report["summary"]
    row = report["rows"][0]

    assert report["scorecard"] == "turbocore_muon_model_shape_aware_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["muon_model_shape_aware_family_batch_ready"] is True, report
    assert report["report_only"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["native_kernel_preconditions"]["native_kernel_precondition_ready"] is True, report
    assert report["native_kernel_preconditions"]["native_kernel_implementation_ready"] is False, report
    assert report["runtime_dispatch_shadow"]["runtime_dispatch_shadow_ready"] is True, report
    assert report["runtime_dispatch_shadow"]["runtime_dispatch_ready"] is False, report
    assert report["dispatch_integration_review"]["dispatch_integration_review_ready"] is True, report
    assert report["dispatch_integration_review"]["owner_approval_recorded"] is False, report
    assert report["dispatch_integration_review"]["release_approval_recorded"] is False, report

    assert summary["optimizer_count"] == 1, summary
    assert summary["capability_available_count"] == 1, summary
    assert summary["param_group_abi_spec_ready_count"] == 1, summary
    assert summary["param_group_abi_implementation_ready_count"] == 1, summary
    assert summary["param_group_resume_replay_matrix_artifact_ready_count"] == 1, summary
    assert summary["param_group_resume_replay_matrix_row_count"] == 3, summary
    assert summary["param_group_resume_replay_matrix_implementation_ready_count"] == 1, summary
    assert summary["param_group_resume_replay_row_implementation_ready_count"] == 3, summary
    assert summary["native_kernel_precondition_ready_count"] == 1, summary
    assert summary["runtime_dispatch_shadow_ready_count"] == 1, summary
    assert summary["dispatch_integration_review_ready_count"] == 1, summary
    assert summary["native_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["default_behavior_changed_count"] == 0, summary
    assert summary["unsafe_claim_count"] == 0, summary

    assert row["optimizer_type"] == TARGET_OPTIMIZER, row
    assert row["native_route_family"] == "model_or_shape_aware", row
    assert row["batch_status"] == "dispatch_review_ready_report_only", row
    assert row["param_group_abi_spec_ready"] is True, row
    assert row["param_group_abi_implementation_ready"] is True, row
    assert row["native_kernel_precondition_ready"] is True, row
    assert row["runtime_dispatch_shadow_ready"] is True, row
    assert row["dispatch_integration_review_ready"] is True, row
    assert row["adamw_native_simple_kernel_compatible"] is False, row
    assert row["native_simple_kernel_reusable"] is False, row
    assert row["training_path_enabled"] is False, row
    assert row["runtime_dispatch_ready"] is False, row
    assert row["native_dispatch_allowed"] is False, row
    assert row["native_kernel_ready"] is False, row
    artifact = row["param_group_resume_replay_matrix_artifact"]
    assert artifact["artifact_kind"] == "builtin_muon_param_group_resume_replay_matrix_v0", artifact
    assert artifact["artifact_ready"] is True, artifact
    assert artifact["artifact_status"] == "implementation_ready", artifact
    assert artifact["implementation_ready"] is True, artifact
    assert len(artifact["rows"]) == 3, artifact
    for artifact_row in artifact["rows"]:
        assert artifact_row["artifact_status"] == "implementation_ready", artifact_row
        assert artifact_row["implementation_ready"] is True, artifact_row
        assert artifact_row["required_payload"], artifact_row
        assert artifact_row["replay_assertions"], artifact_row
        assert artifact_row["native_dispatch_allowed"] is False, artifact_row

    return {
        "schema_version": 1,
        "probe": "turbocore_muon_model_shape_aware_family_batch_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
