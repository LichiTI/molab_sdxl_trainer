"""Smoke for the compact TurboCore native-update owner-release handoff."""

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

from core.turbocore_native_update_owner_release_handoff_summary import (  # noqa: E402
    build_native_update_owner_release_handoff_summary,
)
from core.turbocore_native_update_representative_performance_importer import (  # noqa: E402
    build_native_update_representative_performance_import,
)
from core.turbocore_optimizer_product_training_route_binding_config_adapter import (  # noqa: E402
    build_optimizer_product_training_route_binding_config_adapter,
)
from core.turbocore_optimizer_product_training_route_binding_preflight import (  # noqa: E402
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_optimizer_product_training_route_binding_training_loop_contract import (  # noqa: E402
    build_optimizer_product_training_route_binding_training_loop_contract,
)


def run_smoke() -> dict[str, Any]:
    build_native_update_representative_performance_import(write_artifacts=True)
    build_optimizer_product_training_route_binding_preflight(write_artifact=True)
    build_optimizer_product_training_route_binding_training_loop_contract(write_artifact=True)
    build_optimizer_product_training_route_binding_config_adapter(write_artifact=True)
    report = build_native_update_owner_release_handoff_summary(write_artifact=True)
    summary = report["summary"]
    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design.md", report
    assert report["ok"] is True, report
    assert report["technical_evidence_ready"] is True, report
    assert report["representative_performance_evidence_complete"] is True, report
    assert report["representative_performance_source_evidence_quality"], report
    assert report["representative_performance_fresh_live_run"] is False, report
    assert report["ready_for_owner_release_review"] is True, report
    assert report["release_review_recorded"] is False, report
    assert report["owner_action_required"] is True, report
    assert report["decision"] == "native_update_release_review_hold_for_owner_review_default_off", report
    assert "native_update_release_owner_review_missing" in report["blocked_reasons"], report
    assert report["review_template_for_owner"]["approve_native_update_release_review_package"] is False, report
    assert report["review_template_for_owner"]["acknowledged_gate_count"] == 12, report
    assert report["review_template_for_owner"]["acknowledged_supplemental_gate_count"] == 2, report
    assert report["required_gate_acknowledgement_count"] == 12, report
    assert "optimizer_family_coverage" in report["required_supplemental_acknowledgements"], report
    assert "native_update_optimizer_multitensor_release_hold" in report["required_supplemental_acknowledgements"], report
    assert report["product_exposure_allowed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["training_launch_executed"] is False, report
    assert summary["expected_gate_count"] == 12, report
    assert summary["present_gate_count"] == 12, report
    assert summary["default_off_gate_count"] == 12, report
    assert summary["supplemental_gate_count"] == 2, report
    assert summary["present_supplemental_gate_count"] == 2, report
    assert summary["default_off_supplemental_gate_count"] == 2, report
    assert summary["plugin_optimizer_count"] == 124, report
    assert summary["plugin_selected_native_ready_count"] == 0, report
    assert summary["optimizer_inventory_source_ready_count"] == 124, report
    assert summary["optimizer_inventory_probe_ready_count"] == 124, report
    assert summary["optimizer_family_contract_ready_count"] == 10, report
    assert summary["multitensor_native_kernel_launch_count"] == 2, report
    assert summary["multitensor_training_parameter_mutation_count"] == 2, report
    assert summary["multitensor_top_level_native_dispatch_allowed_count"] == 0, report
    assert summary["representative_performance_artifact_present_count"] == 1, report
    assert summary["representative_performance_gate_ready_count"] == 1, report
    assert summary["representative_performance_fresh_live_run_count"] == 0, report
    assert summary["representative_performance_training_matrix_steps"] >= 20, report
    assert summary["native_readiness_runtime_launch_coverage_ready_family_count"] == 10, report
    assert summary["native_readiness_runtime_launch_adapter_ready_family_count"] == 6, report
    assert summary["native_readiness_runtime_launch_adapter_ready_optimizer_count"] == 72, report
    assert summary["native_readiness_owner_release_hold_ready_family_count"] == 10, report
    assert summary["native_readiness_request_schema_ui_non_exposure_ready_family_count"] == 10, report
    assert summary["native_readiness_family_specific_runtime_launch_missing_count"] == 0, report
    assert summary["product_route_binding_preflight_ready_count"] == 0, report
    assert summary["product_route_binding_candidate_count"] == 0, report
    assert summary["product_route_binding_owner_approval_recorded_count"] == 0, report
    assert summary["product_route_binding_exposure_decision_recorded_count"] == 0, report
    assert summary["training_loop_route_candidate_switch_count"] == 3, report
    assert summary["training_loop_route_open_training_path_enabled_count"] == 1, report
    assert summary["training_loop_route_request_fields_emitted_count"] == 0, report
    assert summary["training_loop_route_schema_exposure_allowed_count"] == 0, report
    assert summary["training_loop_route_ui_exposure_allowed_count"] == 0, report
    assert summary["route_binding_config_patch_ready_count"] == 0, report
    assert summary["route_binding_constructor_switch_field_count"] == 3, report
    assert summary["route_binding_kwargs_patch_field_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_owner_release_handoff_summary_smoke",
        "ok": True,
        "roadmap": report["roadmap"],
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
