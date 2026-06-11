"""Smoke checks for selected plugin state-adapter-special family batch scorecard."""

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

from core.turbocore_plugin_state_adapter_special_family_batch_scorecard import (  # noqa: E402
    STATE_ADAPTER_SPECIAL_ROUTE_FAMILY,
    build_plugin_state_adapter_special_family_batch_scorecard,
)


EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS = ("demo", "sgdsai", "spam")


def run_smoke() -> dict[str, Any]:
    report = build_plugin_state_adapter_special_family_batch_scorecard()
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]

    assert report["scorecard"] == "turbocore_plugin_state_adapter_special_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_state_adapter_special_family_batch_ready"] is True, report
    assert report["report_only"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report
    assert report["native_compatibility"]["adapter_abi_spec_ready"] is True, report
    assert report["native_compatibility"]["adapter_abi_implementation_ready"] is True, report
    assert report["native_compatibility"]["adapter_resume_matrix_artifact_ready"] is True, report
    assert report["native_compatibility"]["adapter_resume_matrix_implementation_ready"] is True, report
    assert report["execution_matrix"]["execution_matrix_ready"] is True, report

    assert set(rows) == set(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), rows
    assert report["request_contract"]["selected_optimizer_names"] == sorted(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), report
    assert summary["selected_optimizer_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["selector_state_adapter_special_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["selector_classified_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["resume_proven_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["special_optimizer_state_adapter_required_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["resume_state_adapter_required_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["param_ownership_abi_required_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["adapter_abi_spec_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["adapter_abi_implementation_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["param_ownership_abi_spec_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["state_adapter_role_spec_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["resume_translation_scope_spec_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["quality_safety_guard_spec_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["native_kernel_precondition_spec_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["adapter_resume_matrix_artifact_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["adapter_resume_matrix_implementation_ready_count"] == len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS), summary
    assert summary["adapter_resume_replay_case_planned_count"] == 15, summary
    assert summary["adapter_resume_translation_case_planned_count"] == 12, summary
    assert summary["adapter_resume_replay_case_implementation_ready_count"] == 15, summary
    assert summary["adapter_resume_translation_case_implementation_ready_count"] == 12, summary
    assert summary["native_kernel_precondition_implementation_ready_count"] == len(
        EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS
    ), summary
    assert summary["adamw_kernel_compatible_count"] == 0, summary
    assert summary["simple_kernel_compatible_count"] == 0, summary
    assert summary["native_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["product_native_dispatch_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["default_behavior_changed_count"] == 0, summary
    assert summary["plugin_selected_native_ready_count"] == 0, summary
    assert summary["exact_adamw_product_native_route_count_delta"] == 0, summary
    assert summary["missing_selector_classification_count"] == 0, summary
    assert summary["unsafe_claim_count"] == 0, summary

    for name, row in rows.items():
        assert row["native_route_family"] == STATE_ADAPTER_SPECIAL_ROUTE_FAMILY, row
        assert row["selector_classified"] is True, row
        assert row["resume_proven"] is True, row
        assert row["batch_status"] == "state_adapter_special_abi_replay_ready_report_only", row
        assert row["native_route"] == "none_report_only", row
        assert row["adamw_state_schema_compatible"] is False, row
        assert row["adamw_kernel_compatible"] is False, row
        assert row["simple_formula_kernel_compatible"] is False, row
        assert row["can_reuse_exact_adamw_native_dispatch"] is False, row
        assert row["plugin_selected_native_ready"] is False, row
        assert row["product_native_ready"] is False, row
        assert row["product_native_dispatch_ready"] is False, row
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
        contract = row["state_adapter_contract"]
        assert contract["requires_special_optimizer_state_adapter"] is True, row
        assert contract["requires_resume_state_adapter"] is True, row
        assert contract["requires_param_ownership_abi"] is True, row
        assert contract["adamw_state_schema_compatible"] is False, row
        assert contract["simple_formula_kernel_compatible"] is False, row
        assert row["adapter_abi_spec_ready"] is True, row
        assert row["adapter_abi_implementation_ready"] is True, row
        abi = row["state_adapter_abi_spec"]
        assert abi["report_only"] is True, row
        assert abi["implementation_ready"] is True, row
        assert abi["native_dispatch_allowed"] is False, row
        assert abi["param_ownership"]["spec_ready"] is True, row
        assert abi["state_adapter_role"]["spec_ready"] is True, row
        assert abi["resume_translation_scope"]["spec_ready"] is True, row
        assert abi["quality_safety_guard"]["spec_ready"] is True, row
        assert abi["native_kernel_preconditions"]["spec_ready"] is True, row
        assert abi["quality_safety_guard"]["guards"], row
        assert abi["native_kernel_preconditions"]["required_before_kernel"], row
        matrix = row["adapter_resume_matrix_artifact"]
        assert matrix["artifact_kind"] == "selected_plugin_state_adapter_special_resume_matrix", row
        assert matrix["report_only"] is True, row
        assert matrix["selected_optimizer_name"] == name, row
        assert matrix["spec_ready"] is True, row
        assert matrix["implementation_ready"] is True, row
        assert matrix["training_path_enabled"] is False, row
        assert matrix["runtime_dispatch_ready"] is False, row
        assert matrix["native_dispatch_allowed"] is False, row
        assert matrix["native_kernel_ready"] is False, row
        assert "state_dict_roundtrip_after_step" in matrix["resume_replay_cases"], row
        assert "bridge_payload_restored_after_base_load_state_dict" in matrix["translation_cases"], row
        assert "owner_release_hold" in matrix["blocked_until"], row
        assert matrix["evidence_status"] == "planned_report_only", row
        assert all(status == "implementation_ready" for status in matrix["resume_replay_case_status"].values()), row
        assert all(status == "implementation_ready" for status in matrix["translation_case_status"].values()), row
        if name == "demo":
            assert contract["state_adapter_family"] == "distributed_demo_state_bridge", row
            assert contract["requires_external_demo_state_persistence"] is True, row
            assert abi["param_ownership"]["bridge_payload"] == "lulynx_demo_state", row
            assert abi["resume_translation_scope"]["target"] == "optimizer.demo_state[param] restored after base load_state_dict", row
            assert "demo_state_sidecar_roundtrip" in matrix["resume_replay_cases"], row
            assert "lulynx_demo_state_sidecar_to_optimizer_demo_state" in matrix["translation_cases"], row
        elif name == "sgdsai":
            assert contract["state_adapter_family"] == "warmup_flag_resume_bridge", row
            assert contract["requires_non_state_dict_attribute_restore"] is True, row
            assert abi["param_ownership"]["bridge_payload"] == "lulynx_optimizer_attrs.has_warmup", row
            assert "warmup_phase_resume_parity" in abi["quality_safety_guard"]["guards"], row
            assert "warmup_phase_flag_roundtrip" in matrix["resume_replay_cases"], row
            assert "has_warmup_bool_attr_restore" in matrix["translation_cases"], row
        elif name == "spam":
            assert contract["state_adapter_family"] == "sparse_mask_resume_bridge", row
            assert contract["requires_bool_mask_restore"] is True, row
            assert abi["param_ownership"]["bridge_payload"] == "lulynx_sparse_masks", row
            assert "bool_mask_dtype_guard" in abi["quality_safety_guard"]["guards"], row
            assert "sparse_mask_state_roundtrip" in matrix["resume_replay_cases"], row
            assert "lulynx_sparse_masks_to_optimizer_mask_state" in matrix["translation_cases"], row

    unsafe = build_plugin_state_adapter_special_family_batch_scorecard(
        selector_report=_selector_fixture(runtime_dispatch_ready=True),
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["unsafe_claim_count"] == 1, unsafe
    assert unsafe["plugin_selected_native_ready_count"] == 0, unsafe
    assert unsafe["training_path_enabled"] is False, unsafe
    assert unsafe["default_behavior_changed"] is False, unsafe
    assert unsafe["runtime_dispatch_ready"] is False, unsafe
    assert unsafe["native_dispatch_allowed"] is False, unsafe
    assert unsafe["native_kernel_ready"] is False, unsafe
    assert unsafe["product_native_ready"] is False, unsafe
    assert unsafe["product_native_dispatch_ready"] is False, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_state_adapter_special_family_batch_scorecard_smoke",
        "ok": True,
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
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "summary": {
            "plugin_optimizer_count": len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS),
            "missing_resume_count": 0,
            "route_family_counts": {
                STATE_ADAPTER_SPECIAL_ROUTE_FAMILY: len(EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS)
            },
        },
        "rows": [_selector_row(name) for name in EXPECTED_STATE_ADAPTER_SPECIAL_OPTIMIZERS],
    }


def _selector_row(name: str) -> dict[str, Any]:
    return {
        "optimizer_name": name,
        "selector": "PytorchOptimizer",
        "native_route_family": STATE_ADAPTER_SPECIAL_ROUTE_FAMILY,
        "resume_proven": True,
        "special_handling": f"fixture special handling for {name}",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
