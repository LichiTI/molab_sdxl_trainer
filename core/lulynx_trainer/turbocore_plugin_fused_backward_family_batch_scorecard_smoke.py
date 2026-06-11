"""Smoke checks for selected plugin fused-backward family batch scorecard."""

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

from core.turbocore_plugin_fused_backward_family_batch_scorecard import (  # noqa: E402
    FUSED_BACKWARD_FAMILY,
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_fused_backward_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_fused_backward_family_batch_scorecard()
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]

    assert report["scorecard"] == "turbocore_plugin_fused_backward_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_fused_backward_family_batch_ready"] is True, report
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
    assert summary["selector_fused_backward_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selector_classified_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["fused_backward_required_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["backward_hook_required_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["gradient_ownership_abi_required_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["skip_step_contract_required_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["per_optimizer_abi_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["backward_hook_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["gradient_ownership_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["skip_optimizer_step_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["loss_backward_call_site_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["state_resume_scope_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["native_kernel_preconditions_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["resume_parity_matrix_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["resume_parity_matrix_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["fused_backward_replay_case_planned_count"] == len(TARGET_PLUGIN_OPTIMIZERS) * 5, summary
    assert summary["loss_scale_boundary_case_planned_count"] == len(TARGET_PLUGIN_OPTIMIZERS) * 2, summary
    assert summary["fused_backward_replay_case_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS) * 5, summary
    assert summary["fused_backward_abi_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["native_kernel_preconditions_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
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
    assert summary["missing_expected_selector_count"] == 0, summary
    assert summary["unexpected_selector_count"] == 0, summary
    assert summary["unsafe_claim_count"] == 0, summary

    abi = report["fused_backward_abi"]
    assert abi["requires_fused_backward"] is True, abi
    assert abi["requires_backward_hook_contract"] is True, abi
    assert abi["requires_gradient_ownership_abi"] is True, abi
    assert abi["requires_skip_step_contract"] is True, abi
    assert abi["per_optimizer_abi_spec_ready"] is True, abi
    assert abi["abi_implementation_ready"] is True, abi
    assert abi["adamw_step_kernel_compatible"] is False, abi
    assert abi["simple_formula_kernel_compatible"] is False, abi
    assert abi["can_reuse_exact_adamw_native_dispatch"] is False, abi

    for row in rows.values():
        assert row["native_route_family"] == FUSED_BACKWARD_FAMILY, row
        assert row["selector_classified"] is True, row
        assert row["resume_proven"] is True, row
        assert row["batch_status"] == "fused_backward_abi_replay_ready_report_only", row
        assert row["fused_backward_abi_spec_ready"] is True, row
        assert row["fused_backward_abi_implementation_ready"] is True, row
        assert row["native_kernel_preconditions_implementation_ready"] is True, row
        assert row["native_route"] == "none_report_only", row
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
        assert row["blocked_reasons"], row
        requirements = row["abi_requirements"]
        assert requirements["requires_fused_backward"] is True, row
        assert requirements["requires_backward_hook"] is True, row
        assert requirements["requires_gradient_ownership_abi"] is True, row
        assert requirements["requires_skip_step_contract"] is True, row
        abi_spec = row["abi_spec"]
        assert abi_spec["optimizer_step_policy"] == "forbid_public_optimizer_step_call", row
        backward_hook = abi_spec["backward_hook"]
        assert backward_hook["attachment_point"] == "loss_backward_owner", row
        assert backward_hook["requires_grad_callback_per_parameter"] is True, row
        assert backward_hook["requires_backward_completion_barrier"] is True, row
        gradient_ownership = abi_spec["gradient_ownership"]
        assert gradient_ownership["gradient_owner"] == "fused_backward_hook", row
        assert gradient_ownership["optimizer_step_reads_grad"] is False, row
        assert gradient_ownership["native_update_consumes_gradient_once"] is True, row
        skip_step = abi_spec["skip_optimizer_step"]
        assert skip_step["forbidden_call"] == "optimizer.step", row
        assert skip_step["requires_explicit_skip_evidence"] is True, row
        call_site = abi_spec["loss_backward_call_site"]
        assert call_site["authoritative_call_site"] == "training_loop_loss_backward", row
        assert call_site["requires_backward_invocation_token"] is True, row
        resume_scope = abi_spec["state_resume_scope"]
        assert resume_scope["resume_adapter"] == "plugin_state_dict_plus_fused_backward_owner_state", row
        assert resume_scope["requires_mid_backward_resume_block"] is True, row
        matrix = row["resume_parity_matrix"]
        assert matrix["matrix_spec_ready"] is True, row
        assert matrix["matrix_implementation_ready"] is True, row
        assert len(matrix["planned_cases"]) == 5, row
        assert all(case["status"] == "implementation_ready" for case in matrix["planned_cases"]), row
        assert all(case["native_dispatch_allowed"] is False for case in matrix["planned_cases"]), row
        preconditions = abi_spec["native_kernel_preconditions"]
        assert preconditions["requires_flat_param_buffer"] is True, row
        assert preconditions["requires_flat_grad_buffer"] is True, row
        assert preconditions["requires_backward_hook_ownership_token"] is True, row
        assert preconditions["requires_loss_scale_metadata"] is True, row

    unsafe = build_plugin_fused_backward_family_batch_scorecard(
        selector_report=_selector_fixture(runtime_dispatch_ready=True),
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["unsafe_claim_count"] == 1, unsafe
    assert unsafe["plugin_selected_native_ready_count"] == 0, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_fused_backward_family_batch_scorecard_smoke",
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
        "summary": {
            "plugin_optimizer_count": len(TARGET_PLUGIN_OPTIMIZERS),
            "missing_resume_count": 0,
            "route_family_counts": {FUSED_BACKWARD_FAMILY: len(TARGET_PLUGIN_OPTIMIZERS)},
        },
        "rows": [_selector_row(name) for name in TARGET_PLUGIN_OPTIMIZERS],
    }


def _selector_row(name: str) -> dict[str, Any]:
    return {
        "optimizer_name": name,
        "selector": "PytorchOptimizer",
        "native_route_family": FUSED_BACKWARD_FAMILY,
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
