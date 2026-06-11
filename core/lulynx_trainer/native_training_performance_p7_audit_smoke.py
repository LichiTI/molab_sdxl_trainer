"""Smoke checks for Native Training Performance P7 audit."""

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

from devtools.audit_native_training_performance_p7 import build_p7_optimizer_expansion_audit  # noqa: E402


def run_smoke() -> dict[str, Any]:
    audit = build_p7_optimizer_expansion_audit(quick=True)
    assert audit["audit"] == "native_training_performance_p7_audit_v0", audit
    assert audit["ok"] is True, audit
    assert audit["milestone_completed"] is True, audit
    assert "simple_formula_optimizer_reference" in audit["sections"], audit
    assert "simple_formula_optimizer_kind_abi" in audit["sections"], audit
    assert "simple_formula_kernel_registry_dry_run" in audit["sections"], audit
    assert "simple_formula_native_kernel_parity" in audit["sections"], audit
    assert "simple_formula_runtime_canary" in audit["sections"], audit
    assert "simple_formula_runtime_tensor_launch" in audit["sections"], audit
    assert "simple_formula_training_executor" in audit["sections"], audit
    assert "simple_formula_dispatch_runtime" in audit["sections"], audit
    assert "simple_formula_training_loop_canary" in audit["sections"], audit
    assert "simple_formula_e2e_no_regression" in audit["sections"], audit
    section = audit["sections"]["simple_formula_optimizer_reference"]
    assert section["ok"] is True, section
    assert section["first_stage_ready"] is True, section
    assert section["promotion_ready"] is False, section
    assert section["training_path_enabled"] is False, section
    abi_section = audit["sections"]["simple_formula_optimizer_kind_abi"]
    assert abi_section["ok"] is True, abi_section
    assert abi_section["first_abi_stage_ready"] is True, abi_section
    assert abi_section["training_path_enabled"] is False, abi_section
    registry_section = audit["sections"]["simple_formula_kernel_registry_dry_run"]
    assert registry_section["ok"] is True, registry_section
    assert registry_section["registry_stage_ready"] is True, registry_section
    assert registry_section["training_path_enabled"] is False, registry_section
    kernel_section = audit["sections"]["simple_formula_native_kernel_parity"]
    assert kernel_section["ok"] is True, kernel_section
    assert kernel_section["lion_native_kernel_parity"] is True, kernel_section
    assert kernel_section["sgd_nesterov_native_kernel_parity"] is True, kernel_section
    assert kernel_section["kernel_parity_stage_ready"] is True, kernel_section
    assert kernel_section["training_path_enabled"] is False, kernel_section
    runtime_section = audit["sections"]["simple_formula_runtime_canary"]
    assert runtime_section["ok"] is True, runtime_section
    assert runtime_section["runtime_canary_ready"] is True, runtime_section
    assert runtime_section["native_route_hit_count"] == 2, runtime_section
    assert runtime_section["training_path_enabled"] is False, runtime_section
    launch_section = audit["sections"]["simple_formula_runtime_tensor_launch"]
    assert launch_section["ok"] is True, launch_section
    assert launch_section["runtime_launch_stage_ready"] is True, launch_section
    assert launch_section["summary"]["runtime_launch_ready_count"] == 2, launch_section
    assert launch_section["training_path_enabled"] is False, launch_section
    executor_section = audit["sections"]["simple_formula_training_executor"]
    assert executor_section["ok"] is True, executor_section
    assert executor_section["summary"]["native_step_count"] == 8, executor_section
    for case in executor_section["cases"]:
        assert case["ok"] is True, case
        assert case["native_step_count"] == 4, case
        assert case["training_path_enabled"] is False, case
    dispatch_section = audit["sections"]["simple_formula_dispatch_runtime"]
    assert dispatch_section["ok"] is True, dispatch_section
    assert dispatch_section["summary"]["native_step_count"] == 2, dispatch_section
    assert dispatch_section["summary"]["skip_pytorch_count"] == 2, dispatch_section
    for case in dispatch_section["cases"]:
        assert case["ok"] is True, case
        assert case["native_step_executed"] is True, case
        assert case["should_call_pytorch_optimizer_step"] is False, case
        assert case["training_path_enabled"] is False, case
    training_loop_section = audit["sections"]["simple_formula_training_loop_canary"]
    assert training_loop_section["ok"] is True, training_loop_section
    assert training_loop_section["summary"]["native_step_count"] == 2, training_loop_section
    assert training_loop_section["training_path_enabled"] is False, training_loop_section
    assert training_loop_section["default_behavior_changed"] is False, training_loop_section
    for case in training_loop_section["cases"]:
        assert case["ok"] is True, case
        assert case["native_step_executed"] is True, case
        assert case["native_kernel_launched"] is True, case
        assert case["training_path_enabled"] is False, case
    e2e_section = audit["sections"]["simple_formula_e2e_no_regression"]
    assert e2e_section["ok"] is True, e2e_section
    assert e2e_section["e2e_no_regression_ready"] is True, e2e_section
    assert e2e_section["training_path_enabled"] is False, e2e_section
    assert audit["progress_gates"]["simple_formula_optimizer_reference"] is True, audit
    assert audit["progress_gates"]["simple_formula_optimizer_kind_abi"] is True, audit
    assert audit["progress_gates"]["simple_formula_kernel_registry_dry_run"] is True, audit
    assert audit["progress_gates"]["lion_native_kernel_parity"] is True, audit
    assert audit["progress_gates"]["sgd_nesterov_native_kernel_parity"] is True, audit
    assert audit["progress_gates"]["runtime_canary_hit"] is True, audit
    assert audit["progress_gates"]["runtime_tensor_launch"] is True, audit
    assert audit["progress_gates"]["training_executor_native_steps"] is True, audit
    assert audit["progress_gates"]["dispatch_runtime_native_steps"] is True, audit
    assert audit["progress_gates"]["training_loop_native_canary"] is True, audit
    assert audit["progress_gates"]["e2e_no_regression"] is True, audit
    assert audit["milestone_completed"] is True, audit
    assert "lion_native_kernel_parity_missing" not in audit["remaining_blockers"], audit
    assert "sgd_nesterov_native_kernel_parity_missing" not in audit["remaining_blockers"], audit
    assert "runtime_canary_hit_missing" not in audit["remaining_blockers"], audit
    assert "e2e_no_regression_missing" not in audit["remaining_blockers"], audit
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p7_audit_smoke",
        "ok": True,
        "milestone_completed": audit["milestone_completed"],
        "progress_gates": audit["progress_gates"],
        "recommended_next_step": audit["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
