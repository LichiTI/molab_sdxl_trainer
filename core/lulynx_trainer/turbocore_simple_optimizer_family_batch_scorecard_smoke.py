"""Smoke checks for simple-formula optimizer family batch scorecard."""

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

from core.turbocore_simple_optimizer_family_batch_scorecard import (  # noqa: E402
    build_simple_optimizer_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    payload = build_simple_optimizer_family_batch_scorecard(workspace_root=REPO_ROOT)
    assert payload["scorecard"] == "turbocore_simple_optimizer_family_batch_scorecard_v0", payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["summary"]["exact_target_count"] == 2, payload
    assert payload["summary"]["pending_variant_count"] == 5, payload
    assert payload["summary"]["variant_layout_spec_ready_count"] == 3, payload
    assert payload["summary"]["variant_native_abi_spec_ready_count"] == 5, payload
    assert payload["summary"]["variant_formula_parity_matrix_artifact_ready_count"] == 5, payload
    assert payload["summary"]["variant_formula_parity_matrix_implementation_ready_count"] == 3, payload
    assert payload["summary"]["variant_resume_parity_matrix_artifact_ready_count"] == 5, payload
    assert payload["summary"]["variant_resume_parity_matrix_implementation_ready_count"] == 5, payload
    assert payload["summary"]["variant_quantized_resume_parity_ready_count"] == 3, payload
    assert payload["summary"]["variant_schedule_free_resume_parity_ready_count"] == 2, payload
    assert payload["summary"]["variant_quantized_formula_parity_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_native_scratch_kernel_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_runtime_canary_manifest_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_training_loop_canary_manifest_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_training_loop_canary_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_e2e_no_regression_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_product_state_sync_review_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_product_optimizer_state_sync_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_optimizer_state_sync_state_tensor_count"] == 6, payload
    assert payload["summary"]["variant_quantized_optimizer_state_sync_parameter_tensor_count"] == 3, payload
    assert payload["summary"]["variant_quantized_rollout_policy_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_dispatch_integration_review_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_owner_approval_hold_ready_count"] == 3, payload
    assert payload["summary"]["variant_quantized_native_canary_pending_count"] == 0, payload
    assert payload["summary"]["variant_native_kernel_ready_count"] == 3, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    assert rows["Lion"]["batch_status"] in {
        "simple_formula_native_batch_canary_ready",
        "simple_formula_batch_blocked",
    }, rows["Lion"]
    assert rows["SGDNesterov"]["batch_status"] == rows["Lion"]["batch_status"], payload
    assert (
        rows["Lion8bit"]["batch_status"] == "simple_formula_variant_quantized_owner_approval_hold_ready"
    ), rows["Lion8bit"]
    assert (
        rows["PagedLion8bit"]["batch_status"] == "simple_formula_variant_quantized_owner_approval_hold_ready"
    ), rows["PagedLion8bit"]
    assert (
        rows["SGDNesterov8bit"]["batch_status"] == "simple_formula_variant_quantized_owner_approval_hold_ready"
    ), rows["SGDNesterov8bit"]
    assert rows["Lion8bit"]["native_abi_spec_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["quantized_formula_parity_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["native_scratch_kernel_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["runtime_canary_manifest_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["training_loop_canary_manifest_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["training_loop_canary_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["e2e_no_regression_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["product_state_sync_review_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["product_optimizer_state_sync_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["canary_rollout_policy_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["dispatch_integration_review_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["owner_approval_hold_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["formula_parity_matrix_artifact_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["resume_parity_matrix_artifact_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["resume_parity_matrix_implementation_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["native_kernel_ready"] is True, rows["Lion8bit"]
    assert rows["PagedLion8bit"]["native_dispatch_allowed"] is False, rows["PagedLion8bit"]
    if payload["variant_state_layout_scorecard"]["ok"]:
        assert payload["summary"]["variant_state_machine_reference_ready_count"] == 2, payload
        if payload["variant_native_canary_scorecard"]["ok"]:
            assert payload["summary"]["variant_schedule_free_native_canary_ready_count"] == 2, payload
            assert (
                rows["RAdamScheduleFree"]["batch_status"]
                == "simple_formula_variant_schedule_free_native_canary_ready"
            ), rows["RAdamScheduleFree"]
            assert (
                rows["SGDScheduleFree"]["batch_status"]
                == "simple_formula_variant_schedule_free_native_canary_ready"
            ), rows["SGDScheduleFree"]
            assert rows["SGDScheduleFree"]["runtime_canary_ready"] is True, rows["SGDScheduleFree"]
        else:
            assert (
                rows["RAdamScheduleFree"]["batch_status"]
                == "simple_formula_variant_native_abi_spec_ready"
            ), rows["RAdamScheduleFree"]
        assert rows["RAdamScheduleFree"]["native_abi_spec_ready"] is True, rows["RAdamScheduleFree"]
    else:
        assert rows["RAdamScheduleFree"]["native_kernel_ready"] is False, rows["RAdamScheduleFree"]
        assert rows["SGDScheduleFree"]["native_kernel_ready"] is False, rows["SGDScheduleFree"]
    if payload["simple_formula_native_batch_canary_ready"]:
        assert payload["summary"]["batch_canary_ready_count"] == 2, payload
        assert rows["Lion"]["native_kernel_ready"] is True, rows["Lion"]
        assert rows["SGDNesterov"]["training_loop_canary_ready"] is True, rows["SGDNesterov"]
    else:
        assert payload["summary"]["batch_canary_ready_count"] == 0, payload
        assert payload["blocked_reasons"], payload
    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_family_batch_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_family_batch_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
