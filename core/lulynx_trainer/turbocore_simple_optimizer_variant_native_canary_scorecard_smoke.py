"""Smoke checks for simple optimizer variant native canary scorecard."""

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

from core.turbocore_simple_optimizer_variant_native_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_variant_native_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_variant_native_canary_scorecard()
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["scorecard"] == "turbocore_simple_optimizer_variant_native_canary_scorecard_v0", report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["summary"]["target_optimizer_count"] == 5, report
    assert report["summary"]["native_abi_spec_ready_count"] == 5, report
    assert report["summary"]["quantized_formula_parity_ready_count"] == 3, report
    assert report["summary"]["quantized_native_scratch_kernel_ready_count"] == 3, report
    assert report["summary"]["quantized_runtime_canary_manifest_ready_count"] == 3, report
    assert report["summary"]["quantized_training_loop_canary_manifest_ready_count"] == 3, report
    assert report["summary"]["quantized_training_loop_canary_ready_count"] == 3, report
    assert report["summary"]["quantized_e2e_no_regression_ready_count"] == 3, report
    assert report["summary"]["quantized_product_state_sync_review_ready_count"] == 3, report
    assert report["summary"]["quantized_product_optimizer_state_sync_ready_count"] == 3, report
    assert report["summary"]["quantized_optimizer_state_sync_state_tensor_count"] == 6, report
    assert report["summary"]["quantized_optimizer_state_sync_parameter_tensor_count"] == 3, report
    assert report["summary"]["quantized_rollout_policy_ready_count"] == 3, report
    assert report["summary"]["quantized_dispatch_integration_review_ready_count"] == 3, report
    assert report["summary"]["quantized_owner_approval_hold_ready_count"] == 3, report
    assert report["summary"]["quantized_native_canary_pending_count"] == 0, report
    assert report["summary"]["native_kernel_ready_count"] == 3, report
    assert rows["Lion8bit"]["variant_status"] == "quantized_owner_approval_hold_ready", rows["Lion8bit"]
    assert rows["Lion8bit"]["formula_parity_ready"] is True, rows["Lion8bit"]
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
    assert rows["Lion8bit"]["native_canary_ready"] is True, rows["Lion8bit"]
    assert rows["Lion8bit"]["native_kernel_ready"] is True, rows["Lion8bit"]
    assert rows["PagedLion8bit"]["native_abi_spec_ready"] is True, rows["PagedLion8bit"]
    assert rows["PagedLion8bit"]["native_scratch_kernel_ready"] is True, rows["PagedLion8bit"]
    assert rows["SGDNesterov8bit"]["native_canary_ready"] is True, rows["SGDNesterov8bit"]
    assert rows["SGDNesterov8bit"]["runtime_canary_ready"] is True, rows["SGDNesterov8bit"]
    assert rows["SGDNesterov8bit"]["native_kernel_ready"] is True, rows["SGDNesterov8bit"]
    if report["ok"]:
        assert report["variant_schedule_free_native_canary_ready"] is True, report
        assert report["summary"]["schedule_free_native_canary_ready_count"] == 2, report
        assert rows["RAdamScheduleFree"]["variant_status"] == "schedule_free_native_canary_ready", rows["RAdamScheduleFree"]
        assert rows["SGDScheduleFree"]["variant_status"] == "schedule_free_native_canary_ready", rows["SGDScheduleFree"]
        assert rows["RAdamScheduleFree"]["native_kernel_launch_count"] == 1, rows["RAdamScheduleFree"]
        assert rows["SGDScheduleFree"]["native_step_count"] == 1, rows["SGDScheduleFree"]
    else:
        assert report["blocked_reasons"], report
        assert rows["RAdamScheduleFree"]["native_dispatch_allowed"] is False, rows["RAdamScheduleFree"]
        assert rows["SGDScheduleFree"]["training_path_enabled"] is False, rows["SGDScheduleFree"]
    _write_real_artifact(report)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_variant_native_canary_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_variant_native_canary_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
