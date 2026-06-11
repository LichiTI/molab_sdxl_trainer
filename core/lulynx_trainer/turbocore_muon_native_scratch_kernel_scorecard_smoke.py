"""Smoke checks for Muon native scratch-kernel scorecard."""

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
    build_muon_model_shape_aware_family_batch_scorecard,
)
from core.turbocore_muon_native_scratch_kernel_scorecard import (  # noqa: E402
    ENTRYPOINT,
    build_muon_native_scratch_kernel_scorecard,
)


def run_smoke() -> dict[str, Any]:
    model_shape = build_muon_model_shape_aware_family_batch_scorecard(write_artifact=True)
    report = build_muon_native_scratch_kernel_scorecard(
        muon_model_shape_report=model_shape,
        write_artifact=True,
    )
    summary = report["summary"]
    case = report["case"]

    assert report["scorecard"] == "turbocore_muon_native_scratch_kernel_scorecard_v0", report
    assert report["entrypoint"] == ENTRYPOINT, report
    assert report["ok"] is True, report
    assert report["native_scratch_kernel_ready"] is True, report
    assert report["native_kernel_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["runtime_canary_ready"] is False, report
    assert report["runtime_canary_hit"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_ready_count"] == 0, report

    assert summary["optimizer_count"] == 1, summary
    assert summary["native_scratch_kernel_ready_count"] == 1, summary
    assert summary["native_kernel_ready_count"] == 1, summary
    assert summary["kernel_executed_count"] == 1, summary
    assert summary["parameters_mutated_count"] == 1, summary
    assert summary["model_shape_precondition_ready_count"] == 1, summary
    assert summary["runtime_canary_ready_count"] == 0, summary
    assert summary["runtime_canary_hit_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["default_behavior_changed_count"] == 0, summary

    assert case["optimizer_type"] == "Muon", case
    assert case["native_route_family"] == "model_or_shape_aware", case
    assert case["native_kernel_name"] == "muon_flat_fp32_cuda_v0", case
    assert case["model_shape_precondition_ready"] is True, case
    assert case["native_scratch_kernel_ready"] is True, case
    assert case["native_kernel_ready"] is True, case
    assert case["kernel_executed"] is True, case
    assert case["parameters_mutated"] is True, case
    assert case["scratch_buffers_only"] is True, case
    assert case["training_tensor_binding"] is False, case
    assert case["training_dispatch"] is False, case
    assert case["parity_ok"] is True, case
    assert case["training_path_enabled"] is False, case
    assert case["native_dispatch_allowed"] is False, case
    assert case["runtime_dispatch_ready"] is False, case
    assert case["product_native_ready"] is False, case

    return {
        "schema_version": 1,
        "probe": "turbocore_muon_native_scratch_kernel_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
