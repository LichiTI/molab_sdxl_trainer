"""Smoke checks for Muon live tensor-binding canary scorecard."""

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

from core.turbocore_muon_native_scratch_kernel_scorecard import (  # noqa: E402
    build_muon_native_scratch_kernel_scorecard,
)
from core.turbocore_muon_training_tensor_binding_canary_scorecard import (  # noqa: E402
    ENTRYPOINT,
    build_muon_training_tensor_binding_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    scratch = build_muon_native_scratch_kernel_scorecard(write_artifact=True)
    report = build_muon_training_tensor_binding_canary_scorecard(
        native_scratch_report=scratch,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    summary = report["summary"]
    live = report["live_probe"]
    case = live["cases"][0]

    assert report["scorecard"] == "turbocore_muon_training_tensor_binding_canary_scorecard_v0", report
    assert report["gate"] == "muon_model_shape_aware_training_tensor_binding_canary", report
    assert report["entrypoint"] == ENTRYPOINT, report
    assert report["ok"] is True, report
    assert report["training_tensor_binding_canary_ready"] is True, report
    assert report["training_tensor_binding_parity_ready"] is True, report
    assert report["native_scratch_kernel_ready"] is True, report
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
    assert summary["training_tensor_binding_canary_ready_count"] == 1, summary
    assert summary["training_tensor_binding_parity_ready_count"] == 1, summary
    assert summary["kernel_executed_count"] == 1, summary
    assert summary["training_parameters_mutated_count"] == 1, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary

    assert live["status"] == "passed", live
    assert live["training_tensor_binding_parity_passed"] is True, live
    assert live["training_dispatch"] is False, live
    assert live["training_path_enabled"] is False, live
    assert live["native_dispatch_allowed"] is False, live
    assert case["kernel_executed"] is True, case
    assert case["native_live_tensor_binding"] is True, case
    assert case["training_tensor_binding"] is True, case
    assert case["training_dispatch"] is False, case
    assert case["training_path_enabled"] is False, case
    assert case["training_parameters_mutated"] is True, case
    assert case["max_abs_diff"] <= case["tolerance"], case

    return {
        "schema_version": 1,
        "probe": "turbocore_muon_training_tensor_binding_canary_scorecard_smoke",
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
