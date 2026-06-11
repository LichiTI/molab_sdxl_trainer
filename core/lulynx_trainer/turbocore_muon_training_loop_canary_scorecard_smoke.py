"""Smoke checks for Muon TrainingLoop canary scorecard."""

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

from core.turbocore_muon_training_loop_canary_scorecard import (  # noqa: E402
    build_muon_training_loop_canary_scorecard,
)
from core.turbocore_muon_training_tensor_binding_canary_scorecard import (  # noqa: E402
    build_muon_training_tensor_binding_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    binding = build_muon_training_tensor_binding_canary_scorecard(workspace_root=REPO_ROOT, write_artifact=True)
    report = build_muon_training_loop_canary_scorecard(
        training_tensor_binding_report=binding,
        write_artifact=True,
    )
    summary = report["summary"]
    case = report["family_cases"][0]
    row = report["rows"][0]

    assert report["scorecard"] == "turbocore_muon_training_loop_canary_scorecard_v0", report
    assert report["gate"] == "muon_model_shape_aware_training_loop_canary", report
    assert report["ok"] is True, report
    assert report["training_loop_canary_ready"] is True, report
    assert report["training_loop_canary_hit"] is True, report
    assert report["training_tensor_binding_canary_ready"] is True, report
    assert report["training_tensor_binding_parity_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["product_native_ready"] is False, report

    assert case["native_step_executed"] is True, case
    assert case["native_kernel_launched"] is True, case
    assert case["training_parameters_mutated"] is True, case
    assert case["should_call_pytorch_optimizer_step"] is False, case
    assert case["training_executor_called"] is True, case
    assert case["training_executor_ok"] is True, case
    assert case["executor_optimizer_kind"] == "muon", case
    assert "momentum_buffer" in case["state_keys"], case

    assert row["optimizer_type"] == "Muon", row
    assert row["training_loop_canary_ready"] is True, row
    assert row["product_native_ready"] is False, row

    assert summary["optimizer_count"] == 1, summary
    assert summary["training_loop_canary_ready_count"] == 1, summary
    assert summary["native_step_count"] == 1, summary
    assert summary["native_kernel_launch_count"] == 1, summary
    assert summary["training_executor_called_count"] == 1, summary
    assert summary["training_parameters_mutated_count"] == 1, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary

    return {
        "schema_version": 1,
        "probe": "turbocore_muon_training_loop_canary_scorecard_smoke",
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
