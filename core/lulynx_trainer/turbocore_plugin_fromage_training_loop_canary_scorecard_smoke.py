"""Smoke checks for selected plugin Fromage TrainingLoop native canary."""

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

from core.turbocore_plugin_fromage_training_loop_canary_scorecard import (  # noqa: E402
    build_plugin_fromage_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_fromage_training_loop_canary_scorecard()
    assert report["scorecard"] == "turbocore_plugin_fromage_training_loop_canary_scorecard_v0", report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["selected_optimizer_name"] == "fromage", report
    assert report["optimizer_family"] == "simple_formula", report
    assert report["selected_native_canary_ready"] is True, report
    assert report["ok"] is True, report
    case = report["case"]
    assert case["optimizer_class"] == "Fromage", case
    assert case["native_step_executed"] is True, case
    assert case["native_kernel_launched"] is True, case
    assert case["training_parameters_mutated"] is True, case
    assert case["should_call_pytorch_optimizer_step"] is False, case
    assert case["training_executor_called"] is True, case
    assert case["training_executor_ok"] is True, case
    assert case["executor_result_ok"] is True, case
    assert case["executor_optimizer_kind"] == "fromage", case
    assert case["state_keys"] == [], case
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_fromage_training_loop_canary_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
