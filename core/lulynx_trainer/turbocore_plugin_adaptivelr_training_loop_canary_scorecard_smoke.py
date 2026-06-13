"""Smoke for selected plugin adaptive-LR TrainingLoop native canaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_plugin_adaptivelr_family_batch_scorecard import TARGET_PLUGIN_OPTIMIZERS  # noqa: E402
from core.turbocore_plugin_adaptivelr_training_loop_canary_scorecard import (  # noqa: E402
    ROADMAP,
    build_plugin_adaptivelr_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adaptivelr_training_loop_canary_scorecard(write_artifact=True)
    summary = report["summary"]
    assert report["roadmap"] == ROADMAP, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["ok"] is True, report
    assert summary["selected_optimizer_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["case_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["native_step_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["native_kernel_launch_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["training_executor_called_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    assert summary["skip_pytorch_count"] == len(TARGET_PLUGIN_OPTIMIZERS), report
    cases = {str(case["selected_optimizer_name"]): case for case in report["cases"]}
    assert set(cases) == set(TARGET_PLUGIN_OPTIMIZERS), cases
    assert cases["prodigy"]["native_family_alias"] == "prodigy", cases["prodigy"]
    for name in TARGET_PLUGIN_OPTIMIZERS:
        case = cases[name]
        assert case["native_step_executed"] is True, case
        assert case["native_kernel_launched"] is True, case
        assert case["training_executor_called"] is True, case
        assert case["training_executor_ok"] is True, case
        assert case["should_call_pytorch_optimizer_step"] is False, case
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adaptivelr_training_loop_canary_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
