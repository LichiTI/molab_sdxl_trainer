"""Smoke checks for selected plugin Adamax TrainingLoop native canary."""

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

from core.turbocore_plugin_adamax_training_loop_canary_scorecard import (  # noqa: E402
    build_plugin_adamax_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adamax_training_loop_canary_scorecard()
    case = report["case"]
    assert report["scorecard"] == "turbocore_plugin_adamax_training_loop_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["state_roles"] == {"exp_avg_sq": "exp_inf"}, report
    assert case["optimizer_class"] == "AdaMax", case
    assert case["native_step_executed"] is True, case
    assert case["native_kernel_launched"] is True, case
    assert case["should_call_pytorch_optimizer_step"] is False, case
    assert case["step_after_native"] == 2, case
    assert case["state_roles"] == {"exp_avg_sq": "exp_inf"}, case
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adamax_training_loop_canary_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
