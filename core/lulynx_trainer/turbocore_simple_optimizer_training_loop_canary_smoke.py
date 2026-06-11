"""Smoke for simple optimizer TrainingLoop native canary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_training_loop_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_training_loop_canary_scorecard()
    assert report["scorecard"] == "turbocore_simple_optimizer_training_loop_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["summary"]["native_step_count"] == 2, report
    assert report["summary"]["native_kernel_launch_count"] == 2, report
    cases = {str(case["optimizer_kind"]): case for case in report["cases"]}
    assert cases["lion"]["native_step_executed"] is True, cases["lion"]
    assert cases["sgd_nesterov"]["native_step_executed"] is True, cases["sgd_nesterov"]
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_training_loop_canary_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
