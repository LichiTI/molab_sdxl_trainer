"""Smoke checks for selected plugin Ranger TrainingLoop canary."""

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

from core.turbocore_plugin_ranger_training_loop_canary_scorecard import (  # noqa: E402
    build_plugin_ranger_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_ranger_training_loop_canary_scorecard()
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_plugin_ranger_training_loop_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_native_canary_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["selected_optimizer_name"] == "ranger", report
    assert summary["native_step_count"] == 1, report
    assert summary["native_kernel_launch_count"] == 1, report
    assert summary["step_after_native"] == 2, report
    assert summary["optimizer_class"] == "Ranger", report
    assert summary["executor_optimizer_kind"] == "ranger", report
    assert summary["lookahead_k"] == 2, report
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_ranger_training_loop_canary_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
