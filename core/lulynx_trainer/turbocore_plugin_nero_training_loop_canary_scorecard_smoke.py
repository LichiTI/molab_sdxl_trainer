"""Smoke checks for selected plugin Nero TrainingLoop native canary."""

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

from core.turbocore_plugin_nero_training_loop_canary_scorecard import (  # noqa: E402
    build_plugin_nero_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_nero_training_loop_canary_scorecard()
    assert report["scorecard"] == "turbocore_plugin_nero_training_loop_canary_scorecard_v0", report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["selected_optimizer_name"] == "nero", report
    assert report["selected_native_canary_ready"] is True, report
    case = report["case"]
    assert case["optimizer_class"] == "Nero", case
    assert case["executor_optimizer_kind"] == "nero", case
    for key in ("exp_avg_sq", "scale"):
        assert key in case["state_keys"], case
    return {"schema_version": 1, "probe": "turbocore_plugin_nero_training_loop_canary_scorecard_smoke", "ok": True}


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
