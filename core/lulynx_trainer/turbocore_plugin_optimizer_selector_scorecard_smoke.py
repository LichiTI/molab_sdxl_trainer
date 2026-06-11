"""Smoke checks for plugin optimizer selector scorecard."""

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

from core.turbocore_plugin_optimizer_selector_scorecard import (  # noqa: E402
    build_plugin_optimizer_selector_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_optimizer_selector_scorecard()
    rows = {str(row["optimizer_name"]): row for row in report["rows"]}
    families = report["summary"]["route_family_counts"]
    assert report["ok"] is True, report
    assert report["plugin_selector_classification_ready"] is True, report
    assert report["selector_boundary_ready"] is True, report
    assert report["all_discovered_plugins_resume_proven"] is True, report
    assert report["missing_classification_count"] == 0, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["summary"]["plugin_optimizer_count"] >= 100, report
    assert report["summary"]["missing_resume_count"] == 0, report
    assert rows["prodigy"]["native_route_family"] == "adaptive_lr_state_machine", rows["prodigy"]
    assert rows["adafactor"]["native_route_family"] == "factored_memory_layout", rows["adafactor"]
    assert rows["schedulefreeadamw"]["native_route_family"] == "schedule_free_state_machine", rows["schedulefreeadamw"]
    assert rows["lion"]["native_route_family"] == "simple_formula", rows["lion"]
    assert rows["lbfgs"]["native_route_family"] == "closure_or_second_order", rows["lbfgs"]
    assert rows["lomo"]["native_route_family"] == "fused_backward", rows["lomo"]
    assert rows["muon"]["native_route_family"] == "model_or_shape_aware", rows["muon"]
    assert "adam_like_formula" in families, families
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_optimizer_selector_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
