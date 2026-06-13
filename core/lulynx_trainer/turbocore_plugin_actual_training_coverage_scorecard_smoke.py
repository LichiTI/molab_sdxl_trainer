"""Smoke for selected plugin actual-training coverage matrix."""

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

from core.turbocore_plugin_actual_training_coverage_scorecard import (  # noqa: E402
    EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT,
    ROADMAP,
    build_plugin_actual_training_coverage_scorecard,
)


EXPECTED_PER_OPTIMIZER_NATIVE_TRAINING_COUNT = 124
EXPECTED_TRAINER_RESUME_PARITY_COUNT = 124


def run_smoke() -> dict[str, Any]:
    report = build_plugin_actual_training_coverage_scorecard(write_artifact=True)
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["evidence_ready"] is True, report
    assert report["roadmap"] == ROADMAP, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["actual_training_complete"] is True, report
    assert summary["selected_plugin_optimizer_count"] == EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT, report
    assert summary["trainer_resume_parity_proven_count"] == EXPECTED_TRAINER_RESUME_PARITY_COUNT, report
    assert summary["per_optimizer_native_training_count"] == EXPECTED_PER_OPTIMIZER_NATIVE_TRAINING_COUNT, report
    assert summary["actual_training_gap_count"] == (
        EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT - EXPECTED_PER_OPTIMIZER_NATIVE_TRAINING_COUNT
    ), report
    assert summary["training_path_enabled_count"] == 0, report
    assert summary["native_dispatch_allowed_count"] == 0, report
    assert summary["product_native_ready_count"] == 0, report
    route_counts = summary["route_family_counts"]
    assert sum(route_counts.values()) == EXPECTED_SELECTED_PLUGIN_OPTIMIZER_COUNT, route_counts
    assert route_counts["simple_formula"] == 18, route_counts
    assert route_counts["adam_like_formula"] == 25, route_counts
    assert route_counts["custom_formula"] == 47, route_counts
    status_counts = summary["actual_training_status_counts"]
    assert status_counts["per_optimizer_native_training"] == EXPECTED_PER_OPTIMIZER_NATIVE_TRAINING_COUNT, status_counts
    assert status_counts.get("runtime_precondition_only", 0) == 0, status_counts
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_actual_training_coverage_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "actual_training_complete": report["actual_training_complete"],
        "recommended_next_step": report["recommended_next_step"],
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
