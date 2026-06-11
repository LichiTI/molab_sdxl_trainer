"""Smoke checks for selected schedule-free plugin family batch."""

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

from core.turbocore_plugin_schedulefree_family_batch_scorecard import (  # noqa: E402
    build_plugin_schedulefree_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_schedulefree_family_batch_scorecard(write_artifact=True)
    rows = {str(row["stage"]): row for row in report["rows"]}
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_plugin_schedulefree_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_schedulefree_family_batch_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report
    assert summary["selected_optimizer_count"] == 3, summary
    assert summary["ready_stage_count"] == summary["stage_count"], summary
    assert summary["selected_native_canary_ready_count"] == 3, summary
    assert summary["native_ready_count"] == 3, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["e2e_shadow_case_count"] == 6, summary
    assert rows["native_abi_sketch"]["stage_ready"] is True, rows
    assert rows["training_tensor_binding"]["stage_ready"] is True, rows
    assert rows["e2e_shadow_matrix"]["stage_ready"] is True, rows
    assert rows["canary_rollout_policy"]["stage_ready"] is True, rows
    assert rows["dispatch_integration_review"]["stage_ready"] is True, rows
    assert rows["schedulefreeadamw_training_loop_canary"]["stage_ready"] is True, rows
    assert rows["schedulefreesgd_training_loop_canary"]["stage_ready"] is True, rows
    assert rows["schedulefreeradam_training_loop_canary"]["stage_ready"] is True, rows
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_schedulefree_family_batch_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
