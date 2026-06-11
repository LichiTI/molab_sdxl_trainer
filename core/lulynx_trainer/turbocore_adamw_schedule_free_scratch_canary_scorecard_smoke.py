"""Smoke checks for AdamWScheduleFree scratch formula canary."""

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

from core.turbocore_adamw_schedule_free_scratch_canary_scorecard import (  # noqa: E402
    build_adamw_schedule_free_scratch_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_schedule_free_scratch_canary_scorecard()
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_adamw_schedule_free_scratch_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["scratch_formula_canary_ready"] is True, report
    assert report["native_ready"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["runtime_canary_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert summary["case_count"] == 2, summary
    assert summary["passed_case_count"] == 2, summary
    assert summary["max_param_diff"] <= 1.0e-6, summary
    assert summary["max_z_diff"] <= 1.0e-6, summary
    assert summary["max_exp_avg_sq_diff"] <= 1.0e-6, summary
    for case in report["cases"]:
        assert case["ok"] is True, case
        assert case["training_path_enabled"] is False, case
        assert case["native_dispatch_allowed"] is False, case
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_schedule_free_scratch_canary_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
