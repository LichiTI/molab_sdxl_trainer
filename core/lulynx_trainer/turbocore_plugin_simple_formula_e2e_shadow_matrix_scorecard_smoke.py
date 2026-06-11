"""Smoke checks for selected plugin simple-formula e2e shadow matrix."""

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

from core.turbocore_plugin_simple_formula_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_plugin_simple_formula_e2e_shadow_matrix_scorecard,
)


EXPECTED = {
    "accsgd",
    "aggmo",
    "asgd",
    "fromage",
    "gravity",
    "lars",
    "lion",
    "madgrad",
    "nero",
    "pid",
    "qhm",
    "rmsprop",
    "sgd",
    "sgdp",
    "sgdw",
    "signsgd",
    "tiger",
    "vsgd",
}


def run_smoke() -> dict[str, Any]:
    report = build_plugin_simple_formula_e2e_shadow_matrix_scorecard(write_artifact=True)
    cases = {str(case["selected_optimizer_name"]): case for case in report["matrix_cases"]}
    assert report["scorecard"] == "turbocore_plugin_simple_formula_e2e_shadow_matrix_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["e2e_shadow_matrix_ready"] is True, report
    assert report["live_shadow_matrix_executed"] is False, report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_training_mutates_authority"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert set(cases) == EXPECTED, cases
    assert report["summary"]["case_count"] == 18, report
    assert report["summary"]["ready_case_count"] == 18, report
    assert report["summary"]["product_native_ready_count"] == 0, report
    assert all(case["shadow_matrix_case_ready"] is True for case in cases.values()), cases
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_simple_formula_e2e_shadow_matrix_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
