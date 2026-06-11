"""Smoke checks for selected plugin Adam-like e2e shadow matrix."""

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

from core.turbocore_plugin_adamlike_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_plugin_adamlike_e2e_shadow_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_adamlike_e2e_shadow_matrix_scorecard(
        include_live_canaries=True,
        write_artifact=True,
    )
    cases = {str(case["selected_optimizer_name"]): case for case in report["matrix_cases"]}
    expected = {
        "adam",
        "adamax",
        "adamc",
        "adamg",
        "adamod",
        "adamp",
        "adams",
        "adamw",
        "adamwsn",
        "dualadam",
        "exadam",
        "fadam",
        "flashadamw",
        "grokfastadamw",
        "lamb",
        "nadam",
        "novograd",
        "padam",
        "qhadam",
        "radam",
        "ranger",
        "ranger21",
        "ranger25",
        "stableadamw",
        "yogi",
    }
    assert report["scorecard"] == "turbocore_plugin_adamlike_e2e_shadow_matrix_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["e2e_shadow_matrix_ready"] is True, report
    assert report["live_shadow_matrix_executed"] is False, report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_training_mutates_authority"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert len(cases) == 25, cases
    assert set(cases) == expected, cases
    assert all(case["shadow_matrix_case_ready"] is True for case in cases.values()), cases
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_adamlike_e2e_shadow_matrix_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
}


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
