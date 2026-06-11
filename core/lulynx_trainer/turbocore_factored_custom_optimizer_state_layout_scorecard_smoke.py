"""Smoke checks for factored/custom optimizer state-layout scorecard."""

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

from core.turbocore_factored_custom_optimizer_state_layout_scorecard import (  # noqa: E402
    FACTORED_CUSTOM_OPTIMIZERS,
    build_factored_custom_optimizer_state_layout_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_factored_custom_optimizer_state_layout_scorecard(write_artifact=True)
    artifact_path = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_factored_custom_optimizer_state_layout_scorecard.json"
    )
    cases = {str(case["case"]): case for case in report["cases"]}
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["ok"] is True, report
    assert report["state_layout_reference_ready"] is True, report
    assert report["factored_custom_family_classified"] is True, report
    assert report["quality_guard_documented"] is True, report
    assert report["adamw_kernel_reuse_blocked"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert len(rows) == len(FACTORED_CUSTOM_OPTIMIZERS), rows
    assert all(row["adamw_kernel_compatible"] is False for row in rows.values()), rows
    automagic = cases["automagic_layout_roundtrip"]
    assert automagic["ok"] is True, automagic
    assert automagic["after_step"]["has_required_state"] is True, automagic
    assert automagic["max_resume_diff"] <= automagic["tolerance"], automagic
    anima = cases["anima_factored_layout_memory"]
    assert anima["ok"] is True, anima
    assert anima["after_step"]["is_factored"] is True, anima
    assert anima["after_step"]["estimated_second_moment_saved_mb"] > 0, anima
    assert anima["max_resume_diff"] <= anima["tolerance"], anima
    guard = cases["adamw_kernel_reuse_guard"]
    assert guard["ok"] is True, guard
    assert artifact_path.exists(), artifact_path
    return {
        "schema_version": 1,
        "probe": "turbocore_factored_custom_optimizer_state_layout_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
