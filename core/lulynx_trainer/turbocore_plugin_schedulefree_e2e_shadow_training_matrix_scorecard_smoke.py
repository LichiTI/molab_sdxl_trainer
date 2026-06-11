"""Smoke checks for schedule-free plugin e2e shadow training matrix."""

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

from core.turbocore_plugin_schedulefree_e2e_shadow_training_matrix_scorecard import (  # noqa: E402
    build_plugin_schedulefree_e2e_shadow_training_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_schedulefree_e2e_shadow_training_matrix_scorecard()
    assert report["ok"] is True, report
    assert report["e2e_shadow_training_matrix_ready"] is True, report
    assert report["e2e_shadow_training_matrix_passed"] is True, report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_updates_original"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    summary = report["summary"]
    assert summary["failed_case_count"] == 0, summary
    assert summary["max_param_diff"] <= 1e-5, summary
    assert summary["max_state_tensor_diff"] <= 1e-5, summary
    assert summary["max_original_shadow_mutation_diff"] <= 1e-5, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_schedulefree_e2e_shadow_training_matrix_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
