"""Smoke checks for Automagic++ P31 e2e shadow matrix scaffold."""

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

from core.turbocore_automagicpp_e2e_shadow_matrix_scorecard import (  # noqa: E402
    P23_AUDIT_BUILDER,
    build_automagicpp_e2e_shadow_matrix_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_automagicpp_e2e_shadow_matrix_scorecard()
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["report_only"] is True, report
    assert report["e2e_shadow_matrix_ready"] is True, report
    assert report["report_only_matrix_scaffold_ready"] is True, report
    assert report["live_shadow_matrix_executed"] is False, report
    assert report["native_call_performed_by_p31"] is False, report
    assert report["fallback_backend"] == "python_automagicpp", report
    assert report["fallback_backend_authoritative"] is True, report
    assert report["native_shadow_training_mutates_authority"] is False, report
    assert report["runtime_dispatch_not_enabled"] is True, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["default_behavior_unchanged"] is True, report
    assert report["p23_dependency"]["required_builder"] == P23_AUDIT_BUILDER, report
    assert summary["p23_audit_builder"] == P23_AUDIT_BUILDER, summary
    assert summary["failed_case_count"] == 0, summary
    assert all(case["status"] == "report_only" for case in report["matrix_cases"]), report
    assert all(case["native_call_performed"] is False for case in report["matrix_cases"]), report
    return {
        "schema_version": 1,
        "probe": "turbocore_automagicpp_e2e_shadow_matrix_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
