"""Smoke for the RMSProp branch reference scorecard."""

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

from core.turbocore_optimizer_rmsprop_branch_reference_scorecard import (  # noqa: E402
    build_rmsprop_branch_reference_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_rmsprop_branch_reference_scorecard(write_artifact=True)
    summary = report["summary"]
    ready = set(report["branch_reference_ready_branches"])

    assert report["scorecard"] == "turbocore_optimizer_rmsprop_branch_reference_scorecard_v0", report
    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design_v2.md", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert summary["case_count"] == 4, summary
    assert summary["branch_reference_ready_count"] == 2, summary
    assert summary["rmsprop_centered_reference_ready_count"] == 1, summary
    assert summary["rmsprop_momentum_reference_ready_count"] == 1, summary
    assert ready == {"rmsprop_centered", "rmsprop_momentum"}, ready

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_rmsprop_branch_reference_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
