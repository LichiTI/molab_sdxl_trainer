"""Smoke for the SGDP branch reference scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
PYTORCH_OPTIMIZER_ROOT = REPO_ROOT / "plugin" / "pytorch_optimizer-main"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(PYTORCH_OPTIMIZER_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_sgdp_branch_reference_scorecard import (  # noqa: E402
    build_sgdp_branch_reference_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_sgdp_branch_reference_scorecard(write_artifact=True)
    summary = report["summary"]
    ready = set(report["branch_reference_ready_branches"])

    assert report["scorecard"] == "turbocore_optimizer_sgdp_branch_reference_scorecard_v0", report
    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design_v2.md", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert summary["case_count"] == 3, summary
    assert summary["branch_reference_ready_count"] == 2, summary
    assert summary["sgdp_projection_reference_ready_count"] == 1, summary
    assert summary["sgdp_decoupled_decay_reference_ready_count"] == 1, summary
    assert ready == {"sgdp_projection", "sgdp_decoupled_decay"}, ready

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_sgdp_branch_reference_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
