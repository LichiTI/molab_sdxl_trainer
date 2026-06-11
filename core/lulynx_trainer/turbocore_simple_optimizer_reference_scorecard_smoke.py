"""Smoke checks for V2-P7 simple optimizer reference scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_reference_scorecard import (  # noqa: E402
    build_simple_optimizer_reference_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_reference_scorecard(dtype_cases=("float32", "float16", "bfloat16"))
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    assert report["ok"] is True, report
    assert report["first_stage_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["summary"]["target_optimizer_count"] == 7, report
    assert report["summary"]["parity_case_count"] == 12, report
    assert report["summary"]["passed_parity_case_count"] == 12, report
    assert rows["Lion"]["reference_status"] == "formula_reference_ready", rows["Lion"]
    assert rows["Lion8bit"]["reference_status"] == "formula_reference_ready_layout_pending", rows["Lion8bit"]
    assert rows["PagedLion8bit"]["reference_status"] == "formula_reference_ready_layout_pending", rows["PagedLion8bit"]
    assert rows["SGDNesterov"]["reference_status"] == "formula_reference_ready", rows["SGDNesterov"]
    assert rows["SGDNesterov8bit"]["reference_status"] == "formula_reference_ready_layout_pending", rows["SGDNesterov8bit"]
    assert rows["RAdamScheduleFree"]["reference_status"] == "state_machine_pending", rows["RAdamScheduleFree"]
    assert rows["SGDScheduleFree"]["reference_status"] == "state_machine_pending", rows["SGDScheduleFree"]
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_reference_scorecard_smoke",
        "ok": True,
        "first_stage_ready": report["first_stage_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
