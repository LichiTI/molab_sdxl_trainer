"""Smoke checks for V2-P8 AdamW variant state scorecard."""

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

from core.turbocore_adamw_variant_state_scorecard import (  # noqa: E402
    build_adamw_variant_state_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_variant_state_scorecard(run_cuda_optional=False)
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    probes = {str(row["optimizer_type"]): row for row in report["resume_probes"]}
    assert report["ok"] is True, report
    assert report["state_layout_stage_ready"] is True, report
    assert report["resume_matrix_stage_ready"] is True, report
    assert report["memory_speed_matrix_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert rows["AdamW8bit"]["state_schema"]["quantized_state"] is True, rows["AdamW8bit"]
    assert rows["PagedAdamW"]["state_schema"]["paged_state"] is True, rows["PagedAdamW"]
    assert rows["PagedAdamW8bit"]["state_schema"]["quantized_state"] is True, rows["PagedAdamW8bit"]
    assert rows["PagedAdamW8bit"]["state_schema"]["paged_state"] is True, rows["PagedAdamW8bit"]
    assert rows["KahanAdamW8bit"]["state_schema"]["kahan_compensation"] is True, rows["KahanAdamW8bit"]
    assert rows["AdamWScheduleFree"]["state_schema"]["scheduler_coupled"] is True, rows["AdamWScheduleFree"]
    assert probes["KahanAdamW8bit"]["status"] == "passed", probes["KahanAdamW8bit"]
    assert report["summary"]["hard_failure_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_variant_state_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
