"""Smoke for fp32 PagedAdamW TrainingLoop native canary."""

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

from core.turbocore_paged_adamw32_training_loop_canary_scorecard import (  # noqa: E402
    build_paged_adamw32_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw32_training_loop_canary_scorecard()
    assert report["scorecard"] == "turbocore_paged_adamw32_training_loop_canary_scorecard_v0", report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["ok"] is True, report
    assert report["summary"]["case_count"] == 2, report
    assert report["summary"]["native_step_count"] == 2, report
    assert report["summary"]["native_kernel_launch_count"] == 2, report
    cases = {str(case["optimizer_kind"]): case for case in report["cases"]}
    for kind in ("paged_adamw", "paged_adamw32bit"):
        case = cases[kind]
        assert case["native_step_executed"] is True, case
        assert case["native_kernel_launched"] is True, case
        assert case["executor_optimizer_kind"] == kind, case
        assert case["state1_dtype"] == "float32", case
        assert case["state2_dtype"] == "float32", case
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw32_training_loop_canary_smoke",
        "ok": True,
        "summary": report["summary"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
