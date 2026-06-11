"""Smoke checks for AdamWScheduleFree native scratch-kernel scorecard."""

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

from core.turbocore_adamw_schedule_free_native_scratch_kernel_scorecard import (  # noqa: E402
    ENTRYPOINT,
    KERNEL_NAME,
    build_adamw_schedule_free_native_scratch_kernel_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_schedule_free_native_scratch_kernel_scorecard(workspace_root=REPO_ROOT)
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_adamw_schedule_free_native_scratch_kernel_scorecard_v0", report
    assert report["entrypoint"] == ENTRYPOINT, report
    assert report["kernel_name"] == KERNEL_NAME, report
    assert report["ok"] is True, report
    assert report["native_scratch_kernel_parity_ready"] is True, report
    assert report["native_kernel_ready"] is True, report
    assert report["runtime_canary_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert summary["kernel_executed"] is True, summary
    assert summary["parity_ok"] is True, summary
    assert summary["case_count"] == 2, summary
    assert summary["passed_case_count"] == 2, summary
    assert float(summary["max_abs_diff"]) <= 5.0e-6, summary
    assert summary["runtime_canary_ready"] is False, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_schedule_free_native_scratch_kernel_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
