"""Smoke checks for AdamWScheduleFree runtime canary manifest."""

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

from core.turbocore_adamw_schedule_free_runtime_canary_scorecard import (  # noqa: E402
    build_adamw_schedule_free_runtime_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_schedule_free_runtime_canary_scorecard()
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_adamw_schedule_free_runtime_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["runtime_canary_manifest_ready"] is True, report
    assert report["runtime_canary_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert summary["entrypoint_count"] == 1, summary
    assert summary["native_scratch_kernel_ready"] is True, summary
    assert summary["runtime_canary_ready"] is False, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_schedule_free_runtime_canary_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
