"""Smoke checks for simple optimizer executor through native dispatch runtime."""

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

from core.turbocore_simple_optimizer_dispatch_runtime_scorecard import (  # noqa: E402
    build_simple_optimizer_dispatch_runtime_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_dispatch_runtime_scorecard(workspace_root=REPO_ROOT)
    assert report["scorecard"] == "turbocore_simple_optimizer_dispatch_runtime_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["dispatch_runtime_stage_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["summary"]["native_step_count"] == 2, report
    assert report["summary"]["skip_pytorch_count"] == 2, report
    for case in report["cases"]:
        assert case["ok"] is True, case
        assert case["native_step_executed"] is True, case
        assert case["should_call_pytorch_optimizer_step"] is False, case
        assert case["training_path_enabled"] is False, case
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_dispatch_runtime_smoke",
        "ok": True,
        "summary": report["summary"],
        "cases": report["cases"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
