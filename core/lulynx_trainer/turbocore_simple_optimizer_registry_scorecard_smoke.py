"""Smoke checks for V2-P7 simple optimizer native registry scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_registry_scorecard import (  # noqa: E402
    build_simple_optimizer_registry_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_simple_optimizer_registry_scorecard()
    assert report["ok"] is True, report
    assert report["registry_stage_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["summary"]["case_count"] == 2, report
    assert report["summary"]["passed_case_count"] == 2, report
    assert report["summary"]["dry_run_ready_count"] == 2, report
    assert report["summary"]["cpu_reference_guard_ready_count"] == 2, report
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_registry_scorecard_smoke",
        "ok": True,
        "registry_stage_ready": report["registry_stage_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
