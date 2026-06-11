"""Smoke checks for PagedAdamW8bit dispatch selector dry-run."""

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

from core.turbocore_paged_adamw8bit_dispatch_selector_dry_run_scorecard import (  # noqa: E402
    build_paged_adamw8bit_dispatch_selector_dry_run_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_dispatch_selector_dry_run_scorecard(native_training_mode="canary")
    route = report["route_decision"]
    assert report["ok"] is True, report
    assert report["dispatch_selector_dry_run_ready"] is True, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["real_training_matrix_ready"] is False, report
    assert route["decision"] == "blocked_before_real_training_matrix", route
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_dispatch_selector_dry_run_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
