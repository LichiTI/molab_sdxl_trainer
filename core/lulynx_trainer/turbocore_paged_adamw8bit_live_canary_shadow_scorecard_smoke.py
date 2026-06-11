"""Smoke checks for PagedAdamW8bit live canary shadow manifest."""

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

from core.turbocore_paged_adamw8bit_live_canary_shadow_scorecard import (  # noqa: E402
    build_paged_adamw8bit_live_canary_shadow_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_paged_adamw8bit_live_canary_shadow_scorecard(native_training_mode="canary")
    route = report["route_decision"]
    assert report["ok"] is True, report
    assert report["runtime_canary_shadow_ready"] is True, report
    assert report["runtime_canary_ready"] is False, report
    assert report["runtime_canary_hit"] is False, report
    assert route["decision"] == "blocked_before_training_dispatch", route
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_live_canary_shadow_scorecard_smoke",
        "ok": True,
        "manifest_summary": report["manifest_summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
