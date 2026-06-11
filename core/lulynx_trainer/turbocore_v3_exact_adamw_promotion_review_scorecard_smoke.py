"""Smoke checks for V3 exact AdamW promotion review gate."""

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

from core.turbocore_v3_exact_adamw_promotion_review_scorecard import (  # noqa: E402
    build_v3_exact_adamw_promotion_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_exact_adamw_promotion_review_scorecard()
    gates = report["progress_gates"]
    review = report["review_package"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["promotion_review_ready"] is True, report
    assert report["default_rollout_allowed"] is False, report
    assert report["auto_rollout_allowed"] is False, report
    assert report["explicit_canary_allowed"] is True, report
    assert gates["p4_config_adapter_complete"] is True, gates
    assert gates["explicit_canary_review_ready"] is True, gates
    assert gates["default_and_auto_blocked"] is True, gates
    assert gates["manual_review_required"] is True, gates
    assert gates["fallback_rollback_ready"] is True, gates
    assert gates["performance_boundary_recorded"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert review["blocked_modes_until_review"] == ["default", "auto"], review
    assert report["blocked_reasons"] == [], report
    return {
        "schema_version": 1,
        "probe": "turbocore_v3_exact_adamw_promotion_review_scorecard_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
