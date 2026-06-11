"""Smoke checks for Adafactor explicit canary rollout policy scorecard."""

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

from core.turbocore_adafactor_explicit_canary_rollout_policy_scorecard import (  # noqa: E402
    build_adafactor_explicit_canary_rollout_policy_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adafactor_explicit_canary_rollout_policy_scorecard()
    assert report["scorecard"] == "turbocore_adafactor_explicit_canary_rollout_policy_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["report_only"] is True, report
    assert report["explicit_canary_policy_ready"] is True, report
    assert report["canary_auto_enabled"] is False, report
    assert report["manual_review_required"] is True, report
    assert report["fallback_rollback_ready"] is True, report
    assert report["runtime_dispatch_not_enabled"] is True, report
    assert report["default_behavior_unchanged"] is True, report
    return {
        "schema_version": 1,
        "probe": "turbocore_adafactor_explicit_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
