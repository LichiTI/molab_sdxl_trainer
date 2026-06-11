"""Smoke checks for the V3 exact AdamW rollout manifest scorecard."""

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

from core.turbocore_v3_exact_adamw_rollout_manifest_scorecard import (  # noqa: E402
    build_v3_exact_adamw_rollout_manifest_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_exact_adamw_rollout_manifest_scorecard(native_training_mode="canary")
    route = report["route_decision"]
    rollback = report["rollback_policy"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["rollout_manifest_ready"] is True, report
    assert report["requires_explicit_opt_in"] is True, report
    assert report["default_training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["default_dispatch_allowed"] is False, report
    assert report["explicit_canary_allowed"] is True, report
    assert report["auto_rollout_allowed"] is False, report
    assert route["decision"] == "explicit_canary_ready", route
    assert route["request_fields"]["turbocore_native_update_training_path_enabled"] is True, route
    assert rollback["fallback_authoritative"] is True, rollback
    assert rollback["disable_for_run_on_native_error"] is True, rollback
    assert report["blocked_reasons"] == [], report
    return {
        "schema_version": 1,
        "probe": "turbocore_v3_exact_adamw_rollout_manifest_scorecard_smoke",
        "ok": True,
        "manifest_summary": report["manifest_summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
