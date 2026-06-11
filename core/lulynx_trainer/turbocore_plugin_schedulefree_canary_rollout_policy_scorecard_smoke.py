"""Smoke checks for selected schedule-free canary rollout policy."""

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

from core.turbocore_plugin_schedulefree_canary_rollout_policy_scorecard import (  # noqa: E402
    build_plugin_schedulefree_canary_rollout_policy_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_schedulefree_canary_rollout_policy_scorecard()
    policy = report["policy"]
    assert report["ok"] is True, report
    assert report["canary_rollout_policy_ready"] is True, report
    assert policy["canary_enabled_by_default"] is False, policy
    assert policy["explicit_opt_in_required"] is True, policy
    assert policy["max_canary_fraction_default"] == 0.0, policy
    assert policy["rollback_policy"]["fallback_authoritative"] is True, policy
    assert policy["rollback_policy"]["rollback_on_train_eval_mode_mismatch"] is True, policy
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_schedulefree_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
