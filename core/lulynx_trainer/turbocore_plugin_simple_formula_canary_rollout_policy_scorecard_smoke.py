"""Smoke checks for selected plugin simple-formula canary rollout policy."""

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

from core.turbocore_plugin_simple_formula_canary_rollout_policy_scorecard import (  # noqa: E402
    build_plugin_simple_formula_canary_rollout_policy_scorecard,
)


EXPECTED = {
    "accsgd",
    "aggmo",
    "asgd",
    "fromage",
    "gravity",
    "lars",
    "lion",
    "madgrad",
    "nero",
    "pid",
    "qhm",
    "rmsprop",
    "sgd",
    "sgdp",
    "sgdw",
    "signsgd",
    "tiger",
    "vsgd",
}


def run_smoke() -> dict[str, Any]:
    report = build_plugin_simple_formula_canary_rollout_policy_scorecard(write_artifact=True)
    policy = report["policy"]
    assert report["scorecard"] == "turbocore_plugin_simple_formula_canary_rollout_policy_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["canary_rollout_policy_ready"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["canary_auto_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert policy["explicit_opt_in_required"] is True, policy
    assert policy["canary_enabled_by_default"] is False, policy
    assert policy["max_canary_fraction_default"] == 0.0, policy
    assert set(policy["selected_optimizer_names"]) == EXPECTED, policy
    assert report["summary"]["canary_rollout_policy_ready_count"] == 18, report
    assert report["summary"]["product_native_ready_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_simple_formula_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
