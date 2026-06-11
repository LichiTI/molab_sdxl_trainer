"""Smoke checks for built-in simple schedule-free rollout policy scorecard."""

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

from core.turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard import (  # noqa: E402
    build_simple_optimizer_schedulefree_rollout_policy_scorecard,
)
from core.turbocore_simple_optimizer_variant_native_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_variant_native_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    variant = build_simple_optimizer_variant_native_canary_scorecard()
    assert variant["variant_schedule_free_native_canary_ready"] is True, variant
    payload = build_simple_optimizer_schedulefree_rollout_policy_scorecard(
        variant_native_canary_report=variant
    )
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    policy = payload["policy"]
    rollback = policy["rollback_policy"]
    assert payload["scorecard"] == "turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["canary_rollout_policy_ready"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["canary_auto_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["default_behavior_changed"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["product_native_dispatch_ready"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert payload["summary"]["optimizer_count"] == 2, payload
    assert payload["summary"]["product_native_ready_count"] == 0, payload
    assert policy["optimizer_types"] == ["RAdamScheduleFree", "SGDScheduleFree"], policy
    assert policy["allowed_initial_modes"] == ["off", "observe"], policy
    assert policy["blocked_modes_until_review"] == ["canary", "auto"], policy
    assert rollback["fallback_authoritative"] is True, rollback
    assert rows["RAdamScheduleFree"]["rollout_status"] == "schedule_free_rollout_policy_ready", rows
    assert rows["SGDScheduleFree"]["rollout_status"] == "schedule_free_rollout_policy_ready", rows
    assert rows["RAdamScheduleFree"]["native_kernel_launch_count"] == 1, rows
    assert rows["SGDScheduleFree"]["native_step_count"] == 1, rows
    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": payload["summary"],
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
