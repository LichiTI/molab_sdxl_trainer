"""Smoke checks for selected plugin simple-formula dispatch integration review."""

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
from core.turbocore_plugin_simple_formula_dispatch_integration_review_scorecard import (  # noqa: E402
    build_plugin_simple_formula_dispatch_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    policy = build_plugin_simple_formula_canary_rollout_policy_scorecard(write_artifact=True)
    report = build_plugin_simple_formula_dispatch_integration_review_scorecard(
        rollout_policy_report=policy,
        native_training_mode="observe",
        write_artifact=True,
    )
    review = report["review_package"]
    hooks = review["runtime_hook_contract"]
    dispatch = review["dispatch_contract"]
    rollback = review["rollback_policy"]
    numeric = review["numeric_guardrails"]
    assert report["scorecard"] == "turbocore_plugin_simple_formula_dispatch_integration_review_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["review_gate_ready"] is True, report
    assert report["dispatch_integration_review"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert review["manual_review_required"] is True, review
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    assert hooks["training_loop_step_hook"].endswith("TrainingLoop.train_epoch"), hooks
    assert dispatch["fallback_update_authority"] == "python_plugin_selected_optimizer", dispatch
    assert dispatch["requires_native_simple_formula_kernel"] is True, dispatch
    assert dispatch["requires_training_tensor_binding"] is True, dispatch
    assert dispatch["requires_owner_release_approval"] is True, dispatch
    assert numeric["state_authority"] == "selected_plugin_until_review", numeric
    assert rollback["fallback_authoritative"] is True, rollback
    assert rollback["rollback_on_selected_plugin_mismatch"] is True, rollback
    assert report["summary"]["optimizer_count"] == 18, report
    assert report["summary"]["product_native_ready_count"] == 0, report
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_simple_formula_dispatch_integration_review_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
