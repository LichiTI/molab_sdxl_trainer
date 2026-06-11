"""Smoke checks for adaptive-LR dispatch integration review evidence."""

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

from core.turbocore_adaptive_lr_canary_rollout_policy_scorecard import (  # noqa: E402
    build_adaptive_lr_canary_rollout_policy_scorecard,
)
from core.turbocore_adaptive_lr_dispatch_integration_review_scorecard import (  # noqa: E402
    EXPECTED_OPTIMIZERS,
    REVIEW_KIND,
    build_adaptive_lr_dispatch_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    rollout = build_adaptive_lr_canary_rollout_policy_scorecard()
    payload = build_adaptive_lr_dispatch_integration_review_scorecard(rollout_policy_report=rollout)
    summary = payload["summary"]
    review = payload["review_package"]

    assert payload["scorecard"] == "turbocore_adaptive_lr_dispatch_integration_review_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["promotion_ready"] is True, payload
    assert payload["review_gate_ready"] is True, payload
    assert payload["dispatch_integration_review"] is True, payload
    assert payload["manual_review_required"] is True, payload
    assert payload["canary_auto_blocked_until_review"] is True, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["product_native_ready"] is False, payload
    assert payload["request_fields_emitted"] is False, payload
    assert payload["schema_exposure_allowed"] is False, payload
    assert payload["ui_exposure_allowed"] is False, payload
    assert payload["review_kind"] == REVIEW_KIND, payload
    assert payload["fallback_backend_authoritative"] is True, payload

    optimizer_types = set(review["optimizer_types"])
    assert optimizer_types == EXPECTED_OPTIMIZERS, review
    assert review["native_training_mode"] == "observe", review
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    assert review["runtime_dispatch_ready"] is False, review
    assert review["native_dispatch_allowed"] is False, review
    assert review["training_path_enabled"] is False, review
    assert review["default_behavior_changed"] is False, review

    hooks = review["runtime_hook_contract"]
    assert hooks["optimizer_create_hook"].endswith("Trainer._create_optimizer"), hooks
    assert hooks["training_loop_step_hook"].endswith("TrainingLoop.train_epoch"), hooks
    assert hooks["adaptive_lr_training_executor"] == "core.turbocore_adaptive_lr_training_executor", hooks
    assert hooks["checkpoint_state_hook"].endswith("get_turbocore_update_checkpoint_state"), hooks
    assert hooks["resume_state_hook"].endswith("load_turbocore_update_checkpoint_state"), hooks

    dispatch = review["dispatch_contract"]
    assert dispatch["requires_adaptive_lr_training_loop_canary"] is True, dispatch
    assert dispatch["requires_e2e_shadow_matrix"] is True, dispatch
    assert dispatch["requires_rollout_policy"] is True, dispatch
    assert dispatch["requires_manual_review_approval"] is True, dispatch
    assert dispatch["runtime_dispatch_enabled_by_this_gate"] is False, dispatch
    assert dispatch["request_schema_ui_enabled_by_this_gate"] is False, dispatch

    rollback = review["rollback_policy"]
    assert rollback["fallback_authoritative"] is True, rollback
    assert rollback["rollback_on_nonfinite"] is True, rollback
    assert rollback["rollback_on_parity_failure"] is True, rollback
    assert rollback["rollback_on_state_machine_guard_failure"] is True, rollback
    assert rollback["rollback_on_dispatch_route_mismatch"] is True, rollback

    assert summary["review_gate_ready"] is True, summary
    assert summary["dispatch_integration_review"] is True, summary
    assert summary["optimizer_count"] == len(EXPECTED_OPTIMIZERS), summary
    assert summary["runtime_dispatch_ready"] is False, summary
    assert summary["native_dispatch_allowed"] is False, summary
    assert summary["training_path_enabled"] is False, summary
    assert summary["product_native_ready_count"] == 0, summary

    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_dispatch_integration_review_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_dispatch_integration_review_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
