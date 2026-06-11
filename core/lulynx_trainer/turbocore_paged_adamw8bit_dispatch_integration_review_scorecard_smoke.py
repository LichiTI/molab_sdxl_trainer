"""Smoke checks for PagedAdamW8bit dispatch integration review gate."""

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

from core.turbocore_paged_adamw8bit_canary_rollout_policy_scorecard import (  # noqa: E402
    build_paged_adamw8bit_canary_rollout_policy_scorecard,
)
from core.turbocore_paged_adamw8bit_dispatch_integration_review_scorecard import (  # noqa: E402
    build_paged_adamw8bit_dispatch_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    policy = build_paged_adamw8bit_canary_rollout_policy_scorecard(
        run_live_probe=False,
        require_live_matrix=False,
    )
    report = build_paged_adamw8bit_dispatch_integration_review_scorecard(
        rollout_policy_report=policy,
        native_training_mode="observe",
        run_live_probe=False,
        require_live_matrix=False,
    )
    review = report["review_package"]
    hooks = review["runtime_hook_contract"]
    dispatch = review["dispatch_contract"]
    rollback = review["rollback_policy"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["review_gate_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert review["manual_review_required"] is True, review
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    assert hooks["training_loop_step_hook"].endswith("TrainingLoop.train_epoch"), hooks
    assert dispatch["fallback_update_authority"] == "bitsandbytes_paged_adamw8bit", dispatch
    assert dispatch["requires_checkpoint_runtime_adapter"] is True, dispatch
    assert dispatch["requires_bnb_exact_oracle_boundary"] is True, dispatch
    assert review["numeric_guardrails"]["state_uint8_parity_required"] is True, review
    assert rollback["fallback_backend"] == "bitsandbytes_paged_adamw8bit", rollback
    assert rollback["fallback_authoritative"] is True, rollback
    return {
        "schema_version": 1,
        "probe": "turbocore_paged_adamw8bit_dispatch_integration_review_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
