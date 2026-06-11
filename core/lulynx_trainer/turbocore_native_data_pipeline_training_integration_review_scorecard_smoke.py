"""Smoke checks for P6P native data pipeline trainer integration review gate."""

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

from core.turbocore_native_data_pipeline_canary_rollout_policy_scorecard import (  # noqa: E402
    build_native_data_pipeline_canary_rollout_policy_scorecard,
)
from core.turbocore_native_data_pipeline_training_integration_review_scorecard import (  # noqa: E402
    build_native_data_pipeline_training_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    policy = build_native_data_pipeline_canary_rollout_policy_scorecard(native_training_mode="observe")
    report = build_native_data_pipeline_training_integration_review_scorecard(
        policy_report=policy,
        native_training_mode="observe",
    )
    review = report["review_package"]
    hooks = review["trainer_hook_contract"]
    batch = review["batch_semantic_contract"]
    h2d = review["h2d_ownership_contract"]
    cache = review["cache_and_stage_contract"]
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
    assert hooks["training_loop_step_hook"].endswith("TrainingLoop.train_step"), hooks
    assert batch["descriptor_order_parity_required"] is True, batch
    assert batch["filenames_preserved"] is True, batch
    assert h2d["native_pipeline_owns_device_tensor"] is False, h2d
    assert h2d["copy_independent_required"] is True, h2d
    assert cache["staged_resolution_switch_requires_dataloader_rebuild"] is True, cache
    assert rollback["fallback_backend"] == "standardcore_python_dataloader", rollback
    assert rollback["fallback_authoritative"] is True, rollback
    return {
        "schema_version": 1,
        "probe": "turbocore_native_data_pipeline_training_integration_review_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
