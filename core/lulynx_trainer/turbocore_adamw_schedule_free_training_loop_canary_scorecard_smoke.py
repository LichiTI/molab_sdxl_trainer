"""Smoke checks for AdamWScheduleFree TrainingLoop canary manifest."""

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

from core.turbocore_adamw_schedule_free_training_loop_canary_scorecard import (  # noqa: E402
    build_adamw_schedule_free_training_loop_canary_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_adamw_schedule_free_training_loop_canary_scorecard()
    summary = report["summary"]
    blockers = report["promotion_blockers"]
    assert report["scorecard"] == "turbocore_adamw_schedule_free_training_loop_canary_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_loop_canary_manifest_ready"] is True, report
    assert report["training_loop_canary_ready"] is True, report
    assert report["training_loop_canary_hit"] is True, report
    assert report["runtime_canary_manifest_ready"] is True, report
    assert report["runtime_canary_ready"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["training_executor_ready"] is True, report
    assert report["training_tensor_binding_ready"] is True, report
    assert report["native_live_entrypoint_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert summary["formula_canary_ready"] is True, summary
    assert summary["native_scratch_kernel_ready"] is True, summary
    assert summary["runtime_canary_manifest_ready"] is True, summary
    assert summary["training_loop_canary_ready"] is True, summary
    assert summary["native_step_count"] == 1, summary
    assert summary["native_kernel_launch_count"] == 1, summary
    assert summary["primed_pytorch_state"] is True, summary
    assert summary["step_after_native"] == 2, summary
    assert summary["training_executor_ready"] is True, summary
    assert summary["native_live_entrypoint_ready"] is True, summary
    assert summary["training_tensor_binding_ready"] is True, summary
    assert "product_rollout_review_missing" in blockers, blockers
    return {
        "schema_version": 1,
        "probe": "turbocore_adamw_schedule_free_training_loop_canary_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
