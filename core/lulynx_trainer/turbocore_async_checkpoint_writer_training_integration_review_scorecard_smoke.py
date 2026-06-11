"""Smoke checks for P6O async checkpoint writer trainer integration review gate."""

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

from core.turbocore_async_checkpoint_writer_observe_manifest_scorecard import (  # noqa: E402
    build_async_checkpoint_writer_observe_manifest_scorecard,
)
from core.turbocore_async_checkpoint_writer_scorecard import (  # noqa: E402
    build_async_checkpoint_writer_scorecard,
)
from core.turbocore_async_checkpoint_writer_training_integration_review_scorecard import (  # noqa: E402
    build_async_checkpoint_writer_training_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    writer_report = build_async_checkpoint_writer_scorecard(size_bytes=1024 * 1024)
    observe = build_async_checkpoint_writer_observe_manifest_scorecard(
        writer_report=writer_report,
        native_training_mode="observe",
    )
    report = build_async_checkpoint_writer_training_integration_review_scorecard(
        observe_report=observe,
        native_training_mode="observe",
    )
    review = report["review_package"]
    hooks = review["trainer_hook_contract"]
    lifecycle = review["checkpoint_lifecycle_contract"]
    resume = review["resume_parity_matrix"]
    rollback = review["rollback_policy"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["review_gate_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert review["manual_review_required"] is True, review
    assert review["dispatch_review_outcome"] == "pending_manual_review", review
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    assert hooks["save_model_hook"].endswith("Trainer._save_model"), hooks
    assert hooks["save_state_hook"].endswith("Trainer._save_state"), hooks
    assert hooks["load_state_hook"].endswith("Trainer._load_state"), hooks
    assert lifecycle["retention_after_completed_jobs_only"] is True, lifecycle
    assert resume["state_load_uses_safe_torch_load"] is True, resume
    assert rollback["fallback_backend"] == "standardcore_trainer_sync_checkpoint_save", rollback
    assert rollback["fallback_authoritative"] is True, rollback
    return {
        "schema_version": 1,
        "probe": "turbocore_async_checkpoint_writer_training_integration_review_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
