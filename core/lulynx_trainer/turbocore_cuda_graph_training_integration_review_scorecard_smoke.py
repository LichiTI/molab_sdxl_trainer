"""Smoke checks for P6N CUDA graph training integration review gate."""

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

from core.turbocore_cuda_graph_observe_manifest_scorecard import (  # noqa: E402
    build_cuda_graph_observe_manifest_scorecard,
)
from core.turbocore_cuda_graph_training_integration_review_scorecard import (  # noqa: E402
    build_cuda_graph_training_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    observe = build_cuda_graph_observe_manifest_scorecard(native_training_mode="observe")
    report = build_cuda_graph_training_integration_review_scorecard(
        observe_report=observe,
        native_training_mode="observe",
    )
    review = report["review_package"]
    static = review["static_shape_requirements"]
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
    assert static["requires_static_batch"] is True, static
    assert static["requires_static_resolution"] is True, static
    assert static["shape_mismatch_blocked"] is True, static
    assert "dynamic_batch_or_resolution" in review["runtime_incompatibilities"], review
    assert rollback["fallback_backend"] == "standardcore_eager_training_loop", rollback
    assert rollback["fallback_authoritative"] is True, rollback
    return {
        "schema_version": 1,
        "probe": "turbocore_cuda_graph_training_integration_review_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
