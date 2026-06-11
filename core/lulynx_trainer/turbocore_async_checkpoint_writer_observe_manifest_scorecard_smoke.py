"""Smoke checks for P6M async checkpoint writer observe manifest scorecard."""

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


def run_smoke() -> dict[str, Any]:
    writer_report = build_async_checkpoint_writer_scorecard(size_bytes=1024 * 1024)
    report = build_async_checkpoint_writer_observe_manifest_scorecard(
        writer_report=writer_report,
        native_training_mode="observe",
    )
    decision = report["route_decision"]
    manifest = report["manifest"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["observe_manifest_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert decision["decision"] == "would_select_async_checkpoint_writer_observe_but_dispatch_disabled", decision
    assert manifest["candidate_recorded"] is True, manifest
    assert manifest["allowed_initial_modes"] == ["off", "observe"], manifest
    assert "canary" in manifest["blocked_modes_until_review"], manifest
    assert manifest["rollback_policy"]["fallback_authoritative"] is True, manifest
    assert manifest["rollback_policy"]["fallback_writer"] == "standardcore_python_sync_checkpoint_save", manifest
    return {
        "schema_version": 1,
        "probe": "turbocore_async_checkpoint_writer_observe_manifest_scorecard_smoke",
        "ok": True,
        "promotion_ready": report["promotion_ready"],
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
